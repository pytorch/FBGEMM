/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include "cub/device/device_scan.cuh"
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {

/**
 * Ref. http://tensor-compiler.org/kjolstad-oopsla17-tensor-compiler.pdf
 * @param offset the input value points to the offset in the first jagged dim
 *               and output is the final offset to access the value tensor.
 *               It would've been better if we return a pair including this
 *               offset but CUDA doesn't seem to have comprehensive support
 *               on std::pair like std::tie.
 * @returns true if the flattend jagged idx points to zero'ed (masked out)
 *               portion of the jagged tensor
 */
template <int NUM_JAGGED_DIM, typename index_t>
DEVICE_INLINE bool walk_down_tensor_storage_tree_(
    int& offset,
    const int flattened_jagged_idx,
    const int64_t* jagged_dims,
    const std::array<index_t*, NUM_JAGGED_DIM>& x_offsets) {
  // compute coorindates
  int jagged_coords[NUM_JAGGED_DIM];
  int j_temp = flattened_jagged_idx;
#pragma unroll
  for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
    const int jagged_size = jagged_dims[d];
    jagged_coords[d] = j_temp % jagged_size;
    j_temp /= jagged_size;
  }

  // walk down the tree
  bool is_zero = false;
#pragma unroll
  for (int d = 0; d < NUM_JAGGED_DIM; ++d) {
    const int begin = x_offsets[d][offset];
    const int end = x_offsets[d][offset + 1];
    if (jagged_coords[d] >= end - begin) {
      is_zero = true;
      break;
    }
    offset = begin + jagged_coords[d];
  }
  return is_zero;
}

// output = f(x, y) where x is jagged, y is dense, and output is dense.
// A generic elementwise operation between a jagged tensor and a dense tensor
// This kernel assumes jagged dims are clustered together, preceded by outer
// dense dimensions and followed by inner dense dimensions.
// The outer/inner dense dimensions, and jagged dimensions in between are
// assumed to be folded so physically the dense tensor is 3D and the value of
// jagged tensor is 2D.
// To support arbitrary number of jagged dimensions, we pass a vector of
// pointers to offset tensors (this is ugly and probably we can use nested
// tensor here).
// This kernel parallelizes the (folded) inner dense dimension across
// blockDim.x so the inner dense dimension should be similar to or bigger than
// warp size.
// We rely on compiler unrolling the compiler time constant NUM_JAGGED_DIM.
template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
__global__ void jagged_dense_elementwise_dense_output_kernel_(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    const std::array<index_t*, NUM_JAGGED_DIM> x_offsets,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y,
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> output,
    const int64_t* jagged_dims,
    F f,
    const scalar_t padding_value) {
  const int outer_dense_size = y.size(0);
  const int jagged_folded_size = y.size(1);
  const int inner_dense_size = y.size(2);

  const int outer_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int outer_stride = gridDim.x * blockDim.y;
  for (int outer = outer_begin; outer < outer_dense_size * jagged_folded_size;
       outer += outer_stride) {
    const int oidx = outer / jagged_folded_size;
    const int jidx = outer % jagged_folded_size;

    int offset = oidx;
    const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
        offset, jidx, jagged_dims, x_offsets);

    if (is_zero) {
      for (int iidx = threadIdx.x; iidx < inner_dense_size;
           iidx += blockDim.x) {
        output[oidx][jidx][iidx] = f(padding_value, y[oidx][jidx][iidx]);
      }
    } else {
      for (int iidx = threadIdx.x; iidx < inner_dense_size;
           iidx += blockDim.x) {
        output[oidx][jidx][iidx] =
            f(x_values[offset][iidx], y[oidx][jidx][iidx]);
      }
    }
  }
}

std::tuple<dim3, dim3, Tensor> check_shape_and_partition_(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense_tensor) {
  const int outer_dense_size = dense_tensor.size(0);
  TORCH_CHECK(outer_dense_size == offsets[0].numel() - 1);
  const int inner_dense_size = dense_tensor.size(-1);
  TORCH_CHECK(inner_dense_size == values.size(-1));
  const int jagged_folded_size =
      dense_tensor.numel() / (outer_dense_size * inner_dense_size);
  const int jagged_innermost_size = dense_tensor.size(-2);

  const int threads_x = jagged_innermost_size >= kWarpSize / 2
      ? kWarpSize
      : jagged_innermost_size;
  const int threads_y = kMaxThreads / kWarpSize;
  const dim3 blocks(
      div_round_up(outer_dense_size * jagged_folded_size, threads_y));

  const int num_jagged_dim = dense_tensor.dim() - 2;
  const Tensor jagged_dims_tensor =
      at::from_blob(
          const_cast<int64_t*>(dense_tensor.sizes().data() + 1),
          {num_jagged_dim},
          at::kLong)
          .to(offsets[0].device());

  return {dim3(threads_x, threads_y), blocks, jagged_dims_tensor};
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(x_offsets.size() == static_cast<size_t>(num_jagged_dim));

  dim3 threads, blocks;
  Tensor jagged_dims_tensor;
  std::tie(threads, blocks, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, y);

  // Canonicalize y and output to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  Tensor output_reshaped = output.view(y_reshaped.sizes());

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                \
  {                                                                           \
    Tensor x_offsets_contig[num_jagged_dim];                                  \
    std::array<index_t*, NUM_JAGGED_DIM> x_offset_ptrs;                       \
    for (int d = 0; d < num_jagged_dim; ++d) {                                \
      x_offsets_contig[d] = x_offsets[d].contiguous();                        \
      x_offset_ptrs[d] = x_offsets_contig[d].template data_ptr<index_t>();    \
    }                                                                         \
    jagged_dense_elementwise_dense_output_kernel_<NUM_JAGGED_DIM, index_t>    \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(           \
            x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
            x_offset_ptrs,                                                    \
            y_reshaped                                                        \
                .packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),     \
            output_reshaped                                                   \
                .packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),     \
            jagged_dims_tensor.data_ptr<int64_t>(),                           \
            f,                                                                \
            padding_value);                                                   \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#undef INVOKE_KERNEL_WITH_DIM
}

// Almost identical copy of jagged_to_padded_dense in jagged_tensor_ops_cpu.cpp
Tensor jagged_to_padded_dense(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const std::vector<int64_t>& max_lengths,
    const int64_t padding_value) {
  const size_t num_jagged_dim = offsets.size();
  TORCH_CHECK(max_lengths.size() == num_jagged_dim);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const Tensor values_canonicalized = values.view(
      {values.size(0),
       std::accumulate(
           values.sizes().begin() + 1,
           values.sizes().end(),
           1,
           std::multiplies<size_t>())});
  at::DimVector padded_values_shape({offsets[0].size(0) - 1});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());
  if (values.dim() > 1) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values = at::empty(padded_values_shape, values.options());
  Tensor padded_values_view =
      values.dim() == 1 ? padded_values.unsqueeze(-1) : padded_values;

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half,
      values.scalar_type(),
      "jagged_to_padded_dense",
      [&] {
        jagged_dense_elementwise_dense_output_<scalar_t>(
            values_canonicalized,
            offsets,
            padded_values_view, // dummy not used in the lambda function
            padded_values_view,
            [] __device__(scalar_t x, scalar_t /*unused*/) -> scalar_t {
              return x;
            },
            static_cast<scalar_t>(padding_value));
      });

  return padded_values;
}

template <typename scalar_t, typename F>
Tensor jagged_dense_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
  Tensor output = at::empty_like(y);
  jagged_dense_elementwise_dense_output_(
      x_values, x_offsets, y, output, f, padding_value);
  return output;
}

// output = x + y where x is jagged, y and output are dense
Tensor jagged_dense_elementwise_add(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(x_values.get_device());

  Tensor output;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x_values.scalar_type(), "jagged_scalars", [&] {
        output = jagged_dense_elementwise_dense_output_<scalar_t>(
            x_values,
            x_offsets,
            y,
            [] __device__(scalar_t x, scalar_t y) -> scalar_t {
              return x + y;
            });
      });
  return output;
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
__global__ void jagged_dense_elementwise_jagged_output_kernel_(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    const std::array<index_t*, NUM_JAGGED_DIM> x_offsets,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        output_values,
    const int64_t* jagged_dims,
    F f) {
  const int outer_dense_size = y.size(0);
  const int jagged_folded_size = y.size(1);
  const int inner_dense_size = y.size(2);

  const int outer_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int outer_stride = gridDim.x * blockDim.y;
  for (int outer = outer_begin; outer < outer_dense_size * jagged_folded_size;
       outer += outer_stride) {
    const int oidx = outer / jagged_folded_size;
    const int jidx = outer % jagged_folded_size;

    int offset = oidx;
    const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
        offset, jidx, jagged_dims, x_offsets);

    if (!is_zero) {
      for (int iidx = threadIdx.x; iidx < inner_dense_size;
           iidx += blockDim.x) {
        output_values[offset][iidx] =
            f(x_values[offset][iidx], y[oidx][jidx][iidx]);
      }
    }
  }
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(x_offsets.size() == static_cast<size_t>(num_jagged_dim));

  dim3 threads, blocks;
  Tensor jagged_dims_tensor;
  std::tie(threads, blocks, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, y);

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                \
  {                                                                           \
    Tensor x_offsets_contig[num_jagged_dim];                                  \
    std::array<index_t*, NUM_JAGGED_DIM> x_offset_ptrs;                       \
    for (int d = 0; d < num_jagged_dim; ++d) {                                \
      x_offsets_contig[d] = x_offsets[d].contiguous();                        \
      x_offset_ptrs[d] = x_offsets_contig[d].template data_ptr<index_t>();    \
    }                                                                         \
    jagged_dense_elementwise_jagged_output_kernel_<NUM_JAGGED_DIM, index_t>   \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(           \
            x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
            x_offset_ptrs,                                                    \
            y_reshaped                                                        \
                .packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),     \
            output_values                                                     \
                .packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),     \
            jagged_dims_tensor.data_ptr<int64_t>(),                           \
            f);                                                               \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#undef INVOKE_KERNEL_WITH_DIM
}

template <typename scalar_t, typename F>
Tensor jagged_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    F f) {
  Tensor output = at::empty_like(x_values);
  jagged_dense_elementwise_jagged_output_<scalar_t>(
      x_values, x_offsets, y, output, f);
  return output;
}

Tensor jagged_dense_elementwise_mul(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(x_values.get_device());

  Tensor output;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x_values.scalar_type(), "jagged_scalars", [&] {
        output = jagged_dense_elementwise_jagged_output_<scalar_t>(
            x_values,
            x_offsets,
            y,
            [] __device__(scalar_t x, scalar_t y) -> scalar_t {
              return x * y;
            });
      });

  return output;
}

} // namespace

Tensor
jagged_2d_to_dense_forward_cuda(Tensor values, Tensor offsets, int32_t max_L) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  return jagged_to_padded_dense(values, {offsets}, {max_L}, 0);
}

Tensor jagged_2d_to_dense_backward_cuda(
    Tensor grad_padded_values,
    Tensor offsets,
    int32_t total_L) {
  TORCH_CHECK(grad_padded_values.dim() == 3);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(total_L >= 0);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_padded_values.get_device());

  int32_t D = grad_padded_values.size(2);
  auto grad_values = at::zeros({total_L, D}, grad_padded_values.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_padded_values.scalar_type(),
      "jagged_2d_to_dense_backward_kernel",
      [&] {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            grad_values, // dummy not used in the lambda function
            {offsets},
            grad_padded_values,
            grad_values,
            [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
              return y;
            });
      });

  return grad_values;
}

Tensor jagged_1d_to_dense_gpu(
    Tensor values,
    Tensor offsets,
    int64_t max_L,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  return jagged_to_padded_dense(values, {offsets}, {max_L}, padding_value);
}

// stacked ops
std::tuple<std::vector<Tensor>, std::vector<Tensor>>
stacked_jagged_2d_to_dense_forward_cuda(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(lengths.dim() == 2);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const auto lengths_contig = lengths.contiguous();
  int32_t D = values.size(1);
  int32_t B = lengths.size(1);
  int32_t T = lengths.size(0);
  std::vector<Tensor> padded_values_per_key;
  std::vector<Tensor> offsets_tensor_per_key;
  for (int32_t t = 0; t < T; t++) {
    int64_t max_L = max_lengths_per_key[t];
    size_t temp_storage_bytes = 0;
    auto offsets = at::empty({B + 1}, lengths.options());
    offsets[0].zero_();
    AT_DISPATCH_INTEGRAL_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              nullptr,
              temp_storage_bytes,
              &(lengths_contig.data_ptr<scalar_t>()[t * B]),
              offsets.data_ptr<scalar_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        });
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        lengths.options().dtype(at::kByte));
    AT_DISPATCH_INTEGRAL_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper2", [&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              temp_storage.data_ptr(),
              temp_storage_bytes,
              &(lengths_contig.data_ptr<scalar_t>()[t * B]),
              offsets.data_ptr<scalar_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        });
    offsets_tensor_per_key.push_back(offsets);

    padded_values_per_key.push_back(jagged_2d_to_dense_forward_cuda(
        values.slice(0, offset_per_key[t], offset_per_key[t + 1]),
        offsets,
        max_L));
  }

  return std::make_tuple(padded_values_per_key, offsets_tensor_per_key);
}

Tensor stacked_jagged_2d_to_dense_backward_cuda(
    int64_t B,
    int64_t D,
    int64_t total_L,
    const std::vector<Tensor>& grad_padded_values_per_key,
    const std::vector<Tensor>& offsets_tensor_per_key,
    const std::vector<int64_t>& offset_per_key) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_padded_values_per_key[0].get_device());

  auto grad_values =
      at::zeros({total_L, D}, grad_padded_values_per_key[0].options());
  int32_t T = grad_padded_values_per_key.size();
  for (int32_t t = 0; t < T; t++) {
    TORCH_CHECK(grad_padded_values_per_key[t].dim() == 3);
    TORCH_CHECK(grad_padded_values_per_key[t].size(0) == B);
    TORCH_CHECK(grad_padded_values_per_key[t].size(2) == D);

    Tensor grad_values_slice =
        grad_values.slice(0, offset_per_key[t], offset_per_key[t + 1]);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_values.scalar_type(), "jagged_2d_to_dense_backward_kernel", [&] {
          jagged_dense_elementwise_jagged_output_<scalar_t>(
              grad_values_slice, // dummy not used in the lambda function
              {offsets_tensor_per_key[t]},
              grad_padded_values_per_key[t],
              grad_values_slice,
              [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
                return y;
              });
        });
  }

  return grad_values;
}

std::vector<Tensor> stacked_jagged_1d_to_dense_gpu(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(lengths.dim() == 2);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const auto lengths_contig = lengths.contiguous();
  int32_t B = lengths.size(1);
  int32_t T = lengths.size(0);
  auto offsets = at::empty({B + 1}, lengths.options());
  offsets[0].zero_();
  std::vector<Tensor> padded_values_per_key;
  for (int32_t t = 0; t < T; t++) {
    int64_t max_L = max_lengths_per_key[t];
    size_t temp_storage_bytes = 0;
    AT_DISPATCH_INTEGRAL_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              nullptr,
              temp_storage_bytes,
              &(lengths_contig.data_ptr<scalar_t>()[t * B]),
              offsets.data_ptr<scalar_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        });
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        lengths.options().dtype(at::kByte));
    AT_DISPATCH_INTEGRAL_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper2", [&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              temp_storage.data_ptr(),
              temp_storage_bytes,
              &(lengths_contig.data_ptr<scalar_t>()[t * B]),
              offsets.data_ptr<scalar_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        });

    padded_values_per_key.push_back(jagged_1d_to_dense_gpu(
        values.slice(0, offset_per_key[t], offset_per_key[t + 1]),
        offsets,
        max_L,
        padding_value));
  }

  return padded_values_per_key;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA(
      "jagged_to_padded_dense", fbgemm_gpu::jagged_to_padded_dense);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_add", fbgemm_gpu::jagged_dense_elementwise_add);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_mul", fbgemm_gpu::jagged_dense_elementwise_mul);
}
