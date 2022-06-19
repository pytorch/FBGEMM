/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include "cub/device/device_scan.cuh"
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/layout_transform_ops.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "ATen/Parallel.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor recat_embedding_grad_output_cuda(
    Tensor grad_output, // [B_local][T_global][D]
    const std::vector<int64_t>& num_features_per_rank) {
  TENSOR_ON_CUDA_GPU(grad_output);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  TORCH_CHECK(grad_output.is_contiguous());
  const auto B_local = grad_output.size(0);
  const auto T_global = grad_output.size(1);
  const auto D = grad_output.size(2);

  Tensor sharded_grad_output =
      at::empty({grad_output.numel()}, grad_output.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "recat_embedding_gradients", [&] {
        const auto go = grad_output.accessor<scalar_t, 3>();
        auto sgo = sharded_grad_output.accessor<scalar_t, 1>();
        int64_t feature_offset = 0;
        int64_t sgo_offset = 0;
        for (auto num_features : num_features_per_rank) {
          if (num_features == 0) {
            continue;
          }
          AT_CUDA_CHECK(cudaMemcpy2DAsync(
              &sgo[sgo_offset],
              num_features * D * sizeof(scalar_t),
              &go[0][feature_offset][0],
              T_global * D * sizeof(scalar_t),
              num_features * D * sizeof(scalar_t),
              B_local,
              cudaMemcpyDeviceToDevice,
              at::cuda::getCurrentCUDAStream()));
          feature_offset += num_features;
          sgo_offset += B_local * num_features * D;
        }
        TORCH_CHECK(sgo_offset == grad_output.numel());
        TORCH_CHECK(feature_offset == T_global);
      });
  return sharded_grad_output;
}

Tensor recat_embedding_grad_output_mixed_D_cuda(
    const Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank) {
  TENSOR_ON_CUDA_GPU(grad_output);
  TORCH_CHECK(grad_output.is_contiguous());

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const auto B_local = grad_output.size(0);
  const auto global_dim_sum = at::sum_integers(dim_sum_per_rank);

  Tensor sharded_grad_output =
      at::empty({grad_output.numel()}, grad_output.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "recat_embedding_gradients", [&] {
        const auto go = grad_output.accessor<scalar_t, 2>();
        auto sgo = sharded_grad_output.accessor<scalar_t, 1>();
        int64_t sgo_offset = 0;
        int64_t accum_dim_sum = 0;
        for (auto dim_sum : dim_sum_per_rank) {
          if (dim_sum == 0) {
            continue;
          }
          AT_CUDA_CHECK(cudaMemcpy2DAsync(
              &sgo[sgo_offset],
              dim_sum * sizeof(scalar_t),
              &go[0][accum_dim_sum],
              global_dim_sum * sizeof(scalar_t),
              dim_sum * sizeof(scalar_t),
              B_local,
              cudaMemcpyDeviceToDevice,
              at::cuda::getCurrentCUDAStream()));
          sgo_offset += B_local * dim_sum;
          accum_dim_sum += dim_sum;
        }
        TORCH_CHECK(sgo_offset == grad_output.numel());
        TORCH_CHECK(accum_dim_sum == global_dim_sum);
      });

  return sharded_grad_output;
}

Tensor recat_embedding_grad_output_mixed_D_batch_cuda(
    const Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const Tensor& dim_sum_per_rank,
    const Tensor& cumsum_dim_sum_per_rank) {
  TENSOR_ON_CUDA_GPU(grad_output);
  TENSOR_ON_CUDA_GPU(dim_sum_per_rank);
  TENSOR_ON_CUDA_GPU(cumsum_dim_sum_per_rank);
  TORCH_CHECK(grad_output.is_contiguous());

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const auto B_local = grad_output.size(0);
  Tensor sharded_grad_output =
      at::empty({grad_output.numel()}, grad_output.options());
  const auto dim_num = dim_sum_per_rank.size(0);
  const auto dim_sum = grad_output.size(1);

  const dim3 threads(
      fbgemm_gpu::kWarpSize, fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize);
  const dim3 blocks(fbgemm_gpu::div_round_up(
      (B_local * dim_num), fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "recat_embedding_gradients", [&] {
        recat_copy_async_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dim_sum_per_rank.data_ptr<int64_t>(),
                cumsum_dim_sum_per_rank.data_ptr<int64_t>(),
                grad_output.data_ptr<scalar_t>(),
                sharded_grad_output.data_ptr<scalar_t>(),
                dim_num,
                B_local,
                dim_sum);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return sharded_grad_output;
}

namespace {
template <typename scalar_t>
__global__ void multi_dim_split_kernel_(
    const scalar_t* __restrict__ ten_ptr,
    scalar_t* __restrict__ out_ptr,
    const int num_dims,
    const StackArray<int64_t> ten_sizes,
    const StackArray<int64_t> ten_strides,
    const StackArray<int64_t> num_splits) {
  const int split_idx = blockIdx.y;

  SharedMemory<int64_t> smem;

  int64_t* offsets = smem.getPointer();
  int64_t* out_tensor_sizes = smem.getPointer() + num_dims;
  int64_t* out_offset_ptr = smem.getPointer() + 2 * num_dims;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int temp_split_idx = split_idx;
    int64_t out_offset = 0;
    int numel = 1;
    for (int d = num_dims - 1; d >= 0; --d) {
      const auto split_coord = temp_split_idx % num_splits.vals[d];
      temp_split_idx /= num_splits.vals[d];

      const size_t split_size =
          div_round_up(ten_sizes.vals[d], num_splits.vals[d]);
      out_tensor_sizes[d] =
          std::min(split_size, ten_sizes.vals[d] - split_coord * split_size);

      offsets[d] = split_coord * split_size;

      out_offset = out_offset * out_tensor_sizes[d] + offsets[d] * numel;
      numel *= ten_sizes.vals[d];
    }
    *out_offset_ptr = out_offset;
  }
  __syncthreads();

  int out_numel = 1;
  for (int d = 0; d < num_dims; ++d) {
    out_numel *= out_tensor_sizes[d];
  }

  ten_ptr += offsets[num_dims - 1] * ten_strides.vals[num_dims - 1];
  out_ptr += *out_offset_ptr;

  const int i_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int i_stride = gridDim.x * blockDim.y;

  for (int i = i_start; i < out_numel / out_tensor_sizes[num_dims - 1];
       i += i_stride) {
    // Compute coordinate within the split.
    const scalar_t* ten_ptr_temp = ten_ptr;
    auto temp_i = i;
    for (int d = num_dims - 2; d >= 0; --d) {
      auto coord = temp_i % out_tensor_sizes[d];
      temp_i /= out_tensor_sizes[d];

      ten_ptr_temp += (offsets[d] + coord) * ten_strides.vals[d];
    }

    for (int j = threadIdx.x; j < out_tensor_sizes[num_dims - 1];
         j += blockDim.x) {
      out_ptr[i * out_tensor_sizes[num_dims - 1] + j] = ten_ptr_temp[j];
    }
  }
}

std::vector<Tensor> multi_dim_split_gpu(
    const Tensor& ten,
    const std::vector<int64_t>& splits) {
  TENSOR_ON_CUDA_GPU(ten);

  const int num_dims = splits.size();
  TORCH_CHECK(num_dims <= kStackArrayMaxDims);

  StackArray<int64_t> ten_sizes, ten_strides, num_splits;
  ten_sizes.ndim = num_dims;
  ten_strides.ndim = num_dims;
  num_splits.ndim = num_dims;
  for (const auto d : c10::irange(num_dims)) {
    ten_sizes.vals[d] = ten.size(d);
    ten_strides.vals[d] = ten.stride(d);
    num_splits.vals[d] = div_round_up(ten.size(d), splits[d]);
  }
  const int num_total_splits = std::accumulate(
      num_splits.vals, num_splits.vals + num_dims, 1, std::multiplies<int>());
  const auto splits_prod = std::accumulate(
      splits.begin(), splits.end(), 1, std::multiplies<int64_t>());

  std::vector<Tensor> out_tensors;
  out_tensors.reserve(num_dims);
  // To reduce allocation overhead, allocate one big tensor and slice.
  Tensor out_tensor_container = at::empty({ten.numel()}, ten.options());

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      ten.scalar_type(),
      "multi_dim_split",
      [&] {
        dim3 threads(kWarpSize, kMaxThreads / kWarpSize);
        dim3 blocks(div_round_up(splits_prod, threads.y), num_total_splits);
        multi_dim_split_kernel_<scalar_t>
            <<<blocks,
               threads,
               (2 * num_dims + 1) * sizeof(int64_t),
               at::cuda::getCurrentCUDAStream()>>>(
                ten.data_ptr<scalar_t>(),
                out_tensor_container.data_ptr<scalar_t>(),
                num_dims,
                ten_sizes,
                ten_strides,
                num_splits);
      });

  std::vector<int64_t> out_tensor_sizes(num_dims);
  std::vector<int64_t> offsets(num_dims);

  for (const auto split_idx : c10::irange(num_total_splits)) {
    // Compute offset and size of the split.
    // This is intentional redundant of what each CUDA thread block so CPU and
    // GPU can work simultaneously.
    auto temp_split_idx = split_idx;
    int64_t out_offset = 0;
    int numel = 1;
    for (int d = num_dims - 1; d >= 0; --d) {
      const auto split_coord = temp_split_idx % num_splits.vals[d];
      temp_split_idx /= num_splits.vals[d];

      out_tensor_sizes[d] =
          std::min(splits[d], ten.size(d) - split_coord * splits[d]);

      offsets[d] = split_coord * splits[d];

      out_offset = out_offset * out_tensor_sizes[d] + offsets[d] * numel;
      numel *= ten.size(d);
    }

    const auto out_numel = std::accumulate(
        out_tensor_sizes.begin(),
        out_tensor_sizes.end(),
        1,
        std::multiplies<int64_t>());

    Tensor out_tensor =
        out_tensor_container.slice(0, out_offset, out_offset + out_numel)
            .view(out_tensor_sizes);

    out_tensors.push_back(out_tensor);
  }

  return out_tensors;
}

template <typename scalar_t>
__global__ void multi_dim_cat_kernel_(
    const scalar_t** __restrict__ ten_ptrs,
    scalar_t* __restrict__ out_ptr,
    const int num_dims,
    const StackArray<int64_t> out_tensor_sizes,
    const StackArray<int64_t> out_tensor_strides,
    const StackArray<int64_t> num_splits) {
  const int split_idx = blockIdx.y;

  SharedMemory<int64_t> smem;

  int64_t* offsets = smem.getPointer();
  int64_t* ten_sizes = smem.getPointer() + num_dims;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int temp_split_idx = split_idx;
    for (int d = num_dims - 1; d >= 0; --d) {
      const auto split_coord = temp_split_idx % num_splits.vals[d];
      temp_split_idx /= num_splits.vals[d];

      const size_t split_size =
          div_round_up(out_tensor_sizes.vals[d], num_splits.vals[d]);
      ten_sizes[d] = std::min(
          split_size, out_tensor_sizes.vals[d] - split_coord * split_size);

      offsets[d] = split_coord * split_size;
    }
  }
  __syncthreads();

  int out_numel = 1;
  for (int d = 0; d < num_dims; ++d) {
    out_numel *= out_tensor_sizes.vals[d];
  }

  const scalar_t* ten_ptr = ten_ptrs[split_idx];
  out_ptr += offsets[num_dims - 1] * out_tensor_strides.vals[num_dims - 1];

  const int i_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int i_stride = gridDim.x * blockDim.y;

  for (int i = i_start; i < out_numel / out_tensor_sizes.vals[num_dims - 1];
       i += i_stride) {
    // Compute coordinate within the split.
    scalar_t* out_ptr_temp = out_ptr;
    auto temp_i = i;
    for (int d = num_dims - 2; d >= 0; --d) {
      auto coord = temp_i % ten_sizes[d];
      temp_i /= ten_sizes[d];

      out_ptr_temp += (offsets[d] + coord) * out_tensor_strides.vals[d];
    }

    for (int j = threadIdx.x; j < out_tensor_sizes.vals[num_dims - 1];
         j += blockDim.x) {
      out_ptr_temp[j] = ten_ptr[i * ten_sizes[num_dims - 1] + j];
    }
  }
}

Tensor multi_dim_cat_gpu(
    const std::vector<Tensor>& tens,
    const std::vector<int64_t>& num_splits) {
  const size_t num_total_splits = std::accumulate(
      num_splits.begin(), num_splits.end(), 1, std::multiplies<int64_t>());
  TORCH_CHECK(tens.size() == num_total_splits);

  const int num_dims = num_splits.size();
  TORCH_CHECK(num_dims <= kStackArrayMaxDims);
  int multiplier = 1;
  std::vector<int64_t> out_tensor_sizes;
  for (int d = num_dims - 1; d >= 0; --d) {
    int sum = 0;
    for (int i = 0; i < num_splits[d]; ++i) {
      sum += tens[i * multiplier].size(d);
    }
    out_tensor_sizes[d] = sum;
    multiplier *= num_splits[d];
  }

  TORCH_CHECK(tens.size() > 0);
  Tensor out_tensor = at::empty(out_tensor_sizes, tens[0].options());

  StackArray<int64_t> out_tensor_sizes_stack_array,
      out_tensor_strides_stack_array, num_splits_stack_array;
  out_tensor_sizes_stack_array.ndim = num_dims;
  out_tensor_strides_stack_array.ndim = num_dims;
  num_splits_stack_array.ndim = num_dims;
  for (const auto d : c10::irange(num_dims)) {
    out_tensor_sizes_stack_array.vals[d] = out_tensor_sizes[d];
    out_tensor_strides_stack_array.vals[d] = out_tensor.stride(d);
    num_splits_stack_array.vals[d] = num_splits[d];
  }

  Tensor tens_ptr_tensor = at::empty(
      {static_cast<int64_t>(num_total_splits)},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out_tensor.scalar_type(),
      "multi_dim_split",
      [&] {
        const scalar_t** tens_ptr_tensor_data =
            reinterpret_cast<const scalar_t**>(
                tens_ptr_tensor.data_ptr<int64_t>());
        for (auto i : c10::irange(num_total_splits)) {
          TENSOR_ON_CUDA_GPU(tens[i]);
          tens_ptr_tensor_data[i] =
              tens[i].expect_contiguous()->data_ptr<scalar_t>();
        }
        tens_ptr_tensor =
            tens_ptr_tensor.to(out_tensor.device(), /*non_blocking=*/true);

        const dim3 threads(kWarpSize, kMaxThreads / kWarpSize);
        const dim3 blocks(
            div_round_up(tens[0].numel(), threads.y), num_total_splits);
        multi_dim_cat_kernel_<scalar_t>
            <<<blocks,
               threads,
               2 * num_dims * sizeof(int64_t),
               at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t**>(
                    tens_ptr_tensor.data_ptr<int64_t>()),
                out_tensor.data_ptr<scalar_t>(),
                num_dims,
                out_tensor_sizes_stack_array,
                out_tensor_strides_stack_array,
                num_splits_stack_array);
      });

  return out_tensor;
}
} // namespace
} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA("multi_dim_split", fbgemm_gpu::multi_dim_split_gpu);
  DISPATCH_TO_CUDA("multi_dim_cat", fbgemm_gpu::multi_dim_cat_gpu);
}
