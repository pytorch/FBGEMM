/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

/**
 * output = f(x, y) where x and y are jagged (and share x_offsets), and output
 * is dense.
 *
 * @param padding_value padding_value for the output, not for inputs
 */
template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
__global__
__launch_bounds__(kMaxThreads) void jagged_jagged_elementwise_dense_output_kernel_(
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    StackArray<index_t*> x_offsets,
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        y_values,
    pta::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> output,
    StackArray<int64_t> jagged_dims,
    F f,
    const scalar_t padding_value) {
  const int outer_dense_size = output.size(0);
  const int jagged_folded_size = output.size(1);
  const int inner_dense_size = output.size(2);

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
        output[oidx][jidx][iidx] = padding_value;
      }
    } else {
      for (int iidx = threadIdx.x; iidx < inner_dense_size;
           iidx += blockDim.x) {
        output[oidx][jidx][iidx] =
            f(x_values[offset][iidx], y_values[offset][iidx]);
      }
    }
  }
}

template <typename scalar_t, typename F>
void jagged_jagged_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_values,
    const Tensor& output,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = output.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (output.numel() == 0) {
    return;
  }

  dim3 threads, blocks;
  StackArray<int64_t> jagged_dims_tensor;
  std::tie(threads, blocks, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, output);

  // Canonicalize output to 3D, collapsing jagged dimensions.
  Tensor output_reshaped = output.view({output.size(0), -1, output.size(-1)});

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                               \
  {                                                                          \
    std::vector<Tensor> x_offsets_contig;                                    \
    x_offsets_contig.resize(num_jagged_dim);                                 \
    StackArray<index_t*> x_offset_ptrs;                                      \
    x_offset_ptrs.ndim = num_jagged_dim;                                     \
    for (int d = 0; d < num_jagged_dim; ++d) {                               \
      x_offsets_contig[d] = x_offsets[d].contiguous();                       \
      x_offset_ptrs.vals[d] =                                                \
          x_offsets_contig[d].template data_ptr<index_t>();                  \
    }                                                                        \
    [[maybe_unused]] const auto func_name =                                  \
        "jagged_jagged_elementwise_dense_output_kernel_";                    \
    jagged_jagged_elementwise_dense_output_kernel_<NUM_JAGGED_DIM, index_t>  \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(          \
            MAKE_PTA_WITH_NAME(func_name, x_values, scalar_t, 2, 32),        \
            x_offset_ptrs,                                                   \
            MAKE_PTA_WITH_NAME(func_name, y_values, scalar_t, 2, 32),        \
            MAKE_PTA_WITH_NAME(func_name, output_reshaped, scalar_t, 3, 32), \
            jagged_dims_tensor,                                              \
            f,                                                               \
            padding_value);                                                  \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#undef INVOKE_KERNEL_WITH_DIM
}

std::tuple<Tensor, Tensor> jagged_dense_elementwise_mul_backward(
    const Tensor& grad_output,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& x_values) {
  CUDA_DEVICE_GUARD(grad_output);

  Tensor x_values_grad = at::empty_like(grad_output);
  Tensor y_grad = at::empty_like(y);

  FBGEMM_DISPATCH_FLOATING_TYPES(x_values.scalar_type(), "jagged_scalars", [&] {
    jagged_dense_elementwise_jagged_output_<scalar_t>(
        grad_output,
        x_offsets,
        y,
        x_values_grad,
        [] __device__(scalar_t x, scalar_t y) -> scalar_t { return x * y; });

    jagged_jagged_elementwise_dense_output_<scalar_t>(
        grad_output,
        x_offsets,
        x_values,
        y_grad,
        [] __device__(scalar_t x, scalar_t y) -> scalar_t { return x * y; });
  });

  return {x_values_grad, y_grad};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_dense_elementwise_mul_backward",
    fbgemm_gpu::jagged_dense_elementwise_mul_backward);
