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

template <const int THREADS_PER_BLOCK, typename index_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void jagged_softmax_backward_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2> grad_output,
    const at::PackedTensorAccessor32<scalar_t, 2> output,
    const at::PackedTensorAccessor32<index_t, 1> offsets,
    at::PackedTensorAccessor32<scalar_t, 2> grad_input,
    const int max_L) {
  const auto B = offsets.size(0) - 1;
  const auto D = grad_output.size(1);

  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<scalar_t, THREADS_PER_BLOCK> BlockReduceT;

  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  __shared__ scalar_t sum_value;

  const auto tid = threadIdx.x;
  for (uint32_t b = blockIdx.y; b < B; b += gridDim.y) {
    const index_t row_start = offsets[b];
    const index_t row_end = offsets[b + 1];
    const auto length = min(row_end - row_start, (index_t)max_L);

    if (length > 0) {
      const auto num_l_blocks =
          (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

      for (uint32_t d = blockIdx.x; d < D; d += gridDim.x) {
        if (tid == 0) {
          sum_value = 0;
        }

        // Loop through all blocks to calculate the sum value
        // Each block has its own sum, and sum_value is the sum value across all
        // blocks
        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          const auto l = bk_l * blockDim.x + tid;
          scalar_t thread_val = 0;
          if (l < length) {
            thread_val =
                grad_output[row_start + l][d] * output[row_start + l][d];
          }

          // Collectively compute the block-wide sum reduction
          scalar_t block_sum_value = BlockReduceT(temp_storage).Sum(thread_val);
          __syncthreads();

          if (tid == 0) {
            sum_value += block_sum_value;
          }
        }

        // The sum_value was updated by thread 0 in the last loop, sync here to
        // make sure the next loop uses the updated sum_value
        __syncthreads();

        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          const auto l = bk_l * blockDim.x + tid;
          if (l < length) {
            grad_input[row_start + l][d] =
                (grad_output[row_start + l][d] - sum_value) *
                output[row_start + l][d];
          }
        }

        // The sum_value will be reinitialized by thread 0 in the
        // next d iteration, sync here to make sure the last loop still uses the
        // reduced value before reinitialization
        __syncthreads();
      }
    }
  }
}

Tensor jagged_softmax_backward_cuda(
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& offsets,
    const int64_t max_L) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(grad_output, output, offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const auto B = offsets.numel() - 1;
  const auto D = grad_output.size(1);
  auto grad_input = at::empty_like(grad_output);

  if (B > 0 && D > 0) {
    constexpr int THREADS_PER_BLOCK = 128;
    const dim3 grid(D, std::min((int32_t)B, (int32_t)kMaxBlockYDim), 1);

    AT_DISPATCH_INDEX_TYPES(
        offsets.scalar_type(), "jagged_softmax_backward_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              grad_output.scalar_type(),
              "jagged_softmax_backward_kernel_2",
              [&] {
                jagged_softmax_backward_kernel<
                    THREADS_PER_BLOCK,
                    index_t,
                    scalar_t>
                    <<<grid,
                       THREADS_PER_BLOCK,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        grad_output.packed_accessor32<scalar_t, 2>(),
                        output.packed_accessor32<scalar_t, 2>(),
                        offsets.packed_accessor32<index_t, 1>(),
                        grad_input.packed_accessor32<scalar_t, 2>(),
                        (int)max_L);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }
  return grad_input;
}
} // namespace fbgemm_gpu

JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_softmax_backward",
    fbgemm_gpu::jagged_softmax_backward_cuda);
