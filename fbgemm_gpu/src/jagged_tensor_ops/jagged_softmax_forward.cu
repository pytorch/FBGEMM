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
__global__ __launch_bounds__(kMaxThreads) void jagged_softmax_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2> values,
    const at::PackedTensorAccessor32<index_t, 1> offsets,
    at::PackedTensorAccessor32<scalar_t, 2> output,
    const int max_L) {
  const auto B = offsets.size(0) - 1;
  const auto D = output.size(1);

  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<scalar_t, THREADS_PER_BLOCK> BlockReduceT;

  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  __shared__ scalar_t max_value;
  __shared__ scalar_t exp_sum;

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
          max_value = values[row_start][d];
          exp_sum = 0;
        }

        // Loop through all blocks to calculate the max value
        // Each block has its own max value block_max_value, and
        // max_value is the max value across all blocks
        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          const auto l = bk_l * blockDim.x + tid;
          scalar_t thread_val = values[row_start][d];
          if (l < length) {
            thread_val = values[row_start + l][d];
          }

          // Collectively compute the block-wide max reduction
          scalar_t block_max_value =
              BlockReduceT(temp_storage).Reduce(thread_val, cub::Max());
          __syncthreads();

          if (tid == 0) {
            max_value = max(max_value, block_max_value);
          }
        }

        // The max_value was updated by thread 0 in the last loop, sync here to
        // make sure the next loop uses the updated max_value
        __syncthreads();

        // Loop through all blocks to calculate the sum of exp
        // Each block has its own sum block_exp_acc, and
        // exp_sum is the sum across all blocks
        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          auto l = bk_l * blockDim.x + tid;

          scalar_t thread_exp = 0;
          if (l < length) {
            thread_exp = std::exp(values[row_start + l][d] - max_value);
          }

          // Collectively compute the block-wide sum reduction
          scalar_t block_exp_sum = BlockReduceT(temp_storage).Sum(thread_exp);
          __syncthreads();

          if (tid == 0) {
            exp_sum += block_exp_sum;
          }
        }

        // The exp_sum was updated by thread 0 in the last loop, sync here to
        // make sure the next loop uses the updated exp_sum
        __syncthreads();

        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          auto l = bk_l * blockDim.x + tid;
          scalar_t thread_exp = 0;
          if (l < length) {
            thread_exp = std::exp(values[row_start + l][d] - max_value);
            output[row_start + l][d] = thread_exp / exp_sum;
          }
        }

        // The max_value and exp_sum will be reinitialized by thread 0 in the
        // next d iteration, sync here to make sure the last loop still uses the
        // reduced values before reinitialization
        __syncthreads();
      }
    }
  }
}

Tensor jagged_softmax_forward_cuda(
    const Tensor& values,
    const Tensor& offsets,
    const int64_t max_L) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const auto B = offsets.numel() - 1;
  const auto D = values.size(1);
  auto output = at::empty_like(values);

  if (B > 0 && D > 0) {
    constexpr int THREADS_PER_BLOCK = 128;
    const dim3 grid(D, std::min((int32_t)B, (int32_t)kMaxBlockYDim), 1);

    AT_DISPATCH_INDEX_TYPES(
        offsets.scalar_type(), "jagged_softmax_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              values.scalar_type(),
              "jagged_softmax_kernel_2",
              [&] {
                jagged_softmax_kernel<THREADS_PER_BLOCK, index_t, scalar_t>
                    <<<grid,
                       THREADS_PER_BLOCK,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        values.packed_accessor32<scalar_t, 2>(),
                        offsets.packed_accessor32<index_t, 1>(),
                        output.packed_accessor32<scalar_t, 2>(),
                        (int)max_L);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }

  return output;
}
} // namespace fbgemm_gpu

JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_softmax_forward",
    fbgemm_gpu::jagged_softmax_forward_cuda);
