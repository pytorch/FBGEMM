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
 * CUDA kernel to generate repeated arange for jagged tensor operations.
 *
 * This kernel efficiently generates consecutive integers for each sequence
 * by parallelizing across batch items. Each block handles one batch item,
 * with threads within the block processing elements in parallel.
 *
 * Algorithm:
 * 1. Each block handles one batch item (batch_idx = blockIdx.x)
 * 2. Threads within block cooperatively generate arange for that sequence
 * 3. Grid-stride loop handles cases where sequence is longer than block size
 * 4. Each element gets value from 0 to lengths[batch_idx]-1
 *
 * Example:
 *   lengths = [3, 5, 2]
 *   Output: [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]
 *
 * This is much more efficient than the PyTorch implementation which uses:
 * - asynchronous_complete_cumsum (1 kernel)
 * - arange (1 kernel)
 * - repeat_interleave (1 kernel)
 * - subtraction (1 kernel)
 * Total: 4 kernels with multiple intermediate allocations
 *
 * Our approach: 1 fused kernel with no intermediate allocations
 */
template <typename index_t, typename output_t>
__global__ __launch_bounds__(kMaxThreads) void repeat_arange_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        lengths,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    pta::PackedTensorAccessor32<output_t, 1, at::RestrictPtrTraits> output) {
  // Each block handles one batch item
  const index_t batch_idx = blockIdx.x;
  const index_t batch_size = lengths.size(0);

  if (batch_idx >= batch_size) {
    return;
  }

  const index_t length = lengths[batch_idx];
  const index_t offset = offsets[batch_idx];

  // Grid-stride loop: each thread processes multiple elements if needed
  for (index_t local_idx = threadIdx.x; local_idx < length;
       local_idx += blockDim.x) {
    output[offset + local_idx] = static_cast<output_t>(local_idx);
  }
}

Tensor repeat_arange_cuda(const Tensor& lengths) {
  TENSOR_ON_CUDA_GPU(lengths);

  const auto batch_size = lengths.size(0);

  if (batch_size == 0) {
    return at::empty({0}, lengths.options());
  }

  // Compute offsets and total output size in one go
  Tensor offsets = asynchronous_complete_cumsum_gpu(lengths);

  // Get total output size - this is efficient as it's just reading the last
  // element
  int64_t output_size;
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "repeat_arange_get_size", ([&] {
        index_t last_offset;
        cudaMemcpy(
            &last_offset,
            offsets.data_ptr<index_t>() + batch_size,
            sizeof(index_t),
            cudaMemcpyDeviceToHost);
        output_size = static_cast<int64_t>(last_offset);
      }));

  if (output_size == 0) {
    return at::empty({0}, lengths.options());
  }

  // Create output tensor - use same dtype as input for flexibility
  Tensor output = at::empty({output_size}, lengths.options());

  // Launch kernel - one block per batch item
  const int num_blocks = batch_size;

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "repeat_arange_kernel", ([&] {
        // Dispatch on output type as well
        AT_DISPATCH_INTEGRAL_TYPES(
            lengths.scalar_type(), "repeat_arange_kernel_output", ([&] {
              FBGEMM_LAUNCH_KERNEL(
                  (repeat_arange_kernel<index_t, scalar_t>),
                  num_blocks,
                  kMaxThreads,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(lengths, index_t, 1, 32),
                  PTA_B(offsets, index_t, 1, 32),
                  PTA_B(output, scalar_t, 1, 32));
            }));
      }));

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(CUDA, "repeat_arange", fbgemm_gpu::repeat_arange_cuda);
