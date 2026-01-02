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
 * CUDA kernel to generate source mask for jagged tensor operations.
 *
 * This kernel efficiently generates a boolean mask by parallelizing across
 * batch items rather than individual output elements. Each block handles one
 * batch item, with threads within the block processing elements in parallel.
 *
 * Algorithm:
 * 1. Each block handles one batch item (batch_idx = blockIdx.x)
 * 2. Threads within block cooperatively process all elements for that batch
 * 3. Grid-stride loop handles cases where batch item has more elements than
 * threads
 * 4. Set to true if position < num_sources[batch_idx], false otherwise
 *
 * Example:
 *   num_sources = [2, 3]
 *   num_targets = [1, 2]
 *   Output: [True, True, False, True, True, True, False, False]
 */
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void get_source_mask_kernel(
    const index_t* __restrict__ num_sources,
    const index_t* __restrict__ num_targets,
    const index_t* __restrict__ offsets,
    bool* __restrict__ output,
    const index_t batch_size) {
  // Each block handles one batch item
  const index_t batch_idx = blockIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  const index_t ns = num_sources[batch_idx];
  const index_t nt = num_targets[batch_idx];
  const index_t total = ns + nt;
  const index_t offset = offsets[batch_idx];

  // Grid-stride loop: each thread processes multiple elements if needed
  for (index_t local_idx = threadIdx.x; local_idx < total;
       local_idx += blockDim.x) {
    output[offset + local_idx] = (local_idx < ns);
  }
}

Tensor get_source_mask_cuda(
    const Tensor& num_sources,
    const Tensor& num_targets,
    const at::SymInt output_size) {
  TENSOR_ON_CUDA_GPU(num_sources);
  TENSOR_ON_CUDA_GPU(num_targets);

  const auto batch_size = num_sources.size(0);
  TORCH_CHECK(
      num_targets.size(0) == batch_size,
      "num_sources and num_targets must have the same batch size");

  Tensor combined = num_sources + num_targets;
  Tensor offsets = asynchronous_complete_cumsum_gpu(combined);

  // Create output tensor
  Tensor output = at::empty_symint(
      {output_size},
      at::TensorOptions().dtype(at::kBool).device(num_sources.device()));

  if (output_size == 0) {
    return output;
  }

  // Launch kernel - one block per batch item
  const int num_blocks = batch_size;

  AT_DISPATCH_INDEX_TYPES(
      num_sources.scalar_type(), "get_source_mask_kernel", ([&] {
        FBGEMM_LAUNCH_KERNEL(
            (get_source_mask_kernel<index_t>),
            num_blocks,
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream(),
            num_sources.data_ptr<index_t>(),
            num_targets.data_ptr<index_t>(),
            offsets.data_ptr<index_t>(),
            output.mutable_data_ptr<bool>(),
            static_cast<index_t>(batch_size));
      }));

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(CUDA, "get_source_mask", fbgemm_gpu::get_source_mask_cuda);
