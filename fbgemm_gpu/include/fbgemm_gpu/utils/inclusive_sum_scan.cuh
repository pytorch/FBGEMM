/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// clang-format off
#ifdef USE_ROCM
#define HIPCUB_ARCH 1
#include <hipcub/backend/rocprim/block/block_scan.hpp>
#else
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh"
#include <cub/block/block_scan.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh"
#endif
// clang-format on

namespace fbgemm_gpu {

#ifdef USE_ROCM
namespace cub = hipcub;
#endif

/**
 * inclusive_sum_scan_kernel performs intra- and inter-thread block sum scan
 * (i.e., prefix sum scan). We use cub::BlockScan to do inclusive sum within
 * thread block and use a waterfall sync method to perform prefix sum across
 * thread block.
 *
 * @param arr an array of input values. Its length must be fixed to
 *            ITEMS_PER_THREAD
 * @param temp_storage a shared memory struct for cub::BlockScan
 * @param block_flags a global flag buffer for inter-block sync (must be
 *                    initialized with zeros)
 * @param block_sums a global sum buffer for inter-block sync
 * @param block_prev a shared memory pointer for sharing sum from the previous
 *                   block within a block
 * @param num_entries_per_block a number of input entries for this block
 * @param block_id a relative thread block ID (the first block that contains
 *                 the first set of input entries has block_id = 0)
 * @param is_multi_block a boolean to indicate if inter-block sum scan has to
 *                       be performed
 * @param signal If the value of block_flags of the previous block is equal to
 *               signal, it means that the previous block has written its sum
 *               to block_sums. We have thread blocks increment the value of
 *               block_flags by one after they write their sums to block_sums.
 *               We increment the flag instead of setting the flag to a single
 *               value to support multiple sequential inclusive_sum_scan_kernel
 *               calls (e.g., in the AUC kernel). signal is the order that
 *               inclusive_sum_scan_kernel is called. Since we intialize
 *               block_flags with zeros, the signal of the first call should be
 *               one.
 */
template <typename scalar_t, int ITEMS_PER_THREAD, int NUM_THREADS_PER_BLOCK>
__inline__ __device__ void inclusive_sum_scan_kernel(
    scalar_t (&arr)[ITEMS_PER_THREAD],
    typename cub::BlockScan<scalar_t, NUM_THREADS_PER_BLOCK>::TempStorage&
        temp_storage,
    int* block_flags,
    // Declared as volatile to prevent the compiler from register-allocating
    // the accesses to block_sums
    volatile scalar_t* block_sums,
    scalar_t* block_prev,
    const int num_entries_per_block,
    const int block_id,
    const bool is_multi_block,
    const int signal) {
  // Perform scan within a block
  cub::BlockScan<scalar_t, NUM_THREADS_PER_BLOCK>(temp_storage)
      .InclusiveSum(arr, arr);

  // Perform stream scan across blocks
  if (is_multi_block) {
    // The thread that holds the last entry in the block does synchronization
    if (threadIdx.x == (num_entries_per_block - 1) / ITEMS_PER_THREAD) {
      scalar_t block_prev_local = 0;
      if (block_id != 0) {
        // Spin wait for the previous block to write the sum value
        while (atomicAdd(&block_flags[block_id - 1], 0) < signal)
          ;

        // Get sum from the previous block
        *block_prev = block_prev_local = block_sums[block_id - 1];
      }

      // Write sum to global memory for the next block to consume
      const int scope = (num_entries_per_block - 1) % ITEMS_PER_THREAD;
      block_sums[block_id] = block_prev_local + arr[scope];
      __threadfence();
      // Set a flag to notify the next block
      atomicAdd(&block_flags[block_id], 1);
    }

    __syncthreads();

    if (block_id != 0) {
      scalar_t block_prev_local = *block_prev;
      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        arr[i] += block_prev_local;
      }
    }
  }
}

} // namespace fbgemm_gpu
