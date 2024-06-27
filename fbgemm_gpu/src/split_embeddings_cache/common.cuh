/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "common.h"

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>
#include <cub/block/block_reduce.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;

namespace {

constexpr size_t kCacheMaxThreads = 512;
constexpr int32_t kCacheLocationMissing = -1;
constexpr int64_t kCacheStateInvalid = -1;

constexpr int32_t kCacheSetBits = 24;
constexpr int32_t kLFUCounterBits = 40;
static_assert(kCacheSetBits + kLFUCounterBits == 8 * sizeof(int64_t), "");

// // TODO: do we care about 64-bit indices? Currently we just ignore.
// __host__ DEVICE_INLINE uint32_t cache_slot(int32_t h_in, int32_t C) {
//   // MurmorHash3 32-bit mixing function.
//   uint32_t h = (uint32_t)h_in;
//   h ^= h >> 16;
//   h *= 0x85ebca6b;
//   h ^= h >> 13;
//   h *= 0xc2b2ae35;
//   h ^= h >> 16;
//   //
//   https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
//   return ((uint64_t)h * (uint64_t)C) >> 32;
// }

__host__ DEVICE_INLINE uint32_t
cache_slot(const int64_t h_in, const int32_t C) {
  // MurmurHash3 64-bit mixing function.
  uint64_t h = (uint64_t)h_in;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccd;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53;
  h ^= h >> 33;

  return h % (uint32_t)C;
}

// Experiments showed that performance of lru/lxu_cache_find_uncached_kernel is
// not sensitive to grid size as long as the number thread blocks per SM is not
// too small nor too big.
constexpr int MAX_THREAD_BLOCKS_PER_SM_FOR_CACHE_KERNELS = 16;

int get_max_thread_blocks_for_cache_kernels_() {
  return get_device_sm_cnt_() * MAX_THREAD_BLOCKS_PER_SM_FOR_CACHE_KERNELS;
}

} // namespace

namespace fbgemm_gpu {

void lfu_update_counts_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    Tensor unique_indices_count,
    Tensor lfu_state);

std::pair<Tensor, Tensor> lfu_cache_find_uncached_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    int64_t max_indices,
    Tensor lxu_cache_state,
    Tensor lfu_state);

} // namespace fbgemm_gpu
