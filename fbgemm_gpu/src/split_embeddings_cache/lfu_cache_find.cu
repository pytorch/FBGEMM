/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

namespace {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lfu_update_counts_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const int32_t* __restrict__ N_unique,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        unique_indices_count,
    pta::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lfu_state) {
  CUDA_KERNEL_LOOP(n, *N_unique) {
    const auto idx = unique_indices[n];
    lfu_state[idx] += unique_indices_count[n];
  }
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lfu_cache_find_uncached_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const int32_t* __restrict__ N_unique,
    int64_t max_indices,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    uint64_t* __restrict__ cache_sets,
    const pta::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        lfu_state) {
  const int32_t C = lxu_cache_state.size(0);

  for (int32_t n = blockIdx.x * blockDim.y + threadIdx.y; n < *N_unique;
       n += gridDim.x * blockDim.y) {
    const int64_t idx = unique_indices[n];
    if (idx == max_indices) {
      // cache_sets are initialized with sentinel values in
      // lfu_cache_find_uncached_cuda
      continue;
    }
    const uint32_t cache_set = cache_slot(idx, C);

    const auto slot = threadIdx.x;
    const bool found = ::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx;

#ifdef USE_ROCM
    if (!__any_sync(0xFFFFFFFFFFFFFFFF, found)) {
#else
    if (!__any_sync(0xFFFFFFFF, found)) {
#endif
      if (threadIdx.x == 0) {
        // sort so the highest LFUs come first in the segment.
        // assume lfu_state[idx] <= 2^40 - 1 and cache_set < 2^24 -1
        cache_sets[n] =
            ((static_cast<uint64_t>(cache_set) << kLFUCounterBits)) |
            ((static_cast<uint64_t>(1) << kLFUCounterBits) - 1 -
             lfu_state[idx]);
      }
    }
  }
}

} // namespace

namespace fbgemm_gpu {

void lfu_update_counts_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    Tensor unique_indices_count,
    Tensor lfu_state) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      unique_indices, unique_indices_length, unique_indices_count, lfu_state);

  CUDA_DEVICE_GUARD(unique_indices);

  const int32_t N = unique_indices.size(0);
  AT_DISPATCH_INDEX_TYPES(
      unique_indices.scalar_type(), "lfu_update_counts_cuda", [&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "lfu_update_counts_kernel";
#endif
        lfu_update_counts_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads),
                get_max_thread_blocks_for_cache_kernels_()),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            MAKE_PTA_WITH_NAME(func_name, unique_indices, index_t, 1, 32),
            unique_indices_length.data_ptr<int32_t>(),
            MAKE_PTA_WITH_NAME(func_name, unique_indices_count, int32_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, lfu_state, int64_t, 1, 64));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

std::pair<Tensor, Tensor> lfu_cache_find_uncached_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    int64_t max_indices,
    Tensor lxu_cache_state,
    Tensor lfu_state) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      unique_indices, unique_indices_length, lxu_cache_state, lfu_state);

  CUDA_DEVICE_GUARD(unique_indices);

  auto cache_sets = full_like(
      unique_indices,
      static_cast<int64_t>(
          static_cast<uint64_t>(lxu_cache_state.size(0)) << kLFUCounterBits),
      unique_indices.options().dtype(at::kLong));
  const int32_t N = unique_indices.numel();
  auto sorted_cache_sets = empty_like(cache_sets);
  auto cache_set_sorted_unique_indices = empty_like(unique_indices);

  AT_DISPATCH_INDEX_TYPES(
      unique_indices.scalar_type(), "lfu_cache_find_uncached_cuda", [&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "lfu_cache_find_uncached_kernel";
#endif
        // Find uncached indices
        lfu_cache_find_uncached_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads / kWarpSize),
                get_max_thread_blocks_for_cache_kernels_()),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            MAKE_PTA_WITH_NAME(func_name, unique_indices, index_t, 1, 32),
            unique_indices_length.data_ptr<int32_t>(),
            max_indices,
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_state, int64_t, 2, 32),
            (uint64_t*)cache_sets.data_ptr<int64_t>(),
            MAKE_PTA_WITH_NAME(func_name, lfu_state, int64_t, 1, 64));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // Sort the cache sets and ids
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            (uint64_t*)cache_sets.data_ptr<int64_t>(),
            (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1) + kLFUCounterBits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            unique_indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            (uint64_t*)cache_sets.data_ptr<int64_t>(),
            (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1) + kLFUCounterBits,
            at::cuda::getCurrentCUDAStream(),
            false));
      });
  return {sorted_cache_sets, cache_set_sorted_unique_indices};
}

} // namespace fbgemm_gpu
