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

__global__ __launch_bounds__(kMaxThreads) void emulate_cache_miss_kernel(
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const int64_t enforced_misses_per_256,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats) {
  const int32_t N = lxu_cache_locations.size(0);
  int64_t n_enforced_misses = 0;
  CUDA_KERNEL_LOOP(n, N) {
    if ((n & 0x00FF) < enforced_misses_per_256) {
      if (lxu_cache_locations[n] >= 0) {
        n_enforced_misses++;
      }
      lxu_cache_locations[n] = kCacheLocationMissing;
    }
  }
  if (gather_cache_stats && n_enforced_misses > 0) {
    atomicAdd(
        &uvm_cache_stats[uvm_cache_stats_index::num_conflict_misses],
        n_enforced_misses);
  }
}

} // namespace

DLL_PUBLIC Tensor emulate_cache_miss(
    Tensor lxu_cache_locations,
    const int64_t enforced_misses_per_256,
    const bool gather_cache_stats,
    Tensor uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      lxu_cache_locations, uvm_cache_stats);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lxu_cache_locations.get_device());

  const auto N = lxu_cache_locations.numel();
  if (N == 0) {
    // nothing to do
    return lxu_cache_locations;
  }

  const dim3 blocks(std::min(
      div_round_up(N, kMaxThreads),
      get_max_thread_blocks_for_cache_kernels_()));

#ifdef FBGEMM_GPU_MEMCHECK
  const char* func_name = "emulate_cache_miss_kernel";
#endif

  emulate_cache_miss_kernel<<<
      blocks,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32),
      enforced_misses_per_256,
      gather_cache_stats,
      MAKE_PTA_WITH_NAME(func_name, uvm_cache_stats, int32_t, 1, 32));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return lxu_cache_locations;
}

namespace {
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lru_cache_find_uncached_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const int32_t* __restrict__ N_unique,
    int64_t max_indices,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> cache_sets,
    int64_t time_stamp,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats,
    const bool lock_cache_line,
    pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits>
        lxu_cache_locking_counter) {
  if (gather_cache_stats) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_calls], 1); // N_called.
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_requested_indices],
          unique_indices.size(0)); // N_requested_indices.
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_unique_indices],
          *N_unique); // N_unique_indices.
    }
  }

  const int32_t C = lxu_cache_state.size(0);
  int32_t n_misses = 0;

  for (int32_t n = blockIdx.x * blockDim.y + threadIdx.y; n < *N_unique;
       n += gridDim.x * blockDim.y) {
    int64_t idx = unique_indices[n];
    if (idx == max_indices) {
      // cache_sets are initialized with sentinel values in
      // lru_cache_find_uncached_cuda
      continue;
    }
    int32_t cache_set = cache_slot(idx, C);

    const auto slot = threadIdx.x;
    const bool found = ::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx;
    if (found) {
      // mark it as recently accessed so we don't evict.
      lru_state[cache_set][slot] = time_stamp;
      if (lock_cache_line) {
        lxu_cache_locking_counter[cache_set][slot] += 1;
      }
    }

#ifdef USE_ROCM
    if (!__any_sync(0xFFFFFFFFFFFFFFFF, found)) {
#else
    if (!__any_sync(0xFFFFFFFF, found)) {
#endif
      if (threadIdx.x == 0) {
        cache_sets[n] = cache_set;
        n_misses++;
      }
    }
  }
  if (gather_cache_stats && threadIdx.x == 0) {
    atomicAdd(
        &uvm_cache_stats[uvm_cache_stats_index::num_unique_misses],
        n_misses); // N_unique_misses.
  }
}

} // namespace

DLL_PUBLIC std::pair<Tensor, Tensor> lru_cache_find_uncached_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    int64_t max_indices,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    Tensor lru_state,
    bool gather_cache_stats,
    Tensor uvm_cache_stats,
    bool lock_cache_line,
    Tensor lxu_cache_locking_counter) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lru_state,
      uvm_cache_stats,
      lxu_cache_locking_counter);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(unique_indices.get_device());

  // Fill with sentinel value
  auto cache_sets = full_like(
      unique_indices,
      lxu_cache_state.size(0),
      unique_indices.options().dtype(at::kInt));
  const int32_t N = unique_indices.numel();
  auto sorted_cache_sets = empty_like(cache_sets);
  auto cache_set_sorted_unique_indices = empty_like(unique_indices);

  AT_DISPATCH_INDEX_TYPES(
      unique_indices.scalar_type(), "lru_cache_find_uncached_cuda", [&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "lru_cache_find_uncached_kernel";
#endif
        // Find uncached indices
        lru_cache_find_uncached_kernel<<<
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
            MAKE_PTA_WITH_NAME(func_name, cache_sets, int32_t, 1, 32),
            time_stamp,
            MAKE_PTA_WITH_NAME(func_name, lru_state, int64_t, 2, 32),
            gather_cache_stats,
            MAKE_PTA_WITH_NAME(func_name, uvm_cache_stats, int32_t, 1, 32),
            lock_cache_line,
            MAKE_PTA_WITH_NAME(
                func_name, lxu_cache_locking_counter, int32_t, 2, 32));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // Sort the cache sets and ids
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            cache_sets.data_ptr<int32_t>(),
            sorted_cache_sets.data_ptr<int32_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<index_t>(temp_storage_bytes)},
            unique_indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            cache_sets.data_ptr<int32_t>(),
            sorted_cache_sets.data_ptr<int32_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
      });
  return {sorted_cache_sets, cache_set_sorted_unique_indices};
}
