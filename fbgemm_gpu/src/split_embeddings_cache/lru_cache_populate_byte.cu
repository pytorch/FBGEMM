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
__global__
__launch_bounds__(kMaxThreads) void direct_mapped_lru_cache_find_uncached_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> cache_sets,
    const int64_t max_indices,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    const int64_t time_stamp,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_miss_timestamp) {
  const int32_t N = linear_cache_indices.size(0);
  const int32_t C = lxu_cache_state.size(0);

  if (gather_cache_stats) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_calls], 1); // N_called.
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_requested_indices],
          N); // N_requested_indices.
    }
  }

  CUDA_KERNEL_LOOP(n, N) {
    int64_t idx = linear_cache_indices[n];
    if (idx == max_indices) {
      // Invalid or pruned row: set it to sentinel value.
      // 32-way uses C as the sentinel value to reduce the maximum value during
      // radix sort to make it faster but for direct_mapped we use -1
      cache_sets[n] = -1;
      continue;
    }
    int32_t cache_set = cache_slot(idx, C);

    const bool found = ::__ldg((&lxu_cache_state[cache_set][0])) == idx;
    if (found) {
      // After all threads run, timestamp will be current timestamp
      // if any idx was hit
      // +1 because AMD doesn't have atomicMax for signed long so we should
      // initialize lxu_cache_miss_timestamp with 0 vs. -1.
      lru_state[cache_set][0] = time_stamp;
      cache_sets[n] = -1; // sentinel value
    } else {
      // There is no atomicMax for int64_t...
#ifdef USE_ROCM
      auto addr = reinterpret_cast<unsigned long long*>(
          &lxu_cache_miss_timestamp[cache_set][0]);
      auto val = static_cast<unsigned long long>(time_stamp + 1);
      auto old = static_cast<int64_t>(atomicMax(addr, val));
#else
      auto addr = reinterpret_cast<long long int*>(
          &lxu_cache_miss_timestamp[cache_set][0]);
      auto val = static_cast<long long int>(time_stamp + 1);
      auto old = static_cast<int64_t>(atomicMax(addr, val));
#endif

      if (old < time_stamp + 1) {
        // This is the lucky thread that gets to insert its idx in the cache
        // slot. So the number of elements in cache_sets array that has the
        // value of cache_set is 1 at maximum
        cache_sets[n] = cache_set;
      } else {
        // Otherwise (too late to get this set)
        // set it to sentinel value.
        cache_sets[n] = -1;
      }
    }
  }
}

Tensor direct_mapped_lru_cache_find_uncached_cuda(
    Tensor linear_cache_indices,
    int64_t max_indices,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    Tensor lru_state,
    Tensor lxu_cache_miss_timestamp,
    bool gather_cache_stats,
    Tensor uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_cache_indices,
      lxu_cache_state,
      lru_state,
      lxu_cache_miss_timestamp);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(linear_cache_indices.get_device());

  const int32_t N = linear_cache_indices.numel();

  auto cache_sets = empty_like(
      linear_cache_indices, linear_cache_indices.options().dtype(at::kInt));

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(),
      "direct_mapped_lru_cache_find_uncached_cuda",
      [&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "direct_mapped_lru_cache_find_uncached_kernel";
#endif
        // Find uncached indices
        direct_mapped_lru_cache_find_uncached_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads),
                get_max_thread_blocks_for_cache_kernels_()),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            MAKE_PTA_WITH_NAME(func_name, linear_cache_indices, index_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, cache_sets, int32_t, 1, 32),
            max_indices,
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_state, int64_t, 2, 32),
            time_stamp,
            MAKE_PTA_WITH_NAME(func_name, lru_state, int64_t, 2, 32),
            gather_cache_stats,
            MAKE_PTA_WITH_NAME(func_name, uvm_cache_stats, int32_t, 1, 32),
            MAKE_PTA_WITH_NAME(
                func_name, lxu_cache_miss_timestamp, int64_t, 2, 32));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return cache_sets;
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lru_cache_insert_byte_kernel(
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> weights,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const pta::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const pta::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
        weights_tys,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_cache_sets,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    int64_t time_stamp,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats,
    const int64_t row_alignment) {
  const int32_t C = lxu_cache_state.size(0);
  int64_t n_conflict_misses = 0;
  for (int32_t n = blockIdx.x * blockDim.y + threadIdx.y; n < *N_unique;
       n += gridDim.x * blockDim.y) {
    // check if this warp is responsible for this whole segment.
    const bool segment_start =
        (n == 0 || sorted_cache_sets[n - 1] != sorted_cache_sets[n]);

    if (!segment_start) {
      // don't have *warp* divergence since we launch full warps in blockDim.x,
      // so we can just exit this warp entirely.
      continue;
    }
    const int32_t cache_set = sorted_cache_sets[n];
    if (cache_set == C) {
      // ignore the already-existing elements
      continue;
    }

    int32_t SL = 1;
    while (n + SL < *N_unique && sorted_cache_sets[n + SL] == cache_set) {
      SL += 1;
    }
    int64_t n_inserted = 0;

    // now, we need to insert the (unique!) values in indices[n:n + SL] into
    // our slots.
    const int32_t slot = threadIdx.x;
    const int64_t slot_time = lru_state[cache_set][slot];
    int64_t costs[1] = {slot_time};
    int32_t slots[1] = {slot};

    BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
    const int32_t sorted_slot = slots[0];
    const int64_t sorted_lru_cost = costs[0];

    for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
      const int32_t insert_slot = shfl_sync(sorted_slot, l);
      const int64_t insert_current_lru_cost = shfl_sync(sorted_lru_cost, l);
      if (insert_current_lru_cost == time_stamp) {
        break;
      }
      index_t insert_idx = cache_set_sorted_indices[n + l];
      const int32_t t_insert = cache_index_table_map[insert_idx];
      SparseType weight_ty_insert =
          static_cast<SparseType>(weights_tys[t_insert]);
      const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
      const int64_t weights_offset_insert = weights_offsets[t_insert];
      const int32_t D_start_insert = D_offsets[t_insert];
      const int32_t D_end_insert = D_offsets[t_insert + 1];
      const int32_t D_insert = D_end_insert - D_start_insert;

      const int32_t D_insert_bytes = nbit::padded_row_size_in_bytes(
          D_insert, weight_ty_insert, row_alignment);

      // insert into cache. Note that nbit::padded_row_size_in_bytes pad each
      // row with row_alignment (16 bytes on GPUs) So each row will be multiple
      // of 16 bytes (uint4 = 32bit x 4 = 16 bytes).
      auto row = reinterpret_cast<const uint4*>(
          &weights[weights_offset_insert + idx_insert * D_insert_bytes + 0]);
      auto cache_row = reinterpret_cast<uint4*>(
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0]);
      for (int32_t d = threadIdx.x; d * sizeof(uint4) < D_insert_bytes;
           d += blockDim.x) {
        cache_row[d] = row[d];
      }

      if (threadIdx.x == 0) {
        lxu_cache_state[cache_set][insert_slot] = insert_idx;
        lru_state[cache_set][insert_slot] = time_stamp;
      }
      n_inserted++;
    }
    n_conflict_misses += (SL - n_inserted);
  }
  if (gather_cache_stats && n_conflict_misses > 0 && threadIdx.x == 0) {
    atomicAdd(
        &uvm_cache_stats[uvm_cache_stats_index::num_conflict_unique_misses],
        n_conflict_misses);
  }
}

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void direct_mapped_lru_cache_insert_byte_kernel(
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> weights,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const pta::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const pta::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
        weights_tys,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    int64_t time_stamp,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_miss_timestamp,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> cache_sets,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats,
    const int64_t row_alignment) {
  const int32_t N = cache_sets.size(0);

  // one warp for each set (multiple times)
  // (no divergence for each control branch)
  for (int32_t pos = blockIdx.x * blockDim.y + threadIdx.y; pos < N;
       pos += gridDim.x * blockDim.y) {
    auto cache_set = cache_sets[pos];

    if (cache_set == -1) {
      // Cache hit, index invalid (e.g., pruned), or too late to grab this set.
      continue;
    }

    if (lru_state[cache_set][0] == time_stamp) {
      // we have a missing index but
      // current cache row is a hit
      // so abort unnecessary insert
      continue;
    }

    // no need to check because cache_sets[pos] != -1 only when it was the
    // first one to set the buffer time_stamp
    // if (lxu_cache_miss_timestamp[cache_set][0] != time_stamp) {
    //   continue;
    // }

    if (gather_cache_stats && threadIdx.x == 0) {
      // We are using this slot for a slightly different purpose.
      // In 32 way:
      //    UVM traffic for insert
      //    = # of inserted rows
      //    = # of unique misses - # of unique misses that were not inserted
      //    = uvm_cache_stats_index::num_unique_misses
      //      - uvm_cache_stats_index::num_conflict_unique_misses
      // In Direct Mapped (here):
      //    UVM traffic for insert
      //    = # of inserted rows
      //    = uvm_cache_stats_index::num_conflict_unique_misses
      //      (just store here directly)
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_conflict_unique_misses],
          1);
    }

    // insert the index in the buffer into our only slot
    const int32_t insert_slot = 0;

    int64_t insert_idx = linear_cache_indices[pos];
    const int32_t t_insert = cache_index_table_map[insert_idx];
    SparseType weight_ty_insert =
        static_cast<SparseType>(weights_tys[t_insert]);
    const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
    const int64_t weights_offset_insert = weights_offsets[t_insert];
    const int32_t D_start_insert = D_offsets[t_insert];
    const int32_t D_end_insert = D_offsets[t_insert + 1];
    const int32_t D_insert = D_end_insert - D_start_insert;
    const int32_t D_insert_bytes = nbit::padded_row_size_in_bytes(
        D_insert, weight_ty_insert, row_alignment);

    // insert into cache. Note that nbit::padded_row_size_in_bytes pad each
    // row with row_alignment (16 bytes on GPUs) So each row will be multiple
    // of 16 bytes (uint4 = 32bit x 4 = 16 bytes).
    auto row = reinterpret_cast<const uint4*>(
        &weights[weights_offset_insert + idx_insert * D_insert_bytes + 0]);
    auto cache_row = reinterpret_cast<uint4*>(&lxu_cache_weights[cache_set][0]);
    for (int32_t d = threadIdx.x; d * sizeof(uint4) < D_insert_bytes;
         d += blockDim.x) {
      cache_row[d] = row[d];
    }

    if (threadIdx.x == 0) {
      lxu_cache_state[cache_set][insert_slot] = insert_idx;
      lru_state[cache_set][insert_slot] = time_stamp;
    }
  }
}

void lru_cache_insert_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    bool gather_cache_stats,
    Tensor uvm_cache_stats,
    int64_t row_alignment) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state,
      uvm_cache_stats);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const int32_t N = cache_set_sorted_unique_indices.numel();

  AT_DISPATCH_INDEX_TYPES(
      cache_set_sorted_unique_indices.scalar_type(),
      "lru_cache_insert_byte_cuda",
      [&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "lru_cache_insert_byte_kernel";
#endif
        lru_cache_insert_byte_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads / kWarpSize),
                get_max_thread_blocks_for_cache_kernels_()),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            MAKE_PTA_WITH_NAME(func_name, weights, uint8_t, 1, 64),
            MAKE_PTA_WITH_NAME(
                func_name, cache_hash_size_cumsum, int64_t, 1, 32),
            MAKE_PTA_WITH_NAME(
                func_name, cache_index_table_map, int32_t, 1, 64),
            MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, weights_tys, uint8_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, sorted_cache_sets, int32_t, 1, 32),
            MAKE_PTA_WITH_NAME(
                func_name, cache_set_sorted_unique_indices, index_t, 1, 32),
            unique_indices_length.data_ptr<int32_t>(),
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_state, int64_t, 2, 32),
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, uint8_t, 2, 64),
            time_stamp,
            MAKE_PTA_WITH_NAME(func_name, lru_state, int64_t, 2, 32),
            gather_cache_stats,
            MAKE_PTA_WITH_NAME(func_name, uvm_cache_stats, int32_t, 1, 32),
            row_alignment);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

void direct_mapped_lru_cache_insert_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    Tensor linear_cache_indices,
    Tensor lxu_cache_miss_timestamp,
    Tensor cache_sets,
    bool gather_cache_stats,
    Tensor uvm_cache_stats,
    int64_t row_alignment) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state,
      linear_cache_indices,
      lxu_cache_miss_timestamp);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const int32_t N = cache_sets.size(0);

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(),
      "direct_mapped_lru_cache_insert_byte_cuda",
      [&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "direct_mapped_lru_cache_insert_byte_kernel";
#endif
        direct_mapped_lru_cache_insert_byte_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads / kWarpSize),
                get_max_thread_blocks_for_cache_kernels_()),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            MAKE_PTA_WITH_NAME(func_name, weights, uint8_t, 1, 64),
            MAKE_PTA_WITH_NAME(
                func_name, cache_hash_size_cumsum, int64_t, 1, 32),
            MAKE_PTA_WITH_NAME(
                func_name, cache_index_table_map, int32_t, 1, 64),
            MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, weights_tys, uint8_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_state, int64_t, 2, 32),
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, uint8_t, 2, 64),
            time_stamp,
            MAKE_PTA_WITH_NAME(func_name, lru_state, int64_t, 2, 32),
            MAKE_PTA_WITH_NAME(func_name, linear_cache_indices, index_t, 1, 32),
            MAKE_PTA_WITH_NAME(
                func_name, lxu_cache_miss_timestamp, int64_t, 2, 32),
            MAKE_PTA_WITH_NAME(func_name, cache_sets, int32_t, 1, 32),
            gather_cache_stats,
            MAKE_PTA_WITH_NAME(func_name, uvm_cache_stats, int32_t, 1, 32),
            row_alignment);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace

DLL_PUBLIC void lru_cache_populate_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    int64_t row_alignment,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      linear_cache_indices,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state);

  Tensor uvm_cache_stats_ = at::empty({0}, weights.options().dtype(at::kInt));
  if (gather_cache_stats) {
    TORCH_CHECK(uvm_cache_stats.has_value());
    uvm_cache_stats_ = uvm_cache_stats.value();
    TENSOR_ON_CUDA_GPU(uvm_cache_stats_);
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  // Get unqiue indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(
          linear_cache_indices, total_cache_hash_size, false);

  // Find uncached indices
  Tensor lxu_cache_locking_counter =
      at::empty({0, 0}, lxu_cache_state.options().dtype(at::kInt));
  auto cache_sets_and_unique_indices = lru_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      time_stamp,
      lru_state,
      gather_cache_stats,
      uvm_cache_stats_,
      false, // lock_cache_line
      lxu_cache_locking_counter);
  auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  auto cache_set_sorted_unique_indices = cache_sets_and_unique_indices.second;

  // insert caching weights
  lru_cache_insert_byte_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      time_stamp,
      lru_state,
      gather_cache_stats,
      uvm_cache_stats_,
      row_alignment);
}

DLL_PUBLIC void direct_mapped_lru_cache_populate_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    Tensor lxu_cache_miss_timestamp,
    int64_t row_alignment,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      linear_cache_indices,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state,
      lxu_cache_miss_timestamp);

  if (gather_cache_stats) {
    TORCH_CHECK(uvm_cache_stats.has_value());
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        uvm_cache_stats, lxu_cache_weights);
  }
  auto uvm_cache_stats_ = uvm_cache_stats.value_or(
      at::empty({0}, weights.options().dtype(at::kInt)));

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  /*
  populate_byte normal flow:
  (1) get_unique (sort, dedup)
  (2) find_uncached
  (3) sort by set_idx
  (4) insert rows

  merged kernels flow:
  (1) find_uncached
        No need for sorting.
        Each hit idx will just update the timestamp in lru_state.
        Only one of miss indices will atomically set miss_timestamp,
                                      and have cache_sets[pos] = set
                                          where pos is the position of that idx
                                          in the linear_cache_indices array
        After this, for each set, we either have
          (a) lru_state timestamp is recent (hit) => no need to insert row
          (b) lru_state timestamp is not recent (no hit)
              (b-1) miss_timestamp is recent
                    => insert row for idx = linear_cache_indices[pos]
              (b-2) insert_timestamp_buffer is not recent
                    => no need to insert since there was no miss idx this time
  (2) insert rows
        Use buffer info to insert rows as the above logic.
  */

  auto cache_sets = direct_mapped_lru_cache_find_uncached_cuda(
      linear_cache_indices,
      total_cache_hash_size,
      lxu_cache_state,
      time_stamp,
      lru_state,
      lxu_cache_miss_timestamp,
      gather_cache_stats,
      uvm_cache_stats_);

  // insert caching weights
  direct_mapped_lru_cache_insert_byte_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      lxu_cache_state,
      lxu_cache_weights,
      time_stamp,
      lru_state,
      linear_cache_indices,
      lxu_cache_miss_timestamp,
      cache_sets,
      gather_cache_stats,
      uvm_cache_stats_,
      row_alignment);
}
