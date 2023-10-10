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

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kMaxThreads) void lru_cache_insert_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const pta::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_cache_sets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const int64_t time_stamp,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    const bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats,
    const bool lock_cache_line,
    pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits>
        lxu_cache_locking_counter) {
  const int32_t C = lxu_cache_state.size(0);
  int32_t n_conflict_misses = 0;
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
    int32_t n_inserted = 0; // also used as index to insert

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
      if (lock_cache_line) {
        auto count = lxu_cache_locking_counter[cache_set][insert_slot];
        if (count > 0) {
          continue; // cache slot is in use
        }
      }
      const int64_t insert_current_lru_cost = shfl_sync(sorted_lru_cost, l);
      if (insert_current_lru_cost == time_stamp) {
        break;
      }
      const int64_t insert_idx = cache_set_sorted_indices[n + n_inserted];
      const int32_t t_insert = cache_index_table_map[insert_idx];
      const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
      const int64_t weights_offset_insert = weights_offsets[t_insert];
      const int32_t D_start_insert = D_offsets[t_insert];
      const int32_t D_end_insert = D_offsets[t_insert + 1];
      const int32_t D_insert = D_end_insert - D_start_insert;

      // ensure that threadIdx.x is the only thread reading/writing to
      // lxu_cache_state
      int64_t current_idx =
          threadIdx.x == 0 ? lxu_cache_state[cache_set][insert_slot] : 0;
      current_idx = shfl_sync(current_idx, 0);

      // not empty
      if (current_idx != static_cast<int64_t>(kCacheStateInvalid)) {
        // evict from slot to backing storage
        const int32_t t_current = cache_index_table_map[current_idx];
        const int64_t idx_current =
            current_idx - cache_hash_size_cumsum[t_current];
        const int64_t weights_offset_current = weights_offsets[t_current];
        const int32_t D_start_current = D_offsets[t_current];
        const int32_t D_end_current = D_offsets[t_current + 1];
        const int32_t D_current = D_end_current - D_start_current;
        int32_t D_emb = D_current;
        if constexpr (std::is_same_v<emb_t, uint8_t>) {
          D_emb += kINT8QparamsBytes;
        }

        auto weight_row = WeightRow<emb_t, cache_t, cache_t>(
            &weights[weights_offset_current + idx_current * D_emb + 0],
            &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
            D_current,
            nullptr);

        weight_row.set_stochastic_rounding(
            stochastic_rounding,
            stochastic_rounding_philox_args,
            (blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
             threadIdx.x) *
                    kWarpSize +
                l);

        weight_row.warp_evict(D_current, blockDim.x, threadIdx.x);
      }

      int32_t D_emb = D_insert;
      if constexpr (std::is_same_v<emb_t, uint8_t>) {
        D_emb += kINT8QparamsBytes;
      }

      auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
          &weights[weights_offset_insert + idx_insert * D_emb + 0],
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
          D_insert,
          nullptr);

      auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
          &weights[weights_offset_insert + idx_insert * D_emb + 0],
          nullptr,
          D_insert,
          nullptr);

      weight_row_emb.warp_copy_to(
          weight_row_cache, D_insert, blockDim.x, threadIdx.x);

      if (threadIdx.x == 0) {
        lxu_cache_state[cache_set][insert_slot] = insert_idx;
        lru_state[cache_set][insert_slot] = time_stamp;
        if (lock_cache_line) {
          lxu_cache_locking_counter[cache_set][insert_slot] += 1;
        }
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

void lru_cache_insert_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    const int64_t time_stamp,
    Tensor lru_state,
    const bool stochastic_rounding,
    bool gather_cache_stats,
    Tensor uvm_cache_stats,
    bool lock_cache_line,
    Tensor lxu_cache_locking_counter) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state,
      uvm_cache_stats,
      lxu_cache_locking_counter);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const int32_t N = cache_set_sorted_unique_indices.numel();

  DISPATCH_EMB_CACHE_TYPES(
      weights.scalar_type(),
      lxu_cache_weights.scalar_type(),
      "lru_cache_insert_kernel_2",
      ([&] {
        at::PhiloxCudaState rng_engine_inputs;
        if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
          auto gen = at::cuda::detail::getDefaultCUDAGenerator();
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)
                                  ->philox_cuda_state(4);
        }

        // During concurrent prefetch, cache lines are locked and we use less
        // SMs for some of the prefetch kernels (e.g. insert)
        // since it is not SM bound. It leaves SMs for main stream to overlap
        constexpr int ALL_TO_PREFETCH_SM_RATIO = 8;

        auto grid_size = lock_cache_line
            ? div_round_up(get_device_sm_cnt_(), ALL_TO_PREFETCH_SM_RATIO)
            : div_round_up(N, kMaxThreads / kWarpSize);

#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "lru_cache_insert_kernel";
#endif
        lru_cache_insert_kernel<emb_t, cache_t>
            <<<grid_size,
               dim3(kWarpSize, kMaxThreads / kWarpSize),
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                MAKE_PTA_WITH_NAME(func_name, weights, emb_t, 1, 64),
                MAKE_PTA_WITH_NAME(
                    func_name, cache_hash_size_cumsum, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(
                    func_name, cache_index_table_map, int32_t, 1, 64),
                MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(
                    func_name, sorted_cache_sets, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(
                    func_name, cache_set_sorted_unique_indices, int64_t, 1, 32),
                unique_indices_length.data_ptr<int32_t>(),
                MAKE_PTA_WITH_NAME(func_name, lxu_cache_state, int64_t, 2, 32),
                MAKE_PTA_WITH_NAME(
                    func_name, lxu_cache_weights, cache_t, 2, 64),
                time_stamp,
                MAKE_PTA_WITH_NAME(func_name, lru_state, int64_t, 2, 32),
                stochastic_rounding,
                rng_engine_inputs,
                gather_cache_stats,
                MAKE_PTA_WITH_NAME(func_name, uvm_cache_stats, int32_t, 1, 32),
                lock_cache_line,
                MAKE_PTA_WITH_NAME(
                    func_name, lxu_cache_locking_counter, int32_t, 2, 32));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

} // namespace

DLL_PUBLIC void lru_cache_populate_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    const int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    const int64_t time_stamp,
    Tensor lru_state,
    const bool stochastic_rounding,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats,
    bool lock_cache_line,
    c10::optional<Tensor> lxu_cache_locking_counter) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
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

  Tensor lxu_cache_locking_counter_ =
      at::empty({0, 0}, lxu_cache_state.options().dtype(at::kInt));
  if (lock_cache_line) {
    TORCH_CHECK(lxu_cache_locking_counter.has_value());
    lxu_cache_locking_counter_ = lxu_cache_locking_counter.value();
    TENSOR_ON_CUDA_GPU(lxu_cache_locking_counter_);
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

  auto cache_sets_and_unique_indices = lru_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      time_stamp,
      lru_state,
      gather_cache_stats,
      uvm_cache_stats_,
      lock_cache_line,
      lxu_cache_locking_counter_);
  auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  auto cache_set_sorted_unique_indices = cache_sets_and_unique_indices.second;

  // insert caching weights
  lru_cache_insert_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      time_stamp,
      lru_state,
      stochastic_rounding,
      gather_cache_stats,
      uvm_cache_stats_,
      lock_cache_line,
      lxu_cache_locking_counter_);
}
