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

DLL_PUBLIC int64_t host_lxu_cache_slot(int64_t h_in, int64_t C) {
  return static_cast<int64_t>(cache_slot(h_in, static_cast<int32_t>(C)));
}

namespace {

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_flush_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const pta::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  const int32_t B = lxu_cache_weights.size(0);
  const int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  const int32_t slot = b % kWarpSize;
  const int32_t cache_set = b / kWarpSize;
  const int64_t current_idx = lxu_cache_state[cache_set][slot];
  if (current_idx != static_cast<int64_t>(kCacheStateInvalid)) {
    // evict from slot to backing storage
    const int32_t t_current = cache_index_table_map[current_idx];
    const int64_t idx_current = current_idx - cache_hash_size_cumsum[t_current];
    const int64_t weights_offset_current = weights_offsets[t_current];
    const int32_t D_start_current = D_offsets[t_current];
    const int32_t D_end_current = D_offsets[t_current + 1];
    const int32_t D_current = D_end_current - D_start_current;

    int32_t D_emb = D_current;
    if constexpr (std::is_same_v<emb_t, uint8_t>) {
      D_emb += kINT8QparamsBytes;
    }
    StochasticRoundingRNGState state;
    auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
        &weights[weights_offset_current + idx_current * D_emb + 0],
        &lxu_cache_weights[b][0],
        D_current,
        stochastic_rounding ? &state : nullptr,
        &stochastic_rounding_philox_args,
        blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x);

    float2 qparams;
    if (std::is_same<emb_t, uint8_t>::value) {
      qparams =
          thrust_find_qparams<cache_t>(&lxu_cache_weights[b][0], D_current);
      if (threadIdx.x == 0) {
        weight_row.store_qparams(qparams);
      }
    }
    for (int32_t d = threadIdx.x * 4; d < D_current; d += blockDim.x * 4) {
      weight_row.evict_cache(d, qparams);
    }
  }
}

} // namespace

DLL_PUBLIC void lxu_cache_flush_cuda(
    Tensor uvm_weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    bool stochastic_rounding) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      uvm_weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      lxu_cache_state,
      lxu_cache_weights);

  CUDA_DEVICE_GUARD(lxu_cache_weights);

  const int32_t T = D_offsets.numel() - 1;
  const int32_t S = lxu_cache_weights.size(0);
  const int32_t tx = std::min<int32_t>(total_D / 4 / T, kMaxThreads);
  const dim3 threads(tx, kMaxThreads / tx);
  const dim3 blocks(div_round_up(S, kMaxThreads / tx));

  DISPATCH_EMB_CACHE_TYPES(
      uvm_weights.scalar_type(),
      lxu_cache_weights.scalar_type(),
      "lxu_cache_flush_kernel_2",
      ([&] {
        // Stochastic rounding is required only when emb_t and cache_t are
        // not the same type and emb_t is not float
        const bool stochastic_rounding_ = stochastic_rounding &&
            !std::is_same<emb_t, float>::value &&
            !std::is_same<emb_t, cache_t>::value;

        at::PhiloxCudaState rng_engine_inputs;
        if (stochastic_rounding_) {
          auto gen = at::cuda::detail::getDefaultCUDAGenerator();
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)
                                  ->philox_cuda_state(4);
        }
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "lxu_cache_flush_kernel";
#endif
        lxu_cache_flush_kernel<emb_t, cache_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                MAKE_PTA_WITH_NAME(
                    func_name, cache_hash_size_cumsum, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(
                    func_name, cache_index_table_map, int32_t, 1, 64),
                MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name, lxu_cache_state, int64_t, 2, 32),
                MAKE_PTA_WITH_NAME(
                    func_name, lxu_cache_weights, cache_t, 2, 64),
                stochastic_rounding_,
                rng_engine_inputs);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

namespace {

// count the number of times that a cache_slot appears in lxu_cache_locations
// we actually only care about whether the number is 0 or > 0.
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_locations_count_kernel(
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits> count,
    FixedDivisor fd) {
  const int32_t N = lxu_cache_locations.size(0);
  CUDA_KERNEL_LOOP(n, N) {
    if (lxu_cache_locations[n] >= 0) {
      int32_t cache_set;
      int32_t slot;
      fd.DivMod(lxu_cache_locations[n], &cache_set, &slot);
      atomicAdd(&count[cache_set][slot], 1);
    }
  }
}

// if a cache_slot is in lxu_cache_locations (count > 0),
// decrement the counter of that cache_slot.
__global__
__launch_bounds__(kMaxThreads) void lxu_cache_locking_counter_decrement_kernel(
    pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits>
        lxu_cache_locking_counter,
    pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits> count) {
  const int32_t C = lxu_cache_locking_counter.size(0);
  for (int32_t i = blockIdx.x * blockDim.y + threadIdx.y; i < C;
       i += gridDim.x * blockDim.y) {
    const auto j = threadIdx.x;
    if (count[i][j] > 0) {
      lxu_cache_locking_counter[i][j] -= 1;
    }
  }
}

} // namespace

// for any cache_slot in lxu_cache_locations,
// decrement the counter of that cache_slot.
// duplicate cache_slot only decrement once.
void lxu_cache_locking_counter_decrement_cuda(
    at::Tensor lxu_cache_locking_counter,
    at::Tensor lxu_cache_locations) {
  TENSOR_ON_CUDA_GPU(lxu_cache_locking_counter);
  TENSOR_ON_CUDA_GPU(lxu_cache_locations);

  CUDA_DEVICE_GUARD(lxu_cache_locations);

  const auto N = lxu_cache_locations.numel();
  if (N == 0) {
    return;
  }

  auto count = at::zeros_like(lxu_cache_locking_counter);
  const int32_t C = lxu_cache_locking_counter.size(0);
  TORCH_CHECK(lxu_cache_locking_counter.size(1) == kWarpSize);
  auto fd = FixedDivisor(kWarpSize);

  const dim3 blocks(std::min(
      div_round_up(N, kMaxThreads),
      get_max_thread_blocks_for_cache_kernels_()));

#ifdef FBGEMM_GPU_MEMCHECK
  const char* func_name = "lxu_cache_locations_count_kernel";
#endif

  lxu_cache_locations_count_kernel<<<
      blocks,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, count, int32_t, 2, 32),
      fd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef FBGEMM_GPU_MEMCHECK
  const char* func_name2 = "lxu_cache_locking_counter_decrement_kernel";
#endif

  lxu_cache_locking_counter_decrement_kernel<<<
      std::min(
          div_round_up(C, kMaxThreads / kWarpSize),
          get_max_thread_blocks_for_cache_kernels_()),
      dim3(kWarpSize, kMaxThreads / kWarpSize),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      MAKE_PTA_WITH_NAME(func_name2, lxu_cache_locking_counter, int32_t, 2, 32),
      MAKE_PTA_WITH_NAME(func_name2, count, int32_t, 2, 32));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_lookup_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    int64_t invalid_index,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats,
    const int32_t* N_unique) {
  const int32_t C = lxu_cache_state.size(0);
  const int32_t N =
      N_unique == nullptr ? linear_cache_indices.size(0) : *N_unique;
  const int32_t n0 =
      blockIdx.x * blockDim.y * blockDim.x + threadIdx.y * blockDim.x;
  if (n0 >= N) {
    return;
  }

  int32_t cache_location = kCacheLocationMissing;
  int32_t n_indices = 0;
  int32_t n_hits = 0;
  const auto slot = threadIdx.x;
  for (int i = 0; i < blockDim.x; ++i) {
    int32_t n = n0 + i;
    if (n >= N) {
      continue;
    }
    const int64_t idx = linear_cache_indices[n];
    if (idx == invalid_index) {
      continue;
    }
    const int32_t cache_set = cache_slot(idx, C);
    n_indices++;
    const bool found =
        (::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx);
#ifdef USE_ROCM
    // FIXME: __ballot_sync with mask isn't supported by HIP yet.
    // See https://fburl.com/fvy7j0lq for the similar context.
    // assert false here with https://fburl.com/pfm7enw2
    assert(false);
    const auto bitmap = __ballot(found);
    if (bitmap) {
      const auto way = __ffsll(bitmap) - 1;
#else
    const auto bitmap = __ballot_sync(0xFFFFFFFF, found);
    if (bitmap) {
      // LSB == 1 hence we need to subtract one to get lane ID.
      const auto way = __ffs(bitmap) - 1;
#endif
      if (i == threadIdx.x) {
        cache_location = cache_set * kWarpSize + way;
      }
      n_hits++;
    }
  }

  const int32_t n = n0 + threadIdx.x;
  if (n < N) {
    lxu_cache_locations[n] = cache_location;
  }
  if (gather_cache_stats && threadIdx.x == 0 && n_indices > n_hits) {
    atomicAdd(
        &uvm_cache_stats[uvm_cache_stats_index::num_conflict_misses],
        (n_indices - n_hits));
  }
}

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void direct_mapped_lxu_cache_lookup_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    int64_t invalid_index,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats) {
  const int32_t C = lxu_cache_state.size(0);
  const int32_t N = linear_cache_indices.size(0);

  int32_t n_indices = 0;
  int32_t n_hits = 0;

  CUDA_KERNEL_LOOP(n, N) {
    int32_t cache_location = kCacheLocationMissing;
    const auto slot = 0;

    const int64_t idx = linear_cache_indices[n];
    if (idx == invalid_index) {
      continue;
    }

    const int32_t cache_set = cache_slot(idx, C);
    n_indices++;
    const bool found =
        (::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx);
    if (found) {
      cache_location = cache_set;
      n_hits++;
    }
    lxu_cache_locations[n] = cache_location;
  }

  if (gather_cache_stats) {
    typedef cub::BlockReduce<int32_t, kMaxThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    const int32_t conflict_miss = n_indices - n_hits;
    const int32_t conflict_miss_sum = BlockReduce(temp).Sum(conflict_miss);

    if (threadIdx.x == 0) {
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_conflict_misses],
          conflict_miss_sum);
    }
  }
}

} // namespace

/// Lookup the cache locations for each linear cache indices in
/// linear_cache_indices and return lxu_cache_locations
///
/// lxu_cache_locations A 1D tensor with the same length as
///                     linear_cache_indices.  It contains the cache locations
///                     (the row indices in the cache) of the corresponding
///                     indices in linear_cache_indices, i.e.,
///                     lxu_cache_locations[i] is the cache location for
///                     linear_cache_indices[i], where 0 <= i <
///                     linear_cache_indices.numel().
///
/// @param linear_cache_indices        Linear cache indices tensor (1D)
/// @param lxu_cache_state             LXU cache state tensor (2D tensor of
///                                    shape (# of cache sets, # of cache
///                                    slots per set)).  It contains linear
///                                    indices of rows that are in the
///                                    corresponding cache slots. If the cache
///                                    slot is empty, a sentinel value is
///                                    stored.
/// @param invalid_index               A sentinel value for linear cache
///                                    indices.  A cache index is skipped if it
///                                    is a sentinel value.
/// @param gather_cache_stats          A flag to enable/disable cache stats
///                                    collection.
/// @param uvm_cache_stats             A tensor for storing cache stats.
/// @param num_uniq_cache_indices      An optional GPU tensor that contains the
///                                    number of unique cache indices.  If this
///                                    tensor is passed, the kernel will only
///                                    lookup num_uniq_cache_indices number of
///                                    indices instead of looking up the entire
///                                    linear_cache_indices.
/// @param lxu_cache_locations_output  An optional output tensor.  If the
///                                    tensor is passed, the operator will not
///                                    allocate a new output tensor and use
///                                    this tensor as an output tensor.
DLL_PUBLIC Tensor lxu_cache_lookup_cuda(
    const Tensor linear_cache_indices,
    const Tensor lxu_cache_state,
    const int64_t invalid_index,
    const bool gather_cache_stats,
    const c10::optional<Tensor> uvm_cache_stats,
    const c10::optional<Tensor> num_uniq_cache_indices,
    const c10::optional<Tensor> lxu_cache_locations_output) {
  const auto uniq_lookup = num_uniq_cache_indices.has_value();
  // TODO: Support gather_cache_stats=true when uniq_lookup=true
  TORCH_CHECK(
      !uniq_lookup || !gather_cache_stats,
      "Unique lxu_cache_locations generation does not support gather_cache_stats=true");
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_cache_indices, lxu_cache_state, num_uniq_cache_indices);
  Tensor uvm_cache_stats_ =
      at::empty({0}, linear_cache_indices.options().dtype(at::kInt));
  if (gather_cache_stats) {
    TORCH_CHECK(uvm_cache_stats.has_value());
    uvm_cache_stats_ = uvm_cache_stats.value();
  }

  CUDA_DEVICE_GUARD(linear_cache_indices);

  const auto lxu_cache_locations =
      lxu_cache_locations_output.value_or(empty_like(
          linear_cache_indices,
          linear_cache_indices.options().dtype(at::kInt)));

  const auto N = linear_cache_indices.numel();
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return lxu_cache_locations;
  }

  const dim3 threads(kWarpSize, kMaxThreads / kWarpSize);
  const dim3 blocks(div_round_up(N, kMaxThreads));

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(), "lxu_cache_lookup_cuda", [&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "lxu_cache_lookup_kernel";
#endif
        lxu_cache_lookup_kernel<<<
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            MAKE_PTA_WITH_NAME(func_name, linear_cache_indices, index_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_state, int64_t, 2, 32),
            invalid_index,
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32),
            gather_cache_stats,
            MAKE_PTA_WITH_NAME(func_name, uvm_cache_stats_, int32_t, 1, 32),
            num_uniq_cache_indices.has_value()
                ? num_uniq_cache_indices.value().data_ptr<int32_t>()
                : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return lxu_cache_locations;
}

DLL_PUBLIC Tensor direct_mapped_lxu_cache_lookup_cuda(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    int64_t invalid_index,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_cache_indices, lxu_cache_state);
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(uvm_cache_stats, lxu_cache_state);

  if (gather_cache_stats) {
    TORCH_CHECK(uvm_cache_stats.has_value());
  }
  auto uvm_cache_stats_ = uvm_cache_stats.value_or(
      at::empty({0}, linear_cache_indices.options().dtype(at::kInt)));

  CUDA_DEVICE_GUARD(linear_cache_indices);

  const auto N = linear_cache_indices.numel();
  auto lxu_cache_locations = empty_like(
      linear_cache_indices, linear_cache_indices.options().dtype(at::kInt));
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return lxu_cache_locations;
  }

  const dim3 blocks(div_round_up(N, kMaxThreads));

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(),
      "direct_mapped_lxu_cache_lookup_cuda",
      [&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "direct_mapped_lxu_cache_lookup_kernel";
#endif
        direct_mapped_lxu_cache_lookup_kernel<<<
            blocks,
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            MAKE_PTA_WITH_NAME(func_name, linear_cache_indices, index_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_state, int64_t, 2, 32),
            invalid_index,
            MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32),
            gather_cache_stats,
            MAKE_PTA_WITH_NAME(func_name, uvm_cache_stats_, int32_t, 1, 32));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return lxu_cache_locations;
}

namespace {

__global__
__launch_bounds__(kMaxThreads) void lxu_cache_locations_update_kernel(
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations_new,
    const int32_t* N_unique) {
  const auto N = N_unique == nullptr ? lxu_cache_locations.size(0) : *N_unique;
  CUDA_KERNEL_LOOP(n, N) {
    if (N_unique != nullptr ||
        (lxu_cache_locations[n] == kCacheLocationMissing &&
         lxu_cache_locations_new[n] >= 0)) {
      lxu_cache_locations[n] = lxu_cache_locations_new[n];
    }
  }
}

} // namespace

DLL_PUBLIC void lxu_cache_locations_update_cuda(
    Tensor lxu_cache_locations,
    Tensor lxu_cache_locations_new,
    c10::optional<Tensor> num_uniq_cache_indices) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      lxu_cache_locations, lxu_cache_locations_new, num_uniq_cache_indices);

  CUDA_DEVICE_GUARD(lxu_cache_locations);

  const auto N = lxu_cache_locations.numel();

  if (N == 0) {
    return;
  }

  const dim3 blocks(std::min(
      div_round_up(N, kMaxThreads),
      get_max_thread_blocks_for_cache_kernels_()));

#ifdef FBGEMM_GPU_MEMCHECK
  const char* func_name = "lxu_cache_locations_update_kernel";
#endif

  lxu_cache_locations_update_kernel<<<
      blocks,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations_new, int32_t, 1, 32),
      num_uniq_cache_indices.has_value()
          ? num_uniq_cache_indices.value().data_ptr<int32_t>()
          : nullptr);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return;
}
