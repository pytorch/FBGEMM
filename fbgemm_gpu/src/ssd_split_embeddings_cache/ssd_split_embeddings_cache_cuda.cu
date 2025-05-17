/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDADeviceAssertion.h>
#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/utils/bitonic_sort.cuh"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "fbgemm_gpu/utils/tensor_utils.h"
#include "fbgemm_gpu/utils/vec4.cuh"

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

int get_masked_index_default_pipeline_sms(int device) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  // The default number of SMs for use_pipeline=true is set based on an
  // empirical study
  if (prop.major == 8) {
    // Assume A100
    return 4;
  } else if (prop.major == 9) {
    // Assume H100
    return 16;
  }
  constexpr int ALL_TO_PREFETCH_SM_RATIO = 8;
  return div_round_up(get_device_sm_cnt_(), ALL_TO_PREFETCH_SM_RATIO);
}

template <typename scalar_t>
DEVICE_INLINE void
vec4_copy(scalar_t* dst, const scalar_t* src, const int32_t D) {
  constexpr int32_t VEC_SIZE = 4;
  const scalar_t* __restrict__ src_ = src;
  scalar_t* __restrict__ dst_ = dst;
  for (auto d = threadIdx.x * VEC_SIZE; d < D; d += blockDim.x * VEC_SIZE) {
    Vec4T<scalar_t>::copy(&src_[d], &dst_[d]);
  }
}

template <>
DEVICE_INLINE void
vec4_copy(uint8_t* dst, const uint8_t* src, const int32_t D) {
  // each row is padded with row_alignment (16 bytes on GPUs), so each row will
  // be multiple of 16 bytes (uint4 = 32bit x 4 = 16 bytes).
  const uint4* __restrict__ src_ = reinterpret_cast<const uint4*>(src);
  uint4* __restrict__ dst_ = reinterpret_cast<uint4*>(dst);
  for (auto d = threadIdx.x; d * sizeof(uint4) < D; d += blockDim.x) {
    dst_[d] = src_[d];
  }
}

template <typename value_t, typename index_t, bool is_index_put>
__global__ __launch_bounds__(kMaxThreads) void masked_index_kernel(
    pta::PackedTensorAccessor64<value_t, 2, at::RestrictPtrTraits> self,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor64<value_t, 2, at::RestrictPtrTraits> values,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        count) {
  const int32_t N = indices.size(0);
  const auto count_ = count[0];
  CUDA_KERNEL_ASSERT(count_ <= N);
  for (auto n = blockIdx.x * blockDim.y + threadIdx.y; n < count_;
       n += blockDim.y * gridDim.x) {
    // idx == -1 if it is conflict miss
    const auto idx = indices[n];
    if (idx < 0) {
      continue;
    }
    const auto D = self.size(1);
    const auto self_idx = is_index_put ? idx : n;
    const auto values_idx = is_index_put ? n : idx;
    vec4_copy(&self[self_idx][0], &values[values_idx][0], D);
  }
}

template <bool is_index_put>
Tensor masked_index_impl(
    const Tensor& self,
    const Tensor& indices,
    const Tensor& values,
    const Tensor& count,
    const bool use_pipeline,
    const int preferred_sms) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(self, indices, values, count);
  TENSOR_CONTIGUOUS(self);
  TENSOR_CONTIGUOUS(indices);
  TENSOR_CONTIGUOUS(values);

  CUDA_DEVICE_GUARD(self);

  const auto N = indices.numel();
  if (N == 0) {
    return self;
  }
  const auto D = self.size(1);
  TORCH_CHECK_EQ(self.size(1), values.size(1));

  const int32_t tx = std::min<int32_t>(D / 4, kMaxThreads);
  const dim3 threads(tx, kMaxThreads / tx);

  const auto full_grid_size = div_round_up(N, kMaxThreads / tx);

  static int masked_index_default_pipeline_sms =
      get_masked_index_default_pipeline_sms(at::cuda::current_device());

  const int pipeline_grid_size =
      preferred_sms == -1 ? masked_index_default_pipeline_sms : preferred_sms;
  TORCH_CHECK(
      !use_pipeline || pipeline_grid_size >= 1, "preferred_sms must >= 1");

  // Use a fraction of SMs if use_pipeline=true
  const auto grid_size = use_pipeline
      ? std::min(pipeline_grid_size, full_grid_size)
      : full_grid_size;

  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      self.scalar_type(),
      is_index_put ? "masked_index_put" : "masked_index_select",
      [&] {
        using value_t = scalar_t;
#ifdef FBGEMM_GPU_MEMCHECK
        const auto func_name = is_index_put ? "masked_index_put_kernel"
                                            : "masked_index_select_kernel";
#endif
        if (std::is_same_v<value_t, uint8_t>) {
          TORCH_CHECK(D % 16 == 0, "D needs to be padded to be multiple of 16");
        }
        FBGEMM_DISPATCH_INTEGRAL_TYPES(
            indices.scalar_type(),
            is_index_put ? "masked_index_put" : "masked_index_select",
            [&] {
              using index_t = scalar_t;
              FBGEMM_LAUNCH_KERNEL(
                  (masked_index_kernel<value_t, index_t, is_index_put>),
                  grid_size,
                  dim3(tx, kMaxThreads / tx),
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(self, value_t, 2, 64),
                  PTA_B(indices, index_t, 1, 32),
                  PTA_B(values, value_t, 2, 64),
                  PTA_B(count, int32_t, 1, 32));
            } // lambda for FBGEMM_DISPATCH_INTEGRAL_TYPES
        );
      } // lambda for FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE
  );

  return self;
}

Tensor masked_index_put_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count,
    const bool use_pipeline,
    const int64_t preferred_sms) {
  return masked_index_impl</*is_index_put=*/true>(
      self, indices, values, count, use_pipeline, preferred_sms);
}

Tensor masked_index_select_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count,
    const bool use_pipeline,
    const int64_t preferred_sms) {
  return masked_index_impl</*is_index_put=*/false>(
      self, indices, values, count, use_pipeline, preferred_sms);
}

__global__ __launch_bounds__(kMaxThreads) void ssd_cache_actions_insert_kernel(
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_cache_sets, // [N = \sum_{b} L_{b} total indices, i.e.
                           // flattened
                           // [B][L]
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices, // [N = \sum_{b} L_{b} total indices, i.e.
                                  // flattened [B][L]
    int64_t time_stamp,
    int64_t prefetch_dist, // Number of batches we can prefetch ahead of a
                           // forward call A value of 1 means that entries where
                           // timestep with insert_time >= time_stamp -
                           // prefetch_dist are locked, and cannot be evicted.
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        assigned_cache_slots,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        evicted_indices,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        actions_count,
    const bool lock_cache_line,
    pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits>
        lxu_cache_locking_counter,
    TORCH_DSA_KERNEL_ARGS) {
  // Number of cache sets
  const int32_t C = lxu_cache_state.size(0);

  const int32_t N = sorted_cache_sets.size(0);
  const auto n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }

  const int32_t cache_set = sorted_cache_sets[n];

  // Set actions_count. It is basically the sum of all SLs. Since cache sets
  // are sorted in sorted_cache_sets, we can count the number of elements that
  // are not C by finding the position of the last cache set that is not C
  if (threadIdx.x == 0) {
    // Zero cache misses (the first sorted_cache_sets is C) or
    // some cache misses (some sorted_cache_sets are C)
    if (cache_set == C && (n == 0 || sorted_cache_sets[n - 1] != C)) {
      actions_count[0] = n;
    }
    // All cache misses (none of sorted_cache_sets is C)
    else if (n == N - 1 && cache_set != C) {
      actions_count[0] = N;
    }
  }

  if (cache_set >= C) {
    if (threadIdx.x == 0) {
      // Ignore the already-existing elements
      evicted_indices[n] = -1;
      assigned_cache_slots[n] = -1;
    }
    return;
  }

  // check if this warp is responsible for this whole segment.
  const bool segment_start = (n == 0 || sorted_cache_sets[n - 1] != cache_set);
  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so
    // we can just exit this warp entirely.
    return;
  }

  int32_t SL = 1;
  while (n + SL < N && sorted_cache_sets[n + SL] == cache_set) {
    SL += 1;
  }

  // Now, we need to insert the (unique!) values in indices[n:n + SL] into
  // our slots.
  const auto slot = threadIdx.x;
  const int64_t slot_time = lru_state[cache_set][slot];

  // Check if the slot is locked
  const bool is_slot_locked =
      lock_cache_line && (lxu_cache_locking_counter[cache_set][slot] > 0);
  // Check if the slot has the inserted row that was a cache hit.
  const int64_t slot_idx = lxu_cache_state[cache_set][slot];
  const bool slot_has_idx = slot_idx != -1 && slot_time == time_stamp;
  // Check if the slot is unavailable: either it is locked or contains
  // a cache hit inserted row
  const bool is_slot_unavailable = is_slot_locked || slot_has_idx;

  // Set the slot cost: if the slot is not available, set it to the
  // maximum timestamp which is the current timestamp. After sorting,
  // the unavailable slots will be in the bottom, while the available
  // slots will be bubbled to the top
  const int64_t slot_cost = is_slot_unavailable ? time_stamp : slot_time;

  // Prepare key-value pair for sorting
  int64_t costs[1] = {slot_cost};
  int64_t slots[1] = {slot};

  // Sort the slots based on their costs
  BitonicSort<int64_t, int64_t, 1, Comparator<int64_t>>::sort(costs, slots);

  // Get the sorted results
  const int64_t insert_slot = slots[0];
  const int64_t insert_cost = costs[0];

  auto l = threadIdx.x;

  // Get the current index
  const int64_t current_idx = shfl_sync(slot_idx, insert_slot);

  // Insert rows
  if (l < SL) {
    // Insert indices
    const int64_t insert_idx = cache_set_sorted_indices[n + l];

    if (insert_cost == time_stamp) {
      // Skip this slot as it is not available
      evicted_indices[n + l] = -1;
      assigned_cache_slots[n + l] = -1;
    } else {
      evicted_indices[n + l] = current_idx; // -1 if not set, >= 0 if valid.
      assigned_cache_slots[n + l] = cache_set * kWarpSize + insert_slot;

      // TODO: Check if we can do contiguous writes here.
      // Update cache states
      lxu_cache_state[cache_set][insert_slot] = insert_idx;
      lru_state[cache_set][insert_slot] = time_stamp;

      // Lock cache line
      if (lock_cache_line) {
        lxu_cache_locking_counter[cache_set][insert_slot] += 1;
      }
    }
  }

  // Conflict misses
  for (auto l = kWarpSize + threadIdx.x; l < SL; l += kWarpSize) {
    evicted_indices[n + l] = -1;
    assigned_cache_slots[n + l] = -1;
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
ssd_cache_populate_actions_cuda(
    Tensor linear_indices,
    int64_t total_hash_size,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    int64_t prefetch_dist,
    Tensor lru_state,
    bool gather_cache_stats,
    std::optional<Tensor> ssd_cache_stats,
    const bool lock_cache_line,
    const std::optional<Tensor>& lxu_cache_locking_counter) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_indices, lxu_cache_state, lru_state, lxu_cache_locking_counter);

  CUDA_DEVICE_GUARD(linear_indices);

  // Get unique indices
  auto
      [unique_indices,
       unique_indices_length,
       unique_indices_count,
       linear_index_inverse_indices] =
          get_unique_indices_with_inverse_cuda(
              linear_indices,
              total_hash_size,
              /*compute_count=*/true,
              /*compute_inverse_indices=*/true);

  TORCH_CHECK(linear_index_inverse_indices.has_value());
  TORCH_CHECK(unique_indices_count.has_value());
  const auto unique_indices_count_cumsum =
      asynchronous_complete_cumsum_gpu(unique_indices_count.value())
          .to(at::kInt);

  TORCH_CHECK_LT(unique_indices.numel(), std::numeric_limits<int32_t>::max());
  const int32_t N = unique_indices.numel();

  auto evicted_indices = empty_like(unique_indices);
  const auto int_options = unique_indices.options().dtype(at::kInt);
  auto assigned_cache_slots = empty_like(unique_indices, int_options);

  Tensor ssd_cache_stats_ = at::empty({0}, int_options);
  if (gather_cache_stats) {
    TORCH_CHECK(ssd_cache_stats.has_value());
    ssd_cache_stats_ = ssd_cache_stats.value();
    TENSOR_ON_CUDA_GPU(ssd_cache_stats_);
  }

  if (unique_indices.numel() == 0) {
    auto actions_count = at::zeros({1}, int_options);
    // these are all of length zero
    return std::make_tuple(
        empty_like(unique_indices),
        evicted_indices,
        assigned_cache_slots,
        actions_count,
        /*linear_index_inverse_indices=*/at::empty({0}, int_options),
        /*unique_indices_count_cumsum=*/at::empty({0}, int_options),
        /*cache_set_inverse_indices=*/at::empty({0}, int_options),
        /*cache_set_inverse_indices=*/at::empty({0}, int_options));
  }

  Tensor lxu_cache_locking_counter_;
  if (lock_cache_line) {
    TORCH_CHECK(lxu_cache_locking_counter.has_value());
    lxu_cache_locking_counter_ = lxu_cache_locking_counter.value();
  } else {
    lxu_cache_locking_counter_ =
        at::empty({0, 0}, lxu_cache_state.options().dtype(at::kInt));
  }

  auto actions_count = at::empty({1}, int_options);
  // Find uncached indices
  auto
      [sorted_cache_sets,
       cache_set_sorted_unique_indices,
       cache_set_inverse_indices] =
          lru_cache_find_uncached_cuda(
              unique_indices,
              unique_indices_length,
              total_hash_size,
              lxu_cache_state,
              time_stamp,
              lru_state,
              gather_cache_stats,
              ssd_cache_stats_,
              lock_cache_line,
              lxu_cache_locking_counter_,
              /*compute_inverse_indices=*/true);

  TORCH_CHECK(cache_set_inverse_indices.has_value());

#ifdef FBGEMM_GPU_MEMCHECK
  const auto func_name = "ssd_cache_actions_insert_kernel";
#endif

  TORCH_DSA_KERNEL_LAUNCH(
      ssd_cache_actions_insert_kernel,
      div_round_up(N, kMaxThreads / kWarpSize),
      dim3(kWarpSize, kMaxThreads / kWarpSize),
      0,
      at::cuda::getCurrentCUDAStream(),
      MAKE_PTA_WITH_NAME(func_name, lxu_cache_state, int64_t, 2, 32),
      MAKE_PTA_WITH_NAME(func_name, sorted_cache_sets, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(
          func_name, cache_set_sorted_unique_indices, int64_t, 1, 32),
      time_stamp,
      prefetch_dist,
      MAKE_PTA_WITH_NAME(func_name, lru_state, int64_t, 2, 32),
      MAKE_PTA_WITH_NAME(func_name, assigned_cache_slots, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, evicted_indices, int64_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, actions_count, int32_t, 1, 32),
      lock_cache_line,
      MAKE_PTA_WITH_NAME(
          func_name, lxu_cache_locking_counter_, int32_t, 2, 32));

  return std::make_tuple(
      cache_set_sorted_unique_indices,
      evicted_indices,
      assigned_cache_slots,
      actions_count,
      linear_index_inverse_indices.value(),
      unique_indices_count_cumsum,
      cache_set_inverse_indices.value(),
      unique_indices_length);
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void ssd_generate_row_addrs_kernel(
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        ssd_row_addrs,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        post_bwd_evicted_indices,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        assigned_cache_slots,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        linear_index_inverse_indices,
    // TODO: Use int64_t here
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        unique_indices_count_cumsum,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        cache_set_inverse_indices,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_unique_indices,
    const uint64_t lxu_cache_weights_addr,
    const uint64_t inserted_ssd_weights_addr,
    const int* N_unique,
    const uint64_t cache_row_bytes // has to be 64 bits to prevent overflow
) {
  const auto n = blockDim.y * blockIdx.x + threadIdx.y;
  if (n >= *N_unique) {
    return;
  }

  const auto cache_set_id = cache_set_inverse_indices[n];
  const auto segment_start = unique_indices_count_cumsum[cache_set_id];
  const auto segment_end = unique_indices_count_cumsum[cache_set_id + 1];
  // Cache locations
  const auto cache_loc =
      lxu_cache_locations[linear_index_inverse_indices[segment_start]];

  const uint64_t ptr_addr = (cache_loc == -1)
      // Conflict miss
      ? (inserted_ssd_weights_addr + (n * cache_row_bytes))
      // Not conflict miss
      : (lxu_cache_weights_addr + (cache_loc * cache_row_bytes));

  // Set post backward evicted indices
  if (assigned_cache_slots[n] == -1 && cache_loc == -1) {
    post_bwd_evicted_indices[n] = cache_set_sorted_unique_indices[n];
  } else {
    post_bwd_evicted_indices[n] = -1;
  }

  // Set pointer address
  for (auto l = segment_start + threadIdx.x; l < segment_end; l += blockDim.x) {
    auto dst = linear_index_inverse_indices[l];
    *reinterpret_cast<uint64_t*>(&ssd_row_addrs[dst]) = ptr_addr;
  }
}

std::tuple<Tensor, Tensor> ssd_generate_row_addrs_cuda(
    const Tensor& lxu_cache_locations,
    const Tensor& assigned_cache_slots,
    const Tensor& linear_index_inverse_indices,
    const Tensor& unique_indices_count_cumsum,
    const Tensor& cache_set_inverse_indices,
    const Tensor& lxu_cache_weights,
    const Tensor& inserted_ssd_weights,
    const Tensor& unique_indices_length,
    const Tensor& cache_set_sorted_unique_indices) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      lxu_cache_locations,
      assigned_cache_slots,
      linear_index_inverse_indices,
      unique_indices_count_cumsum,
      cache_set_inverse_indices,
      lxu_cache_weights,
      inserted_ssd_weights,
      unique_indices_length,
      cache_set_sorted_unique_indices);

  CUDA_DEVICE_GUARD(lxu_cache_locations);

  const auto ssd_row_addrs = at::zeros(
      {lxu_cache_locations.numel()},
      lxu_cache_locations.options().dtype(at::kLong));
  const auto post_bwd_evicted_indices = at::empty_like(ssd_row_addrs);

  constexpr auto kNumWarps = kMaxThreads / kWarpSize;
  const auto cache_row_bytes =
      lxu_cache_weights.size(1) * lxu_cache_weights.element_size();
  const auto lxu_cache_weights_addr =
      reinterpret_cast<uint64_t>(lxu_cache_weights.data_ptr());

  // All rows are hit in the cache
  if (lxu_cache_locations.numel() == 0) {
    // TODO: make this more efficient
    return {ssd_row_addrs, post_bwd_evicted_indices};
  }

  FBGEMM_DISPATCH_INTEGRAL_TYPES(
      assigned_cache_slots.scalar_type(),
      "ssd_generate_row_addrs",
      [&] {
        using index_t = scalar_t;
        FBGEMM_LAUNCH_KERNEL(
            (ssd_generate_row_addrs_kernel<index_t>),
            div_round_up(lxu_cache_locations.numel(), kNumWarps),
            dim3(kWarpSize, kNumWarps),
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(ssd_row_addrs, int64_t, 1, 32),
            PTA_B(post_bwd_evicted_indices, int64_t, 1, 32),
            PTA_B(lxu_cache_locations, int32_t, 1, 32),
            PTA_B(assigned_cache_slots, index_t, 1, 32),
            PTA_B(linear_index_inverse_indices, int32_t, 1, 32),
            PTA_B(unique_indices_count_cumsum, int32_t, 1, 32),
            PTA_B(cache_set_inverse_indices, int32_t, 1, 32),
            PTA_B(cache_set_sorted_unique_indices, int64_t, 1, 32),
            lxu_cache_weights_addr,
            reinterpret_cast<uint64_t>(inserted_ssd_weights.data_ptr()),
            unique_indices_length.data_ptr<int32_t>(),
            cache_row_bytes);
      } // lambda
  );

  return {ssd_row_addrs, post_bwd_evicted_indices};
}

__global__ __launch_bounds__(kMaxThreads) void ssd_update_row_addrs_kernel(
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        ssd_row_addrs_curr,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        ssd_curr_next_map,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations_curr,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        linear_index_inverse_indices_curr,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        unique_indices_count_cumsum_curr,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        cache_set_inverse_indices_curr,
    const uint64_t lxu_cache_weights_addr,
    const uint64_t inserted_ssd_weights_addr_next,
    const int* N_unique_curr,
    const uint64_t cache_row_bytes // has to be 64 bits to prevent overflow
) {
  const auto n_curr = blockDim.y * blockIdx.x + threadIdx.y;
  if (n_curr >= *N_unique_curr) {
    return;
  }

  // Find mapping between n_curr and n_next
  const auto n_next = ssd_curr_next_map[n_curr];

  // Return if the row is not used in both previous and next iterations
  if (n_next < 0) {
    return;
  }

  // Find out if the row gets moved to the nextent iteration's scratch pad or
  // L1 by checking the lxu_cache_locations_curr
  const auto cache_set_id_curr = cache_set_inverse_indices_curr[n_curr];
  const auto segment_start_curr =
      unique_indices_count_cumsum_curr[cache_set_id_curr];
  const auto segment_end_curr =
      unique_indices_count_cumsum_curr[cache_set_id_curr + 1];
  const auto cache_loc_curr = lxu_cache_locations_curr
      [linear_index_inverse_indices_curr[segment_start_curr]];

  const uint64_t ptr_addr = (cache_loc_curr == -1)
      // The row is moved from the previous iteration's scratch pad to the
      // next iteration's scratch pad
      ? (inserted_ssd_weights_addr_next + (n_next * cache_row_bytes))
      // The row is moved from the previous iteration's scratch pad to L1 cache
      : (lxu_cache_weights_addr + (cache_loc_curr * cache_row_bytes));

  // Set pointer address
  for (auto l = segment_start_curr + threadIdx.x; l < segment_end_curr;
       l += blockDim.x) {
    auto dst = linear_index_inverse_indices_curr[l];
    *reinterpret_cast<uint64_t*>(&ssd_row_addrs_curr[dst]) = ptr_addr;
  }
}

void ssd_update_row_addrs_cuda(
    const Tensor& ssd_row_addrs_curr,
    const Tensor& ssd_curr_next_map,
    const Tensor& lxu_cache_locations_curr,
    const Tensor& linear_index_inverse_indices_curr,
    const Tensor& unique_indices_count_cumsum_curr,
    const Tensor& cache_set_inverse_indices_curr,
    const Tensor& lxu_cache_weights,
    const Tensor& inserted_ssd_weights_next,
    const Tensor& unique_indices_length_curr) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      ssd_row_addrs_curr,
      ssd_curr_next_map,
      lxu_cache_locations_curr,
      linear_index_inverse_indices_curr,
      unique_indices_count_cumsum_curr,
      cache_set_inverse_indices_curr,
      lxu_cache_weights,
      inserted_ssd_weights_next,
      unique_indices_length_curr);

  CUDA_DEVICE_GUARD(ssd_row_addrs_curr);

  const auto lxu_cache_weights_addr =
      reinterpret_cast<uint64_t>(lxu_cache_weights.data_ptr());
  const auto inserted_ssd_weights_addr_next =
      reinterpret_cast<uint64_t>(inserted_ssd_weights_next.data_ptr());
  const auto cache_row_bytes =
      lxu_cache_weights.size(1) * lxu_cache_weights.element_size();
  constexpr auto kNumWarps = kMaxThreads / kWarpSize;

  FBGEMM_LAUNCH_KERNEL(
      (ssd_update_row_addrs_kernel),
      div_round_up(ssd_row_addrs_curr.numel(), kNumWarps),
      dim3(kWarpSize, kNumWarps),
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(ssd_row_addrs_curr, int64_t, 1, 32),
      PTA_B(ssd_curr_next_map, int32_t, 1, 32),
      PTA_B(lxu_cache_locations_curr, int32_t, 1, 32),
      PTA_B(linear_index_inverse_indices_curr, int32_t, 1, 32),
      PTA_B(unique_indices_count_cumsum_curr, int32_t, 1, 32),
      PTA_B(cache_set_inverse_indices_curr, int32_t, 1, 32),
      lxu_cache_weights_addr,
      inserted_ssd_weights_addr_next,
      unique_indices_length_curr.data_ptr<int32_t>(),
      cache_row_bytes);
}

template <typename index_t, typename offset_t>
__global__ __launch_bounds__(kMaxThreads) void compact_indices_kernel(
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        compact_indices,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        compact_count,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<offset_t, 1, at::RestrictPtrTraits>
        offsets,
    const pta::PackedTensorAccessor32<offset_t, 1, at::RestrictPtrTraits> masks,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        count) {
  const auto n = blockIdx.x * blockDim.x + threadIdx.x;
  const auto N = masks.size(0);
  const auto count_ = count[0];
  CUDA_KERNEL_ASSERT(count_ <= N);
  if (n == count_) {
    compact_count[0] = offsets[count_];
  }
  if (n >= N || n >= count[0]) {
    return;
  }
  if (masks[n] == 1) {
    compact_indices[offsets[n]] = indices[n];
  }
}

void compact_indices_cuda(
    std::vector<Tensor> compact_indices,
    Tensor compact_count,
    std::vector<Tensor> indices,
    Tensor masks,
    Tensor count) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(compact_count, masks, count);
  CUDA_DEVICE_GUARD(masks);

  const auto num_tensors = indices.size();
  TORCH_CHECK(
      num_tensors > 0, "compact_indices expects at least 1 indices tensor");
  TORCH_CHECK(
      compact_indices.size() == num_tensors,
      "compact_indices and indices must have the same shape");

  for (auto i = 0; i < num_tensors; ++i) {
    const auto& indices_ = indices[i];
    const auto& compact_indices_ = compact_indices[i];
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(masks, indices_, compact_indices_);
    TENSORS_HAVE_SAME_NUMEL(masks, indices_);
  }

  const auto N = masks.numel();
  if (N == 0) {
    compact_count.fill_(0);
    return;
  }

  const auto offsets = asynchronous_complete_cumsum_gpu(masks);
#ifdef FBGEMM_GPU_MEMCHECK
  const auto func_name = "compact_indices_kernel";
#endif

  for (auto i = 0; i < num_tensors; ++i) {
    FBGEMM_DISPATCH_INTEGRAL_TYPES(
        indices[i].scalar_type(),
        "compact_indices",
        [&] {
          using index_t = scalar_t;
          FBGEMM_DISPATCH_INTEGRAL_TYPES(
              offsets.scalar_type(), "compact_indices", [&] {
                using offset_t = scalar_t;
                FBGEMM_LAUNCH_KERNEL(
                    (compact_indices_kernel<index_t, offset_t>),
                    // Launch N + 1 threads because we need at least one thread
                    // to set compact_count
                    div_round_up(N + 1, kMaxThreads),
                    kMaxThreads,
                    0,
                    at::cuda::getCurrentCUDAStream(),
                    PTA_B(compact_indices[i], index_t, 1, 32),
                    PTA_B(compact_count, int32_t, 1, 32),
                    PTA_B(indices[i], index_t, 1, 32),
                    PTA_B(offsets, offset_t, 1, 32),
                    PTA_B(masks, offset_t, 1, 32),
                    PTA_B(count, int32_t, 1, 32));
              });
        } // lambda
    );
  }

  return;
}
