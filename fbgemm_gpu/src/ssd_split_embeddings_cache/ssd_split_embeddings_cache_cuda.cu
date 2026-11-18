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
#include <c10/util/Float8_e4m3fn.h>
#include <limits>
#include <optional>
// clang-format off
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh"
#include <cub/block/block_scan.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh"
// clang-format on
#include <ATen/cuda/Atomic.cuh>
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/utils/bitonic_sort.cuh"
#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/cuda_utilities.cuh"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/find_qparams.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "fbgemm_gpu/utils/tensor_utils.h"
#include "fbgemm_gpu/utils/vec4.cuh"

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

int get_masked_index_default_pipeline_sms(int device) {
  cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

#ifdef USE_ROCM
  // AMD GPUs: CU count varies by SKU. Use a conservative ratio.
  // MI300X has 304 CUs, MI350 has ~320 CUs.
  constexpr int ALL_TO_PREFETCH_CU_RATIO = 8;
  return div_round_up(prop.multiProcessorCount, ALL_TO_PREFETCH_CU_RATIO);
#else
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
#endif
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

template <>
DEVICE_INLINE void vec4_copy(int8_t* dst, const int8_t* src, const int32_t D) {
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
  CUDA_KERNEL_ASSERT(count_ <= N && "masked index count exceeds input size");
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

  const auto full_grid_size = utils::cuda::cap_grid_dim_x(
      div_round_up(N, kMaxThreads / tx),
      kMaxThreads,
      at::cuda::getCurrentCUDAStream());

  static int masked_index_default_pipeline_sms =
      get_masked_index_default_pipeline_sms(at::cuda::current_device());

  const int pipeline_grid_size =
      preferred_sms == -1 ? masked_index_default_pipeline_sms : preferred_sms;
  TORCH_CHECK(
      !use_pipeline || pipeline_grid_size >= 1, "preferred_sms must >= 1");

  // Use a fraction of SMs if use_pipeline=true.
  // cap_grid_dim_x returns uint32_t, so pin the comparison type explicitly;
  // pipeline_grid_size is >= 1 here (enforced by the TORCH_CHECK above) so the
  // conversion to uint32_t is safe.
  const auto grid_size = use_pipeline
      ? std::min<uint32_t>(pipeline_grid_size, full_grid_size)
      : full_grid_size;

  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      self.scalar_type(),
      is_index_put ? "masked_index_put" : "masked_index_select",
      [&] {
        using value_t = scalar_t;
        if constexpr (
            std::is_same_v<value_t, uint8_t> ||
            std::is_same_v<value_t, int8_t>) {
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
        atomicAdd(&lxu_cache_locking_counter[cache_set][insert_slot], 1);
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
  at::Tensor sorted_cache_sets;
  at::Tensor cache_set_sorted_unique_indices;
  std::optional<at::Tensor> cache_set_inverse_indices;
  std::tie(
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      cache_set_inverse_indices) =
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

  FBGEMM_LAUNCH_DSA_KERNEL(
      ssd_cache_actions_insert_kernel,
      div_round_up(N, kMaxThreads / kWarpSizeHost()),
      dim3(kWarpSizeHost(), kMaxThreads / kWarpSizeHost()),
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(lxu_cache_state, int64_t, 2, 32),
      PTA_B(sorted_cache_sets, int32_t, 1, 32),
      PTA_B(cache_set_sorted_unique_indices, int64_t, 1, 32),
      time_stamp,
      prefetch_dist,
      PTA_B(lru_state, int64_t, 2, 32),
      PTA_B(assigned_cache_slots, int32_t, 1, 32),
      PTA_B(evicted_indices, int64_t, 1, 32),
      PTA_B(actions_count, int32_t, 1, 32),
      lock_cache_line,
      PTA_B(lxu_cache_locking_counter_, int32_t, 2, 32));

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
#ifdef USE_ROCM
  for (auto n = blockDim.y * blockIdx.x + threadIdx.y; n < *N_unique;
       n += blockDim.y * gridDim.x) {
#else
  const auto n = blockDim.y * blockIdx.x + threadIdx.y;
  if (n >= *N_unique) {
    return;
  }
#endif

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
    for (auto l = segment_start + threadIdx.x; l < segment_end;
         l += blockDim.x) {
      auto dst = linear_index_inverse_indices[l];
      *reinterpret_cast<uint64_t*>(&ssd_row_addrs[dst]) = ptr_addr;
    }
#ifdef USE_ROCM
  } // for n (grid-stride loop, ROCm only)
#endif
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

  const auto kNumWarps = kMaxThreads / kWarpSizeHost();
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
            utils::cuda::cap_grid_dim_x(
                div_round_up(lxu_cache_locations.numel(), kNumWarps),
                kMaxThreads,
                at::cuda::getCurrentCUDAStream()),
            dim3(kWarpSizeHost(), kNumWarps),
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
#ifdef USE_ROCM
  for (auto n_curr = blockDim.y * blockIdx.x + threadIdx.y;
       n_curr < *N_unique_curr;
       n_curr += blockDim.y * gridDim.x) {
#else
  const auto n_curr = blockDim.y * blockIdx.x + threadIdx.y;
  if (n_curr >= *N_unique_curr) {
    return;
  }
#endif

    // Find mapping between n_curr and n_next
    const auto n_next = ssd_curr_next_map[n_curr];

    // Return if the row is not used in both previous and next iterations
    if (n_next < 0) {
#ifdef USE_ROCM
      continue;
#else
    return;
#endif
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
        // The row is moved from the previous iteration's scratch pad to L1
        // cache
        : (lxu_cache_weights_addr + (cache_loc_curr * cache_row_bytes));

    // Set pointer address
    for (auto l = segment_start_curr + threadIdx.x; l < segment_end_curr;
         l += blockDim.x) {
      auto dst = linear_index_inverse_indices_curr[l];
      *reinterpret_cast<uint64_t*>(&ssd_row_addrs_curr[dst]) = ptr_addr;
    }
#ifdef USE_ROCM
  } // for n_curr (grid-stride loop, ROCm only)
#endif
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
  const auto kNumWarps = kMaxThreads / kWarpSizeHost();

  FBGEMM_LAUNCH_KERNEL(
      (ssd_update_row_addrs_kernel),
      utils::cuda::cap_grid_dim_x(
          div_round_up(ssd_row_addrs_curr.numel(), kNumWarps),
          kMaxThreads,
          at::cuda::getCurrentCUDAStream()),
      dim3(kWarpSizeHost(), kNumWarps),
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
  const auto N = masks.size(0);
  const auto count_ = count[0];
  CUDA_KERNEL_ASSERT(count_ <= N && "compact index count exceeds mask size");
#ifdef USE_ROCM
  // Cap the grid on ROCm (HIP 2^32 threads-per-launch limit); grid-stride over
  // n (bound N + 1 to preserve the n == count_ side-effect write).
  for (auto n = blockIdx.x * blockDim.x + threadIdx.x; n < N + 1;
       n += blockDim.x * gridDim.x) {
#else
  const auto n = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (n == count_) {
      compact_count[0] = offsets[count_];
    }
    if (n >= N || n >= count[0]) {
#ifdef USE_ROCM
      continue;
#else
    return;
#endif
    }
    if (masks[n] == 1) {
      compact_indices[offsets[n]] = indices[n];
    }
#ifdef USE_ROCM
  } // for n (grid-stride loop, ROCm only)
#endif
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
                    utils::cuda::cap_grid_dim_x_from_workload(
                        N + 1, kMaxThreads, at::cuda::getCurrentCUDAStream()),
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

namespace {

DEVICE_INLINE int64_t
atomic_cas_int64(int64_t* address, const int64_t compare, const int64_t value) {
  return static_cast<int64_t>(atomicCAS(
      reinterpret_cast<unsigned long long*>(address),
      static_cast<unsigned long long>(compare),
      static_cast<unsigned long long>(value)));
}

DEVICE_INLINE void atomic_exchange_int64(
    int64_t* address,
    const int64_t value) {
  atomicExch(
      reinterpret_cast<unsigned long long*>(address),
      static_cast<unsigned long long>(value));
}

// The set-associative layout packs 16 int64 tags in one 128-byte line.
constexpr int32_t kOwnerCacheWays = 16;

DEVICE_INLINE int32_t owner_cache_set_base(
    const int64_t key,
    const int32_t num_slots,
    const bool set_associative) {
  const auto ways = set_associative ? kOwnerCacheWays : 1;
  return static_cast<int32_t>(cache_slot(key, num_slots / ways)) * ways;
}

DEVICE_INLINE int32_t owner_cache_find(
    const int64_t* __restrict__ tags,
    const int64_t key,
    const int32_t num_slots,
    const bool set_associative) {
  const auto base = owner_cache_set_base(key, num_slots, set_associative);
  if (!set_associative) {
    return ::__ldg(&tags[base]) == key ? base : -1;
  }
  const auto* const line = reinterpret_cast<const longlong2*>(&tags[base]);
#pragma unroll
  for (int32_t pair = 0; pair < kOwnerCacheWays / 2; ++pair) {
    const auto two = ::__ldg(&line[pair]);
    if (two.x == key) {
      return base + 2 * pair;
    }
    if (two.y == key) {
      return base + 2 * pair + 1;
    }
  }
  return -1;
}

// A claimed slot stays pending (-key - 2) until its row is published.
// Fill-only inserts keep existing keys; canonical row refreshes replace them.
DEVICE_INLINE int32_t owner_cache_claim(
    int64_t* tags,
    const int32_t* last_access,
    const int32_t num_slots,
    const int64_t key,
    const int32_t now,
    const uint32_t lane,
    int64_t* victim,
    const bool set_associative,
    const bool update_existing) {
  const auto base = owner_cache_set_base(key, num_slots, set_associative);
  const auto pending = -key - 2;
  if (!set_associative) {
    int32_t slot = -1;
    if (lane == 0) {
      const auto observed =
          *reinterpret_cast<volatile const int64_t*>(&tags[base]);
      if ((update_existing ? observed >= -1 : observed == -1) &&
          atomic_cas_int64(&tags[base], observed, pending) == observed) {
        slot = base;
      }
    }
    *victim = -1;
    return shfl_sync(slot, 0);
  }
  const bool reader = lane < kOwnerCacheWays;
  for (int32_t attempt = 0; attempt < kOwnerCacheWays; ++attempt) {
    int64_t tag = -1;
    int32_t touched = now;
    if (reader) {
      tag = *reinterpret_cast<volatile const int64_t*>(&tags[base + lane]);
      if (last_access != nullptr) {
        touched = *reinterpret_cast<volatile const int32_t*>(
            &last_access[base + lane]);
      }
    }
    if (fbgemm_gpu::ballot_sync(reader && tag == pending)) {
      return -1;
    }
    int32_t way = -1;
    int64_t observed = -1;
    const auto existing = fbgemm_gpu::ballot_sync(reader && tag == key);
    const auto empty = fbgemm_gpu::ballot_sync(reader && tag == -1);
    if (existing) {
      if (!update_existing) {
        return -1;
      }
      way = __ffsll(static_cast<unsigned long long>(existing)) - 1;
      observed = key;
    } else if (empty) {
      way = __ffsll(static_cast<unsigned long long>(empty)) - 1;
    } else {
      const bool evictable = reader && tag >= 0 && touched != now;
      int64_t oldest = evictable
          ? (static_cast<int64_t>(static_cast<uint32_t>(touched)) << 8) | lane
          : std::numeric_limits<int64_t>::max();
      for (int32_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const auto other = shfl_xor(oldest, offset);
        oldest = other < oldest ? other : oldest;
      }
      if (oldest == std::numeric_limits<int64_t>::max()) {
        return -1;
      }
      way = static_cast<int32_t>(oldest & 0xFF);
      observed = shfl_sync(tag, way);
    }
    int32_t claimed = 0;
    if (static_cast<int32_t>(lane) == way) {
      claimed =
          atomic_cas_int64(&tags[base + way], observed, pending) == observed;
      if (claimed && observed >= 0 && observed != key &&
          last_access != nullptr &&
          *reinterpret_cast<volatile const int32_t*>(
              &last_access[base + way]) == now) {
        atomic_exchange_int64(&tags[base + way], observed);
        claimed = 0;
      }
    }
    if (shfl_sync(claimed, way)) {
      *victim = observed == key ? -1 : observed;
      return base + way;
    }
  }
  return -1;
}

// All-null victim outputs disable eviction from full sets.
struct OwnerCacheVictims {
  int64_t* keys;
  int32_t* last_access;
  uint8_t* rows;
  int32_t* count;
};

template <typename value_t>
__global__
__launch_bounds__(kMaxThreads) void ssd_gather_from_row_addresses_kernel(
    pta::PackedTensorAccessor64<value_t, 2, at::RestrictPtrTraits> output,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        row_addresses) {
  const auto N = row_addresses.size(0);
  for (auto n = blockIdx.x * blockDim.y + threadIdx.y; n < N;
       n += blockDim.y * gridDim.x) {
    const auto row_address = row_addresses[n];
    CUDA_KERNEL_ASSERT(row_address != 0 && "SSD row address must be nonzero");
    const auto* const source =
        reinterpret_cast<const value_t*>(static_cast<uint64_t>(row_address));
    vec4_copy(&output[n][0], source, output.size(1));
  }
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void kvzch_direct_cache_lookup_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        cache_state,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        action_indices,
    int32_t* __restrict__ actions_count,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        row_addresses,
    const uint64_t cache_weights_address,
    const uint64_t inserted_rows_address,
    const uint64_t cache_row_bytes,
    const uint64_t inserted_row_bytes,
    const uint64_t hit_tag,
    const bool set_associative) {
  using BlockScan = cub::BlockScan<int32_t, kMaxThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int32_t block_action_base;

  const int32_t N = indices.size(0);
  const int32_t C = cache_state.size(0);
  const auto grid_stride = blockDim.x * gridDim.x;
  for (auto block_start = blockIdx.x * blockDim.x; block_start < N;
       block_start += grid_stride) {
    const auto n = block_start + threadIdx.x;
    int64_t key = -1;
    int32_t cache_slot_index = -1;
    if (n < N) {
      key = static_cast<int64_t>(indices[n]);
      if (key >= 0) {
        cache_slot_index =
            owner_cache_find(&cache_state[0][0], key, C, set_associative);
      }
    }

    const int32_t is_action = n < N && key >= 0 && cache_slot_index < 0;
    int32_t local_action_position;
    int32_t block_action_count;
    BlockScan(scan_storage)
        .ExclusiveSum(is_action, local_action_position, block_action_count);
    if (threadIdx.x == 0 && block_action_count > 0) {
      block_action_base = atomicAdd(actions_count, block_action_count);
    }
    __syncthreads();

    if (n < N) {
      if (is_action) {
        const int32_t action_position =
            block_action_base + local_action_position;
        action_indices[action_position] = static_cast<index_t>(key);
        row_addresses[n] = inserted_rows_address +
            static_cast<uint64_t>(action_position) * inserted_row_bytes;
      } else {
        const int32_t row = cache_slot_index < 0 ? 0 : cache_slot_index;
        row_addresses[n] = (cache_weights_address +
                            static_cast<uint64_t>(row) * cache_row_bytes) |
            hit_tag;
      }
    }
    __syncthreads();
  }
}

template <typename value_t, typename index_t>
__global__ __launch_bounds__(kMaxThreads) void kvzch_direct_cache_insert_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor64<value_t, 2, at::RestrictPtrTraits> rows,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> cache_state,
    pta::PackedTensorAccessor64<value_t, 2, at::RestrictPtrTraits>
        cache_weights,
    const int32_t* __restrict__ count,
    int32_t* last_access,
    const int32_t now,
    const bool set_associative,
    const bool update_existing) {
  constexpr int32_t kWarpsPerBlock = kMaxThreads / kWarpSize;
  const auto lane = threadIdx.x;
  const auto warp = threadIdx.y;
  const int32_t N = count == nullptr ? indices.size(0) : *count;
  CUDA_KERNEL_ASSERT(
      N >= 0 && N <= indices.size(0) &&
      "direct-cache insert count exceeds input size");
  const int32_t C = cache_state.size(0);

  for (auto n = blockIdx.x * kWarpsPerBlock + warp; n < N;
       n += gridDim.x * kWarpsPerBlock) {
    const auto key = static_cast<int64_t>(indices[n]);
    bool lane_has_nonzero = false;
    if (key >= 0) {
      for (auto d = lane; d < rows.size(1); d += kWarpSize) {
        lane_has_nonzero |= rows[n][d] != value_t{0};
      }
    }
    // Canonical DRAM reads zero-fill rows that are not present and do not
    // expose a found mask at this layer.  Keep all-zero rows out of the
    // positive cache: publishing one would turn a transient backend miss into
    // an unbounded cache hit.  Legitimate all-zero rows remain semantically
    // correct but are deliberately re-read until the caller provides an
    // explicit validity signal.
    if (fbgemm_gpu::ballot_sync(lane_has_nonzero) == 0) {
      continue;
    }

    // A copy of DRAM rows: an evicted row still has its DRAM copy.
    int64_t victim = -1;
    const int32_t slot = owner_cache_claim(
        &cache_state[0][0],
        last_access,
        C,
        key,
        now,
        lane,
        &victim,
        set_associative,
        update_existing);
    if (slot < 0) {
      continue;
    }
    for (auto d = lane; d < rows.size(1); d += kWarpSize) {
      cache_weights[slot][d] = rows[n][d];
    }
    if (lane == 0 && last_access != nullptr) {
      last_access[slot] = now;
    }
    syncwarp();
    if (lane == 0) {
      __threadfence();
      atomic_exchange_int64(&cache_state[slot][0], key);
    }
  }
}

DEVICE_INLINE uint32_t
quantize_uint8(const float x, const float bias, const float scale) {
  return static_cast<uint32_t>(
      fminf(fmaxf(rintf((x - bias) / scale), 0.0f), 255.0f));
}

template <typename value_t, typename index_t>
__global__
__launch_bounds__(kMaxThreads) void kvzch_direct_cache_insert_int8_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor64<value_t, 2, at::RestrictPtrTraits> rows,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> cache_state,
    pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        cache_weights,
    const int32_t* __restrict__ count,
    int32_t* last_access,
    const int32_t now,
    const bool set_associative,
    const bool update_existing) {
  constexpr int32_t kWarpsPerBlock = kMaxThreads / kWarpSize;
  const auto lane = threadIdx.x;
  const auto warp = threadIdx.y;
  const int32_t N = count == nullptr ? indices.size(0) : *count;
  CUDA_KERNEL_ASSERT(
      N >= 0 && N <= indices.size(0) &&
      "direct-cache insert count exceeds input size");
  const int32_t C = cache_state.size(0);
  const int32_t D = rows.size(1);
  const int32_t num_vecs = D / 4;

  for (auto n = blockIdx.x * kWarpsPerBlock + warp; n < N;
       n += gridDim.x * kWarpsPerBlock) {
    const auto key = static_cast<int64_t>(indices[n]);
    bool lane_has_nonzero = false;
    float row_min = std::numeric_limits<float>::max();
    float row_max = std::numeric_limits<float>::lowest();
    if (key >= 0) {
      for (auto v = lane; v < num_vecs; v += kWarpSize) {
        const Vec4T<float> x(&rows[n][4 * v]);
        lane_has_nonzero |= x.acc.x != 0.0f || x.acc.y != 0.0f ||
            x.acc.z != 0.0f || x.acc.w != 0.0f;
        row_min = fminf(row_min, x.vmin());
        row_max = fmaxf(row_max, x.vmax());
      }
    }
    if (fbgemm_gpu::ballot_sync(lane_has_nonzero) == 0) {
      continue;
    }

    // A copy of DRAM rows: an evicted row still has its DRAM copy.
    int64_t victim = -1;
    const int32_t slot = owner_cache_claim(
        &cache_state[0][0],
        last_access,
        C,
        key,
        now,
        lane,
        &victim,
        set_associative,
        update_existing);
    if (slot < 0) {
      continue;
    }

    row_min = warp_reduce_min(row_min);
    row_max = warp_reduce_max(row_max);
    const float bias = row_min;
    const float range = row_max - row_min;
    const float scale = range == 0.0f ? 1.0f : range / 255.0f;
    uint8_t* const dst = &cache_weights[slot][0];
    for (auto v = lane; v < num_vecs; v += kWarpSize) {
      const Vec4T<float> x(&rows[n][4 * v]);
      reinterpret_cast<uint32_t*>(dst)[v] =
          quantize_uint8(x.acc.x, bias, scale) |
          (quantize_uint8(x.acc.y, bias, scale) << 8) |
          (quantize_uint8(x.acc.z, bias, scale) << 16) |
          (quantize_uint8(x.acc.w, bias, scale) << 24);
    }
    if (lane == 0) {
      auto* const qparams = reinterpret_cast<float*>(dst + D);
      qparams[0] = scale;
      qparams[1] = bias;
      if (last_access != nullptr) {
        last_access[slot] = now;
      }
    }
    syncwarp();
    if (lane == 0) {
      __threadfence();
      atomic_exchange_int64(&cache_state[slot][0], key);
    }
  }
}

template <typename value_t>
__global__
__launch_bounds__(kMaxThreads) void kvzch_direct_cache_gather_int8_kernel(
    pta::PackedTensorAccessor64<value_t, 2, at::RestrictPtrTraits> output,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        row_addresses) {
  const auto N = row_addresses.size(0);
  const int32_t D = output.size(1);
  for (auto n = blockIdx.x * blockDim.y + threadIdx.y; n < N;
       n += blockDim.y * gridDim.x) {
    const auto row_address = static_cast<uint64_t>(row_addresses[n]);
    CUDA_KERNEL_ASSERT(row_address != 0 && "row address must be nonzero");
    if (row_address & 1) {
      const auto* const codes =
          reinterpret_cast<const uint8_t*>(row_address & ~uint64_t{1});
      const auto* const qparams = reinterpret_cast<const float*>(codes + D);
      const float scale = qparams[0];
      const float bias = qparams[1];
      for (auto d = threadIdx.x * 4; d < D; d += blockDim.x * 4) {
        const uint32_t q = *reinterpret_cast<const uint32_t*>(codes + d);
        Vec4T<float> x;
        x.acc.x = static_cast<float>(q & 0xFF) * scale + bias;
        x.acc.y = static_cast<float>((q >> 8) & 0xFF) * scale + bias;
        x.acc.z = static_cast<float>((q >> 16) & 0xFF) * scale + bias;
        x.acc.w = static_cast<float>(q >> 24) * scale + bias;
        x.store(&output[n][d]);
      }
    } else {
      vec4_copy(
          &output[n][0], reinterpret_cast<const value_t*>(row_address), D);
    }
  }
}

// Rounds x to row_t and back, as storing a row_t output and widening it with
// `Tensor.to(float32)` does: float->half/bfloat16 is round-to-nearest-even
// (the Vec4T stores), and the widening is exact.
template <typename row_t>
DEVICE_INLINE float round_through(const float x) {
  if constexpr (std::is_same_v<row_t, at::Half>) {
    return __half2float(__float2half_rn(x));
  } else if constexpr (std::is_same_v<row_t, at::BFloat16>) {
    return static_cast<float>(at::BFloat16(x));
  } else {
    return x;
  }
}

// Same as kvzch_direct_cache_gather_int8_kernel<row_t> followed by a float32
// cast of its output, in one pass: hit rows are dequantized and rounded to
// row_t, miss rows are row_t rows widened to float.
template <typename row_t>
__global__
__launch_bounds__(kMaxThreads) void kvzch_direct_cache_gather_int8_float_kernel(
    pta::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> output,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        row_addresses) {
  const auto N = row_addresses.size(0);
  const int32_t D = output.size(1);
  for (auto n = blockIdx.x * blockDim.y + threadIdx.y; n < N;
       n += blockDim.y * gridDim.x) {
    const auto row_address = static_cast<uint64_t>(row_addresses[n]);
    CUDA_KERNEL_ASSERT(row_address != 0 && "row address must be nonzero");
    if (row_address & 1) {
      const auto* const codes =
          reinterpret_cast<const uint8_t*>(row_address & ~uint64_t{1});
      const auto* const qparams = reinterpret_cast<const float*>(codes + D);
      const float scale = qparams[0];
      const float bias = qparams[1];
      for (auto d = threadIdx.x * 4; d < D; d += blockDim.x * 4) {
        const uint32_t q = *reinterpret_cast<const uint32_t*>(codes + d);
        Vec4T<float> x;
        x.acc.x =
            round_through<row_t>(static_cast<float>(q & 0xFF) * scale + bias);
        x.acc.y = round_through<row_t>(
            static_cast<float>((q >> 8) & 0xFF) * scale + bias);
        x.acc.z = round_through<row_t>(
            static_cast<float>((q >> 16) & 0xFF) * scale + bias);
        x.acc.w =
            round_through<row_t>(static_cast<float>(q >> 24) * scale + bias);
        x.store(&output[n][d]);
      }
    } else {
      const auto* const row = reinterpret_cast<const row_t*>(row_address);
      for (auto d = threadIdx.x * 4; d < D; d += blockDim.x * 4) {
        const Vec4T<row_t> x(&row[d]);
        x.store(&output[n][d]);
      }
    }
  }
}

// The fp16 value of one group-wise INT4 code, as the CPU decode
// (grouped_int4_e4m3::decode, then fp16) and serving produce it: code * scale
// is exact in float, so only the fp16 rounding remains.
DEVICE_INLINE float decode_grouped_int4(
    const uint32_t nibble,
    const uint8_t scale_bits) {
  const int32_t code = static_cast<int32_t>(nibble ^ 0x8) - 0x8;
  const float scale = static_cast<float>(
      c10::Float8_e4m3fn(scale_bits, c10::Float8_e4m3fn::from_bits()));
  return __half2float(__float2half_rn(static_cast<float>(code) * scale));
}

// Gathers rows whose hits are tagged rows in the DRAM storage format and whose
// misses are row_t rows. With a group_size, a hit is a g8 row (D / 2 code
// bytes, two's complement nibbles with the even column low, then D /
// group_size float8_e4m3fn scales) that decodes to the fp16 values the DRAM
// backend and serving decode, rounded through row_t; without one, a hit is a
// row_t row. Every value is then converted to out_t, as gather + to() would.
template <typename out_t, typename row_t>
__global__
__launch_bounds__(kMaxThreads) void kvzch_direct_cache_gather_raw_kernel(
    pta::PackedTensorAccessor64<out_t, 2, at::RestrictPtrTraits> output,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        row_addresses,
    const int32_t group_size) {
  const auto N = row_addresses.size(0);
  const int32_t D = output.size(1);
  for (auto n = blockIdx.x * blockDim.y + threadIdx.y; n < N;
       n += blockDim.y * gridDim.x) {
    const auto row_address = static_cast<uint64_t>(row_addresses[n]);
    CUDA_KERNEL_ASSERT(row_address != 0 && "row address must be nonzero");
    if (group_size > 0 && (row_address & 1)) {
      const auto* const codes =
          reinterpret_cast<const uint8_t*>(row_address & ~uint64_t{1});
      const auto* const scales = codes + D / 2;
      for (auto d = threadIdx.x * 8; d < D; d += blockDim.x * 8) {
        const uint32_t q = *reinterpret_cast<const uint32_t*>(codes + d / 2);
        float v[8];
#pragma unroll
        for (int32_t k = 0; k < 8; ++k) {
          v[k] = round_through<row_t>(decode_grouped_int4(
              (q >> (4 * k)) & 0xF, scales[(d + k) / group_size]));
        }
        Vec4T<float> lo;
        lo.acc = make_float4(v[0], v[1], v[2], v[3]);
        lo.store(&output[n][d]);
        Vec4T<float> hi;
        hi.acc = make_float4(v[4], v[5], v[6], v[7]);
        hi.store(&output[n][d + 4]);
      }
    } else {
      const auto* const row =
          reinterpret_cast<const row_t*>(row_address & ~uint64_t{1});
      for (auto d = threadIdx.x * 4; d < D; d += blockDim.x * 4) {
        const Vec4T<float> x(&row[d]);
        x.store(&output[n][d]);
      }
    }
  }
}

// The gather kernels grid-stride over rows, so any grid size gives the same
// output. A cap below the SM count keeps whole SMs free for kernels that cannot
// share an SM with a gather CTA, such as NCCL collectives on another stream.
uint32_t cap_gather_grid(const uint32_t grid_size, const int64_t max_ctas) {
  TORCH_CHECK_GE(max_ctas, 0);
  if (max_ctas == 0) {
    return grid_size;
  }
  return static_cast<uint32_t>(std::min<int64_t>(grid_size, max_ctas));
}

} // namespace

Tensor ssd_gather_from_row_addresses_cuda(
    Tensor output,
    Tensor row_addresses,
    const int64_t max_ctas) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(output, row_addresses);
  TENSOR_CONTIGUOUS(output);
  TENSOR_CONTIGUOUS(row_addresses);
  TORCH_CHECK_EQ(output.dim(), 2);
  TORCH_CHECK_EQ(row_addresses.dim(), 1);
  TORCH_CHECK_EQ(output.size(0), row_addresses.numel());
  TORCH_CHECK_EQ(row_addresses.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_GT(output.size(1), 0);
  TORCH_CHECK_EQ(output.size(1) % 4, 0);

  if (row_addresses.numel() == 0) {
    return output;
  }

  CUDA_DEVICE_GUARD(output);
  const auto D = output.size(1);
  const int32_t threads_per_row = std::min<int32_t>(D / 4, kMaxThreads);
  const dim3 threads(threads_per_row, kMaxThreads / threads_per_row);
  const auto grid_size = cap_gather_grid(
      utils::cuda::cap_grid_dim_x(
          div_round_up(row_addresses.numel(), threads.y),
          kMaxThreads,
          at::cuda::getCurrentCUDAStream()),
      max_ctas);

  FBGEMM_DISPATCH_FLOATING_TYPES(
      output.scalar_type(), "ssd_gather_from_row_addresses", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (ssd_gather_from_row_addresses_kernel<scalar_t>),
            grid_size,
            threads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(output, scalar_t, 2, 64),
            PTA_B(row_addresses, int64_t, 1, 32));
      });
  return output;
}

namespace {

Tensor kvzch_direct_cache_lookup_impl(
    Tensor indices,
    Tensor cache_state,
    Tensor cache_weights,
    Tensor action_indices,
    Tensor actions_count,
    Tensor inserted_rows,
    Tensor row_addresses,
    const int64_t cache_row_bytes,
    const int64_t inserted_row_bytes,
    const uint64_t hit_tag,
    const bool set_associative) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      indices,
      cache_state,
      cache_weights,
      action_indices,
      actions_count,
      inserted_rows,
      row_addresses);
  TENSOR_CONTIGUOUS(indices);
  TENSOR_CONTIGUOUS(cache_state);
  TENSOR_CONTIGUOUS(cache_weights);
  TENSOR_CONTIGUOUS(action_indices);
  TENSOR_CONTIGUOUS(actions_count);
  TENSOR_CONTIGUOUS(inserted_rows);
  TENSOR_CONTIGUOUS(row_addresses);
  TORCH_CHECK_EQ(indices.dim(), 1);
  TORCH_CHECK_EQ(cache_state.dim(), 2);
  TORCH_CHECK_EQ(cache_state.size(1), 1);
  TORCH_CHECK_EQ(cache_weights.dim(), 2);
  TORCH_CHECK_EQ(action_indices.dim(), 1);
  TORCH_CHECK_EQ(actions_count.dim(), 1);
  TORCH_CHECK_EQ(inserted_rows.dim(), 2);
  TORCH_CHECK_EQ(row_addresses.dim(), 1);
  TORCH_CHECK_EQ(cache_state.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(action_indices.scalar_type(), indices.scalar_type());
  TORCH_CHECK_EQ(actions_count.scalar_type(), at::ScalarType::Int);
  TORCH_CHECK_EQ(row_addresses.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(cache_weights.size(0), cache_state.size(0));
  TORCH_CHECK_GE(action_indices.numel(), indices.numel());
  TORCH_CHECK_GE(inserted_rows.size(0), indices.numel());
  TORCH_CHECK_EQ(row_addresses.numel(), indices.numel());
  TORCH_CHECK_GT(cache_state.size(0), 0);
  TORCH_CHECK_EQ(
      cache_state.size(0) % (set_associative ? kOwnerCacheWays : 1), 0);
  TORCH_CHECK_LE(cache_state.size(0), std::numeric_limits<int32_t>::max());
  TORCH_CHECK_LE(indices.numel(), std::numeric_limits<int32_t>::max());

  CUDA_DEVICE_GUARD(indices);
  actions_count.zero_();
  if (indices.numel() == 0) {
    return row_addresses;
  }

  const auto grid_size = utils::cuda::cap_grid_dim_x(
      div_round_up(indices.numel(), kMaxThreads),
      kMaxThreads,
      at::cuda::getCurrentCUDAStream());
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "kvzch_direct_cache_lookup", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (kvzch_direct_cache_lookup_kernel<index_t>),
            grid_size,
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(indices, index_t, 1, 32),
            PTA_B(cache_state, int64_t, 2, 32),
            PTA_B(action_indices, index_t, 1, 32),
            actions_count.data_ptr<int32_t>(),
            PTA_B(row_addresses, int64_t, 1, 32),
            reinterpret_cast<uint64_t>(cache_weights.data_ptr()),
            reinterpret_cast<uint64_t>(inserted_rows.data_ptr()),
            static_cast<uint64_t>(cache_row_bytes),
            static_cast<uint64_t>(inserted_row_bytes),
            hit_tag,
            set_associative);
      });
  return row_addresses;
}

} // namespace

Tensor kvzch_direct_cache_lookup_cuda(
    Tensor indices,
    Tensor cache_state,
    Tensor cache_weights,
    Tensor action_indices,
    Tensor actions_count,
    Tensor inserted_rows,
    Tensor row_addresses,
    const bool set_associative) {
  TORCH_CHECK_EQ(cache_weights.dim(), 2);
  TORCH_CHECK_EQ(inserted_rows.dim(), 2);
  TORCH_CHECK_EQ(cache_weights.scalar_type(), inserted_rows.scalar_type());
  TORCH_CHECK_EQ(cache_weights.size(1), inserted_rows.size(1));
  const auto row_bytes = cache_weights.size(1) * cache_weights.element_size();
  return kvzch_direct_cache_lookup_impl(
      indices,
      cache_state,
      cache_weights,
      action_indices,
      actions_count,
      inserted_rows,
      row_addresses,
      row_bytes,
      row_bytes,
      0,
      set_associative);
}

Tensor kvzch_direct_cache_lookup_int8_cuda(
    Tensor indices,
    Tensor cache_state,
    Tensor cache_weights,
    Tensor action_indices,
    Tensor actions_count,
    Tensor inserted_rows,
    Tensor row_addresses,
    const bool set_associative) {
  TORCH_CHECK_EQ(cache_weights.dim(), 2);
  TORCH_CHECK_EQ(inserted_rows.dim(), 2);
  TORCH_CHECK_EQ(cache_weights.scalar_type(), at::ScalarType::Byte);
  const auto inserted_dtype = inserted_rows.scalar_type();
  TORCH_CHECK(
      inserted_dtype == at::ScalarType::Float ||
          inserted_dtype == at::ScalarType::Half ||
          inserted_dtype == at::ScalarType::BFloat16,
      "inserted_rows must be float, half, or bfloat16");
  const auto D = inserted_rows.size(1);
  TORCH_CHECK_EQ(D % 4, 0);
  TORCH_CHECK_EQ(cache_weights.size(1), D + kINT8QparamsBytes);
  // Bit 0 of every row address is the hit tag, so both row bases must be even.
  TORCH_CHECK_EQ(reinterpret_cast<uintptr_t>(cache_weights.data_ptr()) % 4, 0);
  TORCH_CHECK_EQ(reinterpret_cast<uintptr_t>(inserted_rows.data_ptr()) % 4, 0);
  return kvzch_direct_cache_lookup_impl(
      indices,
      cache_state,
      cache_weights,
      action_indices,
      actions_count,
      inserted_rows,
      row_addresses,
      cache_weights.size(1),
      D * inserted_rows.element_size(),
      1,
      set_associative);
}

namespace {

void check_owner_cache_last_access(
    const Tensor& cache_state,
    const std::optional<Tensor>& last_access,
    const bool set_associative) {
  TORCH_CHECK_EQ(
      cache_state.size(0) % (set_associative ? kOwnerCacheWays : 1), 0);
  if (last_access.has_value()) {
    TENSOR_CONTIGUOUS(last_access.value());
    TORCH_CHECK_EQ(last_access->scalar_type(), at::ScalarType::Int);
    TORCH_CHECK_EQ(last_access->dim(), 1);
    TORCH_CHECK_EQ(last_access->size(0), cache_state.size(0));
  }
}

// The victim outputs of an exclusive insert of up to `num_rows` rows of
// `row_bytes`: all given (host-mapped in production) or none.
OwnerCacheVictims owner_cache_victims(
    const std::optional<Tensor>& victim_keys,
    const std::optional<Tensor>& victim_access,
    const std::optional<Tensor>& victim_rows,
    const std::optional<Tensor>& victim_count,
    const int64_t num_rows,
    const int64_t row_bytes) {
  const bool any = victim_keys.has_value() || victim_access.has_value() ||
      victim_rows.has_value() || victim_count.has_value();
  if (!any) {
    return {nullptr, nullptr, nullptr, nullptr};
  }
  TORCH_CHECK(
      victim_keys.has_value() && victim_access.has_value() &&
          victim_rows.has_value() && victim_count.has_value(),
      "exclusive insert victims need keys, access times, rows and a count");
  TENSOR_CONTIGUOUS(victim_keys.value());
  TENSOR_CONTIGUOUS(victim_access.value());
  TENSOR_CONTIGUOUS(victim_rows.value());
  TENSOR_CONTIGUOUS(victim_count.value());
  TORCH_CHECK_EQ(victim_keys->scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(victim_access->scalar_type(), at::ScalarType::Int);
  TORCH_CHECK_EQ(victim_rows->scalar_type(), at::ScalarType::Byte);
  TORCH_CHECK_EQ(victim_count->scalar_type(), at::ScalarType::Int);
  TORCH_CHECK_GE(victim_keys->numel(), num_rows);
  TORCH_CHECK_GE(victim_access->numel(), num_rows);
  TORCH_CHECK_EQ(victim_rows->dim(), 2);
  TORCH_CHECK_GE(victim_rows->size(0), num_rows);
  TORCH_CHECK_EQ(victim_rows->size(1), row_bytes);
  TORCH_CHECK_EQ(victim_count->numel(), 1);
  victim_count->zero_();
  return {
      victim_keys->data_ptr<int64_t>(),
      victim_access->data_ptr<int32_t>(),
      victim_rows->data_ptr<uint8_t>(),
      victim_count->data_ptr<int32_t>()};
}

} // namespace

Tensor kvzch_direct_cache_insert_cuda(
    Tensor cache_state,
    Tensor cache_weights,
    Tensor indices,
    Tensor rows,
    const std::optional<Tensor>& count,
    const std::optional<Tensor>& last_access,
    const int64_t now,
    const bool set_associative,
    const bool update_existing) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cache_state, cache_weights, indices, rows, count, last_access);
  check_owner_cache_last_access(cache_state, last_access, set_associative);
  TENSOR_CONTIGUOUS(cache_state);
  TENSOR_CONTIGUOUS(cache_weights);
  TENSOR_CONTIGUOUS(indices);
  TENSOR_CONTIGUOUS(rows);
  if (count.has_value()) {
    TENSOR_CONTIGUOUS(count.value());
    TORCH_CHECK_EQ(count->scalar_type(), at::ScalarType::Int);
    TORCH_CHECK_EQ(count->numel(), 1);
  }
  TORCH_CHECK_EQ(cache_state.dim(), 2);
  TORCH_CHECK_EQ(cache_state.size(1), 1);
  TORCH_CHECK_EQ(cache_weights.dim(), 2);
  TORCH_CHECK_EQ(indices.dim(), 1);
  TORCH_CHECK_EQ(rows.dim(), 2);
  TORCH_CHECK_EQ(cache_state.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(cache_weights.scalar_type(), rows.scalar_type());
  TORCH_CHECK_EQ(cache_weights.size(0), cache_state.size(0));
  TORCH_CHECK_EQ(cache_weights.size(1), rows.size(1));
  TORCH_CHECK_EQ(rows.size(0), indices.numel());
  TORCH_CHECK_GT(cache_state.size(0), 0);
  TORCH_CHECK_LE(cache_state.size(0), std::numeric_limits<int32_t>::max());
  TORCH_CHECK_LE(indices.numel(), std::numeric_limits<int32_t>::max());

  CUDA_DEVICE_GUARD(cache_weights);
  if (indices.numel() == 0) {
    return cache_weights;
  }

  const int32_t kWarpsPerBlock = kMaxThreads / kWarpSizeHost();
  const dim3 threads(kWarpSizeHost(), kWarpsPerBlock);
  const auto grid_size = utils::cuda::cap_grid_dim_x(
      div_round_up(indices.numel(), kWarpsPerBlock),
      kMaxThreads,
      at::cuda::getCurrentCUDAStream());
  // Byte rows are raw rows in the DRAM storage format (g8 bytes).
  FBGEMM_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Byte,
      cache_weights.scalar_type(),
      "kvzch_direct_cache_insert_value",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "kvzch_direct_cache_insert_index", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (kvzch_direct_cache_insert_kernel<scalar_t, index_t>),
                  grid_size,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(indices, index_t, 1, 32),
                  PTA_B(rows, scalar_t, 2, 64),
                  PTA_B(cache_state, int64_t, 2, 32),
                  PTA_B(cache_weights, scalar_t, 2, 64),
                  count.has_value() ? count->data_ptr<int32_t>() : nullptr,
                  last_access.has_value() ? last_access->data_ptr<int32_t>()
                                          : nullptr,
                  static_cast<int32_t>(now),
                  set_associative,
                  update_existing);
            });
      });
  return cache_weights;
}

Tensor kvzch_direct_cache_insert_int8_cuda(
    Tensor cache_state,
    Tensor cache_weights,
    Tensor indices,
    Tensor rows,
    const std::optional<Tensor>& count,
    const std::optional<Tensor>& last_access,
    const int64_t now,
    const bool set_associative,
    const bool update_existing) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cache_state, cache_weights, indices, rows, count, last_access);
  check_owner_cache_last_access(cache_state, last_access, set_associative);
  TENSOR_CONTIGUOUS(cache_state);
  TENSOR_CONTIGUOUS(cache_weights);
  TENSOR_CONTIGUOUS(indices);
  TENSOR_CONTIGUOUS(rows);
  if (count.has_value()) {
    TENSOR_CONTIGUOUS(count.value());
    TORCH_CHECK_EQ(count->scalar_type(), at::ScalarType::Int);
    TORCH_CHECK_EQ(count->numel(), 1);
  }
  TORCH_CHECK_EQ(cache_state.dim(), 2);
  TORCH_CHECK_EQ(cache_state.size(1), 1);
  TORCH_CHECK_EQ(cache_weights.dim(), 2);
  TORCH_CHECK_EQ(indices.dim(), 1);
  TORCH_CHECK_EQ(rows.dim(), 2);
  TORCH_CHECK_EQ(cache_state.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(cache_weights.scalar_type(), at::ScalarType::Byte);
  TORCH_CHECK_EQ(cache_weights.size(0), cache_state.size(0));
  TORCH_CHECK_EQ(rows.size(1) % 4, 0);
  TORCH_CHECK_EQ(cache_weights.size(1), rows.size(1) + kINT8QparamsBytes);
  TORCH_CHECK_EQ(reinterpret_cast<uintptr_t>(cache_weights.data_ptr()) % 4, 0);
  TORCH_CHECK_EQ(
      reinterpret_cast<uintptr_t>(rows.data_ptr()) % (4 * rows.element_size()),
      0);
  TORCH_CHECK_EQ(rows.size(0), indices.numel());
  TORCH_CHECK_GT(cache_state.size(0), 0);
  TORCH_CHECK_LE(cache_state.size(0), std::numeric_limits<int32_t>::max());
  TORCH_CHECK_LE(indices.numel(), std::numeric_limits<int32_t>::max());

  CUDA_DEVICE_GUARD(cache_weights);
  if (indices.numel() == 0) {
    return cache_weights;
  }

  const int32_t kWarpsPerBlock = kMaxThreads / kWarpSizeHost();
  const dim3 threads(kWarpSizeHost(), kWarpsPerBlock);
  const auto grid_size = utils::cuda::cap_grid_dim_x(
      div_round_up(indices.numel(), kWarpsPerBlock),
      kMaxThreads,
      at::cuda::getCurrentCUDAStream());
  FBGEMM_DISPATCH_FLOATING_TYPES(
      rows.scalar_type(), "kvzch_direct_cache_insert_int8_value", [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "kvzch_direct_cache_insert_int8_index", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (kvzch_direct_cache_insert_int8_kernel<scalar_t, index_t>),
                  grid_size,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(indices, index_t, 1, 32),
                  PTA_B(rows, scalar_t, 2, 64),
                  PTA_B(cache_state, int64_t, 2, 32),
                  PTA_B(cache_weights, uint8_t, 2, 64),
                  count.has_value() ? count->data_ptr<int32_t>() : nullptr,
                  last_access.has_value() ? last_access->data_ptr<int32_t>()
                                          : nullptr,
                  static_cast<int32_t>(now),
                  set_associative,
                  update_existing);
            });
      });
  return cache_weights;
}

Tensor kvzch_direct_cache_gather_int8_cuda(
    Tensor output,
    Tensor row_addresses,
    at::ScalarType row_dtype,
    const int64_t max_ctas) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(output, row_addresses);
  TENSOR_CONTIGUOUS(output);
  TENSOR_CONTIGUOUS(row_addresses);
  TORCH_CHECK_EQ(output.dim(), 2);
  TORCH_CHECK_EQ(row_addresses.dim(), 1);
  TORCH_CHECK_EQ(output.size(0), row_addresses.numel());
  TORCH_CHECK_EQ(row_addresses.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK(
      output.scalar_type() == row_dtype,
      "kvzch_direct_cache_gather_int8 requires output.dtype == row_dtype; "
      "use kvzch_direct_cache_gather_int8_float to widen rows to float32");
  TORCH_CHECK_GT(output.size(1), 0);
  TORCH_CHECK_EQ(output.size(1) % 4, 0);

  if (row_addresses.numel() == 0) {
    return output;
  }

  CUDA_DEVICE_GUARD(output);
  const auto D = output.size(1);
  const int32_t threads_per_row = std::min<int32_t>(D / 4, kMaxThreads);
  const dim3 threads(threads_per_row, kMaxThreads / threads_per_row);
  const auto grid_size = cap_gather_grid(
      utils::cuda::cap_grid_dim_x(
          div_round_up(row_addresses.numel(), threads.y),
          kMaxThreads,
          at::cuda::getCurrentCUDAStream()),
      max_ctas);

  FBGEMM_DISPATCH_FLOATING_TYPES(
      output.scalar_type(), "kvzch_direct_cache_gather_int8", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (kvzch_direct_cache_gather_int8_kernel<scalar_t>),
            grid_size,
            threads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(output, scalar_t, 2, 64),
            PTA_B(row_addresses, int64_t, 1, 32));
      });
  return output;
}

Tensor kvzch_direct_cache_gather_int8_float_cuda(
    Tensor output,
    Tensor row_addresses,
    at::ScalarType row_dtype,
    const int64_t max_ctas) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(output, row_addresses);
  TENSOR_CONTIGUOUS(output);
  TENSOR_CONTIGUOUS(row_addresses);
  TORCH_CHECK_EQ(output.dim(), 2);
  TORCH_CHECK_EQ(row_addresses.dim(), 1);
  TORCH_CHECK_EQ(output.size(0), row_addresses.numel());
  TORCH_CHECK_EQ(row_addresses.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(output.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK_GT(output.size(1), 0);
  TORCH_CHECK_EQ(output.size(1) % 4, 0);

  if (row_addresses.numel() == 0) {
    return output;
  }

  CUDA_DEVICE_GUARD(output);
  const auto D = output.size(1);
  const int32_t threads_per_row = std::min<int32_t>(D / 4, kMaxThreads);
  const dim3 threads(threads_per_row, kMaxThreads / threads_per_row);
  const auto grid_size = cap_gather_grid(
      utils::cuda::cap_grid_dim_x(
          div_round_up(row_addresses.numel(), threads.y),
          kMaxThreads,
          at::cuda::getCurrentCUDAStream()),
      max_ctas);

  FBGEMM_DISPATCH_FLOATING_TYPES(
      row_dtype, "kvzch_direct_cache_gather_int8_float", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (kvzch_direct_cache_gather_int8_float_kernel<scalar_t>),
            grid_size,
            threads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(output, float, 2, 64),
            PTA_B(row_addresses, int64_t, 1, 32));
      });
  return output;
}

namespace {

// A g8 row holds D / 2 code bytes then D / group_size scale bytes; the gather
// reads codes 4 bytes (8 values) at a time from 4-byte aligned rows.
void check_g8_rows(const int64_t D, const int64_t group_size) {
  TORCH_CHECK(
      group_size > 0 && group_size % 2 == 0 && D % group_size == 0,
      "g8 rows need an even group size dividing the dim ",
      D,
      ", got ",
      group_size);
  TORCH_CHECK_EQ(D % 8, 0);
  TORCH_CHECK_EQ((D / 2 + D / group_size) % 4, 0);
}

} // namespace

Tensor kvzch_direct_cache_lookup_raw_cuda(
    Tensor indices,
    Tensor cache_state,
    Tensor cache_weights,
    Tensor action_indices,
    Tensor actions_count,
    Tensor inserted_rows,
    Tensor row_addresses,
    const bool set_associative) {
  TORCH_CHECK_EQ(cache_weights.dim(), 2);
  TORCH_CHECK_EQ(inserted_rows.dim(), 2);
  TORCH_CHECK_EQ(cache_weights.scalar_type(), at::ScalarType::Byte);
  const auto inserted_dtype = inserted_rows.scalar_type();
  TORCH_CHECK(
      inserted_dtype == at::ScalarType::Float ||
          inserted_dtype == at::ScalarType::Half ||
          inserted_dtype == at::ScalarType::BFloat16,
      "inserted_rows must be float, half, or bfloat16");
  const auto D = inserted_rows.size(1);
  TORCH_CHECK_EQ(D % 4, 0);
  TORCH_CHECK_EQ(cache_weights.size(1) % 4, 0);
  // Bit 0 of every row address is the hit tag, so both row bases must be even.
  TORCH_CHECK_EQ(reinterpret_cast<uintptr_t>(cache_weights.data_ptr()) % 4, 0);
  TORCH_CHECK_EQ(reinterpret_cast<uintptr_t>(inserted_rows.data_ptr()) % 4, 0);
  return kvzch_direct_cache_lookup_impl(
      indices,
      cache_state,
      cache_weights,
      action_indices,
      actions_count,
      inserted_rows,
      row_addresses,
      cache_weights.size(1),
      D * inserted_rows.element_size(),
      1,
      set_associative);
}

Tensor kvzch_direct_cache_gather_raw_cuda(
    Tensor output,
    Tensor row_addresses,
    at::ScalarType row_dtype,
    const int64_t group_size,
    const int64_t max_ctas) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(output, row_addresses);
  TENSOR_CONTIGUOUS(output);
  TENSOR_CONTIGUOUS(row_addresses);
  TORCH_CHECK_EQ(output.dim(), 2);
  TORCH_CHECK_EQ(row_addresses.dim(), 1);
  TORCH_CHECK_EQ(output.size(0), row_addresses.numel());
  TORCH_CHECK_EQ(row_addresses.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_GE(group_size, 0);
  if (group_size > 0) {
    check_g8_rows(output.size(1), group_size);
  } else {
    TORCH_CHECK_EQ(output.size(1) % 4, 0);
  }

  if (row_addresses.numel() == 0) {
    return output;
  }

  CUDA_DEVICE_GUARD(output);
  const auto D = output.size(1);
  const int32_t threads_per_row =
      std::min<int32_t>(D / (group_size > 0 ? 8 : 4), kMaxThreads);
  const dim3 threads(threads_per_row, kMaxThreads / threads_per_row);
  const auto grid_size = cap_gather_grid(
      utils::cuda::cap_grid_dim_x(
          div_round_up(row_addresses.numel(), threads.y),
          kMaxThreads,
          at::cuda::getCurrentCUDAStream()),
      max_ctas);

  FBGEMM_DISPATCH_FLOATING_TYPES(
      output.scalar_type(), "kvzch_direct_cache_gather_raw_output", [&] {
        using out_t = scalar_t;
        FBGEMM_DISPATCH_FLOATING_TYPES(
            row_dtype, "kvzch_direct_cache_gather_raw_row", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (kvzch_direct_cache_gather_raw_kernel<out_t, scalar_t>),
                  grid_size,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(output, out_t, 2, 64),
                  PTA_B(row_addresses, int64_t, 1, 32),
                  static_cast<int32_t>(group_size));
            });
      });
  return output;
}

// Exclusive owner-local cache: the HBM copy of a row is its only copy, so every
// claimed key is reported for the backend to drop its DRAM row, and a row
// evicted from a full set is reported (with its bytes) for the backend to write
// back to DRAM.
namespace {

template <typename value_t, typename index_t, bool RawRows>
__global__
__launch_bounds__(kMaxThreads) void kvzch_exclusive_cache_insert_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor64<value_t, 2, at::RestrictPtrTraits> rows,
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> cache_state,
    pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        cache_weights,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> last_access,
    const int32_t* __restrict__ count,
    int64_t* __restrict__ promoted_keys,
    int32_t* __restrict__ promoted_count,
    const int32_t now,
    const OwnerCacheVictims victims,
    const bool set_associative) {
  constexpr int32_t kWarpsPerBlock = kMaxThreads / kWarpSize;
  const auto lane = threadIdx.x;
  const auto warp = threadIdx.y;
  const int32_t N = count == nullptr ? indices.size(0) : *count;
  CUDA_KERNEL_ASSERT(
      N >= 0 && N <= indices.size(0) &&
      "exclusive-cache insert count exceeds input size");
  const int32_t C = cache_state.size(0);
  const int32_t row_bytes = cache_weights.size(1);
  const int32_t D = row_bytes - kINT8QparamsBytes;
  const int32_t num_vecs = D / 4;
  // Lane i holds one pending claim. Host-mapped key/count outputs are
  // reported once per warp batch, for both promotions and victims.
  int64_t batch_key = 0;
  int64_t batch_victim = -1;
  int32_t batch_slot = -1;
  int32_t batch_n = 0;
  int32_t batch_access = 0;
  float batch_min = 0;
  float batch_max = 0;
  int32_t batch_size = 0;
  const auto report_batch = [&]() {
    const auto victim_mask = fbgemm_gpu::ballot_sync(
        static_cast<int32_t>(lane) < batch_size && batch_victim >= 0);
    int32_t victim_base = 0;
    if (lane == 0 && victim_mask != 0) {
      victim_base = atomicAdd(victims.count, __popc(victim_mask));
    }
    victim_base = shfl_sync(victim_base, 0);
    if (static_cast<int32_t>(lane) < batch_size && batch_victim >= 0) {
      const auto position =
          victim_base + __popc(victim_mask & ((uint32_t{1} << lane) - 1));
      victims.keys[position] = batch_victim;
      victims.last_access[position] = batch_access;
    }
    for (int32_t i = 0; i < batch_size; ++i) {
      const auto slot = shfl_sync(batch_slot, i);
      const auto n = shfl_sync(batch_n, i);
      auto* const dst = &cache_weights[slot][0];
      if ((victim_mask & (uint32_t{1} << i)) != 0) {
        const auto position =
            victim_base + __popc(victim_mask & ((uint32_t{1} << i) - 1));
        auto* const victim_row = reinterpret_cast<uint32_t*>(
            victims.rows + static_cast<int64_t>(position) * row_bytes);
        const auto* const old_row = reinterpret_cast<const uint32_t*>(dst);
        for (auto v = lane; v < row_bytes / 4; v += kWarpSize) {
          victim_row[v] = old_row[v];
        }
        syncwarp();
      }
      if constexpr (RawRows || std::is_same_v<value_t, uint8_t>) {
        const auto* const src = reinterpret_cast<const uint32_t*>(&rows[n][0]);
        for (auto v = lane; v < row_bytes / 4; v += kWarpSize) {
          reinterpret_cast<uint32_t*>(dst)[v] = src[v];
        }
      } else {
        const float bias = shfl_sync(batch_min, i);
        const float range = shfl_sync(batch_max, i) - bias;
        const float scale = range == 0.0f ? 1.0f : range / 255.0f;
        for (auto v = lane; v < num_vecs; v += kWarpSize) {
          const Vec4T<float> x(&rows[n][4 * v]);
          reinterpret_cast<uint32_t*>(dst)[v] =
              quantize_uint8(x.acc.x, bias, scale) |
              (quantize_uint8(x.acc.y, bias, scale) << 8) |
              (quantize_uint8(x.acc.z, bias, scale) << 16) |
              (quantize_uint8(x.acc.w, bias, scale) << 24);
        }
        if (lane == 0) {
          auto* const qparams = reinterpret_cast<float*>(dst + D);
          qparams[0] = scale;
          qparams[1] = bias;
        }
      }
      if (lane == 0) {
        last_access[slot] = now;
      }
      syncwarp();
      const auto key = shfl_sync(batch_key, i);
      if (lane == 0) {
        __threadfence();
        atomic_exchange_int64(&cache_state[slot][0], key);
      }
    }
    int32_t base = 0;
    if (lane == 0) {
      base = atomicAdd(promoted_count, batch_size);
    }
    base = shfl_sync(base, 0);
    if (static_cast<int32_t>(lane) < batch_size) {
      promoted_keys[base + lane] = batch_key;
    }
    batch_size = 0;
  };

  for (auto n = blockIdx.x * kWarpsPerBlock + warp; n < N;
       n += gridDim.x * kWarpsPerBlock) {
    const auto key = static_cast<int64_t>(indices[n]);
    bool lane_has_nonzero = false;
    float row_min = std::numeric_limits<float>::max();
    float row_max = std::numeric_limits<float>::lowest();
    if (key >= 0) {
      if constexpr (RawRows) {
        const auto* const src = reinterpret_cast<const uint32_t*>(&rows[n][0]);
        for (auto v = lane; v < row_bytes / 4; v += kWarpSize) {
          lane_has_nonzero |= src[v] != 0;
        }
      } else if constexpr (std::is_same_v<value_t, uint8_t>) {
        const auto* qparams = reinterpret_cast<const float*>(&rows[n][D]);
        if (!isfinite(qparams[0]) || !isfinite(qparams[1]) || qparams[0] < 0) {
          continue;
        }
        lane_has_nonzero = qparams[1] != 0.0f;
        for (auto d = lane; d < D; d += kWarpSize) {
          lane_has_nonzero |= qparams[0] != 0.0f && rows[n][d] != 0;
        }
      } else {
        for (auto v = lane; v < num_vecs; v += kWarpSize) {
          const Vec4T<float> x(&rows[n][4 * v]);
          lane_has_nonzero |= x.acc.x != 0.0f || x.acc.y != 0.0f ||
              x.acc.z != 0.0f || x.acc.w != 0.0f;
          row_min = fminf(row_min, x.vmin());
          row_max = fmaxf(row_max, x.vmax());
        }
      }
    }
    if (fbgemm_gpu::ballot_sync(lane_has_nonzero) == 0) {
      continue;
    }
    int64_t victim = -1;
    const auto slot = owner_cache_claim(
        &cache_state[0][0],
        victims.count == nullptr ? nullptr : &last_access[0],
        C,
        key,
        now,
        lane,
        &victim,
        set_associative,
        /*update_existing=*/false);
    if (slot < 0) {
      continue;
    }
    if constexpr (!RawRows && !std::is_same_v<value_t, uint8_t>) {
      row_min = warp_reduce_min(row_min);
      row_max = warp_reduce_max(row_max);
    }
    if (static_cast<int32_t>(lane) == batch_size) {
      batch_key = key;
      batch_victim = victim;
      batch_slot = slot;
      batch_n = n;
      batch_access = last_access[slot];
      batch_min = row_min;
      batch_max = row_max;
    }
    if (++batch_size == kWarpSize) {
      report_batch();
    }
  }
  if (batch_size > 0) {
    report_batch();
  }
}

__global__
__launch_bounds__(kMaxThreads) void kvzch_exclusive_cache_touch_kernel(
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        row_addresses,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> last_access,
    const uint64_t cache_weights_address,
    const uint64_t cache_row_bytes,
    const int32_t now) {
  const auto N = row_addresses.size(0);
  const auto num_slots = static_cast<uint64_t>(last_access.size(0));
  for (auto n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
       n += blockDim.x * gridDim.x) {
    // Hits point into the cache rows (bit 0 may tag them); misses point into
    // the inserted rows.
    const auto offset =
        (static_cast<uint64_t>(row_addresses[n]) & ~uint64_t{1}) -
        cache_weights_address;
    if (offset < num_slots * cache_row_bytes) {
      last_access[offset / cache_row_bytes] = now;
    }
  }
}

} // namespace

Tensor kvzch_exclusive_cache_insert_int8_cuda(
    Tensor cache_state,
    Tensor cache_weights,
    Tensor last_access,
    Tensor indices,
    Tensor rows,
    const std::optional<Tensor>& count,
    Tensor promoted_keys,
    Tensor promoted_count,
    const int64_t now,
    const bool input_is_packed,
    const std::optional<Tensor>& victim_keys,
    const std::optional<Tensor>& victim_access,
    const std::optional<Tensor>& victim_rows,
    const std::optional<Tensor>& victim_count,
    const bool set_associative) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cache_state,
      cache_weights,
      last_access,
      indices,
      rows,
      count,
      promoted_keys,
      promoted_count);
  check_owner_cache_last_access(cache_state, last_access, set_associative);
  TENSOR_CONTIGUOUS(cache_state);
  TENSOR_CONTIGUOUS(cache_weights);
  TENSOR_CONTIGUOUS(last_access);
  TENSOR_CONTIGUOUS(indices);
  TENSOR_CONTIGUOUS(rows);
  TENSOR_CONTIGUOUS(promoted_keys);
  TENSOR_CONTIGUOUS(promoted_count);
  if (count.has_value()) {
    TENSOR_CONTIGUOUS(count.value());
    TORCH_CHECK_EQ(count->scalar_type(), at::ScalarType::Int);
    TORCH_CHECK_EQ(count->numel(), 1);
  }
  TORCH_CHECK_EQ(cache_state.dim(), 2);
  TORCH_CHECK_EQ(cache_state.size(1), 1);
  TORCH_CHECK_EQ(cache_weights.dim(), 2);
  TORCH_CHECK_EQ(last_access.dim(), 1);
  TORCH_CHECK_EQ(indices.dim(), 1);
  TORCH_CHECK_EQ(rows.dim(), 2);
  TORCH_CHECK_EQ(cache_state.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(cache_weights.scalar_type(), at::ScalarType::Byte);
  TORCH_CHECK_EQ(last_access.scalar_type(), at::ScalarType::Int);
  TORCH_CHECK_EQ(promoted_keys.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(promoted_count.scalar_type(), at::ScalarType::Int);
  TORCH_CHECK_EQ(promoted_count.numel(), 1);
  TORCH_CHECK_EQ(cache_weights.size(0), cache_state.size(0));
  TORCH_CHECK_EQ(last_access.size(0), cache_state.size(0));
  TORCH_CHECK_EQ(rows.size(1) % 4, 0);
  TORCH_CHECK_EQ(
      cache_weights.size(1),
      rows.size(1) + (input_is_packed ? 0 : kINT8QparamsBytes));
  if (input_is_packed) {
    TORCH_CHECK_EQ(rows.scalar_type(), at::kByte);
    TORCH_CHECK_GT(rows.size(1), kINT8QparamsBytes);
  }
  TORCH_CHECK_EQ(reinterpret_cast<uintptr_t>(cache_weights.data_ptr()) % 4, 0);
  TORCH_CHECK_EQ(
      reinterpret_cast<uintptr_t>(rows.data_ptr()) % (4 * rows.element_size()),
      0);
  TORCH_CHECK_EQ(rows.size(0), indices.numel());
  TORCH_CHECK_GE(promoted_keys.numel(), indices.numel());
  TORCH_CHECK_GT(cache_state.size(0), 0);
  TORCH_CHECK_LE(cache_state.size(0), std::numeric_limits<int32_t>::max());
  TORCH_CHECK_LE(indices.numel(), std::numeric_limits<int32_t>::max());

  CUDA_DEVICE_GUARD(cache_weights);
  promoted_count.zero_();
  const auto victims = owner_cache_victims(
      victim_keys,
      victim_access,
      victim_rows,
      victim_count,
      indices.numel(),
      cache_weights.size(1));
  if (indices.numel() == 0) {
    return promoted_count;
  }

  constexpr int32_t kWarpsPerBlock = kMaxThreads / kWarpSize;
  const dim3 threads(kWarpSize, kWarpsPerBlock);
  // A few blocks per SM, so each warp strides over many rows and reports up
  // to kWarpSize claims per atomic on the host-mapped count.
  constexpr int64_t kBlocksPerSm = 4;
  const auto grid_size = std::min<int64_t>(
      div_round_up(indices.numel(), kWarpsPerBlock),
      kBlocksPerSm * get_device_sm_cnt_());
  const auto launch = [&](auto value) {
    using value_t = decltype(value);
    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "kvzch_exclusive_cache_insert_int8_index", [&] {
          FBGEMM_LAUNCH_KERNEL(
              (kvzch_exclusive_cache_insert_kernel<value_t, index_t, false>),
              grid_size,
              threads,
              0,
              at::cuda::getCurrentCUDAStream(),
              PTA_B(indices, index_t, 1, 32),
              PTA_B(rows, value_t, 2, 64),
              PTA_B(cache_state, int64_t, 2, 32),
              PTA_B(cache_weights, uint8_t, 2, 64),
              PTA_B(last_access, int32_t, 1, 32),
              count.has_value() ? count->data_ptr<int32_t>() : nullptr,
              promoted_keys.data_ptr<int64_t>(),
              promoted_count.data_ptr<int32_t>(),
              static_cast<int32_t>(now),
              victims,
              set_associative);
        });
  };
  if (input_is_packed) {
    launch(uint8_t{});
  } else {
    FBGEMM_DISPATCH_FLOATING_TYPES(
        rows.scalar_type(), "kvzch_exclusive_cache_insert_int8_value", [&] {
          launch(scalar_t{});
        });
  }
  return promoted_count;
}

Tensor kvzch_exclusive_cache_insert_raw_cuda(
    Tensor cache_state,
    Tensor cache_weights,
    Tensor last_access,
    Tensor indices,
    Tensor rows,
    const std::optional<Tensor>& count,
    Tensor promoted_keys,
    Tensor promoted_count,
    const int64_t now,
    const std::optional<Tensor>& victim_keys,
    const std::optional<Tensor>& victim_access,
    const std::optional<Tensor>& victim_rows,
    const std::optional<Tensor>& victim_count,
    const bool set_associative) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cache_state,
      cache_weights,
      last_access,
      indices,
      rows,
      count,
      promoted_keys,
      promoted_count);
  check_owner_cache_last_access(cache_state, last_access, set_associative);
  TENSOR_CONTIGUOUS(cache_state);
  TENSOR_CONTIGUOUS(cache_weights);
  TENSOR_CONTIGUOUS(last_access);
  TENSOR_CONTIGUOUS(indices);
  TENSOR_CONTIGUOUS(rows);
  TENSOR_CONTIGUOUS(promoted_keys);
  TENSOR_CONTIGUOUS(promoted_count);
  if (count.has_value()) {
    TENSOR_CONTIGUOUS(count.value());
    TORCH_CHECK_EQ(count->scalar_type(), at::ScalarType::Int);
    TORCH_CHECK_EQ(count->numel(), 1);
  }
  TORCH_CHECK_EQ(cache_state.dim(), 2);
  TORCH_CHECK_EQ(cache_state.size(1), 1);
  TORCH_CHECK_EQ(cache_weights.dim(), 2);
  TORCH_CHECK_EQ(last_access.dim(), 1);
  TORCH_CHECK_EQ(indices.dim(), 1);
  TORCH_CHECK_EQ(rows.dim(), 2);
  TORCH_CHECK_EQ(cache_state.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(cache_weights.scalar_type(), at::ScalarType::Byte);
  TORCH_CHECK_EQ(rows.scalar_type(), at::ScalarType::Byte);
  TORCH_CHECK_EQ(last_access.scalar_type(), at::ScalarType::Int);
  TORCH_CHECK_EQ(promoted_keys.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(promoted_count.scalar_type(), at::ScalarType::Int);
  TORCH_CHECK_EQ(promoted_count.numel(), 1);
  TORCH_CHECK_EQ(cache_weights.size(0), cache_state.size(0));
  TORCH_CHECK_EQ(last_access.size(0), cache_state.size(0));
  TORCH_CHECK_EQ(rows.size(1), cache_weights.size(1));
  TORCH_CHECK_EQ(rows.size(1) % 4, 0);
  TORCH_CHECK_EQ(reinterpret_cast<uintptr_t>(cache_weights.data_ptr()) % 4, 0);
  TORCH_CHECK_EQ(reinterpret_cast<uintptr_t>(rows.data_ptr()) % 4, 0);
  TORCH_CHECK_EQ(rows.size(0), indices.numel());
  TORCH_CHECK_GE(promoted_keys.numel(), indices.numel());
  TORCH_CHECK_GT(cache_state.size(0), 0);
  TORCH_CHECK_LE(cache_state.size(0), std::numeric_limits<int32_t>::max());
  TORCH_CHECK_LE(indices.numel(), std::numeric_limits<int32_t>::max());

  CUDA_DEVICE_GUARD(cache_weights);
  promoted_count.zero_();
  const auto victims = owner_cache_victims(
      victim_keys,
      victim_access,
      victim_rows,
      victim_count,
      indices.numel(),
      cache_weights.size(1));
  if (indices.numel() == 0) {
    return promoted_count;
  }

  constexpr int32_t kWarpsPerBlock = kMaxThreads / kWarpSize;
  const dim3 threads(kWarpSize, kWarpsPerBlock);
  constexpr int64_t kBlocksPerSm = 4;
  const auto grid_size = std::min<int64_t>(
      div_round_up(indices.numel(), kWarpsPerBlock),
      kBlocksPerSm * get_device_sm_cnt_());
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "kvzch_exclusive_cache_insert_raw", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (kvzch_exclusive_cache_insert_kernel<uint8_t, index_t, true>),
            grid_size,
            threads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(indices, index_t, 1, 32),
            PTA_B(rows, uint8_t, 2, 64),
            PTA_B(cache_state, int64_t, 2, 32),
            PTA_B(cache_weights, uint8_t, 2, 64),
            PTA_B(last_access, int32_t, 1, 32),
            count.has_value() ? count->data_ptr<int32_t>() : nullptr,
            promoted_keys.data_ptr<int64_t>(),
            promoted_count.data_ptr<int32_t>(),
            static_cast<int32_t>(now),
            victims,
            set_associative);
      });
  return promoted_count;
}

Tensor kvzch_exclusive_cache_touch_cuda(
    Tensor row_addresses,
    Tensor cache_weights,
    Tensor last_access,
    const int64_t now) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      row_addresses, cache_weights, last_access);
  TENSOR_CONTIGUOUS(row_addresses);
  TENSOR_CONTIGUOUS(cache_weights);
  TENSOR_CONTIGUOUS(last_access);
  TORCH_CHECK_EQ(row_addresses.dim(), 1);
  TORCH_CHECK_EQ(row_addresses.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(cache_weights.dim(), 2);
  TORCH_CHECK_EQ(last_access.dim(), 1);
  TORCH_CHECK_EQ(last_access.scalar_type(), at::ScalarType::Int);
  TORCH_CHECK_EQ(last_access.size(0), cache_weights.size(0));

  CUDA_DEVICE_GUARD(last_access);
  if (row_addresses.numel() == 0) {
    return last_access;
  }
  const auto grid_size = utils::cuda::cap_grid_dim_x(
      div_round_up(row_addresses.numel(), kMaxThreads),
      kMaxThreads,
      at::cuda::getCurrentCUDAStream());
  FBGEMM_LAUNCH_KERNEL(
      kvzch_exclusive_cache_touch_kernel,
      grid_size,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(row_addresses, int64_t, 1, 32),
      PTA_B(last_access, int32_t, 1, 32),
      reinterpret_cast<uint64_t>(cache_weights.data_ptr()),
      static_cast<uint64_t>(
          cache_weights.size(1) * cache_weights.element_size()),
      static_cast<int32_t>(now));
  return last_access;
}

namespace {

__global__
__launch_bounds__(kMaxThreads) void kvzch_owner_cache_clear_tags_kernel(
    pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> cache_state,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> slots,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> keys) {
  for (auto n = blockIdx.x * blockDim.x + threadIdx.x; n < slots.size(0);
       n += blockDim.x * gridDim.x) {
    atomic_cas_int64(&cache_state[slots[n]][0], keys[n], -1);
  }
}

} // namespace

Tensor kvzch_owner_cache_clear_tags_cuda(
    Tensor cache_state,
    Tensor slots,
    Tensor keys) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(cache_state, slots, keys);
  TENSOR_CONTIGUOUS(cache_state);
  TENSOR_CONTIGUOUS(slots);
  TENSOR_CONTIGUOUS(keys);
  TORCH_CHECK_EQ(cache_state.dim(), 2);
  TORCH_CHECK_EQ(cache_state.size(1), 1);
  TORCH_CHECK_EQ(cache_state.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(slots.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(keys.scalar_type(), at::ScalarType::Long);
  TORCH_CHECK_EQ(slots.numel(), keys.numel());
  CUDA_DEVICE_GUARD(cache_state);
  if (slots.numel() == 0) {
    return cache_state;
  }
  FBGEMM_LAUNCH_KERNEL(
      kvzch_owner_cache_clear_tags_kernel,
      utils::cuda::cap_grid_dim_x(
          div_round_up(slots.numel(), kMaxThreads),
          kMaxThreads,
          at::cuda::getCurrentCUDAStream()),
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(cache_state, int64_t, 2, 32),
      PTA_B(slots, int64_t, 1, 32),
      PTA_B(keys, int64_t, 1, 32));
  return cache_state;
}
