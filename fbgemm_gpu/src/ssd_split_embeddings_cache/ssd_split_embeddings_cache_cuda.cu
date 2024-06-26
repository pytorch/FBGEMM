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
#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/utils/bitonic_sort.cuh"

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

template <typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void masked_index_put_kernel(
    pta::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> self,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        values,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> count,
    TORCH_DSA_KERNEL_ARGS) {
  const int32_t N = indices.size(0);
  const int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }
  const auto count_ = count[0];
  if (n >= count_) {
    return;
  }
  const auto idx = indices[n];
  if (idx < 0) {
    return;
  }
  const auto D = self.size(1);
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t>::copy((&values[n][0]) + d * 4, (&self[idx][0]) + d * 4);
  }
}

template <>
__global__ __launch_bounds__(kMaxThreads) void masked_index_put_kernel(
    pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits> self,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<uint8_t, 2, at::RestrictPtrTraits> values,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> count,
    TORCH_DSA_KERNEL_ARGS) {
  const int32_t N = indices.size(0);
  const int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }
  const auto count_ = count[0];
  if (n >= count_) {
    return;
  }
  const auto idx = indices[n];
  if (idx < 0) {
    return;
  }
  const auto D = self.size(1);
  // each row is padded with row_alignment (16 bytes on GPUs), so each row will
  // be multiple of 16 bytes (uint4 = 32bit x 4 = 16 bytes).
  CUDA_KERNEL_ASSERT2(
      D % 16 == 0 && "D needs to be padded to be multiple of 16");
  auto vec_self = reinterpret_cast<uint4*>(&self[idx][0]);
  auto vec_values = reinterpret_cast<const uint4*>(&values[n][0]);
  for (int32_t d = threadIdx.x; d * sizeof(uint4) < D; d += blockDim.x) {
    vec_self[d] = vec_values[d];
  }
}

Tensor masked_index_put_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(self, indices, values, count);

  CUDA_DEVICE_GUARD(self);

  const auto N = indices.numel();
  if (N == 0) {
    return self;
  }
  const auto D = self.size(1);
  TORCH_CHECK_EQ(self.size(1), values.size(1));

  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      self.scalar_type(),
      "masked_index_put",
      [&] {
        const int32_t tx = std::min<int32_t>(D / 4, kMaxThreads);
        const dim3 threads(tx, kMaxThreads / tx);
#ifdef FBGEMM_GPU_MEMCHECK
        const auto func_name = "masked_index_put_kernel";
#endif
        TORCH_DSA_KERNEL_LAUNCH(
            masked_index_put_kernel<scalar_t>,
            div_round_up(N, kMaxThreads / tx),
            dim3(tx, kMaxThreads / tx),
            0,
            at::cuda::getCurrentCUDAStream(),
            MAKE_PTA_WITH_NAME(func_name, self, scalar_t, 2, 64),
            MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, values, scalar_t, 2, 32),
            MAKE_PTA_WITH_NAME(func_name, count, int32_t, 1, 32));
      } // lambda
  );

  return self;
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
    TORCH_DSA_KERNEL_ARGS) {
  // Number of cache sets
  const int32_t C = lxu_cache_state.size(0);

  const int32_t N = sorted_cache_sets.size(0);
  const int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
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

  // now, we need to insert the (unique!) values in indices[n:n + SL] into
  // our slots.
  const int32_t slot = threadIdx.x;
  const int64_t slot_time = lru_state[cache_set][slot];
  int64_t costs[1] = {slot_time};
  int32_t slots[1] = {slot};

  BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
  const int32_t sorted_slot = slots[0];
  const int64_t sorted_time = costs[0];

  auto l = threadIdx.x;

  // Insert rows
  if (l < SL) {
    // Insert indices
    const int32_t insert_slot = sorted_slot;
    const int64_t insert_time = sorted_time;

    const int64_t insert_idx = cache_set_sorted_indices[n + l];
    const int64_t current_idx = lxu_cache_state[cache_set][insert_slot];

#if 0
    // TODO: Check whether to uncomment this
    // Only check insert_time if tag is for valid entry
    if (current_idx != -1) {
      // We need to ensure if prefetching (prefetch_dist) batches ahead
      // No entries that are younger than (time_stamp - prefetch_dist) are
      // evicted from the cache. This will break the guarantees required
      // for the SSD embedding.
      // If you hit this assert, increase the cache size.
      CUDA_KERNEL_ASSERT2(insert_time < (time_stamp - prefetch_dist));
    }
#endif

    if (current_idx != -1 && insert_time == time_stamp) {
      // Skip this slot as the inserted row was a cache hit
      // This is conflict miss
      evicted_indices[n + l] = -1;
      assigned_cache_slots[n + l] = -1;
    } else {
      evicted_indices[n + l] = current_idx; // -1 if not set, >= 0 if valid.
      assigned_cache_slots[n + l] = cache_set * kWarpSize + insert_slot;
      lxu_cache_state[cache_set][insert_slot] = insert_idx;
      lru_state[cache_set][insert_slot] = time_stamp;
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
    Tensor lru_state) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_indices, lxu_cache_state, lru_state);

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

  auto actions_count = at::empty({1}, int_options);
  // Find uncached indices
  Tensor uvm_cache_stats = at::empty({0}, int_options);
  Tensor lxu_cache_locking_counter = at::empty({0, 0}, int_options);
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
              /*gather_cache_stats=*/false,
              uvm_cache_stats,
              /*lock_cache_line=*/false,
              lxu_cache_locking_counter,
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
      MAKE_PTA_WITH_NAME(func_name, actions_count, int32_t, 1, 32));

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

__global__ __launch_bounds__(kMaxThreads) void ssd_generate_row_addrs_kernel(
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> ssd_row_addrs,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        post_bwd_evicted_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        assigned_cache_slots,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        linear_index_inverse_indices,
    // TODO: Use int64_t here
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        unique_indices_count_cumsum,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        cache_set_inverse_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
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

  ssd_generate_row_addrs_kernel<<<
      div_round_up(lxu_cache_locations.numel(), kNumWarps),
      dim3(kWarpSize, kNumWarps),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      ssd_row_addrs.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      post_bwd_evicted_indices
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      lxu_cache_locations
          .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      assigned_cache_slots
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      linear_index_inverse_indices
          .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      unique_indices_count_cumsum
          .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      cache_set_inverse_indices
          .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      cache_set_sorted_unique_indices
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      lxu_cache_weights_addr,
      reinterpret_cast<uint64_t>(inserted_ssd_weights.data_ptr()),
      unique_indices_length.data_ptr<int32_t>(),
      cache_row_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {ssd_row_addrs, post_bwd_evicted_indices};
}
