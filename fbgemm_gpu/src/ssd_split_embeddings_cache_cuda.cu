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
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

template <typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void masked_index_put_kernel(
    at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> self,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> values,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> count,
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
  const auto D = self.size(1);
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t>::copy((&values[n][0]) + d * 4, (&self[idx][0]) + d * 4);
  }
}

template <>
__global__ __launch_bounds__(kMaxThreads) void masked_index_put_kernel(
    at::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits> self,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<uint8_t, 2, at::RestrictPtrTraits> values,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> count,
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

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(self.get_device());
  const auto N = indices.numel();
  if (N == 0) {
    return self;
  }
  const auto D = self.size(1);
  TORCH_CHECK_EQ(self.size(1), values.size(1));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::Byte,
      self.scalar_type(),
      "masked_index_put",
      [&] {
        const int32_t tx = std::min<int32_t>(D / 4, kMaxThreads);
        const dim3 threads(tx, kMaxThreads / tx);
        TORCH_DSA_KERNEL_LAUNCH(
            masked_index_put_kernel<scalar_t>,
            div_round_up(N, kMaxThreads / tx),
            dim3(tx, kMaxThreads / tx),
            0,
            at::cuda::getCurrentCUDAStream(),
            self.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
            indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            count.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());
      } // lambda
  );

  return self;
}

__global__ __launch_bounds__(kMaxThreads) void ssd_cache_actions_insert_kernel(
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_cache_sets, // [N = \sum_{b} L_{b} total indices, i.e.
                           // flattened
                           // [B][L]
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices, // [N = \sum_{b} L_{b} total indices, i.e.
                                  // flattened [B][L]
    int64_t time_stamp,
    int64_t prefetch_dist, // Number of batches we can prefetch ahead of a
                           // forward call A value of 1 means that entries where
                           // timestep with insert_time >= time_stamp -
                           // prefetch_dist are locked, and cannot be evicted.
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        assigned_cache_slots,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        evicted_indices,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> actions_count,
    TORCH_DSA_KERNEL_ARGS) {
  const int32_t C = lxu_cache_state.size(0);

  const int32_t N = sorted_cache_sets.size(0);
  const int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }

  const int32_t cache_set = sorted_cache_sets[n];
  if (cache_set >= C) {
    // ignore the already-existing elements
    evicted_indices[n] = -1;
    assigned_cache_slots[n] = -1;
    return;
  }

  // check if this warp is responsible for this whole segment.
  const bool segment_start =
      (n == 0 || sorted_cache_sets[n - 1] != sorted_cache_sets[n]);

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

  // This will mean that we can't insert all the indices for our segment,
  // which will break the guarantees required for the SSD embedding.
  // If you hit this, increase the cache size.
  CUDA_KERNEL_ASSERT2(SL <= kWarpSize);
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
  if (l >= SL) {
    return;
  }

  const int32_t insert_slot = sorted_slot;
  const int64_t insert_time = sorted_time;

  const int64_t insert_idx = cache_set_sorted_indices[n + l];
  const int64_t current_idx = lxu_cache_state[cache_set][insert_slot];

  // Only check insert_time if tag is for valid entry
  if (current_idx != -1) {
    // We need to ensure if prefetching (prefetch_dist) batches ahead
    // No entries that are younger than (time_stamp - prefetch_dist) are
    // evicted from the cache. This will break the guarantees required
    // for the SSD embedding.
    // If you hit this assert, increase the cache size.
    CUDA_KERNEL_ASSERT2(insert_time < (time_stamp - prefetch_dist));
  }

  evicted_indices[n + l] = current_idx; // -1 if not set, >= 0 if valid.
  assigned_cache_slots[n + l] = cache_set * kWarpSize + insert_slot;
  lxu_cache_state[cache_set][insert_slot] = insert_idx;
  lru_state[cache_set][insert_slot] = time_stamp;

  if (threadIdx.x == 0) {
    gpuAtomicAdd(&actions_count[0], SL);
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor> ssd_cache_populate_actions_cuda(
    Tensor linear_indices,
    int64_t total_hash_size,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    int64_t prefetch_dist,
    Tensor lru_state) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_indices, lxu_cache_state, lru_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(linear_indices.get_device());

  // Get unique indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(linear_indices, total_hash_size, false);

  TORCH_CHECK_LT(unique_indices.numel(), std::numeric_limits<int32_t>::max());
  const int32_t N = unique_indices.numel();

  auto evicted_indices = empty_like(unique_indices);
  auto assigned_cache_slots =
      empty_like(unique_indices, unique_indices.options().dtype(at::kInt));
  auto actions_count = at::zeros({1}, unique_indices.options().dtype(at::kInt));

  if (unique_indices.numel() == 0) {
    // these are all of length zero
    return std::make_tuple(
        empty_like(unique_indices),
        evicted_indices,
        assigned_cache_slots,
        actions_count);
  }
  // Find uncached indices
  Tensor uvm_cache_stats =
      at::empty({0}, linear_indices.options().dtype(at::kInt));
  auto cache_sets_and_unique_indices = lru_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_hash_size,
      lxu_cache_state,
      time_stamp,
      lru_state,
      false, // gather_cache_stats
      uvm_cache_stats);
  auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  auto cache_set_sorted_unique_indices = cache_sets_and_unique_indices.second;
  TORCH_DSA_KERNEL_LAUNCH(
      ssd_cache_actions_insert_kernel,
      div_round_up(N, kMaxThreads / kWarpSize),
      dim3(kWarpSize, kMaxThreads / kWarpSize),
      0,
      at::cuda::getCurrentCUDAStream(),
      lxu_cache_state.packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
      sorted_cache_sets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      cache_set_sorted_unique_indices
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      time_stamp,
      prefetch_dist,
      lru_state.packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
      assigned_cache_slots
          .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      evicted_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      actions_count.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());

  return std::make_tuple(
      cache_set_sorted_unique_indices,
      evicted_indices,
      assigned_cache_slots,
      actions_count);
}
