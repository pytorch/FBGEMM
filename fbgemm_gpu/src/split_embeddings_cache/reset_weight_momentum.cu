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

int get_sm_count_() {
  cudaDeviceProp* deviceProp =
      at::cuda::getDeviceProperties(c10::cuda::current_device());
  return deviceProp->multiProcessorCount;
}

__global__ __launch_bounds__(kMaxThreads) void get_cache_indices_kernel(
    int32_t blocks_per_table,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_indices,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_indices_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        logical_table_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        buffer_ids,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        linear_cache_indices) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  const int32_t t_i = blockIdx.x / blocks_per_table;
  const int32_t threads_per_table = blocks_per_table * blockDim.x;
  const int32_t idx_table = index % threads_per_table;
  const int32_t logical_id = logical_table_ids[t_i];
  const int32_t buffer_id = buffer_ids[t_i];

  const int64_t num_indices =
      pruned_indices_offsets[buffer_id + 1] - pruned_indices_offsets[buffer_id];

  if (num_indices <= 0) {
    return;
  }

  const int64_t indices_per_thread =
      div_round_up(num_indices, threads_per_table);
  const int64_t start = idx_table * indices_per_thread;
  const int64_t end = min(start + indices_per_thread, num_indices);

  if (start >= num_indices) {
    return;
  }

  const int64_t pruned_indices_offset = pruned_indices_offsets[buffer_id];
  const int64_t* pruned_indices_table = &pruned_indices[pruned_indices_offset];
  int64_t* linear_cache_indices_table =
      &linear_cache_indices[pruned_indices_offset];

  const auto max_offset =
      ::__ldg(&cache_hash_size_cumsum[cache_hash_size_cumsum.size(0) - 1]);
  const auto curr_offset = ::__ldg(&cache_hash_size_cumsum[logical_id]);

  for (int64_t i = start; i < end; i++) {
    if (curr_offset >= 0) {
      linear_cache_indices_table[i] = curr_offset + pruned_indices_table[i];
    } else {
      linear_cache_indices_table[i] = max_offset;
    }
  }
}

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kMaxThreads) void reset_weight_momentum_kernel(
    int32_t blocks_per_table,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    pta::PackedTensorAccessor64<
        at::acc_type<cache_t, true>,
        1,
        at::RestrictPtrTraits> momentum1_dev,
    pta::PackedTensorAccessor64<
        at::acc_type<cache_t, true>,
        1,
        at::RestrictPtrTraits> momentum1_uvm,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        momentum1_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        momentum1_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_indices,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_indices_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        logical_table_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        buffer_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  const int32_t t_i = blockIdx.x / blocks_per_table;
  const int32_t buffer_id = buffer_ids[t_i];
  const int64_t num_indices =
      pruned_indices_offsets[buffer_id + 1] - pruned_indices_offsets[buffer_id];

  if (num_indices <= 0) {
    return;
  }

  const int32_t logical_id = logical_table_ids[t_i];
  int32_t D = D_offsets[logical_id + 1] - D_offsets[logical_id];
  const int32_t chunk4s_per_row = D / 4;
  const int64_t total_chunk4s_per_table = num_indices * chunk4s_per_row;

  const int32_t threads_per_table = blocks_per_table * blockDim.x;
  const int64_t chunk4s_per_thread =
      div_round_up(total_chunk4s_per_table, threads_per_table);
  const int32_t idx_table = index % threads_per_table;
  const int64_t start = idx_table * chunk4s_per_thread;
  const int64_t end = min(start + chunk4s_per_thread, total_chunk4s_per_table);

  if (start >= total_chunk4s_per_table) {
    return;
  }

  int32_t D_emb = D;
  if constexpr (std::is_same_v<emb_t, uint8_t>) {
    D_emb += kINT8QparamsBytes;
  }

  at::acc_type<cache_t, true>* __restrict__ momentum1;
  const auto momentum1_placement =
      static_cast<PlacementType>(momentum1_placements[logical_id]);
  int64_t momentum1_offset = momentum1_offsets[logical_id];
  if (momentum1_placement == PlacementType::DEVICE) {
    momentum1 = &momentum1_dev[momentum1_offset];
  } else {
    momentum1 = &momentum1_uvm[momentum1_offset];
  }

  emb_t* __restrict__ weights{nullptr};
  cache_t* __restrict__ cache_weights{nullptr};
  const auto weights_placement =
      static_cast<PlacementType>(weights_placements[logical_id]);
  int64_t weights_offset = weights_offsets[logical_id];

  const int64_t pruned_indices_offset = pruned_indices_offsets[buffer_id];
  const int64_t* pruned_indices_table = &pruned_indices[pruned_indices_offset];

  for (int64_t i = start; i < end; i++) {
    int64_t idx = i / chunk4s_per_row;
    int64_t pruned_index = pruned_indices_table[idx];

    if (weights_placement == PlacementType::DEVICE) {
      weights = &dev_weights[weights_offset + pruned_index * D_emb];
    } else {
      weights = &uvm_weights[weights_offset + pruned_index * D_emb];
    }
    if (weights_placement == PlacementType::MANAGED_CACHING) {
      int32_t cache_idx = lxu_cache_locations[pruned_indices_offset + idx];
      if (cache_idx != kCacheLocationMissing) {
        cache_weights = &lxu_cache_weights[cache_idx][0];
      }
    }

    auto weight_row_template =
        WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
            weights, cache_weights, D, nullptr);

    // reset momentum1
    const int32_t d = (i % chunk4s_per_row) * 4;
    if (d == 0) {
      momentum1[pruned_index] = 0;
    }

    // reset weight
    float2 qparams_new = {1.0, 0.0}; // scaler=1.0, and offset=0.0, for int8.
    Vec4T<at::acc_type<cache_t, true>> weight_new; // 0 weight
    weight_row_template.store(
        weight_new,
        d,
        qparams_new); // qparams_new not used if type is not int8
  }
}

} // namespace

DLL_PUBLIC void reset_weight_momentum_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor momentum1_dev,
    Tensor momentum1_uvm,
    Tensor momentum1_placements,
    Tensor momentum1_offsets,
    Tensor D_offsets,
    Tensor pruned_indices,
    Tensor pruned_indices_offsets,
    Tensor logical_table_ids,
    Tensor buffer_ids,
    Tensor cache_hash_size_cumsum,
    Tensor lxu_cache_state,
    int64_t total_cache_hash_size) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      dev_weights,
      uvm_weights,
      lxu_cache_weights,
      weights_placements,
      weights_offsets,
      momentum1_dev,
      momentum1_uvm,
      momentum1_placements,
      momentum1_offsets,
      D_offsets,
      pruned_indices,
      pruned_indices_offsets,
      logical_table_ids,
      buffer_ids,
      cache_hash_size_cumsum,
      lxu_cache_state);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(dev_weights.get_device());

  const int64_t num_pruned_indices = pruned_indices.size(0);
  const int32_t num_pruned_tables = buffer_ids.size(0);
  const int32_t blocks_per_table = get_sm_count_();

  auto lxu_cache_locations =
      at::zeros({num_pruned_indices}, pruned_indices.options().dtype(at::kInt));
  lxu_cache_locations.fill_(kCacheLocationMissing);

  if (total_cache_hash_size > 0) {
    // Get corresponding cache indices of pruned indices
    auto linear_cache_indices = at::zeros(
        {num_pruned_indices}, pruned_indices.options().dtype(at::kLong));

#ifdef FBGEMM_GPU_MEMCHECK
    const char* func_name = "get_cache_indices_kernel";
#endif

    get_cache_indices_kernel<<<
        num_pruned_tables * blocks_per_table,
        kMaxThreads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        blocks_per_table,
        MAKE_PTA_WITH_NAME(func_name, cache_hash_size_cumsum, int64_t, 1, 32),
        MAKE_PTA_WITH_NAME(func_name, pruned_indices, int64_t, 1, 32),
        MAKE_PTA_WITH_NAME(func_name, pruned_indices_offsets, int64_t, 1, 32),
        MAKE_PTA_WITH_NAME(func_name, logical_table_ids, int32_t, 1, 32),
        MAKE_PTA_WITH_NAME(func_name, buffer_ids, int32_t, 1, 32),
        MAKE_PTA_WITH_NAME(func_name, linear_cache_indices, int64_t, 1, 32));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Look up cache locations
    Tensor uvm_cache_stats =
        at::empty({0}, lxu_cache_weights.options().dtype(at::kInt));
    lxu_cache_locations = lxu_cache_lookup_cuda(
        linear_cache_indices,
        lxu_cache_state,
        total_cache_hash_size,
        false, // gather_cache_stats
        uvm_cache_stats);
  }

  // Reset weight and momentum of pruned rows
  DISPATCH_EMB_CACHE_TYPES(
      dev_weights.scalar_type(),
      lxu_cache_weights.scalar_type(),
      "reset_weight_momentum_kernel",
      ([&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name2 = "get_cache_indices_kernel";
#endif
        reset_weight_momentum_kernel<emb_t, cache_t>
            <<<num_pruned_tables * blocks_per_table,
               kMaxThreads,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                blocks_per_table,
                MAKE_PTA_WITH_NAME(func_name2, dev_weights, emb_t, 1, 64),
                MAKE_PTA_WITH_NAME(func_name2, uvm_weights, emb_t, 1, 64),
                MAKE_PTA_WITH_NAME(
                    func_name2, lxu_cache_weights, cache_t, 2, 64),
                MAKE_PTA_WITH_NAME(
                    func_name2, weights_placements, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, weights_offsets, int64_t, 1, 32),
                MAKE_PTA_ACC_WITH_NAME(
                    func_name2, momentum1_dev, cache_t, 1, 64),
                MAKE_PTA_ACC_WITH_NAME(
                    func_name2, momentum1_uvm, cache_t, 1, 64),
                MAKE_PTA_WITH_NAME(
                    func_name2, momentum1_placements, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(
                    func_name2, momentum1_offsets, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, D_offsets, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, pruned_indices, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(
                    func_name2, pruned_indices_offsets, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(
                    func_name2, logical_table_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name2, buffer_ids, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(
                    func_name2, lxu_cache_locations, int32_t, 1, 32));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}
