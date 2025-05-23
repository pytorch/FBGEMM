/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fbgemm_gpu/embedding_inplace_update.h"
#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

constexpr int32_t kCacheLocationMissing = -1;

template <typename index_t>
inline __device__ void embedding_inplace_update_kernel_impl(
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> uvm_weights,
    const PlacementType& weights_placement,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const pta::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
        weights_tys,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits>
        update_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        update_table_idx,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        update_row_idx,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        update_offsets,
    const int64_t row_alignment,
    pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations) {
  // each row is updated by one warp of threads
  // blockIdx.x: block idx, threadIdx.x: thread idx in the warp,
  // threadIdx.y: warp idx in the block.
  // blockDim.x = warpSize, blockDim.y = warpsPerBlock.
  const int64_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= update_row_idx.size(0)) {
    return;
  }
  const int32_t table_idx = update_table_idx[i];
  const auto row_idx = update_row_idx[i];

  // TODO: We don't need to compute these here.
  const int32_t D_start = D_offsets[table_idx];
  const int32_t D_end = D_offsets[table_idx + 1];
  const int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[table_idx]);
  const int32_t D_bytes =
      nbit::padded_row_size_in_bytes(D, weight_ty, row_alignment);

  const int64_t weight_offset = weights_offsets[table_idx];
  uint8_t* __restrict__ weight_row;
  if (weights_placement == PlacementType::DEVICE) {
    weight_row =
        &dev_weights
            [weight_offset +
             static_cast<int64_t>(D_bytes) * static_cast<int64_t>(row_idx)];
  } else {
    weight_row =
        &uvm_weights
            [weight_offset +
             static_cast<int64_t>(D_bytes) * static_cast<int64_t>(row_idx)];
  }

  // padded_row_size_in_bytes pad each row with row_alignment (16 bytes on GPUs)
  // So each row will be multiple of 16 bytes (uint4 = 32bit x 4 = 16 bytes)
  auto vec_weight_row = reinterpret_cast<uint4*>(weight_row);
  const int64_t update_weight_offset = update_offsets[i];
  auto update_weight_row =
      reinterpret_cast<const uint4*>(&update_weights[update_weight_offset]);
  // Do wider loads/stores so that each 16 Byte segment in the row can be
  // updated in a single memory transaction
  for (auto d = threadIdx.x; d * sizeof(uint4) < D_bytes; d += blockDim.x) {
    vec_weight_row[d] = update_weight_row[d];
  }

  bool cache_valid = (weights_placement == PlacementType::MANAGED_CACHING);
  int32_t cache_idx =
      cache_valid ? lxu_cache_locations[i] : kCacheLocationMissing;
  if (cache_valid && cache_idx != kCacheLocationMissing) {
    auto vec_cache_row = reinterpret_cast<uint4*>(
        &lxu_cache_weights[static_cast<int64_t>(cache_idx)][0]);

    for (auto d = threadIdx.x; d * sizeof(uint4) < D_bytes; d += blockDim.x) {
      vec_cache_row[d] = update_weight_row[d];
    }
  }
}

template <typename index_t>
__launch_bounds__(kMaxThreads) __global__
    void embedding_inplace_update_kernel_1(
        pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits>
            dev_weights,
        pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits>
            uvm_weights,
        const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            weights_placements,
        const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
            weights_offsets,
        const pta::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
            weights_tys,
        const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            D_offsets,
        const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits>
            update_weights,
        const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            update_table_idx,
        const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
            update_row_idx,
        const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
            update_offsets,
        const int64_t row_alignment,
        pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
            lxu_cache_weights,
        const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            lxu_cache_locations) {
  // each row is updated by one warp of threads
  // blockIdx.x: block idx, threadIdx.x: thread idx in the warp,
  // threadIdx.y: warp idx in the block.
  // blockDim.x = warpSize, blockDim.y = warpsPerBlock.
  const int64_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= update_row_idx.size(0)) {
    return;
  }
  const int32_t table_idx = update_table_idx[i];
  const auto placement =
      static_cast<PlacementType>(weights_placements[table_idx]);
  embedding_inplace_update_kernel_impl(
      dev_weights,
      uvm_weights,
      placement,
      weights_offsets,
      weights_tys,
      D_offsets,
      update_weights,
      update_table_idx,
      update_row_idx,
      update_offsets,
      row_alignment,
      lxu_cache_weights,
      lxu_cache_locations);
}

template <typename index_t>
__launch_bounds__(kMaxThreads) __global__
    void embedding_inplace_update_kernel_2(
        pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits>
            dev_weights,
        pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits>
            uvm_weights,
        const PlacementType weights_placement,
        const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
            weights_offsets,
        const pta::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
            weights_tys,
        const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            D_offsets,
        const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits>
            update_weights,
        const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            update_table_idx,
        const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
            update_row_idx,
        const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
            update_offsets,
        const int64_t row_alignment,
        pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
            lxu_cache_weights,
        const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            lxu_cache_locations) {
  embedding_inplace_update_kernel_impl(
      dev_weights,
      uvm_weights,
      weights_placement,
      weights_offsets,
      weights_tys,
      D_offsets,
      update_weights,
      update_table_idx,
      update_row_idx,
      update_offsets,
      row_alignment,
      lxu_cache_weights,
      lxu_cache_locations);
}

void embedding_inplace_update_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor update_weights,
    Tensor update_table_idx,
    Tensor update_row_idx,
    Tensor update_offsets,
    const int64_t row_alignment,
    std::optional<Tensor> lxu_cache_weights,
    std::optional<Tensor> lxu_cache_locations) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      dev_weights,
      uvm_weights,
      weights_placements,
      weights_offsets,
      weights_tys,
      D_offsets,
      update_weights,
      update_offsets,
      update_table_idx,
      update_row_idx,
      lxu_cache_weights,
      lxu_cache_locations);

  CUDA_DEVICE_GUARD(dev_weights);

  const int64_t N = update_row_idx.numel();
  if (N == 0) {
    return;
  }
  TORCH_CHECK_EQ(N, update_table_idx.numel());

  const int32_t warpsPerBlock = kMaxThreads / kWarpSize;

  auto lxu_cache_weights_value = lxu_cache_weights.value_or(
      at::empty({0, 0}, dev_weights.options().dtype(at::kByte)));

  auto lxu_cache_locations_value = lxu_cache_locations.value_or(
      at::empty({0}, dev_weights.options().dtype(at::kInt)));

  AT_DISPATCH_INDEX_TYPES(
      update_row_idx.scalar_type(), "embedding_inplace_update_kernel_1", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (embedding_inplace_update_kernel_1<index_t>),
            nbit::div_round_up(N, warpsPerBlock), // number of blocks needed
            dim3(kWarpSize, warpsPerBlock), // shape of each block
            0,
            at::cuda::getCurrentCUDAStream(),

            PTA_B(dev_weights, uint8_t, 1, 64),
            PTA_B(uvm_weights, uint8_t, 1, 64),
            PTA_B(weights_placements, int32_t, 1, 32),
            PTA_B(weights_offsets, int64_t, 1, 32),
            PTA_B(weights_tys, uint8_t, 1, 32),
            PTA_B(D_offsets, int32_t, 1, 32),
            PTA_B(update_weights, uint8_t, 1, 64),
            PTA_B(update_table_idx, int32_t, 1, 32),
            PTA_B(update_row_idx, index_t, 1, 32),
            PTA_B(update_offsets, int64_t, 1, 32),
            row_alignment,
            PTA_B(lxu_cache_weights_value, uint8_t, 2, 64),
            PTA_B(lxu_cache_locations_value, int32_t, 1, 32));
      });
}

void embedding_inplace_update_single_placement_cuda(
    Tensor& dev_weights,
    Tensor& uvm_weights,
    const PlacementType& weights_placement,
    const Tensor& weights_offsets,
    const Tensor& weights_tys,
    const Tensor& D_offsets,
    const Tensor& update_weights,
    const Tensor& update_table_idx,
    const Tensor& update_row_idx,
    const Tensor& update_offsets,
    const int64_t row_alignment,
    std::optional<Tensor> lxu_cache_weights,
    std::optional<Tensor> lxu_cache_locations) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      dev_weights,
      uvm_weights,
      weights_offsets,
      weights_tys,
      D_offsets,
      update_weights,
      update_offsets,
      update_table_idx,
      update_row_idx,
      lxu_cache_weights,
      lxu_cache_locations);

  CUDA_DEVICE_GUARD(dev_weights);

  const int64_t N = update_row_idx.numel();
  if (N == 0) {
    return;
  }
  TORCH_CHECK_EQ(N, update_table_idx.numel());

  const int32_t warpsPerBlock = kMaxThreads / kWarpSize;

  auto lxu_cache_weights_value = lxu_cache_weights.value_or(
      at::empty({0, 0}, dev_weights.options().dtype(at::kByte)));

  auto lxu_cache_locations_value = lxu_cache_locations.value_or(
      at::empty({0}, dev_weights.options().dtype(at::kInt)));

  AT_DISPATCH_INDEX_TYPES(
      update_row_idx.scalar_type(), "embedding_inplace_update_kernel_2", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (embedding_inplace_update_kernel_2<index_t>),
            nbit::div_round_up(N, warpsPerBlock), // number of blocks needed
            dim3(kWarpSize, warpsPerBlock), // shape of each block
            0,
            at::cuda::getCurrentCUDAStream(),

            PTA_B(dev_weights, uint8_t, 1, 64),
            PTA_B(uvm_weights, uint8_t, 1, 64),
            weights_placement,
            PTA_B(weights_offsets, int64_t, 1, 32),
            PTA_B(weights_tys, uint8_t, 1, 32),
            PTA_B(D_offsets, int32_t, 1, 32),
            PTA_B(update_weights, uint8_t, 1, 64),
            PTA_B(update_table_idx, int32_t, 1, 32),
            PTA_B(update_row_idx, index_t, 1, 32),
            PTA_B(update_offsets, int64_t, 1, 32),
            row_alignment,
            PTA_B(lxu_cache_weights_value, uint8_t, 2, 64),
            PTA_B(lxu_cache_locations_value, int32_t, 1, 32));
      });
}

template <typename index_t, typename remap_t>
__global__
__launch_bounds__(kMaxThreads) void pruned_array_lookup_from_row_idx_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        update_row_indices,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        update_table_indices,
    const pta::PackedTensorAccessor32<remap_t, 1, at::RestrictPtrTraits>
        index_remappings,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        index_remappings_offsets,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        dense_indices) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= update_row_indices.size(0)) {
    return;
  }
  const auto row_idx = update_row_indices[idx];
  if (idx >= update_table_indices.size(0)) {
    return;
  }
  const int table_idx = update_table_indices[idx];

  const auto index_remappings_start = index_remappings_offsets[table_idx];
  const auto index_remappings_end = index_remappings_offsets[table_idx + 1];
  const auto capacity = index_remappings_end - index_remappings_start;

  if (capacity > 0) {
    dense_indices[idx] = static_cast<index_t>(
        index_remappings[index_remappings_start + row_idx]);
  } else {
    dense_indices[idx] = row_idx;
  }
}

Tensor pruned_array_lookup_from_row_idx_cuda(
    const Tensor& update_row_indices,
    const Tensor& update_table_indices,
    const Tensor& index_remappings,
    const Tensor& index_remappings_offsets) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      update_row_indices,
      update_table_indices,
      index_remappings,
      index_remappings_offsets);
  CUDA_DEVICE_GUARD(update_table_indices);

  auto dense_indices = at::empty_like(update_row_indices);
  const int32_t T = index_remappings_offsets.size(0) - 1;

  const auto num_indices = update_row_indices.numel();
  if (num_indices == 0) {
    return dense_indices;
  }

  TORCH_CHECK_LT(index_remappings.size(0), std::numeric_limits<int64_t>::max());
  TORCH_CHECK(
      update_row_indices.dim() == 1, "Tensor dim: ", update_row_indices.dim());
  TORCH_CHECK(
      update_table_indices.dim() == 1,
      "Tensor dim: ",
      update_table_indices.dim());
  TORCH_CHECK(
      index_remappings.dim() == 1, "Tensor dim: ", index_remappings.dim());
  TORCH_CHECK(
      index_remappings_offsets.dim() == 1,
      "Tensor dim: ",
      index_remappings_offsets.dim());
  TORCH_CHECK(dense_indices.dim() == 1, "Tensor dim: ", dense_indices.dim());
  constexpr size_t kForwardMaxThreads = 256;

  AT_DISPATCH_INDEX_TYPES(
      index_remappings.scalar_type(),
      "pruned_array_lookup_from_row_idx_cuda_0",
      [&] {
        using remap_t = index_t;

        AT_DISPATCH_INDEX_TYPES(
            update_row_indices.scalar_type(),
            "pruned_array_lookup_from_row_idx_cuda_1",
            [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (pruned_array_lookup_from_row_idx_kernel<index_t, remap_t>),
                  nbit::div_round_up(num_indices, kForwardMaxThreads),
                  kForwardMaxThreads,
                  0,
                  at::cuda::getCurrentCUDAStream(),

                  PTA_B(update_row_indices, index_t, 1, 32),
                  PTA_B(update_table_indices, int32_t, 1, 32),
                  PTA_B(index_remappings, remap_t, 1, 32),
                  PTA_B(index_remappings_offsets, int64_t, 1, 32),
                  PTA_B(dense_indices, index_t, 1, 32));
            });
      });
  return dense_indices;
}

} // namespace fbgemm_gpu
