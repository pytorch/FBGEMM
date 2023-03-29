/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>

#include "fbgemm_gpu/embedding_inplace_update.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

constexpr int32_t kCacheLocationMissing = -1;

template <typename index_t>
__launch_bounds__(kMaxThreads) __global__ void embedding_inplace_update_kernel(
    at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> dev_weights,
    at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
        weights_tys,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits>
        update_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        update_table_idx,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        update_row_idx,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        update_offsets,
    const int64_t row_alignment,
    at::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
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

  const int32_t D_start = D_offsets[table_idx];
  const int32_t D_end = D_offsets[table_idx + 1];
  const int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[table_idx]);
  const int32_t D_bytes =
      nbit::padded_row_size_in_bytes(D, weight_ty, row_alignment);

  const int64_t weight_offset = weights_offsets[table_idx];
  uint8_t* __restrict__ weight_row;
  const auto placement =
      static_cast<PlacementType>(weights_placements[table_idx]);
  if (placement == PlacementType::DEVICE) {
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
  for (int32_t d = threadIdx.x; d * sizeof(uint4) < D_bytes; d += blockDim.x) {
    vec_weight_row[d] = update_weight_row[d];
  }

  bool cache_valid = (placement == PlacementType::MANAGED_CACHING);
  int32_t cache_idx =
      cache_valid ? lxu_cache_locations[i] : kCacheLocationMissing;
  if (cache_valid && cache_idx != kCacheLocationMissing) {
    auto vec_cache_row = reinterpret_cast<uint4*>(
        &lxu_cache_weights[static_cast<int64_t>(cache_idx)][0]);

    for (int32_t d = threadIdx.x; d * sizeof(uint4) < D_bytes;
         d += blockDim.x) {
      vec_cache_row[d] = update_weight_row[d];
    }
  }
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
    c10::optional<Tensor> lxu_cache_weights,
    c10::optional<Tensor> lxu_cache_locations) {
  TENSOR_ON_CUDA_GPU(dev_weights);
  TENSOR_ON_CUDA_GPU(uvm_weights);
  TENSOR_ON_CUDA_GPU(weights_placements);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(weights_tys);
  TENSOR_ON_CUDA_GPU(D_offsets);

  TENSOR_ON_CUDA_GPU(update_weights);
  TENSOR_ON_CUDA_GPU(update_offsets);
  TENSOR_ON_CUDA_GPU(update_table_idx);
  TENSOR_ON_CUDA_GPU(update_row_idx);

  if (lxu_cache_weights.has_value()) {
    TENSOR_ON_CUDA_GPU(lxu_cache_weights);
  }
  if (lxu_cache_locations.has_value()) {
    TENSOR_ON_CUDA_GPU(lxu_cache_locations);
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(dev_weights.get_device());

  const int64_t N = update_row_idx.numel();
  if (N == 0) {
    return;
  }
  TORCH_CHECK(N == update_table_idx.numel());

  const int32_t warpsPerBlock = kMaxThreads / kWarpSize;

  auto lxu_cache_weights_value = lxu_cache_weights.value_or(
      at::empty({0, 0}, dev_weights.options().dtype(at::kByte)));

  auto lxu_cache_locations_value = lxu_cache_locations.value_or(
      at::empty({0}, dev_weights.options().dtype(at::kInt)));

  AT_DISPATCH_INDEX_TYPES(
      update_row_idx.scalar_type(), "embedding_inplace_update_kernel", [&] {
        embedding_inplace_update_kernel<<<
            nbit::div_round_up(N, warpsPerBlock), // number of blocks needed
            dim3(kWarpSize, warpsPerBlock), // shape of each block
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(),
            uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(),
            weights_placements
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            weights_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            update_weights
                .packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(),
            update_table_idx
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            update_row_idx
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            update_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            row_alignment,
            lxu_cache_weights_value
                .packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(),
            lxu_cache_locations_value
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void pruned_array_lookup_from_row_idx_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        update_row_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        update_table_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        index_remappings,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        index_remappings_offsets,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
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

  const int64_t index_remappings_start = index_remappings_offsets[table_idx];
  const int64_t index_remappings_end = index_remappings_offsets[table_idx + 1];
  const int64_t capacity = index_remappings_end - index_remappings_start;

  if (capacity > 0) {
    dense_indices[idx] = index_remappings[index_remappings_start + row_idx];
  } else {
    dense_indices[idx] = row_idx;
  }
}

Tensor pruned_array_lookup_from_row_idx_cuda(
    const Tensor& update_row_indices,
    const Tensor& update_table_indices,
    const Tensor& index_remappings,
    const Tensor& index_remappings_offsets) {
  TENSOR_ON_CUDA_GPU(update_row_indices);
  TENSOR_ON_CUDA_GPU(update_table_indices);
  TENSOR_ON_CUDA_GPU(index_remappings);
  TENSOR_ON_CUDA_GPU(index_remappings_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(update_table_indices.get_device());
  auto dense_indices = at::empty_like(update_row_indices);
  const int32_t T = index_remappings_offsets.size(0) - 1;

  const auto num_indices = update_row_indices.numel();
  if (num_indices == 0) {
    return dense_indices;
  }

  TORCH_CHECK(index_remappings.size(0) < std::numeric_limits<int64_t>::max());
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
      update_row_indices.scalar_type(),
      "pruned_array_lookup_from_row_idx_kernel",
      [&] {
        pruned_array_lookup_from_row_idx_kernel<<<
            nbit::div_round_up(num_indices, kForwardMaxThreads),
            kForwardMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            update_row_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            update_table_indices
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            index_remappings
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            index_remappings_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            dense_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return dense_indices;
}

} // namespace fbgemm_gpu
