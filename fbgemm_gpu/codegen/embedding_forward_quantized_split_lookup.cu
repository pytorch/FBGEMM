/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "codegen/embedding_forward_template_helpers.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"

using namespace fbgemm_gpu;
using Tensor = at::Tensor;

namespace nbit {

__global__
__launch_bounds__(kMaxThreads) void int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_kernel(
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        offsets,
    const pta::PackedTensorAccessor64<int32_t, 2, at::RestrictPtrTraits>
        hash_table,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        hash_table_offsets,
    const int32_t B,
    const int32_t T,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        dense_indices) {
  // uint32_t capacity = hash_table.size(0);
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t t = b_t / B;
  const int32_t b = b_t % B;
  if (b_t >= B * T) {
    return;
  }
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  const int32_t L = indices_end - indices_start;

  const int64_t table_start = hash_table_offsets[t];
  const int64_t table_end = hash_table_offsets[t + 1];
  const int64_t capacity = table_end - table_start;

  if (capacity == 0) {
    // No pruning applied on the indices associated with this table.
    for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
      dense_indices[indices_start + l] = indices[indices_start + l];
    }
    return;
  }

  const uint32_t subwarp_id = threadIdx.x / 4;
  const uint32_t subwarp_tid = threadIdx.x % 4;
#ifdef USE_ROCM
  const uint64_t subwarp_mask = static_cast<uint64_t>(0xF) << (4 * subwarp_id);
#else
  const uint32_t subwarp_mask = static_cast<uint32_t>(0xF) << (4 * subwarp_id);
#endif
  for (int32_t l_start = 0; l_start + subwarp_id < L;
       l_start += kWarpSize / 4) {
    const int32_t idx = indices[indices_start + l_start + subwarp_id];
    uint32_t slot_start =
        pruned_hash_function(static_cast<uint32_t>(idx)) % capacity;
    while (true) {
      const uint32_t slot = (slot_start + subwarp_tid) % capacity;
      const int2 val = *reinterpret_cast<const int2*>(
          &hash_table[table_start + static_cast<int64_t>(slot)][0]);
      const int32_t slot_sparse_idx = val.x;
      const int32_t slot_dense_idx = val.y;

      bool found = false;
      bool empty = false;
      if (slot_sparse_idx == -1) {
        empty = true;
      } else if (slot_sparse_idx == idx) {
        found = true;
        dense_indices[indices_start + l_start + subwarp_id] = slot_dense_idx;
      }
      if (__any_sync(subwarp_mask, found)) {
        break;
      } else if (__any_sync(subwarp_mask, empty)) {
        dense_indices[indices_start + l_start + subwarp_id] = -1;
        break;
      }
      slot_start += 4;
    }
  }
}

__global__
__launch_bounds__(kMaxThreads) void int_nbit_split_embedding_codegen_forward_pruned_array_lookup_kernel(
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        index_remappings,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        index_remappings_offsets,
    const int32_t B,
    const int32_t T,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        dense_indices) {
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t t = b_t / B;
  const int32_t b = b_t % B;
  if (b_t >= B * T) {
    return;
  }
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  const int32_t L = indices_end - indices_start;

  const int64_t index_remappings_start = index_remappings_offsets[t];
  const int64_t index_remappings_end = index_remappings_offsets[t + 1];
  const int64_t capacity = index_remappings_end - index_remappings_start;

  if (capacity > 0) {
    for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
      int32_t idx = indices[indices_start + l];
      dense_indices[indices_start + l] =
          index_remappings[index_remappings_start + idx];
    }
  } else {
    for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
      dense_indices[indices_start + l] = indices[indices_start + l];
    }
  }
}

} // namespace nbit

Tensor pruned_hashmap_lookup_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      indices, offsets, hash_table, hash_table_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());
  auto dense_indices = at::empty_like(indices);
  const int32_t T = hash_table_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B > 0);
  TORCH_CHECK(hash_table.size(0) < std::numeric_limits<int32_t>::max());
  constexpr size_t kForwardMaxThreads = 256;

#ifdef FBGEMM_GPU_MEMCHECK
  const auto func_name =
      "int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_kernel";
#endif

  nbit::int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_kernel<<<
      nbit::div_round_up(B * T + 1, kForwardMaxThreads / kWarpSize),
      dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      MAKE_PTA_WITH_NAME(func_name, indices, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, offsets, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, hash_table, int32_t, 2, 64),
      MAKE_PTA_WITH_NAME(func_name, hash_table_offsets, int64_t, 1, 32),
      B,
      T,
      MAKE_PTA_WITH_NAME(func_name, dense_indices, int32_t, 1, 32));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dense_indices;
}

Tensor pruned_array_lookup_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor index_remappings,
    Tensor index_remappings_offsets) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      indices, offsets, index_remappings, index_remappings_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());
  auto dense_indices = at::empty_like(indices);
  const int32_t T = index_remappings_offsets.size(0) - 1;
  TORCH_CHECK(
      (offsets.size(0) - 1) % T == 0,
      "offsets.size() - 1 is not divisible by T! offsets.size: ",
      offsets.size(0),
      "T: ",
      T);
  const int32_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(
      B > 0, "offsets.size(): ", offsets.size(0), ", T: ", T, ", B: ", B);
  TORCH_CHECK(index_remappings.size(0) < std::numeric_limits<int64_t>::max());
  TORCH_CHECK(indices.dim() == 1, "Tensor dim: ", indices.dim());
  TORCH_CHECK(offsets.dim() == 1, "Tensor dim: ", offsets.dim());
  TORCH_CHECK(
      index_remappings.dim() == 1, "Tensor dim: ", index_remappings.dim());
  TORCH_CHECK(
      index_remappings_offsets.dim() == 1,
      "Tensor dim: ",
      index_remappings_offsets.dim());
  TORCH_CHECK(dense_indices.dim() == 1, "Tensor dim: ", dense_indices.dim());
  constexpr size_t kForwardMaxThreads = 256;

#ifdef FBGEMM_GPU_MEMCHECK
  const auto func_name =
      "int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_kernel";
#endif

  nbit::int_nbit_split_embedding_codegen_forward_pruned_array_lookup_kernel<<<
      nbit::div_round_up(offsets.size(0), kForwardMaxThreads / kWarpSize),
      dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      MAKE_PTA_WITH_NAME(func_name, indices, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, offsets, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, index_remappings, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, index_remappings_offsets, int64_t, 1, 32),
      B,
      T,
      MAKE_PTA_WITH_NAME(func_name, dense_indices, int32_t, 1, 32));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dense_indices;
}
