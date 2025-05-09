/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/embedding_forward_template_helpers.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

using namespace fbgemm_gpu;
using Tensor = at::Tensor;

namespace nbit {

template <typename index_t, typename hash_t>
__global__
__launch_bounds__(kMaxThreads) void int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    const pta::PackedTensorAccessor64<hash_t, 2, at::RestrictPtrTraits>
        hash_table,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        hash_table_offsets,
    const int32_t B,
    const int32_t T,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        dense_indices) {
  // uint32_t capacity = hash_table.size(0);
  const auto b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t t = b_t / B;
  const int32_t b = b_t % B;
  if (b_t >= B * T) {
    return;
  }
  const auto indices_start = offsets[t * B + b];
  const auto indices_end = offsets[t * B + b + 1];
  const auto L = indices_end - indices_start;

  const auto table_start = hash_table_offsets[t];
  const auto table_end = hash_table_offsets[t + 1];
  const auto capacity = table_end - table_start;

  if (capacity == 0) {
    // No pruning applied on the indices associated with this table.
    for (auto l = threadIdx.x; l < L; l += blockDim.x) {
      dense_indices[indices_start + l] = indices[indices_start + l];
    }
    return;
  }

  using uidx_t =
      std::conditional_t<std::is_same_v<index_t, int64_t>, uint64_t, uint32_t>;

  // Use nv type of size (hash_t x 2)
  using nv_hash_t =
      std::conditional_t<std::is_same_v<hash_t, int64_t>, longlong2, int2>;

  const uint32_t subwarp_id = threadIdx.x / 4;
  const uint32_t subwarp_tid = threadIdx.x % 4;
#ifdef USE_ROCM
  const uint64_t subwarp_mask = static_cast<uint64_t>(0xF) << (4 * subwarp_id);
#else
  const uint32_t subwarp_mask = static_cast<uint32_t>(0xF) << (4 * subwarp_id);
#endif

  for (int32_t l_start = 0; l_start + subwarp_id < L;
       l_start += kWarpSize / 4) {
    const index_t idx = indices[indices_start + l_start + subwarp_id];
    auto slot_start = pruned_hash_function(static_cast<uidx_t>(idx)) % capacity;

    while (true) {
      const auto slot = (slot_start + subwarp_tid) % capacity;

      const nv_hash_t val = *reinterpret_cast<const nv_hash_t*>(
          &hash_table[table_start + static_cast<int64_t>(slot)][0]);
      const auto slot_sparse_idx = val.x;
      const auto slot_dense_idx = val.y;

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

template <typename index_t, typename remap_t>
__global__
__launch_bounds__(kMaxThreads) void int_nbit_split_embedding_codegen_forward_pruned_array_lookup_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    const pta::PackedTensorAccessor64<remap_t, 1, at::RestrictPtrTraits>
        index_remappings,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        index_remappings_offsets,
    const int32_t B,
    const int32_t T,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        dense_indices) {
  const auto b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t t = b_t / B;
  const int32_t b = b_t % B;
  if (b_t >= B * T) {
    return;
  }
  const auto indices_start = offsets[t * B + b];
  const auto indices_end = offsets[t * B + b + 1];
  const auto L = indices_end - indices_start;

  const auto index_remappings_start = index_remappings_offsets[t];
  const auto index_remappings_end = index_remappings_offsets[t + 1];
  const auto capacity = index_remappings_end - index_remappings_start;

  if (capacity > 0) {
    for (index_t l = threadIdx.x; l < L; l += blockDim.x) {
      const overflow_safe_int_t idx = indices[indices_start + l];
      dense_indices[indices_start + l] =
          static_cast<index_t>(index_remappings[index_remappings_start + idx]);
    }
  } else {
    for (index_t l = threadIdx.x; l < L; l += blockDim.x) {
      dense_indices[indices_start + l] = indices[indices_start + l];
    }
  }
}

} // namespace nbit

using namespace nbit;

Tensor pruned_hashmap_lookup_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      indices, offsets, hash_table, hash_table_offsets);
  TENSORS_HAVE_SAME_SCALAR_TYPE(indices, offsets);

  CUDA_DEVICE_GUARD(indices);

  auto dense_indices = at::empty_like(indices);
  const int32_t T = hash_table_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B > 0);
  TORCH_CHECK(hash_table.size(0) < std::numeric_limits<int32_t>::max());
  constexpr size_t kForwardMaxThreads = 256;

  AT_DISPATCH_INDEX_TYPES(
      hash_table.scalar_type(), "pruned_hashmap_lookup_cuda_0", [&] {
        using hash_t = index_t;

        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "pruned_hashmap_lookup_cuda_1", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_kernel<
                      index_t,
                      hash_t>),
                  nbit::div_round_up(B * T + 1, kForwardMaxThreads / kWarpSize),
                  dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(indices, index_t, 1, 32),
                  PTA_B(offsets, index_t, 1, 32),
                  PTA_B(hash_table, hash_t, 2, 64),
                  PTA_B(hash_table_offsets, int64_t, 1, 32),
                  B,
                  T,
                  PTA_B(dense_indices, index_t, 1, 32));
            });
      });

  return dense_indices;
}

Tensor pruned_array_lookup_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor index_remappings,
    Tensor index_remappings_offsets) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      indices, offsets, index_remappings, index_remappings_offsets);
  TENSORS_HAVE_SAME_SCALAR_TYPE(indices, offsets);

  CUDA_DEVICE_GUARD(indices);

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

  AT_DISPATCH_INDEX_TYPES(
      index_remappings.scalar_type(), "pruned_array_lookup_cuda_0", [&] {
        using remap_t = index_t;

        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "pruned_array_lookup_cuda_1", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (int_nbit_split_embedding_codegen_forward_pruned_array_lookup_kernel<
                      index_t,
                      remap_t>),
                  nbit::div_round_up(
                      offsets.size(0), kForwardMaxThreads / kWarpSize),
                  dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(indices, index_t, 1, 32),
                  PTA_B(offsets, index_t, 1, 32),
                  PTA_B(index_remappings, remap_t, 1, 64),
                  PTA_B(index_remappings_offsets, int64_t, 1, 32),
                  B,
                  T,
                  PTA_B(dense_indices, index_t, 1, 32));
            });
      });

  return dense_indices;
}
