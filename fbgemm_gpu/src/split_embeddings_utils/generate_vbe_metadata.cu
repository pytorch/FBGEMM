/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/ops_utils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

namespace {

__global__
__launch_bounds__(kMaxThreads) void generate_vbe_metadata_foreach_sample_kernel(
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        row_output_offsets,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        B_offsets,
    const at::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits>
        B_offsets_rank_per_feature,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        output_offsets_feature_rank,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const int32_t D,
    const bool nobag,
    FixedDivisor fd_max_B,
    FixedDivisor fd_max_B_T,
    const int32_t info_B_num_bits) {
  const auto r_b_t = blockIdx.x * blockDim.x + threadIdx.x;
  const auto T = B_offsets.size(0) - 1; // Num tables
  const auto R = B_offsets_rank_per_feature.size(1) - 1; // Num ranks

  int32_t b_t;
  int32_t r; // Rank ID
  int32_t t; // Table ID
  int32_t b; // Relative sample ID in the rank-table matrix

  fd_max_B_T.DivMod(r_b_t, &r, &b_t);
  if (r >= R) {
    return;
  }

  fd_max_B.DivMod(b_t, &t, &b);
  if (t >= T) {
    return;
  }

  const auto B_start_r_t = B_offsets_rank_per_feature[t][r];
  const auto B_r_t = B_offsets_rank_per_feature[t][r + 1] - B_start_r_t;
  if (b >= B_r_t) {
    return;
  }

  const auto B_start_t = B_offsets[t];
  // Update b_t
  b_t = B_start_t + B_start_r_t + b;
  const auto D_ = nobag ? D : D_offsets[t + 1] - D_offsets[t];
  row_output_offsets[b_t] = output_offsets_feature_rank[r * T + t] + b * D_;

  // Relative sample ID in the table
  const auto b_ = B_start_r_t + b;
  // b_t is always positive.
  *reinterpret_cast<uint32_t*>(&b_t_map[b_t]) =
      (reinterpret_cast<uint32_t*>(&t)[0] << info_B_num_bits) |
      reinterpret_cast<const uint32_t*>(&b_)[0];
}

} // namespace

/// Generate VBE metadata namely output_offsets and b_t_map
///
/// row_output_offsets A 1D tensor that contains the output offset of each b
///                    (sample) and t (feature/table) pair. The output
///                    serializes O_r_t where O_r_t is the local output of rank
///                    r and feature/table t (t is the fastest moving index).
/// b_t_map            A 1D tensor that contains the b and t information of the
///                    linearized b and t (b is the fastest moving index).
///
/// @param B_offsets                   Batch size offsets for all features.
/// @param B_offsets_rank_per_feature  Batch size offsets for all ranks (GPUs)
///                                    for each feature.
/// @param output_offsets_feature_rank Output offsets for all features and ranks
///                                    and features.
/// @param D_offsets                   Embedding dimension offsets. Required if
///                                    nobag is false.
/// @param D                           The embedding dimension. Required if
///                                    nobag is true.
/// @param nobag                       A boolean to indicate if TBE is pooled
///                                    (false) or sequence (true).
/// @param info_B_num_bits             The number of bits used to encode a
///                                    sample ID. (Used for populating b_t_map).
/// @param total_B                     The total number of samples (i.e., the
///                                    total number of b and t pairs).
DLL_PUBLIC std::tuple<Tensor /*row_output_offsets*/, Tensor /*b_t_map*/>
generate_vbe_metadata(
    const Tensor& B_offsets,
    const Tensor& B_offsets_rank_per_feature,
    const Tensor& output_offsets_feature_rank,
    const Tensor& D_offsets,
    const int64_t D,
    const bool nobag,
    const int64_t max_B_feature_rank,
    const int64_t info_B_num_bits,
    const int64_t total_B) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      B_offsets, B_offsets_rank_per_feature, output_offsets_feature_rank);

  TENSOR_NDIM_EQUALS(B_offsets, 1);
  TENSOR_NDIM_EQUALS(B_offsets_rank_per_feature, 2);
  TENSOR_NDIM_EQUALS(output_offsets_feature_rank, 1);

  const int32_t T = B_offsets.numel() - 1;
  if (!nobag) {
    TENSOR_ON_CUDA_GPU(D_offsets);
    TENSORS_ON_SAME_DEVICE(B_offsets, D_offsets);
    TORCH_CHECK(D_offsets.numel() == T + 1)
  }

  const auto num_ranks = B_offsets_rank_per_feature.size(1) - 1;
  TORCH_CHECK(B_offsets_rank_per_feature.size(0) == T);
  TORCH_CHECK(output_offsets_feature_rank.numel() == num_ranks * T + 1);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(B_offsets.get_device());

  Tensor row_output_offsets =
      at::empty({total_B}, output_offsets_feature_rank.options());
  Tensor b_t_map = at::empty({total_B}, B_offsets.options());

  // Over allocate total number of threads to avoid using binary search
  generate_vbe_metadata_foreach_sample_kernel<<<
      div_round_up(max_B_feature_rank * T * num_ranks, kMaxThreads),
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      row_output_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      b_t_map.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      B_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      B_offsets_rank_per_feature
          .packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
      output_offsets_feature_rank
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      D,
      nobag,
      FixedDivisor(max_B_feature_rank),
      FixedDivisor(max_B_feature_rank * T),
      info_B_num_bits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {row_output_offsets, b_t_map};
}
