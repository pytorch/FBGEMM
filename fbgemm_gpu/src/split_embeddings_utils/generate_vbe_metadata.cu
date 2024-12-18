/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/split_embeddings_utils.cuh" // @manual
#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/fixed_divisor.cuh"
#include "fbgemm_gpu/utils/ops_utils.h" // @manual
#include "fbgemm_gpu/utils/tensor_accessor.h" // @manual
#include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

namespace {

__global__
__launch_bounds__(kMaxThreads) void generate_vbe_metadata_foreach_sample_kernel(
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        row_output_offsets,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        B_offsets,
    const pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits>
        B_offsets_rank_per_feature,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        output_offsets_feature_rank,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const int32_t D,
    const bool nobag,
    const int32_t info_B_num_bits) {
  // Relative sample ID in the rank-table matrix
  const auto b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Rank ID
  const auto r = blockIdx.y;
  // Table ID
  const auto t = blockIdx.z;
  // Num tables
  const auto T = B_offsets.size(0) - 1;

  const auto B_start_r_t = B_offsets_rank_per_feature[t][r];
  const auto B_r_t = B_offsets_rank_per_feature[t][r + 1] - B_start_r_t;
  if (b >= B_r_t) {
    return;
  }

  const auto* __restrict__ output_offsets_feature =
      &output_offsets_feature_rank[r * T];

  const auto B_start_t = B_offsets[t];
  const auto b_t =
      static_cast<int64_t>(B_start_t) + static_cast<int64_t>(B_start_r_t) + b;
  const auto D_ = nobag ? D : (D_offsets[t + 1] - D_offsets[t]);
  row_output_offsets[b_t] =
      output_offsets_feature[t] + b * static_cast<int64_t>(D_);

  // Relative sample ID in the table
  const auto b_ = B_start_r_t + b;
  // b_t is always positive.
  *reinterpret_cast<uint32_t*>(&b_t_map[b_t]) =
      (reinterpret_cast<const uint32_t*>(&t)[0] << info_B_num_bits) |
      reinterpret_cast<const uint32_t*>(&b_)[0];
}

} // namespace

std::tuple<int, int, int> get_max_grid_size(int device) {
  static auto max_grid = [&]() -> std::tuple<int, int, int> {
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, at::cuda::current_device()));
    return {prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]};
  }();

  return max_grid;
}

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
/// @param max_B_feature_rank          Maximum number of batches for feature
///                                    ranking
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

  const auto T = B_offsets.numel() - 1;

  if (!nobag) {
    TENSOR_ON_CUDA_GPU(D_offsets);
    TENSORS_ON_SAME_DEVICE(B_offsets, D_offsets);
    TORCH_CHECK(D_offsets.numel() == T + 1)
  }

  const auto num_ranks = B_offsets_rank_per_feature.size(1) - 1;
  TORCH_CHECK(
      num_ranks > 0, "generate_vbe_metadata: Invalid num_ranks ", num_ranks);
  TORCH_CHECK(T > 0, "generate_vbe_metadata: Invalid T ", T);
  TORCH_CHECK(
      max_B_feature_rank > 0,
      "generate_vbe_metadata: Invalid max_B_feature_rank ",
      max_B_feature_rank);

  TORCH_CHECK(B_offsets_rank_per_feature.size(0) == T);
  TORCH_CHECK(output_offsets_feature_rank.numel() == num_ranks * T + 1);

  CUDA_DEVICE_GUARD(B_offsets);

  Tensor row_output_offsets =
      at::empty({total_B}, output_offsets_feature_rank.options());
  Tensor b_t_map = at::empty({total_B}, B_offsets.options());

  const auto grid_dim_x = div_round_up(max_B_feature_rank, kMaxThreads);
  const dim3 grid_size(grid_dim_x, num_ranks, T);
  const auto& [max_grid_x, max_grid_y, max_grid_z] =
      get_max_grid_size(at::cuda::current_device());
  TORCH_CHECK(
      grid_size.x > 0 && grid_size.x <= max_grid_x,
      "generate_vbe_metadata: Invalid grid_size.x ",
      grid_size.x);
  TORCH_CHECK(
      grid_size.y > 0 && grid_size.y <= max_grid_y,
      "generate_vbe_metadata: Invalid grid_size.y ",
      grid_size.y);
  TORCH_CHECK(
      grid_size.z > 0 && grid_size.z <= max_grid_z,
      "generate_vbe_metadata: Invalid grid_size.z ",
      grid_size.z);

#ifdef FBGEMM_GPU_MEMCHECK
  const auto func_name = "generate_vbe_metadata_foreach_sample_kernel";
#endif
  // Over allocate total number of threads to avoid using binary search
  generate_vbe_metadata_foreach_sample_kernel<<<
      grid_size,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      MAKE_PTA_WITH_NAME(func_name, row_output_offsets, int64_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, b_t_map, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, B_offsets, int32_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, B_offsets_rank_per_feature, int32_t, 2, 32),
      MAKE_PTA_WITH_NAME(
          func_name, output_offsets_feature_rank, int64_t, 1, 32),
      MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
      D,
      nobag,
      info_B_num_bits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {row_output_offsets, b_t_map};
}
