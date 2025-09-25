/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cute/tensor.hpp>

namespace fbgemm_gpu {

enum GroupedGemmInputType {
  // K dynamic
  _2D2D,
  // M dynamic (MoE forward style)
  _2D3D
};

template <
    typename ProblemShape,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ScaleDtype,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename LayoutSFA,
    typename LayoutSFB,
    typename Sm1xxBlkScaledConfig>
__global__ void set_grouped_gemm_args_kernel(
    int64_t G,
    int64_t M,
    int64_t N,
    int64_t K,
    ProblemShape* problem_shape_ptr,
    ElementA* xq,
    const ElementA** xq_ptr,
    ElementB* wq,
    const ElementB** wq_ptr,
    ScaleDtype* x_scale,
    const ScaleDtype** x_scale_ptr,
    ScaleDtype* w_scale,
    const ScaleDtype** w_scale_ptr,
    ElementC* output,
    ElementC** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr,
    int32_t* offsets, // Group end offsets
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB,
    GroupedGemmInputType gemm_type) {
  const uint32_t group_index = blockIdx.x * blockDim.x + threadIdx.x;

  // If this thread corresponds to a valid group, write kernel args to device
  // memory.
  if (group_index < G) {
    // Set problem shapes to empty by default.
    problem_shape_ptr[group_index] = ProblemShape(0, 0, 0);

    // Offsets for this group.
    int64_t xq_offset = 0;
    int64_t wq_offset = 0;
    int64_t output_offset = 0;
    int64_t x_scale_offset = 0;
    int64_t w_scale_offset = 0;

    auto round_up = [](int64_t x, int64_t y) { return ((x + y - 1) / y) * y; };

    // Pre-compute common rounded values to minimize round_up calls
    const int64_t N_rounded = round_up(N, 128);
    const int64_t M_rounded = round_up(M, 128);

    const int64_t scale_factor_block_size = 32;

    // Handle offsets API (torch compliant API for 2D-2D and 2D-3D inputs)
    CUDA_KERNEL_ASSERT(
        offsets != nullptr &&
        "offsets must be set for 2d-2d and 2d-3d grouped GEMMs");
    switch (gemm_type) {
      // In the 2d-2d case, contraction dim (total_K) has variable group
      // sizes. XQ = (M, total_K) WQ = (N, total_K) Main loop defined with WQ
      // @ XQ^T = (N, M) for each group. out = (G, N, M)
      case GroupedGemmInputType::_2D2D: {
        // `offsets` contains end index of each group.
        const int32_t prev_group_end_offset =
            (group_index == 0) ? 0 : offsets[group_index - 1];
        const int32_t curr_group_end_offset = offsets[group_index];
        const int32_t K_group_size =
            curr_group_end_offset - prev_group_end_offset;

        // Validate group offsets.
        const int align = 128 / cutlass::sizeof_bits<ElementA>::value;
        CUDA_KERNEL_ASSERT(
            K_group_size % align == 0 &&
            "for 2d-2d grouped gemm, group sizes along K dim must be non-negative multiple of 16\n");
        CUDA_KERNEL_ASSERT(
            curr_group_end_offset <= K &&
            "for 2d-2d grouped gemm, group end offsets must be non-negative and must be <= K\n");

        // Set starting input offsets for this group.
        // XQ is shape (M,K) with strides (K, 1) and group offsets are along
        // the K dim, so: xq_offset -> prev_group_end_offset * 1
        xq_offset = prev_group_end_offset;

        // WQ is shape (N,K) with strides (K, 1) and group offsets are along
        // the K dim, so: wq_offset -> prev_group_end_offset * 1
        wq_offset = prev_group_end_offset;

        // Output for 2d-2d grouped GEMM is shape (G, M, N)
        // output_offset -> group_index rows with stride of M * N
        output_offset = group_index * M * N;

        // Group sizes are variable and converted to blocked/padded format, so
        // to calculate the starting offset of this group's scales, we do the
        // following: For each previous group
        // - Calculate the expected size of its blocked formatted scales
        // - Increment the scale offsets by that size
        // x_scale shape (M_rounded, total_K_padded_per_group).
        // w_scale has shape (N_rounded, total_K_padded_per_group).
        for (int i = 0; i < group_index; i++) {
          int group_i_size = i == 0 ? offsets[i] : offsets[i] - offsets[i - 1];
          int scale_cols_for_group_i_padded =
              round_up(group_i_size / scale_factor_block_size, 4);
          x_scale_offset += M_rounded * scale_cols_for_group_i_padded;
          w_scale_offset += N_rounded * scale_cols_for_group_i_padded;
        }

        // Only write kernel args if this group is non-empty
        if (K_group_size > 0) {
          // Get index automatically for this group
          int total_K = K; // Name alias for clarity/readability.

          // Set problem shape.
          // Main loop passes inputs in B,A order, so we have: (N, K_group) @
          // (M, K_group)^T = (N, M) for each group.
          problem_shape_ptr[group_index] = ProblemShape(N, M, K_group_size);

          // Set pointers for this group.
          xq_ptr[group_index] = xq + xq_offset;
          wq_ptr[group_index] = wq + wq_offset;
          x_scale_ptr[group_index] = x_scale + x_scale_offset;
          w_scale_ptr[group_index] = w_scale + w_scale_offset;
          output_ptr[group_index] = output + output_offset;

          // Set strides.
          // TODO: make strides configurable to handle all NT/TN/NN/NT layouts
          // that Blackwell supports. For XQ, the group processes a slice (M,
          // K_group_size) but it's part of a larger tensor (M, total_K). The
          // stride needs to reflect that rows are separated by total_K
          // elements in the original tensor.
          stride_a_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideA{}, cute::make_shape(int(M), int(total_K), 1));

          // For WQ, the group processes a slice (N, K_group_size) but it's
          // part of a larger tensor (N, total_K). The stride needs to reflect
          // that rows are separated by total_K elements in the original
          // tensor.
          stride_b_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideB{}, cute::make_shape(int(N), int(total_K), 1));

          // For output of this group, (M, K_group_size) @ (N, K_group_size)^T
          // = (M, N)
          stride_c_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideC{}, cute::make_shape(int(N), int(M), 1));

          // Set layouts for scale factors.
          // Groups of variable size are along the K dim, so we need to
          // calculate the size of the blocked group scale factor here.
          layout_SFA[group_index] =
              Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
                  cute::make_shape(int(M), int(N), int(K_group_size), 1));
          layout_SFB[group_index] =
              Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
                  cute::make_shape(int(M), int(N), int(K_group_size), 1));
        }
        break;
      }
      case GroupedGemmInputType::_2D3D: {
        // `offsets` contains end index of each group.
        const int32_t prev_group_end_offset =
            (group_index == 0) ? 0 : offsets[group_index - 1];
        const int32_t curr_group_end_offset = offsets[group_index];
        const int32_t M_group_size =
            curr_group_end_offset - prev_group_end_offset;

        if (M_group_size > 0) {
          // Validate group offsets.
          CUDA_KERNEL_ASSERT(
              curr_group_end_offset <= M &&
              "for 2d-3d grouped gemm, group end offsets must be non-negative and must be <= M\n");

          // Compute starting offset for this group when M_group size > 0
          int64_t group_offset_M =
              group_index == 0 ? 0 : offsets[group_index - 1];
          int64_t scale_group_offset_M = 0;
          for (int i = 0; i < group_index; i++) {
            // Group offset on XQ along total_M dim is the sum of all previous
            // group sizes.
            int group_i_size =
                i == 0 ? offsets[i] : offsets[i] - offsets[i - 1];

            // Scale group offset on x_scale is sum of all previous scale
            // group sizes.
            int scale_group_rows_padded = round_up(group_i_size, 128);
            scale_group_offset_M += scale_group_rows_padded;
          }

          // wq_offset -> group_offset_M rows with stride of K
          xq_offset = group_offset_M * K;

          // wq_offset -> group_index rows with stride of N * K (3d tensor)
          wq_offset = group_index * N * K;

          // output_offset -> group_offset_M rows with stride of N
          output_offset = group_offset_M * N;

          // x_scale offset -> sum of all padded group sizes (rows) * rounded
          // scale group cols
          const int64_t K_rounded = round_up(K / scale_factor_block_size, 4);
          x_scale_offset = scale_group_offset_M * K_rounded;

          // w_scale_offset -> group_index rows with stride of (N rounded to
          // nearest multiple of 128 * K rounded to nearest multiple of 4)
          w_scale_offset = group_index * N_rounded * K_rounded;

          // Set problem shape
          problem_shape_ptr[group_index] = ProblemShape(N, M_group_size, K);

          // Set pointers
          xq_ptr[group_index] = xq + xq_offset;
          wq_ptr[group_index] = wq + wq_offset;
          x_scale_ptr[group_index] = x_scale + x_scale_offset;
          w_scale_ptr[group_index] = w_scale + w_scale_offset;
          output_ptr[group_index] = output + output_offset;

          // Set strides
          stride_a_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideA{}, cute::make_shape(int(M_group_size), int(K), 1));
          stride_b_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideB{}, cute::make_shape(int(N), int(K), 1));
          stride_c_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideC{}, cute::make_shape(int(N), int(M_group_size), 1));

          // Set layouts for scale factors
          layout_SFA[group_index] =
              Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
                  cute::make_shape(int(M_group_size), int(N), int(K), 1));
          layout_SFB[group_index] =
              Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
                  cute::make_shape(int(M_group_size), int(N), int(K), 1));
        }
        break;
      }
    }
  }
}

} // namespace fbgemm_gpu
