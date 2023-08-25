/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fbgemm_gpu/embedding_common.h"

// These values are adjusted in backward based on B and T
constexpr int DEFAULT_INFO_NUM_BITS = 32;
constexpr int DEFAULT_INFO_B_NUM_BITS = 26;
constexpr uint32_t DEFAULT_INFO_B_MASK = (1u << DEFAULT_INFO_B_NUM_BITS) - 1;
constexpr uint32_t MAX_T =
    (1u << (DEFAULT_INFO_NUM_BITS - DEFAULT_INFO_B_NUM_BITS)) - 1;
constexpr uint32_t MAX_B = (1u << DEFAULT_INFO_B_NUM_BITS) - 1;

/**
 * "Transpose" embedding inputs by sorting indices by their values.
 * Logically this transpose compressed sparse row (CSR) representation
 * stored in indices and offsets to compressed sparse column (CSC).
 */
std::tuple<
    at::Tensor /*linear_indices*/,
    at::Tensor /*linear_indices_sorted*/,
    at::Tensor /*infos_sorted*/,
    at::Tensor /*sorted_linear_indices_run*/,
    at::Tensor /*sorted_linear_indices_run_lengths*/,
    at::Tensor /*sorted_linear_indices_num_runs*/,
    at::Tensor /*sorted_linear_indices_cumulative_run_lengths*/>
transpose_embedding_input(
    at::Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    at::Tensor indices,
    at::Tensor offsets,
    bool nobag = false,
    const c10::optional<at::Tensor>& vbe_b_t_map = c10::optional<at::Tensor>(),
    const int64_t info_B_num_bits = 26,
    const int64_t info_B_mask = 0x2FFFFFF,
    const int64_t total_unique_indices = -1,
    const bool is_index_select = false,
    const c10::optional<at::Tensor>& total_L_offsets =
        c10::optional<at::Tensor>(),
    const int64_t fixed_L_per_warp = 0,
    const int64_t num_warps_per_feature = 0);

std::tuple<int64_t, int64_t>
get_infos_metadata(at::Tensor unused, int64_t B, int64_t T);

std::tuple<int32_t, uint32_t> adjust_info_B_num_bits(int32_t B, int32_t T);

// Use these functions instead of directly calling cub functions
// to reduce code size and compilation time.
// Arguments are the same as cub::DeviceRadixSort::SortPairs
#define DECL_RADIX_SORT_PAIRS_FN(KeyT, ValueT) \
  cudaError_t radix_sort_pairs(                \
      void* d_temp_storage,                    \
      size_t& temp_storage_bytes,              \
      const KeyT* d_keys_in,                   \
      KeyT* d_keys_out,                        \
      const ValueT* d_values_in,               \
      ValueT* d_values_out,                    \
      int num_items,                           \
      int begin_bit = 0,                       \
      int end_bit = sizeof(KeyT) * 8,          \
      cudaStream_t stream = 0,                 \
      bool debug_synchronous = false)

DECL_RADIX_SORT_PAIRS_FN(int64_t, float);
DECL_RADIX_SORT_PAIRS_FN(int64_t, double);
DECL_RADIX_SORT_PAIRS_FN(int64_t, int64_t);
DECL_RADIX_SORT_PAIRS_FN(int64_t, int32_t);

#undef DECL_RADIX_SORT_PAIRS_FN

std::tuple<at::Tensor /*row_output_offsets*/, at::Tensor /*b_t_map*/>
generate_vbe_metadata(
    const at::Tensor& B_offsets,
    const at::Tensor& B_offsets_rank_per_feature,
    const at::Tensor& output_offsets_feature_rank,
    const at::Tensor& D_offsets,
    const int64_t D,
    const bool nobag,
    const int64_t max_B_feature_rank,
    const int64_t info_B_num_bits,
    const int64_t total_B);
