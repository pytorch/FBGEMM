/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    const int64_t info_B_mask = 0x2FFFFFF);

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
