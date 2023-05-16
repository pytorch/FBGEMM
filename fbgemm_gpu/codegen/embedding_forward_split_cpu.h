/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include "fbgemm/Utils.h"

at::Tensor split_embedding_codegen_forward_cpu(
    at::Tensor weights,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    int64_t total_D,
    at::Tensor hash_size_cumsum,
    at::Tensor indices,
    at::Tensor offsets,
    int64_t pooling_mode,
    at::Tensor indice_weights,
    int64_t output_dtype = 0 /* SparseType.FP32 */);

at::Tensor split_embedding_codegen_grad_indice_weights_cpu(
    at::Tensor grad_output,
    at::Tensor weights,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor feature_requires_grad);

namespace internal {
// A compressed sparse column but each sparse matrix is hyper sparse
// meaning there can be many columns without any non-zeros.
struct HyperCompressedSparseColumn {
  int num_non_zero_columns;
  // pointers to the beginning of each column segment in row_indices
  // (length num_non_zero_columns + 1)
  // For a shared table, a column can have multiple segments, each for a
  // feature sharing the table. In this case, the segments will have the
  // same column_segment_indices but different column_segment_ids.
  int* column_segment_ptr = nullptr;
  int* column_segment_indices = nullptr; // length num_non_zero_columns
  int* column_segment_ids = nullptr; // length num_non_zero_columns
  int* row_indices = nullptr; // length column_ptr[num_non_zero_columns]
  float* weights = nullptr; // length column_ptr[num_non_zero_columns]
  ~HyperCompressedSparseColumn() {
    if (column_segment_ptr) {
      fbgemm::fbgemmAlignedFree(column_segment_ptr);
      fbgemm::fbgemmAlignedFree(column_segment_indices);
      fbgemm::fbgemmAlignedFree(column_segment_ids);
      fbgemm::fbgemmAlignedFree(row_indices);
    }
    if (weights) {
      fbgemm::fbgemmAlignedFree(weights);
    }
  }
};

template <typename scalar_t>
void csr2csc(
    HyperCompressedSparseColumn& csc,
    int B,
    const at::TensorAccessor<int64_t, 1>& csr_offsets,
    const at::TensorAccessor<int64_t, 1>& csr_indices,
    const at::TensorAccessor<scalar_t, 1>& csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset,
    int64_t num_embeddings);
} // namespace internal
