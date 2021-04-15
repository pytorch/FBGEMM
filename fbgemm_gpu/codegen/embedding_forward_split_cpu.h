/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

enum PoolingMode { SUM = 0, MEAN = 1, NONE = 2 };

at::Tensor split_embedding_codegen_forward_cpu(
    at::Tensor weights,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    int64_t total_D,
    at::Tensor hash_size_cumsum,
    at::Tensor indices,
    at::Tensor offsets,
    int64_t pooling_mode,
    at::Tensor indice_weights);

at::Tensor split_embedding_codegen_grad_indice_weights_cpu(
    at::Tensor grad_output,
    at::Tensor weights,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor feature_requires_grad);

namespace internal {
// A batch of compressed sparse row but each sparse matrix is hyper sparse
// meaning there can be many columns without any non-zeros.
struct BatchedHyperCompressedSparseColumn {
  int num_tables; // # of matrices (or tables)
  // pointers to the beginning of each table in column_ptr (length T + 1)
  std::vector<int> table_ptr;
  // pointers to the beginning of each column segment in row_indices
  // (length table_ptr[T] + 1)
  // For a shared table, a column can have multiple segments, each for a
  // feature sharing the table. In this case, the segments will have the
  // same column_segment_indices but different column_segment_ids.
  std::vector<int> column_segment_ptr;
  std::vector<int64_t> column_segment_indices; // length table_ptr[T]
  std::vector<int64_t> column_segment_ids; // length table_ptr[T]
  std::vector<int> row_indices; // length column_ptr[table_ptr[T]]
  std::vector<float> weights; // length column_ptr[table_ptr[T]]
};

template <typename scalar_t>
void batched_csr2csc(
    BatchedHyperCompressedSparseColumn& batched_csc,
    int num_tables, // number of tables, not number of features
    int B,
    const at::TensorAccessor<int64_t, 1>& batched_csr_offsets,
    const at::TensorAccessor<int64_t, 1>& batched_csr_indices,
    const at::TensorAccessor<scalar_t, 1>& batched_csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset);
} // namespace internal
