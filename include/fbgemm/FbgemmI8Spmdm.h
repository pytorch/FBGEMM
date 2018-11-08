/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>
#include <vector>
#include "Utils.h"

// #define FBGEMM_MEASURE_TIME_BREAKDOWN

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
#include <chrono>
#include <iostream>
extern double spmdm_initial_time;
extern double spmdm_transpose_uint8_time;
extern double spmdm_transpose_32xN_time;
extern double spmdm_compute_time;
extern double spmdm_transpose_Nx32_time;
extern double spmdm_run_time;
#endif

namespace fbgemm {

/**
 * @brief A class to represent a matrix in Compressed Sparse Column (CSC)
 * format.
 *
 * The second input matrix of matrix multiplication is usually weight and can
 * be sparse, and it's usually more efficient to use CSC format to represent
 * the second input matrix.
 */
class CompressedSparseColumn {
 public:
  CompressedSparseColumn(int num_of_rows, int num_of_cols);

  std::vector<std::int32_t>& ColPtr() {
    return colptr_;
  }
  std::vector<std::int16_t>& RowIdx() {
    return rowidx_;
  }
  std::vector<std::int8_t>& Values() {
    return values_;
  }

  std::size_t NumOfRows() const {
    return num_rows_;
  }
  std::size_t NumOfCols() const {
    return colptr_.size() - 1;
  }
  std::int32_t NumOfNonZeros() const {
    return colptr_.back();
  }

  /**
   * @return Total number of non-zero elements as a fraction of total
   * elements.
   */
  double Density() const;

  /**
   * @return True if the number of non-zeros per row is smaller than a small
   * threshold.
   */
  bool IsHyperSparse() const;

  /**
   * @brief Perform dense-matrix * sparse matrix.
   *
   * C += A (dense matrix) * B (this CSC matrix) if accumulation = true \n
   * C  = A (dense matrix) * B (this CSC matrix) if accumulation = false
   */
  void SpMDM(
      const block_type_t& block,
      const std::uint8_t* A,
      int lda,
      bool accumulation,
      std::int32_t* C,
      int ldc) const;

 private:
  const std::size_t num_rows_;
  std::vector<std::int32_t> colptr_;
  std::vector<std::int16_t> rowidx_;
  std::vector<std::int8_t> values_;

  // Cache IsHyperSparse to minimize its overhead.
  mutable bool hyper_sparse_;

  // Whether we can reuse the cached hyper_sparse_ is determined by checking
  // if NumOfNonZeros() is same as old_nnz_ saved in previous invocation of
  // IsHyperSparse call.
  mutable std::int32_t old_nnz_;
};

} // namespace fbgemm
