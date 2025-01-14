/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmFP32.h"
#include "test/FBGemmFPTest.h"

using FBGemmFP32Test = fbgemm::FBGemmFPTest<float>;

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmFP32Test,
    ::testing::Values(
      std::pair<fbgemm::matrix_op_t, fbgemm::matrix_op_t>(
          fbgemm::matrix_op_t::NoTranspose, fbgemm::matrix_op_t::NoTranspose),
      std::pair<fbgemm::matrix_op_t, fbgemm::matrix_op_t>(
          fbgemm::matrix_op_t::NoTranspose, fbgemm::matrix_op_t::Transpose)/*,
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::Transpose, matrix_op_t::NoTranspose),
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::Transpose, matrix_op_t::Transpose)*/));

TEST_P(FBGemmFP32Test, Test) {
  TestRun();
}

TEST_P(FBGemmFP32Test, Unpack) {
  UnpackTestRun();
}
