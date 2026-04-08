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

#include "bench/BenchUtils.h" // @manual
#include "fbgemm/FbgemmFP32.h"
#include "fbgemm/Utils.h"
#include "test/FBGemmFPTest.h"

using FBGemmFP32Test = fbgemm::FBGemmFPTest<float>;

INSTANTIATE_TEST_SUITE_P(
    InstantiationName,
    FBGemmFP32Test,
    ::testing::Values(
        std::pair{
            fbgemm::matrix_op_t::NoTranspose,
            fbgemm::matrix_op_t::NoTranspose},
        std::pair{
            fbgemm::matrix_op_t::NoTranspose,
            fbgemm::matrix_op_t::Transpose}));

TEST_P(FBGemmFP32Test, Test) {
  TestRun();
}

TEST_P(FBGemmFP32Test, Unpack) {
  UnpackTestRun();
}

TEST_P(FBGemmFP32Test, TestAvx2) {
  TestRunWithIsa(fbgemm::inst_set_t::avx2);
}

TEST_P(FBGemmFP32Test, TestAvx512) {
  if (!fbgemm::fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this CPU";
  }
  TestRunWithIsa(fbgemm::inst_set_t::avx512);
}

TEST_P(FBGemmFP32Test, TestAvx512_256) {
  if (!fbgemm::fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this CPU";
  }
  TestRunWithIsa(fbgemm::inst_set_t::avx512_ymm);
}
