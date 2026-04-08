/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "./FBGemmFPTest.h"
#include "fbgemm/FbgemmFP16.h"
#include "fbgemm/Utils.h"

using FBGemmFP16Test = fbgemm::FBGemmFPTest<fbgemm::float16>;

INSTANTIATE_TEST_SUITE_P(
    InstantiationName,
    FBGemmFP16Test,
    ::testing::Values(
        std::pair{
            fbgemm::matrix_op_t::NoTranspose,
            fbgemm::matrix_op_t::NoTranspose},
        std::pair{
            fbgemm::matrix_op_t::NoTranspose,
            fbgemm::matrix_op_t::Transpose}));

TEST_P(FBGemmFP16Test, Test) {
  TestRun();
}

TEST_P(FBGemmFP16Test, Unpack) {
  UnpackTestRun();
}

TEST_P(FBGemmFP16Test, TestAvx2) {
  TestRunWithIsa(fbgemm::inst_set_t::avx2);
}

TEST_P(FBGemmFP16Test, TestAvx512) {
  if (!fbgemm::fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this CPU";
  }
  TestRunWithIsa(fbgemm::inst_set_t::avx512);
}

TEST_P(FBGemmFP16Test, TestAvx512_256) {
  if (!fbgemm::fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this CPU";
  }
  TestRunWithIsa(fbgemm::inst_set_t::avx512_ymm);
}
