/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmConvert.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {
class FBGemmFloatReluTest : public testing::TestWithParam<bool> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmFloatReluTest,
    ::testing::Bool());

TEST_P(FBGemmFloatReluTest, Test) {
  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    a[i] = i - 50 + 1.23;
  }
  float c[100]; // fp32 type
  FloatRelu_simd(a, c, 100);
  for (int i = 0; i < 100; ++i) {
    float expected = std::max(a[i], 0.0f);
    EXPECT_EQ(expected, c[i]);
  }
}
