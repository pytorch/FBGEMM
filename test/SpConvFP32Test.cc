/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>

#include <random>

#include "./TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSpConv.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {
class SpConvTest : public testing::Test {
 protected:
  vector<tuple<int, int, int, int, float>> GenParams() {
    vector<tuple<int, int, int, int, float>> shapes;
    default_random_engine generator;
    uniform_int_distribution<int> dist_dim(2, 64);
    uniform_real_distribution<float> dist_fnz(0, 1.0);
    for (int i = 0; i < 16; ++i) {
      shapes.push_back({dist_dim(generator),
                        dist_dim(generator),
                        dist_dim(generator),
                        dist_dim(generator),
                        dist_fnz(generator)});
    }
    return shapes;
  }
};
} // anonymous namespace

TEST_F(SpConvTest, fp32) {
  auto shapes = GenParams();
  int Cout, Cin, IY, IX;
  float fnz;
  for (auto s : shapes) {
    tie(Cout, Cin, IY, IX, fnz) = s;

    auto aData = getRandomSparseVector(IY * IX * Cin);
    auto bData = getRandomSparseVector(3 * 3 * Cin * Cout, fnz);
    auto cDataJIT = getRandomSparseVector(IY * IX * Cout);
    aligned_vector<float> cDataNaive = cDataJIT;

    // NHWC -> CNHW layout
    aligned_vector<float> atData = aData;
    transpose_matrix(atData.data(), IY * IX, Cin);
    // RSCK -> RSKC layout
    aligned_vector<float> btData = bData;
    for (int i = 0; i < 3 * 3; ++i) {
      transpose_matrix(btData.data() + i * Cin * Cout, Cin, Cout);
    }

    auto fn = generateSpConv<float>(Cin, Cout, IY, IX, btData.data());
    fn(atData.data(), cDataJIT.data());

    // CNHW -> NHWC layout
    transpose_matrix(cDataJIT.data(), Cout, IY * IX);

    conv_param_t<> conv_p(
        1, /* MB */
        Cin,
        Cout,
        {IY, IX},
        1, /* group */
        {3, 3},
        {1, 1}, /* stride */
        {1, 1, 1, 1}); /* pad */
    conv_ref(conv_p, aData.data(), bData.data(), cDataNaive.data());

    for (int i = 0; i < cDataJIT.size(); ++i) {
      float expected = cDataNaive[i];
      float actual = cDataJIT[i];
      EXPECT_NEAR(expected, actual, 1e-5 * std::abs(expected) + 1e-7)
          << "Results differ at " << i;
    }
  } // for each shape
}
