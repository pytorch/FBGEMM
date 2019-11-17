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
                        // By design, Cin must be a multiple of 4
                        dist_dim(generator) * 4,
                        dist_dim(generator),
                        dist_dim(generator),
                        dist_fnz(generator)});
    }
    return shapes;
  }
};
} // anonymous namespace

TEST_F(SpConvTest, int8) {
  auto shapes = GenParams();
  int Cout, Cin, IY, IX;
  float fnz;
  for (auto s : shapes) {
    tie(Cout, Cin, IY, IX, fnz) = s;

    auto aData = getRandomSparseVector(IY * IX * Cin / 4);
    auto bData = getRandomSparseVector(3 * 3 * Cin * Cout / 4, fnz);
    auto cDataJIT = getRandomSparseVector(IY * IX * Cout);
    aligned_vector<float> cDataNaive = cDataJIT;

    auto bptr = reinterpret_cast<const int8_t*>(bData.data());
    auto aptr = reinterpret_cast<uint8_t*>(aData.data());
    // To avoid saturation
    for (int i = 0; i < aData.size() * 4; ++i) {
      aptr[i] &= 0x7F;
    }

    // NHWC -> CNHWc layout
    aligned_vector<float> atData = aData;
    transpose_matrix(atData.data(), IY * IX, Cin / 4);
    auto atptr = reinterpret_cast<const uint8_t*>(atData.data());
    // RSCK -> RSKC layout
    aligned_vector<float> btData = bData;
    auto btptr = reinterpret_cast<int8_t*>(btData.data());
    for (int i = 0; i < 3 * 3; ++i) {
      transpose_matrix(btptr + i * Cin * Cout, Cin, Cout);
    }
    auto cptrJIT = reinterpret_cast<int32_t*>(cDataJIT.data());

    auto fn = generateSpConv<int32_t>(Cin, Cout, IY, IX, btptr);
    fn(atptr, cptrJIT);

    // CNHW -> NHWC layout
    transpose_matrix(cptrJIT, Cout, IY * IX);

    auto cptrNaive = reinterpret_cast<int32_t*>(cDataNaive.data());
    conv_param_t<> conv_p(
        1, /* MB */
        Cin,
        Cout,
        {IY, IX},
        1, /* group */
        {3, 3},
        {1, 1}, /* stride */
        {1, 1, 1, 1}); /* pad */
    conv_ref(conv_p, aptr, 0 /* A_zero_point */, bptr, cptrNaive);

    for (int i = 0; i < cDataJIT.size(); ++i) {
      float expected = cptrNaive[i];
      float actual = cptrJIT[i];
      EXPECT_EQ(expected, actual) << "Results differ at " << i;
    }
  } // for each shape
}
