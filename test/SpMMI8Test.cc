/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>

#include <random>

#include "TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSpMM.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {
class SpMMTest : public testing::Test {
 protected:
  vector<tuple<int, int, int, float>> GenParams() {
    vector<tuple<int, int, int, float>> shapes;
    default_random_engine generator;
    uniform_int_distribution<int> dist_dim(1, 128);
    uniform_real_distribution<float> dist_fnz(0, 1.0);
    for (int i = 0; i < 256; ++i) {
      shapes.push_back(make_tuple(
          dist_dim(generator),
          dist_dim(generator),
          // By design, k must be a multiple of 4
          dist_dim(generator) * 4,
          dist_fnz(generator)));
    }
    return shapes;
  }
};
} // anonymous namespace

TEST_F(SpMMTest, uint8) {
  auto shapes = GenParams();
  int m, n, k;
  float fnz;
  for (auto s : shapes) {
    tie(m, n, k, fnz) = s;
    auto aData = getRandomSparseVector(m * k / 4);
    auto bData = getRandomSparseVector(k * n / 4, fnz);
    auto cDataJIT = getRandomSparseVector(m * n);
    aligned_vector<float> cDataNaive = cDataJIT;

    auto aptr = reinterpret_cast<uint8_t*>(aData.data());
    auto bptr = reinterpret_cast<const int8_t*>(bData.data());

    // To avoid saturation
    for (int i = 0; i < m * k; ++i) {
      aptr[i] &= 0x7F;
    }

    // To compute A*B where B is sparse matrix, we need to do
    // (B^T*A^T)^T
    aligned_vector<float> aDataPacked = aData;
    // Transpose as if A is float so 4 columns are interleaved
    transpose_matrix(aDataPacked.data(), m, k / 4);
    aligned_vector<float> btData = bData;
    auto btptr = reinterpret_cast<int8_t*>(btData.data());
    transpose_matrix(btptr, k, n);
    auto cptrJIT = reinterpret_cast<int32_t*>(cDataJIT.data());

    auto fn = generateSpMM<int32_t>(n, m, k, btptr, k, m, m);
    fn(reinterpret_cast<const uint8_t*>(aDataPacked.data()),
       cptrJIT,
       0 /* flag */);

    transpose_matrix(cptrJIT, n, m);

    auto cptrNaive = reinterpret_cast<int32_t*>(cDataNaive.data());
    matmul_u8i8acc32_ref(m, n, k, k, n, n, aptr, bptr, cptrNaive);

    for (int i = 0; i < cDataJIT.size(); ++i) {
      float expected = cptrNaive[i];
      float actual = cptrJIT[i];
      EXPECT_EQ(expected, actual) << "Results differ at " << i;
    }
  } // for each shape
}
