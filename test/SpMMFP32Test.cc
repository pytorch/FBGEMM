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
    // The maximum value should be bigger than 192 to test multiple k blocks
    uniform_int_distribution<int> dist_dim(1, 256);
    uniform_real_distribution<float> dist_fnz(0, 1.0);
    for (int i = 0; i < 256; ++i) {
      shapes.push_back(make_tuple(
          dist_dim(generator),
          dist_dim(generator),
          dist_dim(generator),
          dist_fnz(generator)));
    }
    return shapes;
  }
};
} // anonymous namespace

TEST_F(SpMMTest, fp32) {
  auto shapes = GenParams();
  int m, n, k;
  float fnz;
  for (auto s : shapes) {
    tie(m, n, k, fnz) = s;
    auto aData = getRandomSparseVector(m * k);
    auto bData = getRandomSparseVector(k * n, fnz);
    auto cDataJIT = getRandomSparseVector(m * n);
    aligned_vector<float> cDataNaive = cDataJIT;

    // To compute A*B where B is sparse matrix, we need to do
    // (B^T*A^T)^T
    aligned_vector<float> atData = aData;
    transpose_matrix(atData.data(), m, k);
    aligned_vector<float> btData = bData;
    transpose_matrix(btData.data(), k, n);

    auto fn = generateSpMM<float>(n, m, k, btData.data(), k, m, m);
    fn(atData.data(), cDataJIT.data(), 0 /* flag */);

    transpose_matrix(cDataJIT.data(), n, m);

    cblas_sgemm_ref(
        matrix_op_t::NoTranspose,
        matrix_op_t::NoTranspose,
        m,
        n,
        k,
        1.0f,
        aData.data(),
        k,
        bData.data(),
        n,
        0.0f,
        cDataNaive.data(),
        n);

    for (int i = 0; i < cDataJIT.size(); ++i) {
      float expected = cDataNaive[i];
      float actual = cDataJIT[i];
      EXPECT_NEAR(expected, actual, 1e-6 * std::abs(expected) + 1e-7)
          << "Results differ at " << i;
    }
  } // for each shape
}
