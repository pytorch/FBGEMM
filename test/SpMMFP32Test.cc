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
#include "fbgemm/FbgemmSpMM.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {
// The maximum value should be bigger than 192 to test multiple k blocks
uniform_int_distribution<int> dist_dim(1, 256);
default_random_engine generator;

class SpMMTest : public testing::Test {
 protected:
  vector<tuple<int, int, int, float>> GenParams() {
    vector<tuple<int, int, int, float>> shapes;

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
    auto cDataNaive = getRandomSparseVector(m * n);

    // 1. run reference version
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

    // 2. test JIT version
    // Pick arbitrary leading dimensions that are not same as m or k for
    // testing purpose
    int ldat = 2 * m;
    int ldbt = 2 * k;
    int ldct = 2 * m;

    // To compute A*B where B is sparse matrix, we need to do
    // (B^T*A^T)^T
    aligned_vector<float> atData(k * ldat);
    transpose_matrix(m, k, aData.data(), k, atData.data(), ldat);
    aligned_vector<float> btData(n * ldbt);
    transpose_matrix(k, n, bData.data(), n, btData.data(), ldbt);
    auto cDataJIT = getRandomSparseVector(n * ldct);

    auto fn = generateSpMM<float>(n, m, k, btData.data(), ldbt, ldat, ldct);
    fn(atData.data(), cDataJIT.data(), 0 /* accum_flag */);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float expected = cDataNaive[i * n + j];
        float actual = cDataJIT[i + j * ldct];
        EXPECT_NEAR(expected, actual, 1e-6 * std::abs(expected) + 1e-7)
            << "Results differ at (" << i << ", " << j << ")";
      }
    }

    // 3. test JIT version that doesn't depend on dense matrix shapes
    auto fn_varying_n = generateSpMM<float>(n, k, btData.data(), ldbt);
    fn_varying_n(
        atData.data(), cDataJIT.data(), m, ldat, ldct, 0 /* accum_flag */);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float expected = cDataNaive[i * n + j];
        float actual = cDataJIT[i + j * ldct];
        EXPECT_NEAR(expected, actual, 1e-6 * std::abs(expected) + 1e-7)
            << "Results differ at (" << i << ", " << j << ")";
      }
    }

    // 5. test JIT version that doesn't depend on dense matrix shapes
    // with a different A and C
    int new_m = dist_dim(generator);
    ldat = 2 * new_m;
    ldct = 2 * new_m;

    aData = getRandomSparseVector(new_m * k);
    cDataNaive = getRandomSparseVector(new_m * n);

    cblas_sgemm_ref(
        matrix_op_t::NoTranspose,
        matrix_op_t::NoTranspose,
        new_m,
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

    atData.resize(k * ldat);
    transpose_matrix(new_m, k, aData.data(), k, atData.data(), ldat);
    cDataJIT.resize(n * ldct);

    fn_varying_n(
        atData.data(), cDataJIT.data(), new_m, ldat, ldct, 0 /* accum_flag */);

    for (int i = 0; i < new_m; ++i) {
      for (int j = 0; j < n; ++j) {
        float expected = cDataNaive[i * n + j];
        float actual = cDataJIT[i + j * ldct];
        EXPECT_NEAR(expected, actual, 1e-6 * std::abs(expected) + 1e-7)
            << "Results differ at (" << i << ", " << j << ")";
      }
    }
  } // for each shape
}
