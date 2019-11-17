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
          // By design, k must be a multiple of 4
          dist_dim(generator) * 4,
          dist_fnz(generator)));
    }
    return shapes;
  }
};
} // anonymous namespace

TEST_F(SpMMTest, int8) {
  auto shapes = GenParams();
  int m, n, k;
  float fnz;
  for (auto s : shapes) {
    tie(m, n, k, fnz) = s;
    auto aData = getRandomSparseVector(m * k / 4);
    auto bData = getRandomSparseVector(k * n / 4, fnz);
    auto cDataNaive = getRandomSparseVector(m * n);

    auto aptr = reinterpret_cast<uint8_t*>(aData.data());
    auto bptr = reinterpret_cast<const int8_t*>(bData.data());

    // To avoid saturation
    for (int i = 0; i < m * k; ++i) {
      aptr[i] &= 0x7F;
    }

    // 1. run reference version
    auto cptrNaive = reinterpret_cast<int32_t*>(cDataNaive.data());
    matmul_u8i8acc32_ref(m, n, k, k, n, n, aptr, bptr, cptrNaive);

    // 2. test JIT version
    // Pick arbitrary leading dimensions that are not same as m or k for
    // testing purpose
    int ldat = 2 * m;
    int ldbt = 2 * k;
    int ldct = 2 * m;

    // To compute A*B where B is sparse matrix, we need to do
    // (B^T*A^T)^T
    aligned_vector<float> atData(k / 4 * ldat);
    auto atptr = reinterpret_cast<const uint8_t*>(atData.data());
    // Transpose as if A is float so 4 columns are interleaved
    transpose_matrix(m, k / 4, aData.data(), k / 4, atData.data(), ldat);
    aligned_vector<float> btData(n * ldbt);
    auto btptr = reinterpret_cast<int8_t*>(btData.data());
    transpose_matrix(k, n, bptr, n, btptr, ldbt);
    auto cDataJIT = getRandomSparseVector(n * ldct);
    auto cptrJIT = reinterpret_cast<int32_t*>(cDataJIT.data());

    auto fn = generateSpMM<int32_t>(n, m, k, btptr, ldbt, ldat, ldct);
    fn(atptr, cptrJIT, 0 /* accum_flag */);

    transpose_matrix(cptrJIT, n, ldct);

    compare_validate_buffers(cptrNaive, cptrJIT, m, n, n, 0);

    // 3. test JIT version that doesn't depend on dense matrix shapes
    auto fn_varying_n = generateSpMM<int32_t>(n, k, btptr, ldbt);
    fn_varying_n(atptr, cptrJIT, m, ldat, ldct, 0 /* accum_flag */);

    transpose_matrix(cptrJIT, n, ldct);

    compare_validate_buffers(cptrNaive, cptrJIT, m, n, n, 0);

    // 4. test JIT version that doesn't depend on dense matrix shapes
    // with a different A and C
    int new_m = dist_dim(generator);
    ldat = 2 * new_m;
    ldct = 2 * new_m;

    aData = getRandomSparseVector(new_m * k / 4);
    aptr = reinterpret_cast<uint8_t*>(aData.data());
    for (int i = 0; i < new_m * k; ++i) {
      aptr[i] &= 0x7F;
    }
    cDataNaive = getRandomSparseVector(new_m * n);
    cptrNaive = reinterpret_cast<int32_t*>(cDataNaive.data());

    matmul_u8i8acc32_ref(new_m, n, k, k, n, n, aptr, bptr, cptrNaive);

    atData.resize(k / 4 * ldat);
    atptr = reinterpret_cast<const uint8_t*>(atData.data());
    transpose_matrix(new_m, k / 4, aData.data(), k / 4, atData.data(), ldat);
    cDataJIT.resize(n * ldct);
    cptrJIT = reinterpret_cast<int32_t*>(cDataJIT.data());

    fn_varying_n(atptr, cptrJIT, new_m, ldat, ldct, 0 /* accum_flag */);

    transpose_matrix(cptrJIT, n, ldct);

    compare_validate_buffers(cptrNaive, cptrJIT, new_m, n, n, 0);
  } // for each shape
}
