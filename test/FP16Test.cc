/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include "./TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmFP16.h"
#include "src/RefImplementations.h"

#ifdef USE_IACA
#include "iacaMarks.h"
#endif

using namespace std;
using namespace fbgemm;

namespace {
// The template parameter is transpose of A and B
class FBGemmFP16Test
    : public testing::TestWithParam<pair<matrix_op_t, matrix_op_t>> {
 protected:
  vector<vector<int>> GenShapes() const {
    vector<vector<int>> shapes;
    random_device r;
    default_random_engine generator(r());
    uniform_int_distribution<int> dm(1, 256);
    uniform_int_distribution<int> dnk(1, 1024);
    for (int i = 0; i < 10; i++) {
      int m = dm(generator);
      int n = dnk(generator);
      int k = dnk(generator);
      shapes.push_back({m, n, k});
    }
    return shapes;
  }
};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmFP16Test,
    ::testing::Values(
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::NoTranspose, matrix_op_t::NoTranspose),
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::NoTranspose, matrix_op_t::Transpose)/*,
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::Transpose, matrix_op_t::NoTranspose),
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::Transpose, matrix_op_t::Transpose)*/));

TEST_P(FBGemmFP16Test, Test) {
  auto shapes = GenShapes();
  float alpha = 1.f, beta = 0.f;
  matrix_op_t atrans, btrans;
  tie(atrans, btrans) = GetParam();

  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    cerr << "m = " << m << " n = " << n << " k = " << k;
    if (atrans == matrix_op_t::Transpose) {
      cerr << " A_transposed";
    }
    if (btrans == matrix_op_t::Transpose) {
      cerr << " B_transposed";
    }
    cerr << endl;

    // initialize with small numbers
    aligned_vector<int> Aint(m * k);
    aligned_vector<int> Bint(k * n);
    randFill(Aint, 0, 4);
    randFill(Bint, 0, 4);
    aligned_vector<float> A(Aint.begin(), Aint.end());
    aligned_vector<float> B(Bint.begin(), Bint.end());

    aligned_vector<float> C(m * n, NAN);

    aligned_vector<float> A_ref(A), B_ref(B), C_ref(C);

    // Gold via reference sgemm
    cblas_sgemm_ref(
        atrans,
        btrans,
        m,
        n,
        k,
        1.0f,
        A_ref.data(),
        atrans == matrix_op_t::Transpose ? m : k,
        B_ref.data(),
        btrans == matrix_op_t::Transpose ? k : n,
        0.0f,
        C_ref.data(),
        n);

    // fbgemm fp16
    PackedGemmMatrixFP16 Bp(btrans, k, n, alpha, B.data());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int num_threads = fbgemm_get_num_threads();
      int tid = fbgemm_get_thread_num();

      cblas_gemm_compute(
          atrans, m, A.data(), Bp, beta, C.data(), tid, num_threads);
    }

    // correctness check
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float expected = C_ref[i * n + j];
        float actual = C[i * n + j];
        EXPECT_EQ(actual, expected)
            << "GEMM results differ at (" << i << ", " << j << "). ref "
            << expected << " FBGemm " << actual;
      }
    }
  }
}

TEST_P(FBGemmFP16Test, Unpack) {
  auto shapes = GenShapes();
  float alpha = 1.f, beta = 0.f;
  matrix_op_t atrans, btrans;
  tie(atrans, btrans) = GetParam();

  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    cerr << "m = " << m << " n = " << n << " k = " << k;
    if (atrans == matrix_op_t::Transpose) {
      cerr << " A_transposed";
    }
    if (btrans == matrix_op_t::Transpose) {
      cerr << " B_transposed";
    }
    cerr << endl;

    // initialize with small numbers
    aligned_vector<int> Aint(m * k);
    aligned_vector<int> Bint(k * n);
    randFill(Aint, 0, 4);
    randFill(Bint, 0, 4);
    aligned_vector<float> A(Aint.begin(), Aint.end());
    aligned_vector<float> B(Bint.begin(), Bint.end());

    aligned_vector<float> C(m * n, NAN);

    aligned_vector<float> A_ref(A), B_ref(B), C_ref(C);

    // Gold via reference sgemm
    cblas_sgemm_ref(
        atrans,
        btrans,
        m,
        n,
        k,
        1.0f,
        A_ref.data(),
        atrans == matrix_op_t::Transpose ? m : k,
        B_ref.data(),
        btrans == matrix_op_t::Transpose ? k : n,
        0.0f,
        C_ref.data(),
        n);

    // fbgemm fp16
    PackedGemmMatrixFP16 Bp(btrans, k, n, alpha, B.data());
    EXPECT_TRUE(Bp.packed());

    // Test unpack
    aligned_vector<float16> tmp(Bp.matSize());
    memcpy(tmp.data(), Bp.pmat(), Bp.matSize() * sizeof(float16));
    Bp.unpackFromSrc(btrans, tmp.data());
    EXPECT_FALSE(Bp.packed());
    memcpy(tmp.data(), Bp.pmat(), Bp.matSize() * sizeof(float16));
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < n; ++j) {
        EXPECT_EQ(cpu_half2float(tmp[i * n + j]), B[i * n + j]);
      }
    }

    // Pack it back
    Bp.packFromSrc(btrans, tmp.data());
    EXPECT_TRUE(Bp.packed());

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int num_threads = fbgemm_get_num_threads();
      int tid = fbgemm_get_thread_num();

      cblas_gemm_compute(
          atrans, m, A.data(), Bp, beta, C.data(), tid, num_threads);
    }

    // correctness check
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float expected = C_ref[i * n + j];
        float actual = C[i * n + j];
        EXPECT_EQ(actual, expected)
            << "GEMM results differ at (" << i << ", " << j << "). ref "
            << expected << " FBGemm " << actual;
      }
    }
  }
}
