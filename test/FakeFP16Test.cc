#include <gtest/gtest.h>
#include <cmath>
#include <random>

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_BLAS
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmFakeFP16.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {
// The template parameter is transpose of A and B
class FBGemmFloat16Test
    : public testing::TestWithParam<pair<matrix_op_t, matrix_op_t>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmFloat16Test,
    ::testing::Values(
        pair<matrix_op_t, matrix_op_t>(
            matrix_op_t::NoTranspose,
            matrix_op_t::NoTranspose),
        pair<matrix_op_t, matrix_op_t>(
            matrix_op_t::NoTranspose,
            matrix_op_t::Transpose),
        pair<matrix_op_t, matrix_op_t>(
            matrix_op_t::Transpose,
            matrix_op_t::NoTranspose),
        pair<matrix_op_t, matrix_op_t>(
            matrix_op_t::Transpose,
            matrix_op_t::Transpose)));

TEST_P(FBGemmFloat16Test, Test) {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  vector<vector<int>> shapes;
  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<int> dm(1, 256);
  uniform_int_distribution<int> dnk(1, 1024);
  for (int i = 0; i < 5; i++) {
    int m = dm(generator);
    int n = dnk(generator);
    int k = dnk(generator);
    shapes.push_back({m, n, k});
  }

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
    aligned_vector<float> A_ref(m * k);
    aligned_vector<float> B_ref(k * n);
    randFill(A_ref, 0.0f, 4.0f);
    randFill(B_ref, 0.0f, 4.0f);
    // randFill(A_ref, -1.0f, 1.0f);
    // randFill(B_ref, -1.0f, 1.0f);

    aligned_vector<float16> A_float16(m * k);
    aligned_vector<float16> B_float16(k * n);
    aligned_vector<float16> C_float16(m * n);

    aligned_vector<float> C_ref(m * n, NAN);
    aligned_vector<float> C_fb(C_ref);

    FloatToFloat16_ref(A_ref.data(), A_float16.data(), m * k);
    FloatToFloat16_ref(B_ref.data(), B_float16.data(), k * n);

    if (beta != 0.0f) {
      randFill(C_ref, 0.0f, 4.0f);
      FloatToFloat16_ref(C_ref.data(), C_float16.data(), m * n);
      C_fb = C_ref;
    }

    // Gold via reference sgemm
#if defined(USE_MKL) || defined(USE_BLAS)
    cblas_sgemm(
        CblasRowMajor,
        atrans == matrix_op_t::NoTranspose ? CblasNoTrans : CblasTrans,
        btrans == matrix_op_t::NoTranspose ? CblasNoTrans : CblasTrans,
        m,
        n,
        k,
        alpha,
        A_ref.data(),
        atrans == matrix_op_t::NoTranspose ? k : m,
        B_ref.data(),
        btrans == matrix_op_t::NoTranspose ? n : k,
        beta,
        C_ref.data(),
        n);
#else
    cblas_sgemm_ref(
        atrans,
        btrans,
        m,
        n,
        k,
        alpha,
        A_ref.data(),
        atrans == matrix_op_t::NoTranspose ? k : m,
        B_ref.data(),
        btrans == matrix_op_t::NoTranspose ? n : k,
        beta,
        C_ref.data(),
        n);
#endif

    // fbgemm float16
    fbgemmFakeFP16(
        atrans,
        btrans,
        m,
        n,
        k,
        A_float16.data(),
        B_float16.data(),
        beta,
        C_float16.data());
    Float16ToFloat_ref(C_float16.data(), C_fb.data(), m * n);

    // correctness check
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float expected = C_ref[i * n + j];
        float actual = C_fb[i * n + j];
        if (actual != 0) {
          EXPECT_LE(fabs(expected - actual) / actual, k * 1.0 / 1024 * 4)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " FBGemm " << actual;
        }
      }
    }
  }
}
