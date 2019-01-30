/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>
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

#include "AlignedVec.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmFP16.h"

using namespace std;
using namespace fbgemm;

void performance_test() {
  // cache flush
  bool flush = true;
  std::vector<char> llc;
  if (flush) {
    llc.resize(64L * 1024L * 1024L, 1.0);
  }

  float alpha = 1.f, beta = 1.f;
  matrix_op_t btran = matrix_op_t::Transpose;

  using btype = float16;

#define dataset 1

#if dataset == 1
  const int NITER = (flush) ? 10 : 100;
  std::vector<std::vector<int>> shapes;
  for (auto m = 1; m < 120; m++) {
    // shapes.push_back({m, 128, 512});
    shapes.push_back({m, 512, 512});
  }

#elif dataset == 2
  const int NITER = (flush) ? 10 : 100;
#include "shapes_dataset.h"

#else
  flush = false;
  constexpr int NITER = 1;
  std::vector<std::vector<int>> shapes;
  std::random_device r;
  std::default_random_engine generator(r());
  std::uniform_int_distribution<int> dm(1, 100);
  std::uniform_int_distribution<int> dnk(1, 1024);
  for (int i = 0; i < 1000; i++) {
    int m = dm(generator);
    int n = dnk(generator);
    int k = dnk(generator);
    shapes.push_back({m, n, k});
  }
#endif

  std::string type;
  double gflops, gbs, ttot;
  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    aligned_vector<float> C_ref(m * n, 1.f);
    aligned_vector<float> C_fb(m * n, NAN);

    // initialize with small numbers
    aligned_vector<int> Aint(m * k);
    randFill(Aint, 0, 4);
    aligned_vector<float> A(Aint.begin(), Aint.end());

    aligned_vector<int> Bint(k * n);
    randFill(Bint, 0, 4);
    aligned_vector<float> B(Bint.begin(), Bint.end());
    PackedGemmMatrixFP16 Bp(btran, k, n, alpha, B.data());

    if (beta != 0.0f) {
      aligned_vector<int> Cint(C_ref.size());
      randFill(Cint, 0, 4);
      C_ref.assign(Cint.begin(), Cint.end());
      C_fb = C_ref;
    }

    double nflops = 2.0 * (double)m * (double)n * (double)k * (double)NITER;
    double nbytes = (4.0 * (double)m * (double)k + 2.0 * (double)k * (double)n +
                     4.0 * (double)m * (double)n) *
        NITER;

    // warm up MKL and fbgemm
    // check correctness at the same time
    for (auto w = 0; w < 3; w++) {
#if defined(USE_MKL) || defined(USE_BLAS)
      cblas_sgemm(
          CblasRowMajor,
          CblasNoTrans,
          btran == matrix_op_t::Transpose ? CblasTrans : CblasNoTrans,
          m,
          n,
          k,
          alpha,
          A.data(),
          k,
          B.data(),
          (btran == matrix_op_t::NoTranspose) ? n : k,
          beta,
          C_ref.data(),
          n);
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();
        cblas_gemm_compute(
            matrix_op_t::NoTranspose,
            m,
            A.data(),
            Bp,
            beta,
            C_fb.data(),
            tid,
            num_threads);
      }

#if defined(USE_MKL) || defined(USE_BLAS)
      // Compare results
      for (auto i = 0; i < C_ref.size(); i++) {
        if (std::abs(C_ref[i] - C_fb[i]) > 1e-3) {
          fprintf(
              stderr,
              "Error: too high diff between fp32 ref %f and fp16 %f\n",
              C_ref[i],
              C_fb[i]);
          return;
        }
      }
#endif
    }

    chrono::time_point<chrono::system_clock> t_begin, t_end;
#if defined(USE_MKL) || defined(USE_BLAS)
    // Gold via MKL sgemm
#if defined(USE_MKL)
    type = "MKL_FP32";
#else
    type = "BLAS_FP32";
#endif
    ttot = 0;
    for (auto it = -3; it < NITER; it++) {
      if (flush) {
        for (auto i = 0; i < llc.size(); i++) {
          llc[i]++;
        }
      }
      t_begin = chrono::system_clock::now();
      cblas_sgemm(
          CblasRowMajor,
          CblasNoTrans,
          btran == matrix_op_t::Transpose ? CblasTrans : CblasNoTrans,
          m,
          n,
          k,
          alpha,
          A.data(),
          k,
          B.data(),
          (btran == matrix_op_t::NoTranspose) ? n : k,
          beta,
          C_ref.data(),
          n);
      t_end = chrono::system_clock::now();
      if (it >= 0) {
        double dt = chrono::duration<double>(t_end - t_begin).count();
        ttot += dt;
      }
    }
    gflops = nflops / ttot / 1e9;
    gbs = nbytes / ttot / 1e9;
    printf(
        "\n%30s m = %5d n = %5d k = %5d Gflops = %8.4lf GBytes = %8.4lf\n",
        type.c_str(),
        m,
        n,
        k,
        gflops,
        gbs);
    ((volatile char*)(llc.data()));
#endif

    type = "FBP_" + std::string(typeid(btype).name());

    ttot = 0;
    for (auto it = -3; it < NITER; it++) {
      if (flush) {
        for (auto i = 0; i < llc.size(); i++) {
          llc[i]++;
        }
      }

      t_begin = chrono::system_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();

        cblas_gemm_compute(
            matrix_op_t::NoTranspose,
            m,
            A.data(),
            Bp,
            beta,
            C_fb.data(),
            tid,
            num_threads);
      }

      t_end = chrono::system_clock::now();

      if (it >= 0) {
        double dt = chrono::duration<double>(t_end - t_begin).count();
        ttot += dt;
      }
    }
    gflops = nflops / ttot / 1e9;
    gbs = nbytes / ttot / 1e9;
    printf(
        "%30s m = %5d n = %5d k = %5d Gflops = %8.4lf GBytes = %8.4lf\n",
        type.c_str(),
        m,
        n,
        k,
        gflops,
        gbs);
    ((volatile char*)(llc.data()));
  }
}

int main(int /*argc*/, char** /*argv*/) {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  performance_test();
}
