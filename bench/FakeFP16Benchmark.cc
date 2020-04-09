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
#if __APPLE__
// not sure whether need to differentiate TARGET_OS_MAC or TARGET_OS_IPHONE,
// etc.
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bench/AlignedVec.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmFakeFP16.h"

#include <iomanip>
#include <iostream>

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
  matrix_op_t atrans = matrix_op_t::NoTranspose;
  matrix_op_t btrans = matrix_op_t::Transpose;

#define dataset 1

#if dataset == 1
  const int NITER = (flush) ? 10 : 100;
  std::vector<std::vector<int>> shapes;
  for (auto m = 1; m < 120; m++) {
    // shapes.push_back({m, 128, 512});
    // shapes.push_back({m, 512, 512});
    shapes.push_back({m, 1024, 1024});
  }
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

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  cout << "WARNING: the timer may be inaccurate when used by multiple threads."
       << endl;
  cout << "M, "
       << "N, "
       << "K, "
       << "Malloc (ms), "
       << "A fp16->fp32 (ms), "
       << "B fp16->fp32 (ms), "
       << "C fp16->fp32 (ms), "
       << "GEMM fp32 (ms), "
       << "C fp32->fp16 (ms), "
       << "Total (ms), "
       << "GOPS, "
       << "GB/s" << endl;
#else
  cout << setw(7) << "M, " << setw(7) << "N, " << setw(7) << "K, " << setw(32)
       << "Type, " << setw(18) << "GOPS, " << setw(5) << "GB/s" << endl;
#endif

  std::string type;
  double gflops, gbs, ttot;
  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    // initialize with small numbers
    aligned_vector<float> A_ref(m * k);
    aligned_vector<float> B_ref(k * n);
    randFill(A_ref, 0.0f, 1.0f);
    randFill(B_ref, 0.0f, 1.0f);

    aligned_vector<float16> A_float16(m * k);
    aligned_vector<float16> B_float16(k * n);
    aligned_vector<float16> C_float16(m * n);

    FloatToFloat16_ref(A_ref.data(), A_float16.data(), m * k);
    FloatToFloat16_ref(B_ref.data(), B_float16.data(), k * n);

    aligned_vector<float> C_ref(m * n, NAN);
    aligned_vector<float> C_fb(C_ref);

    if (beta != 0.0f) {
      randFill(C_ref, 0.0f, 1.0f);
      FloatToFloat16_ref(C_ref.data(), C_float16.data(), m * n);
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
          atrans == matrix_op_t::Transpose ? CblasTrans : CblasNoTrans,
          btrans == matrix_op_t::Transpose ? CblasTrans : CblasNoTrans,
          m,
          n,
          k,
          alpha,
          A_ref.data(),
          k,
          B_ref.data(),
          (btrans == matrix_op_t::NoTranspose) ? n : k,
          beta,
          C_ref.data(),
          n);
#endif

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

#if defined(USE_MKL) || defined(USE_BLAS)
      // Compare results
      for (auto i = 0; i < C_ref.size(); i++) {
        if (std::fabs(C_ref[i] - C_fb[i]) / C_ref[i] > k * 1.0 / 128) {
          fprintf(
              stderr,
              "Error: too high diff between fp32 ref %f and float16 %f\n",
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
          atrans == matrix_op_t::Transpose ? CblasTrans : CblasNoTrans,
          btrans == matrix_op_t::Transpose ? CblasTrans : CblasNoTrans,
          m,
          n,
          k,
          alpha,
          A_ref.data(),
          k,
          B_ref.data(),
          (btrans == matrix_op_t::NoTranspose) ? n : k,
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

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << type.c_str() << ", " << setw(5) << fixed << setw(5)
         << setprecision(3) << gflops << ", " << setprecision(3) << gbs << endl;
    ((volatile char*)(llc.data()));
#endif

    type = "FBGEMM_Float16";

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    double total_malloc_time = 0.0;
    double total_A_fp16_to_fp32_time = 0.0;
    double total_B_fp16_to_fp32_time = 0.0;
    double total_C_fp16_to_fp32_time = 0.0;
    double total_computing_time = 0.0;
    double total_C_fp32_to_fp16_time = 0.0;
    double total_run_time = 0.0;
#endif
    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << type.c_str() << ", ";

    ttot = 0;
    for (auto it = -3; it < NITER; it++) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      malloc_time = 0.0;
      A_fp16_to_fp32_time = 0.0;
      B_fp16_to_fp32_time = 0.0;
      C_fp16_to_fp32_time = 0.0;
      computing_time = 0.0;
      C_fp32_to_fp16_time = 0.0;
      run_time = 0.0;
#endif

      if (flush) {
        for (auto i = 0; i < llc.size(); i++) {
          llc[i]++;
        }
      }

      t_begin = chrono::system_clock::now();
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
      t_end = chrono::system_clock::now();
      Float16ToFloat_ref(C_float16.data(), C_fb.data(), m * n);
      if (it >= 0) {
        double dt = chrono::duration<double>(t_end - t_begin).count();
        ttot += dt;

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        total_malloc_time += malloc_time;
        total_A_fp16_to_fp32_time += A_fp16_to_fp32_time;
        total_B_fp16_to_fp32_time += B_fp16_to_fp32_time;
        total_C_fp16_to_fp32_time += C_fp16_to_fp32_time;
        total_computing_time += computing_time;
        total_C_fp32_to_fp16_time += C_fp32_to_fp16_time;
        total_run_time += run_time;
#endif
      }
    }
    gflops = nflops / ttot / 1e9;
    gbs = nbytes / ttot / 1e9;

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << fixed << total_malloc_time / (double)NITER / 1e6 << ", "
         << total_A_fp16_to_fp32_time / (double)NITER / 1e6 << ", "
         << total_B_fp16_to_fp32_time / (double)NITER / 1e6 << ", "
         << total_C_fp16_to_fp32_time / (double)NITER / 1e6 << ", "
         << total_computing_time / (double)NITER / 1e6 << ", "
         << total_C_fp32_to_fp16_time / (double)NITER / 1e6 << ", ";
#endif

    cout << setw(5) << fixed << setw(5) << setprecision(3)
         << ttot / NITER * 1000 << ", " << setprecision(3) << gflops << ", "
         << setprecision(3) << gbs << endl;
    cout << endl;

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
