/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h" // @manual
#include "test/QuantizationHelpers.h"

using namespace std;
using namespace fbgemm;

static void
performance_test(const int M, const int N, const int K, const bool timebreak) {
  // clang-format off
  const vector<vector<int>> shapes = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    // m, n, k
    {M?M:64, N?N:800, K?K:320},
    {M?M:64, N?N:768, K?K:512},
    {M?M:16, N?N:256, K?K:512},
    {M?M:128, N?N:128, K?K:128},
    {M?M:256, N?N:512, K?K:256},
    {M?M:1024, N?N:1024, K?K:1024},
  };

  // clang-format on
  bool flush = true;
  std::vector<char> llc;

  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 10;

  if (timebreak) {
    cout
        << "WARNING: the timer may be inaccurate when used by multiple threads."
        << '\n';
    cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(18)
         << "Type, " << setw(18) << "Packing (us), " << setw(18)
         << "Kernel (us), " << setw(18) << "Postproc (us), " << setw(18)
         << "Computation (us)," << setw(18) << "Total (us), " << setw(5)
         << "GOPs" << '\n';
  } else {
    cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(18)
         << "Type, " << setw(5) << "GOPS" << '\n';
  }

  chrono::time_point<chrono::high_resolution_clock> start, end;
  for (const auto& shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    aligned_vector<uint8_t> Aint8(m * k);

    aligned_vector<int8_t> Bint8(k * n);

    aligned_vector<float> Cfp32_mkl(m * n);
    aligned_vector<int32_t> Cint32_mkl(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_ref(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb_acc32(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb_acc16(Cfp32_mkl.size());

    // A matrix
    randFill<uint8_t>(Aint8, 0, 5);
    aligned_vector<float> Afp32(Aint8.begin(), Aint8.end());

    randFill<int8_t>(Bint8, -4, 4);
    avoidOverflow(m, n, k, Aint8.data(), Bint8.data());

    aligned_vector<float> Bfp32(Bint8.begin(), Bint8.end());

    double nops = 2.0 * m * n * k;
    double ttot = 0.0;
    string runType;
#ifdef USE_MKL
    const float alpha = 1.f;
    const float beta = 0.f;
    runType = "MKL_fp32";
    ttot = measureWithWarmup(
        [&]() {
          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              m,
              n,
              k,
              alpha,
              Afp32.data(),
              k,
              Bfp32.data(),
              n,
              beta,
              Cfp32_mkl.data(),
              n);
        },
        NWARMUP,
        NITER,
        [&]() {
          if (flush) {
            llc_flush(llc);
          }
        });
    ttot *= 1e9; // convert to ns

    std::ignore = ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", ";

    if (timebreak) {
      cout << setw(16) << 0 << ", " << setw(16) << 0 << ", " << setw(16) << 0
           << ", " << setw(16) << ttot / 1e3 << ", ";
    }

    cout << setw(5) << fixed << setw(5) << setprecision(1) << nops / ttot
         << '\n';

    for (size_t i = 0; i < Cfp32_mkl.size(); ++i) {
      Cint32_mkl[i] = (int32_t)Cfp32_mkl[i];
    }
#endif

    vector<int32_t> row_offsets(m);

    matmul_u8i8acc32_ref(
        m, n, k, k, n, n, Aint8.data(), Bint8.data(), Cint32_ref.data());

    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Cint32_ref.data(),
    // m, n, n, "C int32");

    PackBMatrix<int8_t> packedB_int32(
        matrix_op_t::NoTranspose, k, n, Bint8.data(), n, nullptr, 1);

    ttot = 0.0;
    runType = "FBGEMM_i8_acc32";

    double packing_time = 0.0, total_packing_time = 0.0;
    double computing_time = 0.0, total_computing_time = 0.0;
    double kernel_time = 0.0, total_kernel_time = 0.0;
    double postprocessing_time = 0.0;
    double total_postprocessing_time = 0.0;
    double run_time = 0.0, total_run_time = 0.0;

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType;

    for (auto i = 0; i < NWARMUP + NITER; ++i) {
      if (timebreak) {
        packing_time = 0.0;
        computing_time = 0.0;
        kernel_time = 0.0;
        postprocessing_time = 0.0;
        run_time = 0.0;
      }
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        PackAMatrix<uint8_t> packA_int32(
            matrix_op_t::NoTranspose, m, k, Aint8.data(), k, nullptr, 1);

        DoNothing<int32_t, int32_t> doNothing32BitObj;
        memCopy<> memcopyObj(doNothing32BitObj);
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();
        // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
        fbgemmPacked(
            packA_int32,
            packedB_int32,
            Cint32_fb_acc32.data(),
            Cint32_fb_acc32.data(),
            n,
            memcopyObj,
            tid,
            num_threads);
      }

      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
        run_time = dur.count();
        if (timebreak) {
          total_packing_time += packing_time;
          total_computing_time += computing_time;
          total_kernel_time += kernel_time;
          total_postprocessing_time += postprocessing_time;
          total_run_time += run_time;
        }
      }
    }
    if (flush) {
      ((volatile char*)(llc.data()))[0] = llc[0] + 1;
    }
    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_fb.data(), m, n, n, "C fb");

    if (timebreak) {
      cout << ", " << setw(16) << total_packing_time / (double)NITER / 1e3
           << ", " << setw(16) << total_kernel_time / (double)NITER / 1e3
           << ", " << setw(16)
           << total_postprocessing_time / (double)NITER / 1e3 << ", "
           << setw(16) << total_computing_time / (double)NITER / 1e3 << ", "
           << setw(16) << total_run_time / (double)NITER / 1e3;
    }

    cout << ", " << setw(5) << fixed << setw(5) << setprecision(1)
         << NITER * nops / ttot << '\n';

    compare_buffers(Cint32_ref.data(), Cint32_fb_acc32.data(), m, n, n, 5);

    PackBMatrix<int8_t, int16_t> packedB_int16(
        matrix_op_t::NoTranspose, k, n, Bint8.data(), n, nullptr, 1);

    ttot = 0.0;
    runType = "FBGEMM_i8_acc16";
    if (timebreak) {
      total_packing_time = 0.0;
      total_computing_time = 0.0;
      total_kernel_time = 0.0;
      total_postprocessing_time = 0.0;
      total_run_time = 0.0;
    }
    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType;

    for (auto i = 0; i < NWARMUP + NITER; ++i) {
      if (timebreak) {
        packing_time = 0.0;
        computing_time = 0.0;
        kernel_time = 0.0;
        postprocessing_time = 0.0;
        run_time = 0.0;
      }
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        PackAMatrix<uint8_t, int16_t> packA_int16(
            matrix_op_t::NoTranspose, m, k, Aint8.data(), k, nullptr, 1);

        DoNothing<int32_t, int32_t> doNothing32BitObj;
        memCopy<> memcopyObj(doNothing32BitObj);
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();
        // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
        fbgemmPacked(
            packA_int16,
            packedB_int16,
            Cint32_fb_acc16.data(),
            Cint32_fb_acc16.data(),
            n,
            memcopyObj,
            tid,
            num_threads);
      }

      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
        run_time = dur.count();
        if (timebreak) {
          total_packing_time += packing_time;
          total_computing_time += computing_time;
          total_kernel_time += kernel_time;
          total_postprocessing_time += postprocessing_time;
          total_run_time += run_time;
        }
      }
    }
    if (flush) {
      ((volatile char*)(llc.data()))[0] = llc[0] + 1;
    }
    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_fb.data(), m, n, n, "C fb");
    // compare_buffers(row_offsets.data(), row_offset_buf.data(),
    // row_offsets.size(), 5);

    if (timebreak) {
      cout << ", " << setw(16) << total_packing_time / (double)NITER / 1e3
           << ", " << setw(16) << total_kernel_time / (double)NITER / 1e3
           << ", " << setw(16)
           << total_postprocessing_time / (double)NITER / 1e3 << ", "
           << setw(16) << total_computing_time / (double)NITER / 1e3 << ", "
           << setw(16) << total_run_time / (double)NITER / 1e3;
    }

    cout << ", " << setw(5) << fixed << setw(5) << setprecision(1)
         << NITER * nops / ttot << '\n';
    cout << '\n';

    compare_buffers(Cint32_ref.data(), Cint32_fb_acc16.data(), m, n, n, 5);
  }
}

int main(int argc, const char** argv) {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  const int M = parseArgumentInt(argc, argv, "--M=", 0, 0);
  const int N = parseArgumentInt(argc, argv, "--N=", 0, 0);
  const int K = parseArgumentInt(argc, argv, "--K=", 0, 0);
  const bool timebreak = parseArgumentBool(argc, argv, "--timebreak", false);

  performance_test(M, N, K, timebreak);
  return 0;
}
