/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
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

#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"
#include "test/QuantizationHelpers.h"
#include "BenchUtils.h"

using namespace std;
using namespace fbgemm;

void performance_test() {
  vector<vector<int>> shapes = {
      {156800, 4, 36},
      {156800, 8, 36},
      {156800, 16, 36},
      {1, 128, 512},
      {1, 1024, 256},
      {1, 2048, 512},
      {1, 4096, 1024},

      {6, 256, 1024},
      {6, 256, 2048},
      {6, 512, 512},
      {6, 1024, 256},
      {6, 2048, 256},
      {6, 2048, 512},
      {6, 4096, 256},
      {6, 4096, 1024},
      {6, 4096, 2048},

      {10, 2048, 256},
      {10, 4096, 1024},

      {20, 2048, 256},
      {20, 4096, 1024},

      {102, 1024, 512},
      {102, 2323, 256},
      {102, 512, 256},

      {1, 800, 3200},
      {1, 800, 8000},

      {16, 256, 1500},
      {16, 256, 1567},
      {1, 128, 2876},
      {16, 128, 1567},
      {1, 128, 2722},
      {16, 256, 512},
  };
  bool flush = true;
  std::vector<char> llc;

  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 10;

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  cout << "WARNING: the timer may be inaccurate when used by multiple threads."
       << endl;
  cout << "M, "
       << "N, "
       << "K, "
       << "Packing (ms), "
       << "Kernel (ms), "
       << "Postprocessing (ms), "
       << "Total (ms), "
       << "GOPs" << endl;
#else
  cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(18)
       << "Type, " << setw(5) << "GOPS" << endl;
#endif

  chrono::time_point<chrono::high_resolution_clock> start, end;
  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    float alpha = 1.f, beta = 0.f;
    aligned_vector<float> Afp32(m * k, 0.0f);
    aligned_vector<uint8_t> Aint8(m * k, 0);

    aligned_vector<float> Bfp32(k * n, 0.0f);
    aligned_vector<int8_t> Bint8(k * n, 0);

    aligned_vector<float> Cfp32_mkl(m * n, 0.0f);
    aligned_vector<int32_t> Cint32_mkl(m * n, 0.0f);
    aligned_vector<int32_t> Cint32_fb(m * n, 0);
    aligned_vector<uint8_t> Cint8_fb(m * n, 0);
    aligned_vector<int32_t> Cint32_local(m * n, 0);
    aligned_vector<int32_t> Cint32_buffer(m * n, 0);
    aligned_vector<uint8_t> Cint8_local(m * n, 0);

    // A matrix
    randFill(Aint8, 0, 255);
    // float Aint8_scale = 0.11;
    int32_t Aint8_zero_point = 43;
    for (auto i = 0; i < Afp32.size(); ++i) {
      Afp32[i] = (float)Aint8[i];
    }

    randFill(Bint8, -128, 127);
    avoidOverflow(m, n, k, Aint8.data(), Bint8.data());

    // float Bint8_scale = 0.49;
    int32_t Bint8_zero_point = -30;
    for (auto i = 0; i < Bfp32.size(); ++i) {
      Bfp32[i] = (float)Bint8[i];
    }

    // computing column offset
    vector<int32_t> col_offsets;
    col_offsets.resize(n);
    col_offsets_with_zero_pt_s8acc32_ref(
        k, n, n, Bint8.data(), Bint8_zero_point, col_offsets.data());

    double nops = 2.0 * static_cast<double>(NITER) * m * n * k;
    double ttot = 0.0;
    string runType;
#ifdef USE_MKL
    runType = "MKL_fp32";
    for (auto i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();
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
      end = chrono::high_resolution_clock::now();
      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
      }
    }
    ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", " << setw(5) << fixed << setw(5)
         << setprecision(1) << nops / ttot << endl;

    for (auto i = 0; i < Cfp32_mkl.size(); ++i) {
      Cint32_mkl[i] = (int32_t)Cfp32_mkl[i];
    }
#endif

    vector<int32_t> row_offsets;
    row_offsets.resize(m);

    float C_multiplier = 0.1234;
    int32_t C_zero_pt = 5;

    matmul_u8i8acc32_ref(
        m, n, k, k, n, n, Aint8.data(), Bint8.data(), Cint32_local.data());

    row_offsets_u8acc32_ref(m, k, k, Aint8.data(), row_offsets.data());

    requantize_u8acc32_ref(
        m,
        n,
        n,
        Cint32_local.data(),
        Cint8_local.data(),
        C_multiplier,
        C_zero_pt,
        Aint8_zero_point,
        Bint8_zero_point,
        row_offsets.data(),
        col_offsets.data(),
        nullptr); // bias
    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Cint32_local.data(),
    // m, n, n, "C int32");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_local.data(), m, n, n, "C requantized");
    // printMatrix(matrix_op_t::NoTranspose, col_offsets.data(), 1, n, n, "col
    // offsets before");

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());

    PackAWithRowOffset<uint8_t> packAN(
        matrix_op_t::NoTranspose,
        m,
        k,
        Aint8.data(),
        k,
        nullptr,
        1,
        Aint8_zero_point,
        row_offset_buf.data());

    PackBMatrix<int8_t> packedBN(
        matrix_op_t::NoTranspose,
        k,
        n,
        Bint8.data(),
        n,
        nullptr,
        1,
        Bint8_zero_point);

    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false> outputProcObj(
        doNothingObj,
        C_multiplier,
        C_zero_pt,
        Aint8_zero_point,
        Bint8_zero_point,
        packAN.getRowOffsetBuffer(),
        col_offsets.data(),
        nullptr);

    ttot = 0.0;
    runType = "FBGEMM_i8_acc32";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    double total_packing_time = 0.0;
    double total_computing_time = 0.0;
    double total_kernel_time = 0.0;
    double total_postprocessing_time = 0.0;
    double total_run_time = 0.0;
#endif
    for (auto i = 0; i < NWARMUP + NITER; ++i) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      packing_time = 0.0;
      computing_time = 0.0;
      kernel_time = 0.0;
      postprocessing_time = 0.0;
      run_time = 0.0;
#endif
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();
      fbgemmPacked(
          packAN,
          packedBN,
          Cint8_fb.data(),
          Cint32_buffer.data(),
          n,
          outputProcObj,
          0,
          1);
      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        total_packing_time += packing_time;
        total_computing_time += computing_time;
        total_kernel_time += kernel_time;
        total_postprocessing_time += postprocessing_time;
        total_run_time += run_time;
#endif
      }
    }
    ((volatile char*)(llc.data()));
    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Cint8_local.data(),
    // m, n, n, "C requantized after");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_fb.data(), m, n, n, "C fb");
    // printMatrix(matrix_op_t::NoTranspose,
    // col_offsets.data(), 1, n, n, "col offsets after");
    // compare_buffers(row_offsets.data(), row_offset_buf.data(),
    // row_offsets.size(), 5);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << fixed << total_packing_time / (double)NITER / 1e6 << ", "
         << total_kernel_time / (double)NITER / 1e6 << ", "
         << total_postprocessing_time / (double)NITER / 1e6 << ", "
         << total_run_time / (double)NITER / 1e6 << ", ";
#endif
    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", " << setw(5) << fixed << setw(5)
         << setprecision(1) << nops / ttot << endl;
    cout << endl;

#ifdef USE_MKL
    compare_buffers(Cint8_local.data(), Cint8_fb.data(), m, n, n, 5);
#endif
  }
}

int main(int /* unused */, char** /* unused */) {
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  performance_test();
  return 0;
}
