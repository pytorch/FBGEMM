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
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

void performance_test() {
  vector<conv_param_t<>> shapes = {
      // MB, IC, OC, IH, IW, G, KH, KW, stride_h, stride_w, pad_h, pad_w
      conv_param_t<>(1, 32, 32, {14, 14}, 1, {3, 3}, {2, 2}, {0, 0, 0, 0}),
      conv_param_t<>(1, 32, 32, {14, 14}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(2, 32, 32, {14, 14}, 1, {3, 3}, {2, 2}, {0, 0, 0, 0}),
      conv_param_t<>(2, 32, 32, {14, 14}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {47, 125}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {64, 125}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {66, 125}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {67, 100}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {75, 75}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {75, 76}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {75, 100}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {94, 75}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {109, 75}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 544, 544, {24, 63}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 544, 544, {33, 63}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 544, 544, {34, 50}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 544, 544, {36, 63}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 544, 544, {38, 38}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 544, 544, {38, 40}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 544, 544, {47, 38}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 1088, 1088, {7, 7}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(51, 1088, 1088, {7, 7}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(100, 1088, 1088, {7, 7}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
      conv_param_t<>(1, 248, 248, {93, 250}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 248, 248, {128, 250}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 248, 248, {133, 200}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 248, 248, {150, 150}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 248, 248, {150, 151}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 248, 248, {150, 158}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 248, 248, {188, 150}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 248, 248, {225, 150}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {47, 125}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {64, 125}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {66, 125}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {67, 100}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {75, 75}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {75, 76}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 272, 272, {94, 75}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 544, 544, {14, 14}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(51, 544, 544, {14, 14}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(100, 544, 544, {14, 14}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
      conv_param_t<>(1, 8, 8, {4, 4}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
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
  cout << "MB, "
       << "IC, "
       << "OC, "
       << "IH, "
       << "IW, "
       << "KH, "
       << "KW, "
       << "stride_h, "
       << "stride_w, "
       << "pad_h, "
       << "pad_w, "
       << "Type, "
       << "M, "
       << "N, "
       << "K, "
       << "Im2Col (ms), "
       << "Packing (ms), "
       << "Kernel (ms), "
       << "Postprocessing (ms), "
       << "fbgemmPacked (ms), "
       << "Total (ms), "
       << "GOPS" << endl;
#else
  cout << setw(8) << "MB, "
       << "IC, "
       << "OC, "
       << "IH, "
       << "IW, "
       << "KH, "
       << "KW, "
       << "stride_h, "
       << "stride_w, "
       << "pad_h, "
       << "pad_w, "
       << "Type, "
       << "M, "
       << "N, "
       << "K, " << setw(5) << "GOPS" << endl;
#endif

  chrono::time_point<chrono::high_resolution_clock> begin, end;
  for (auto conv_p : shapes) {
    aligned_vector<float> Afp32(
        conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IC, 0.0f);
    aligned_vector<uint8_t> Aint8(
        conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IC, 0);

    aligned_vector<uint8_t> Aint8_out(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.K[0] *
            conv_p.K[1] * conv_p.IC,
        0);

    aligned_vector<float> Bfp32(
        conv_p.K[0] * conv_p.K[1] * conv_p.IC * conv_p.OC, 0.0f);
    aligned_vector<int8_t> Bint8(
        conv_p.K[0] * conv_p.K[1] * conv_p.IC * conv_p.OC, 0);

    aligned_vector<int32_t> Cint32_ref(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC, 0);

    aligned_vector<int32_t> Cint32_fb(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC, 0);

    aligned_vector<int32_t> Cint32_fb2(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC, 0);

    // A matrix (input activations)
    randFill(Afp32, 0, 5);
    int32_t Aint8_zero_point = 4;
    for (auto i = 0; i < Afp32.size(); ++i) {
      Aint8[i] = static_cast<uint8_t>(Afp32[i]);
    }

    // B matrix (weights)
    randFill(Bfp32, -4, 4);
    // int32_t Bint8_zero_point = -3;
    for (auto i = 0; i < Bfp32.size(); ++i) {
      Bint8[i] = static_cast<int8_t>(Bfp32[i]);
    }

    // reference implementation
    conv_ref(
        conv_p,
        Aint8.data(),
        Aint8_zero_point,
        Bint8.data(),
        Cint32_ref.data());

    // matrix dimensions after im2col
    int MDim = conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1];
    int NDim = conv_p.OC;
    int KDim = conv_p.K[0] * conv_p.K[1] * conv_p.IC;

    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), KDim, NDim, NDim,
    // "B unpacked");
    // packedB.printPackedMatrix("B Packed");

    double nops = 2.0 * static_cast<double>(NITER) * MDim * NDim * KDim;
    double ttot = 0.0;
    string runType;

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(
        PackAWithIm2Col<uint8_t, int16_t>::rowOffsetBufferSize());

    PackAWithIm2Col<uint8_t, int16_t> packA(
        conv_p, Aint8.data(), nullptr, Aint8_zero_point, row_offset_buf.data());

    PackBMatrix<int8_t, int16_t> packedB(
        matrix_op_t::NoTranspose, KDim, NDim, Bint8.data(), NDim);

    // no-op output process objects
    DoNothing<int32_t, int32_t> doNothing32BitObj;
    memCopy<> memcopyObj(doNothing32BitObj);

    runType = "FusedIm2Col";
    ttot = 0;
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    double im2col_time = 0.0;
    double total_im2col_time = 0.0;
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
      begin = chrono::high_resolution_clock::now();
      fbgemmPacked(
          packA,
          packedB,
          Cint32_fb.data(),
          Cint32_fb.data(),
          NDim,
          memcopyObj,
          0,
          1);
      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - begin);
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

    cout << setw(4) << conv_p.MB << ", " << conv_p.IC << ", " << conv_p.OC
         << ", " << conv_p.IN_DIM[0] << ", " << conv_p.IN_DIM[1] << ", "
         << conv_p.G << ", " << conv_p.K[0] << ", " << conv_p.K[1] << ", "
         << conv_p.stride[0] << ", " << conv_p.stride[1] << ", "
         << conv_p.pad[0] << ", " << conv_p.pad[1] << ", ";

    cout << setw(13) << runType << ", " << setw(5) << fixed << setw(5)
         << setw(6) << MDim << ", " << setw(6) << NDim << ", " << setw(6)
         << KDim << ", ";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << fixed << setprecision(6) << setw(8) << 0 << ", "
         << total_packing_time / (double)NITER / 1e6 << ", "
         << total_kernel_time / (double)NITER / 1e6 << ", "
         << total_postprocessing_time / (double)NITER / 1e6 << ", "
         << total_run_time / (double)NITER / 1e6 << ", "
         << ttot / (double)NITER / 1e6 << ", ";
#endif
    cout << setprecision(2) << nops / ttot << endl;

    compare_buffers(Cint32_ref.data(), Cint32_fb.data(), MDim, NDim, NDim, 5);

    runType = "UnfusedIm2Col";
    row_offset_buf.resize(
        PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize());
    ttot = 0;
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    total_im2col_time = 0.0;
    total_packing_time = 0.0;
    total_computing_time = 0.0;
    total_kernel_time = 0.0;
    total_postprocessing_time = 0.0;
    total_run_time = 0.0;
#endif
    for (auto i = 0; i < NWARMUP + NITER; ++i) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      im2col_time = 0.0;
      packing_time = 0.0;
      computing_time = 0.0;
      kernel_time = 0.0;
      postprocessing_time = 0.0;
      run_time = 0.0;
#endif
      llc_flush(llc);
      begin = chrono::high_resolution_clock::now();

      im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_out.data());

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      end = chrono::high_resolution_clock::now();
      im2col_time =
          chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
#endif

      // printMatrix(matrix_op_t::NoTranspose, Aint8_out.data(), MDim, KDim,
      // KDim, "A_out after im2col unpacked");

      PackAWithRowOffset<uint8_t, int16_t> packAN(
          matrix_op_t::NoTranspose,
          MDim,
          KDim,
          Aint8_out.data(),
          KDim,
          nullptr,
          1,
          row_offset_buf.data());

      fbgemmPacked(
          packAN,
          packedB,
          Cint32_fb2.data(),
          Cint32_fb2.data(),
          NDim,
          memcopyObj,
          0,
          1);
      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - begin);
        ttot += dur.count();
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        total_im2col_time += im2col_time;
        total_packing_time += packing_time;
        total_computing_time += computing_time;
        total_kernel_time += kernel_time;
        total_postprocessing_time += postprocessing_time;
        total_run_time += run_time;
#endif
      }
    }

    ((volatile char*)(llc.data()));

    // packedB.printPackedMatrix("bench B Packed");
    // printMatrix(matrix_op_t::NoTranspose, Cint32_fb.data(), MDim, NDim, NDim,
    // "C fb fp32");
    // printMatrix(matrix_op_t::NoTranspose, Cint32_fb2.data(),
    // MDim, NDim, NDim, "C fb2 fp32");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint32_ref.data(), MDim, NDim, NDim, "C ref fp32");

    cout << setw(4) << conv_p.MB << ", " << conv_p.IC << ", " << conv_p.OC
         << ", " << conv_p.IN_DIM[0] << ", " << conv_p.IN_DIM[1] << ", "
         << conv_p.G << ", " << conv_p.K[0] << ", " << conv_p.K[1] << ", "
         << conv_p.stride[0] << ", " << conv_p.stride[1] << ", "
         << conv_p.pad[0] << ", " << conv_p.pad[1] << ", ";

    cout << setw(13) << runType << ", " << setw(5) << fixed << setw(5)
         << setw(6) << MDim << ", " << setw(6) << NDim << ", " << setw(6)
         << KDim << ", ";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << fixed << setprecision(6) << setw(8)
         << total_im2col_time / (double)NITER / 1e6 << ", "
         << total_packing_time / (double)NITER / 1e6 << ", "
         << total_kernel_time / (double)NITER / 1e6 << ", "
         << total_postprocessing_time / (double)NITER / 1e6 << ", "
         << total_run_time / (double)NITER / 1e6 << ", "
         << ttot / (double)NITER / 1e6 << ", ";
#endif
    cout << setprecision(2) << nops / ttot << endl;

    compare_buffers(Cint32_ref.data(), Cint32_fb2.data(), MDim, NDim, NDim, 5);
  } // shapes
}

int main() {
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  performance_test();
  return 0;
}
