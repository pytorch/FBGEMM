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

#include "./BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

// #define FBGEMM_MEASURE_TIME_BREAKDOWN

void performance_test() {
  // clang-format off
  vector<conv_param_t<1>> shapes = {
    // MB, IC, OC, {IW}, G, {KW}, {stride_w}, pad_l, pad_r
    // Top 10 from Caffee2
    conv_param_t<1>(1, 600, 100, {1}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 600, 100, {2}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 600, 100, {3}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 600, 100, {4}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 600, 100, {5}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 600, 100, {6}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 600, 100, {7}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 600, 100, {8}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 600, 100, {9}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 200, 162, {1}, 1, {3}, {1}, {1, 1}),

    // From Pytorch
    conv_param_t<1>(1, 300, 192, {11}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 256, 162, {23}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 128, 16,  {74}, 2, {4}, {1}, {1, 1}),
    conv_param_t<1>(1, 128, 128, {100},1, {1}, {1}, {0, 0}),
    conv_param_t<1>(1, 160, 55,  {34}, 1, {3}, {1}, {1, 1}),
    conv_param_t<1>(1, 64,  55,  {40}, 1, {3}, {1}, {1, 1}),

    // For testing
    // conv_param_t<1>(1, 6, 9, {2}, 1, {3}, {1}, {1, 1}),
    // conv_param_t<1>(1, 6, 18, {4}, 1, {3}, {1}, {1, 1}),
    // conv_param_t<1>(1, 32, 8, {5}, 1, {3}, {1}, {1, 1}),
    // conv_param_t<1>(1, 4, 4, {5}, 2, {3}, {1}, {1, 1}),
    // conv_param_t<1>(1, 8, 8, {5}, 2, {3}, {1}, {1, 1}),
  };
  // clang-format on

  bool flush = true;
  std::vector<char> llc;

  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 100;

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  cout << "WARNING: the timer may be inaccurate when used by multiple threads."
       << endl;
  cout << "MB, "
       << "IC, "
       << "OC, "
       << "IW, "
       << "G,  "
       << "KW, "
       << "stride_w, "
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
       << "IW, "
       << "G,  "
       << "KW, "
       << "stride_w, "
       << "pad_w, "
       << "Type, "
       << "M, "
       << "N, "
       << "K, " << setw(5) << "GOPS" << endl;
#endif

  chrono::time_point<chrono::high_resolution_clock> begin, end;
  for (auto conv_p : shapes) {
    if (conv_p.IC % conv_p.G != 0) {
      cout << "Error: Number of input channels " << conv_p.IC
           << " is not a multiple of groups " << conv_p.G << endl;
      continue;
    }
    if (conv_p.OC % conv_p.G != 0) {
      cout << "Error: Number of output channels " << conv_p.OC
           << " is not a multiple of groups " << conv_p.G << endl;
      continue;
    }

    int IC_per_G = conv_p.IC / conv_p.G;
    int OC_per_G = conv_p.OC / conv_p.G;

    aligned_vector<uint8_t> Aint8(conv_p.MB * conv_p.IN_DIM[0] * conv_p.IC, 0);

    aligned_vector<int8_t> Bint8(
        conv_p.K[0] * conv_p.G * IC_per_G * OC_per_G, 0);

    aligned_vector<int32_t> Cint32_ref(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OC, 0);

    aligned_vector<uint8_t> Cint8_ref(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OC, 0);

    aligned_vector<int32_t> Cint32_fb_fused(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OC, 0);

    aligned_vector<uint8_t> Cint8_fb_fused(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OC, 0);

    aligned_vector<int32_t> Cint32_fb_direct(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OC, 0);

    aligned_vector<uint8_t> Cint8_fb_direct(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OC, 0);

    // A matrix (input activations)
    randFill<uint8_t>(Aint8, 0, 5);
    int32_t Aint8_zero_point = 1;

    // B matrix (weights)
    randFill<int8_t>(Bint8, -4, 4);
    aligned_vector<int32_t> Bint8_zero_point(1);
    randFill(Bint8_zero_point, -3, -1);

    aligned_vector<float> C_multiplier(Bint8_zero_point.size());
    randFill(C_multiplier, 0.1234f / 2, 0.1234f * 3 / 2);
    int32_t C_zero_pt = 5;

    int S = conv_p.K[0];

    // reference implementation
    conv_ref(
        conv_p,
        Aint8.data(),
        Aint8_zero_point,
        Bint8.data(),
        Cint32_ref.data());

    // matrix dimensions after im2col
    int MDim = conv_p.MB * conv_p.OUT_DIM[0];
    int NDim = conv_p.OC / conv_p.G;
    int KDim = conv_p.K[0] * conv_p.IC;


    // computing row offset
    vector<int32_t> row_offsets(MDim);
    vector<uint8_t> Aint8_im2col(MDim * KDim);
    im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

    // computing column offset
    vector<int32_t> col_offsets(conv_p.OC);
    for (int g = 0; g < conv_p.G; ++g) {
      col_offsets_with_zero_pt_s8acc32_ref(
          S * IC_per_G,
          OC_per_G,
          OC_per_G,
          Bint8.data() + g * S * IC_per_G * OC_per_G,
          Bint8_zero_point.data(),
          col_offsets.data() + g * OC_per_G,
          conv_p.OC);
    }

    for (int g = 0; g < conv_p.G; ++g) {
      row_offsets_u8acc32_ref(
          MDim,
          S * IC_per_G,
          KDim,
          Aint8_im2col.data() + g * S * IC_per_G,
          row_offsets.data());

      requantize_u8acc32_ref(
          MDim,
          NDim,
          conv_p.G * NDim,
          Cint32_ref.data() + g * NDim,
          Cint8_ref.data() + g * NDim,
          C_multiplier.data() + g * NDim / conv_p.OC,
          C_zero_pt,
          Aint8_zero_point,
          Bint8_zero_point.data() + g * NDim / conv_p.OC,
          row_offsets.data(),
          col_offsets.data() + g * NDim,
          nullptr,
          conv_p.OC);
    }
    // printMatrix(matrix_op_t::NoTranspose, Cint8_ref.data(), MDim, NDim, NDim, "B unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), KDim, NDim, NDim, "B unpacked");

    double nops = 2.0 * static_cast<double>(NITER) * MDim * NDim * KDim;
    double ttot = 0.0;
    string runType;

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(
        PackAWithIm2Col<uint8_t, int32_t, 1>::rowOffsetBufferSize());

    PackAWithIm2Col<uint8_t, int32_t, 1> packA(
        conv_p, Aint8.data(), nullptr, Aint8_zero_point, row_offset_buf.data());

    PackBMatrix<int8_t, int32_t> packedB(
        matrix_op_t::NoTranspose,
        KDim,
        NDim,
        Bint8.data(),
        NDim,
        nullptr,
        conv_p.G);

    // no-op output process objects
    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
        doNothingObj,
        C_multiplier.data(),
        C_zero_pt,
        Aint8_zero_point,
        Bint8_zero_point.data(),
        packA.getRowOffsetBuffer(),
        col_offsets.data(),
        nullptr,
        conv_p.G * NDim,
        conv_p.G);

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
          Cint8_fb_fused.data(),
          Cint32_fb_fused.data(),
          conv_p.G * NDim,
          outputProcObj,
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
         << ", " << conv_p.IN_DIM[0] << ", " << conv_p.G << ", " << conv_p.K[0]
         << ", " << conv_p.stride[0] << ", " << conv_p.pad[0] << ", ";

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

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int w = 0; w < conv_p.OUT_DIM[0]; ++w) {
        for (int k = 0; k < conv_p.OC; ++k) {
            int32_t expected = Cint8_ref
                [(n * conv_p.OUT_DIM[0] + w) * conv_p.OC + k];
            int32_t actual = Cint8_fb_fused
                [(n * conv_p.OUT_DIM[0] + w) * conv_p.OC + k];
            if (expected != actual) {
            cout << "Im2Col fused results differ at (" << n << ", " << w << ", "
                 << k << ")."
                 << " expected:" << expected << " actual:" << actual << endl;
            }
        }
      }
    }
    // compare_buffers(Cint32_ref.data(), Cint32_fb_fused.data(), MDim, NDim *
    // conv_p.G, NDim*conv_p.G, 5);

    runType = "direct";
    ttot = 0;

    vector<int32_t> row_offset_buf_direct(rowOffsetBufferSize1DConv(conv_p));

    PackWeightMatrixFor1DConv<int8_t, int32_t> packedWeights(
        matrix_op_t::NoTranspose, conv_p, Bint8.data(), nullptr);

    //printMatrix(matrix_op_t::NoTranspose, packedWeights.getBuf(),
    //    conv_p.G * conv_p.K[0] * ((conv_p.IC + 3) / 4),
    //    (conv_p.OC + myvsize - 1) / myvsize * myvsize * 4,
    //    (conv_p.OC + myvsize - 1) / myvsize * myvsize * 4,
    //    "B packed");

    ReQuantizeOutput<false, QuantizationGranularity::TENSOR> reqObj(
        doNothingObj,
        C_multiplier.data(),
        C_zero_pt,
        Aint8_zero_point,
        Bint8_zero_point.data(),
        row_offset_buf_direct.data(),
        col_offsets.data(),
        nullptr,
        conv_p.OC,
        conv_p.G);

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

      fbgemmFast1DConv(
          conv_p,
          Aint8.data(),
          packedWeights,
          Cint8_fb_direct.data(),
          Cint32_fb_direct.data(),
          reqObj,
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

    cout << setw(4) << conv_p.MB << ", " << conv_p.IC << ", " << conv_p.OC
         << ", " << conv_p.IN_DIM[0] << ", " << conv_p.G << ", " << conv_p.K[0]
         << ", " << conv_p.stride[0] << ", " << conv_p.pad[0] << ", ";

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

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int w = 0; w < conv_p.OUT_DIM[0]; ++w) {
        for (int k = 0; k < conv_p.OC; ++k) {
            int32_t expected = Cint32_ref
                [(n * conv_p.OUT_DIM[0] + w) *
                     conv_p.OC +
                 k];
            int32_t actual = Cint32_fb_direct
                [(n * conv_p.OUT_DIM[0] + w) *
                     conv_p.OC +
                 k];
            if (expected != actual) {
            cout << "direct conv results differ at (" << n << ", " << w << ", "
                 << k << ")."
                 << " expected:" << expected << " actual:" << actual << endl;
            }
        }
      }
    }

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int w = 0; w < conv_p.OUT_DIM[0]; ++w) {
        for (int k = 0; k < conv_p.OC; ++k) {
            int32_t expected = Cint8_ref
                [(n * conv_p.OUT_DIM[0] + w) * conv_p.OC + k];
            int32_t actual = Cint8_fb_direct
                [(n * conv_p.OUT_DIM[0] + w) * conv_p.OC + k];
            if (expected != actual) {
            cout << "direct int8 results differ at (" << n << ", " << w << ", "
                 << k << ")."
                 << " expected:" << expected << " actual:" << actual << endl;
            }
        }
      }
    }

  } // shapes
}

int main() {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  performance_test();
  return 0;
}
