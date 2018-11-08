/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "AlignedVec.h"
#include "src/FbgemmI8Depthwise.h"
#include "fbgemm/Utils.h"
#include "src/RefImplementations.h"
#include "BenchUtils.h"

using namespace std;
using namespace fbgemm;

int main() {
  // From Xray OCR
  vector<vector<int>> shapes = {
    // N,  G, H_in, W_in, stride
    {   1,  272,  47, 125, 1, },
    {   1,  272,  64, 125, 1, },
    {   1,  272,  66, 125, 1, },
    {   1,  272,  67, 100, 1, },
    {   1,  272,  71, 125, 1, },
    {   1,  272,  74, 125, 1, },
    {   1,  272,  75,  75, 1, },
    {   1,  272,  75,  76, 1, },
    {   1,  272,  75,  79, 1, },
    {   1,  272,  75,  85, 1, },
    {   1,  272,  75, 100, 1, },
    {   1,  272,  75, 103, 1, },
    {   1,  272,  75, 111, 1, },
    {   1,  272,  75, 113, 1, },
    {   1,  272,  94,  75, 1, },
    {   1,  272, 109,  75, 1, },
    {   1,  272, 113,  75, 1, },
    {   1,  272, 117,  75, 1, },
    {   1,  544,  24,  63, 1, },
    {   1,  544,  32,  63, 1, },
    {   1,  544,  33,  63, 1, },
    {   1,  544,  34,  50, 1, },
    {   1,  544,  36,  63, 1, },
    {   1,  544,  37,  63, 1, },
    {   1,  544,  38,  38, 1, },
    {   1,  544,  38,  40, 1, },
    {   1,  544,  38,  43, 1, },
    {   1,  544,  38,  50, 1, },
    {   1,  544,  38,  52, 1, },
    {   1,  544,  38,  56, 1, },
    {   1,  544,  38,  57, 1, },
    {   1,  544,  47,  38, 1, },
    {   1,  544,  55,  38, 1, },
    {   1,  544,  57,  38, 1, },
    {   1,  544,  59,  38, 1, },
    {   1, 1088,   7,   7, 1, },
    {  51, 1088,   7,   7, 1, },
    {  59, 1088,   7,   7, 1, },
    {  70, 1088,   7,   7, 1, },
    {  71, 1088,   7,   7, 1, },
    {  77, 1088,   7,   7, 1, },
    {  79, 1088,   7,   7, 1, },
    {  84, 1088,   7,   7, 1, },
    {  85, 1088,   7,   7, 1, },
    {  89, 1088,   7,   7, 1, },
    {  93, 1088,   7,   7, 1, },
    {  96, 1088,   7,   7, 1, },
    { 100, 1088,   7,   7, 1, },

    {   1,  248,  93, 250, 2, },
    {   1,  248, 128, 250, 2, },
    {   1,  248, 132, 250, 2, },
    {   1,  248, 131, 250, 2, },
    {   1,  248, 133, 200, 2, },
    {   1,  248, 141, 250, 2, },
    {   1,  248, 148, 250, 2, },
    {   1,  248, 150, 150, 2, },
    {   1,  248, 150, 151, 2, },
    {   1,  248, 150, 158, 2, },
    {   1,  248, 150, 169, 2, },
    {   1,  248, 150, 200, 2, },
    {   1,  248, 150, 205, 2, },
    {   1,  248, 150, 221, 2, },
    {   1,  248, 150, 225, 2, },
    {   1,  248, 188, 150, 2, },
    {   1,  248, 218, 150, 2, },
    {   1,  248, 225, 150, 2, },
    {   1,  248, 234, 150, 2, },
    {   1,  272,  47, 125, 2, },
    {   1,  272,  64, 125, 2, },
    {   1,  272,  66, 125, 2, },
    {   1,  272,  67, 100, 2, },
    {   1,  272,  71, 125, 2, },
    {   1,  272,  74, 125, 2, },
    {   1,  272,  75,  75, 2, },
    {   1,  272,  75,  76, 2, },
    {   1,  272,  75,  79, 2, },
    {   1,  272,  75,  85, 2, },
    {   1,  272,  75, 100, 2, },
    {   1,  272,  75, 103, 2, },
    {   1,  272,  75, 111, 2, },
    {   1,  272,  75, 113, 2, },
    {   1,  272,  94,  75, 2, },
    {   1,  272, 109,  75, 2, },
    {   1,  272, 113,  75, 2, },
    {   1,  272, 117,  75, 2, },
    {   1,  544,  14,  14, 2, },
    {  51,  544,  14,  14, 2, },
    {  59,  544,  14,  14, 2, },
    {  70,  544,  14,  14, 2, },
    {  71,  544,  14,  14, 2, },
    {  77,  544,  14,  14, 2, },
    {  79,  544,  14,  14, 2, },
    {  84,  544,  14,  14, 2, },
    {  85,  544,  14,  14, 2, },
    {  89,  544,  14,  14, 2, },
    {  93,  544,  14,  14, 2, },
    {  96,  544,  14,  14, 2, },
    { 100,  544,  14,  14, 2, },
  };

  // Depthwise is memory BW bound so we want to flush LLC.
  bool flush = true;
  std::vector<char> llc;
  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }
#define llc_flush()                                                            \
  for (auto i = 0; i < llc.size(); i++) {                                      \
    llc[i]++;                                                                  \
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 16;

  for (auto shape : shapes) {
    int N = shape[0];
    int G = shape[1];
    int H = shape[2];
    int W = shape[3];
    int stride_h = shape[4];
    int stride_w = stride_h;
    constexpr int R = 3, S = 3;
    constexpr int PAD_T = 1, PAD_B = 1, PAD_L = 1, PAD_R = 1;
    int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
    int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;

    aligned_vector<uint8_t> A(N * H * W * G);
    aligned_vector<int8_t> B(G * R * S);
    aligned_vector<int32_t> C_ref(N * H_OUT * W_OUT * G), C(C_ref.size());

    randFill(A, 0, 86);
    int32_t A_zero_point = 43;

    randFill(B, -16, 16);
    int32_t B_zero_point = 5;

    depthwise_3x3_pad_1_ref(
        N,
        H,
        W,
        G,
        stride_h,
        stride_w,
        A_zero_point,
        A.data(),
        B.data(),
        C_ref.data());

    int32_t minimum = *min_element(C_ref.begin(), C_ref.end());
    int32_t maximum = *max_element(C_ref.begin(), C_ref.end());

    float C_multiplier = 255. / (maximum - minimum);

    aligned_vector<int32_t> col_offsets(G);
    aligned_vector<int32_t> bias(G);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);
    int32_t C_zero_point = 5;

    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());
    depthwise_3x3_pad_1_ref(
        N,
        H,
        W,
        G,
        stride_h,
        stride_w,
        A_zero_point,
        A.data(),
        B_zero_point,
        B.data(),
        C_multiplier,
        C_zero_point,
        C_uint8_ref.data(),
        col_offsets.data(),
        bias.data());

    Packed3x3ConvMatrix Bp(G, B.data());

    double ttot = 0;
    double bytes =
        double(NITER) *
        (G * (N * (2 * sizeof(int32_t) * H_OUT * W_OUT + H * W) + R * S));
    double ops = double(NITER) * N * H_OUT * W_OUT * G * R * S * 2;
    chrono::time_point<chrono::system_clock> t_begin, t_end;
    for (int i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush();

      t_begin = chrono::system_clock::now();
#pragma omp parallel
      {
#ifdef _OPENMP
        int num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
#else
        int num_threads = 1;
        int tid = 0;
#endif
        depthwise_3x3_pad_1(
            N,
            H,
            W,
            G,
            stride_h,
            stride_w,
            A_zero_point,
            A.data(),
            Bp,
            C.data(),
            tid,
            num_threads);
      }
      t_end = chrono::system_clock::now();
      if (i >= NWARMUP) {
        double dt = chrono::duration<double>(t_end - t_begin).count();
        ttot += dt;
      }
    }

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H_OUT; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          for (int g = 0; g < G; ++g) {
            int32_t expected = C_ref[((n * H_OUT + h) * W_OUT + w) * G + g];
            int32_t actual = C[((n * H_OUT + h) * W_OUT + w) * G + g];
            if (expected != actual) {
              cerr << "Depthwise 3x3 results differ at (" << n << ", "
                   << h << ", " << w << ", " << g << "). expected "
                   << expected << " actual " << actual << endl;
              return -1;
            }
            assert(expected == actual);
          }
        }
      }
    }

    // Report performance
    printf("N = %d G = %d H = %d W = %d stride = %d\n", N, G, H, W, stride_h);
    printf("GB/s = %f Gops/s = %f\n", bytes / ttot / 1e9, ops / ttot / 1e9);

    ttot = 0;
    for (int i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush();

      t_begin = chrono::system_clock::now();
#pragma omp parallel
      {
#ifdef _OPENMP
        int num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
#else
        int num_threads = 1;
        int tid = 0;
#endif
        depthwise_3x3_pad_1(
            N,
            H,
            W,
            G,
            stride_h,
            stride_w,
            A_zero_point,
            A.data(),
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_uint8.data(),
            col_offsets.data(),
            bias.data(),
            tid,
            num_threads);
      }
      t_end = chrono::system_clock::now();
      if (i >= NWARMUP) {
        double dt = chrono::duration<double>(t_end - t_begin).count();
        ttot += dt;
      }
    }

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H_OUT; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          for (int g = 0; g < G; ++g) {
            uint8_t expected =
                C_uint8_ref[((n * H_OUT + h) * W_OUT + w) * G + g];
            uint8_t actual = C_uint8[((n * H_OUT + h) * W_OUT + w) * G + g];
            if (expected != actual) {
              cerr << "Depthwise 3x3 results differ at (" << n << ", "
                   << h << ", " << w << ", " << g << "). expected "
                   << (int)expected << " actual " << (int)actual << endl;
              return -1;
            }
            assert(expected == actual);
          }
        }
      }
    }

    // Report performance
    printf(
        "N = %d G = %d H = %d W = %d stride = %d with requantization fused\n",
        N, G, H, W, stride_h);
    printf("GB/s = %f Gops/s = %f\n", bytes / ttot / 1e9, ops / ttot / 1e9);
  } // for each shape

  return 0;
}
