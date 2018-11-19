/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "test/I8DepthwiseTest.h"

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
#include "BenchUtils.h"
#include "fbgemm/Utils.h"
#include "src/FbgemmI8Depthwise.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

int main() {
  // Depthwise is memory BW bound so we want to flush LLC.
  bool flush = true;
  std::vector<char> llc;
  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }
#define llc_flush()                       \
  for (auto i = 0; i < llc.size(); i++) { \
    llc[i]++;                             \
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 16;

  for (auto shape : shapes_3d) {
    int N = shape[0];
    int K = shape[1];
    int T = shape[2];
    int H = shape[3];
    int W = shape[4];
    int stride_t = shape[5];
    int stride_h = stride_t;
    int stride_w = stride_t;
    constexpr int K_T = 3, K_H = 3, K_W = 3;
    constexpr int PAD_P = 1, PAD_N = 1, PAD_T = 1, PAD_B = 1, PAD_L = 1,
                  PAD_R = 1;
    int T_OUT = (T + PAD_P + PAD_N - K_T) / stride_t + 1;
    int H_OUT = (H + PAD_T + PAD_B - K_H) / stride_h + 1;
    int W_OUT = (W + PAD_L + PAD_R - K_W) / stride_w + 1;

    aligned_vector<uint8_t> A(N * T * H * W * K);
    aligned_vector<int8_t> B(K * K_T * K_H * K_W);
    aligned_vector<int32_t> C_ref(N * T_OUT * H_OUT * W_OUT * K),
        C(C_ref.size());

    randFill(A, 0, 86);
    int32_t A_zero_point = 43;

    randFill(B, -16, 16);
    int32_t B_zero_point = 5;

    depthwise_3x3x3_pad_1_ref(
        N,
        T,
        H,
        W,
        K,
        stride_t,
        stride_h,
        stride_w,
        A_zero_point,
        A.data(),
        B.data(),
        C_ref.data());

    int32_t minimum = *min_element(C_ref.begin(), C_ref.end());
    int32_t maximum = *max_element(C_ref.begin(), C_ref.end());

    float C_multiplier = 255. / (maximum - minimum);

    aligned_vector<int32_t> col_offsets(K);
    aligned_vector<int32_t> bias(K);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);
    int32_t C_zero_point = 5;

    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());
    depthwise_3x3x3_pad_1_ref(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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

    Packed3x3x3ConvMatrix Bp(K, B.data());

    double ttot = 0;
    double bytes = double(NITER) *
        (K *
         (N * (2. * sizeof(int32_t) * T_OUT * H_OUT * W_OUT + T * H * W) +
          K_T * K_H * K_W));
    double ops =
        double(NITER) * N * T_OUT * H_OUT * W_OUT * K * K_T * K_H * K_W * 2;
    chrono::time_point<chrono::system_clock> t_begin, t_end;
    for (int i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush();

      t_begin = chrono::system_clock::now();
#pragma omp parallel
      {
#if _OPENMP
        int num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
#else
        int num_threads = 1;
        int tid = 0;
#endif
        depthwise_3x3x3_pad_1(
            N,
            T,
            H,
            W,
            K,
            stride_t,
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
      for (int t = 0; t < T_OUT; ++t) {
        for (int h = 0; h < H_OUT; ++h) {
          for (int w = 0; w < W_OUT; ++w) {
            for (int g = 0; g < K; ++g) {
              int32_t expected =
                  C_ref[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + g];
              int32_t actual =
                  C[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + g];
              if (expected != actual) {
                cerr << "Depthwise 3x3 results differ at (" << n << ", " << t
                     << ", " << h << ", " << w << ", " << g << "). expected "
                     << expected << " actual " << actual << endl;
                return -1;
              }
              assert(expected == actual);
            }
          } // w
        } // h
      } // t
    } // n

    // Report performance
    printf(
        "N = %d K = %d T = %d H = %d W = %d stride = %d\n",
        N,
        K,
        T,
        H,
        W,
        stride_h);
    printf("GB/s = %f Gops/s = %f\n", bytes / ttot / 1e9, ops / ttot / 1e9);

    ttot = 0;
    for (int i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush();

      t_begin = chrono::system_clock::now();
#pragma omp parallel
      {
#if _OPENMP
        int num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
#else
        int num_threads = 1;
        int tid = 0;
#endif
        depthwise_3x3x3_pad_1(
            N,
            T,
            H,
            W,
            K,
            stride_t,
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
            false /* fuse_relu */,
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
      for (int t = 0; t < T_OUT; ++t) {
        for (int h = 0; h < H_OUT; ++h) {
          for (int w = 0; w < W_OUT; ++w) {
            for (int g = 0; g < K; ++g) {
              uint8_t expected = C_uint8_ref
                  [(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + g];
              uint8_t actual =
                  C_uint8[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + g];
              if (expected != actual) {
                cerr << "Depthwise 3x3 results differ at (" << n << ", " << t
                     << ", " << h << ", " << w << ", " << g << "). expected "
                     << (int)expected << " actual " << (int)actual << endl;
                return -1;
              }
              assert(expected == actual);
            }
          } // w
        } // h
      } // t
    } // n

    // Report performance
    printf(
        "N = %d K = %d T = %d H = %d W = %d stride = %d with requantization "
        "fused\n",
        N,
        K,
        T,
        H,
        W,
        stride_h);
    printf("GB/s = %f Gops/s = %f\n", bytes / ttot / 1e9, ops / ttot / 1e9);
  } // for each shape

  return 0;
}
