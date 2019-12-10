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

#include "./AlignedVec.h"
#include "./BenchUtils.h"
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"
#include "fbgemm/Utils.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

int main() {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif

  // From Xray OCR
  // clang-format off
  vector<vector<int>> shapes = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    // N,  K, H_in, W_in, stride, kernel
    {   1,  272,  47, 125, 1, 3, },
    {   1,  272,  47, 125, 1, 5, },
    {   1,  272,  64, 125, 1, 3, },
    {   1,  272,  66, 125, 1, 3, },
    {   1,  272,  67, 100, 1, 3, },
    {   1,  272,  71, 125, 1, 3, },
    {   1,  272,  74, 125, 1, 3, },
    {   1,  272,  75,  75, 1, 3, },
    {   1,  272,  75,  76, 1, 3, },
    {   1,  272,  75,  79, 1, 3, },
    {   1,  272,  75,  85, 1, 3, },
    {   1,  272,  75, 100, 1, 3, },
    {   1,  272,  75, 103, 1, 3, },
    {   1,  272,  75, 111, 1, 3, },
    {   1,  272,  75, 113, 1, 3, },
    {   1,  272,  94,  75, 1, 3, },
    {   1,  272, 109,  75, 1, 3, },
    {   1,  272, 113,  75, 1, 3, },
    {   1,  272, 117,  75, 1, 3, },
    {   1,  544,  24,  63, 1, 3, },
    {   1,  544,  32,  63, 1, 3, },
    {   1,  544,  33,  63, 1, 3, },
    {   1,  544,  34,  50, 1, 3, },
    {   1,  544,  36,  63, 1, 3, },
    {   1,  544,  37,  63, 1, 3, },
    {   1,  544,  38,  38, 1, 3, },
    {   1,  544,  38,  40, 1, 3, },
    {   1,  544,  38,  43, 1, 3, },
    {   1,  544,  38,  50, 1, 3, },
    {   1,  544,  38,  52, 1, 3, },
    {   1,  544,  38,  56, 1, 3, },
    {   1,  544,  38,  57, 1, 3, },
    {   1,  544,  47,  38, 1, 3, },
    {   1,  544,  55,  38, 1, 3, },
    {   1,  544,  57,  38, 1, 3, },
    {   1,  544,  59,  38, 1, 3, },
    {   1, 1088,   7,   7, 1, 3, },
    {  51, 1088,   7,   7, 1, 3, },
    {  59, 1088,   7,   7, 1, 3, },
    {  70, 1088,   7,   7, 1, 3, },
    {  71, 1088,   7,   7, 1, 3, },
    {  77, 1088,   7,   7, 1, 3, },
    {  79, 1088,   7,   7, 1, 3, },
    {  84, 1088,   7,   7, 1, 3, },
    {  85, 1088,   7,   7, 1, 3, },
    {  89, 1088,   7,   7, 1, 3, },
    {  93, 1088,   7,   7, 1, 3, },
    {  96, 1088,   7,   7, 1, 3, },
    { 100, 1088,   7,   7, 1, 3, },

    {   1,  248,  93, 250, 2, 3, },
    {   1,  248, 128, 250, 2, 3, },
    {   1,  248, 132, 250, 2, 3, },
    {   1,  248, 131, 250, 2, 3, },
    {   1,  248, 133, 200, 2, 3, },
    {   1,  248, 141, 250, 2, 3, },
    {   1,  248, 148, 250, 2, 3, },
    {   1,  248, 150, 150, 2, 3, },
    {   1,  248, 150, 151, 2, 3, },
    {   1,  248, 150, 158, 2, 3, },
    {   1,  248, 150, 169, 2, 3, },
    {   1,  248, 150, 200, 2, 3, },
    {   1,  248, 150, 205, 2, 3, },
    {   1,  248, 150, 221, 2, 3, },
    {   1,  248, 150, 225, 2, 3, },
    {   1,  248, 188, 150, 2, 3, },
    {   1,  248, 218, 150, 2, 3, },
    {   1,  248, 225, 150, 2, 3, },
    {   1,  248, 234, 150, 2, 3, },
    {   1,  272,  47, 125, 2, 3, },
    {   1,  272,  64, 125, 2, 3, },
    {   1,  272,  66, 125, 2, 3, },
    {   1,  272,  67, 100, 2, 3, },
    {   1,  272,  71, 125, 2, 3, },
    {   1,  272,  74, 125, 2, 3, },
    {   1,  272,  75,  75, 2, 3, },
    {   1,  272,  75,  76, 2, 3, },
    {   1,  272,  75,  79, 2, 3, },
    {   1,  272,  75,  85, 2, 3, },
    {   1,  272,  75, 100, 2, 3, },
    {   1,  272,  75, 103, 2, 3, },
    {   1,  272,  75, 111, 2, 3, },
    {   1,  272,  75, 113, 2, 3, },
    {   1,  272,  94,  75, 2, 3, },
    {   1,  272, 109,  75, 2, 3, },
    {   1,  272, 113,  75, 2, 3, },
    {   1,  272, 117,  75, 2, 3, },
    {   1,  544,  14,  14, 2, 3, },
    {  51,  544,  14,  14, 2, 3, },
    {  59,  544,  14,  14, 2, 3, },
    {  70,  544,  14,  14, 2, 3, },
    {  71,  544,  14,  14, 2, 3, },
    {  77,  544,  14,  14, 2, 3, },
    {  79,  544,  14,  14, 2, 3, },
    {  84,  544,  14,  14, 2, 3, },
    {  85,  544,  14,  14, 2, 3, },
    {  89,  544,  14,  14, 2, 3, },
    {  93,  544,  14,  14, 2, 3, },
    {  96,  544,  14,  14, 2, 3, },
    { 100,  544,  14,  14, 2, 3, },

    {   1,   16, 112, 112, 1, 3, },
    {   1,   24,  56,  56, 1, 3, },
    {   1,   96, 112, 112, 2, 3, },
    {   1,  192,  28,  28, 1, 3, },
    {   1,   96,  28,  28, 1, 5, },
    {   1,  144,  56,  56, 2, 5, },
    {   1,  192,  28,  28, 1, 5, },
    {   1,  192,  28,  28, 2, 5, },
    {   1,  192,  14,  14, 1, 5, },
    {   1,  336,  14,  14, 1, 5, },
    {   1,  384,  14,  14, 1, 5, },
    {   1,  672,  14,  14, 1, 5, },
    {   1,  672,  14,  14, 2, 5, },
    {   1, 1104,   7,   7, 1, 5, },

    {   1,   32, 112, 112, 1, 3, },
    {   1,  144,  56,  56, 1, 3, },
    {   1,  240,  28,  28, 2, 3, },
    {   1,  480,  14,  14, 1, 3, },
    {   1, 1152,   7,   7, 1, 3, },
    {   1,  240,  28,  28, 1, 5, },
    {   1,  480,  14,  14, 1, 5, },
    {   1,  576,  14,  14, 1, 5, },
    {   1,  768,  14,  14, 2, 5, },
    {   1, 1104,   7,   7, 1, 3, },
    {   1, 1152,   7,   7, 1, 5, },

    {   1,   32, 400, 400, 1, 3, },
    {   1,   96, 400, 400, 2, 3, },
    {   1,  144, 200, 200, 1, 3, },
    {   1,  144, 200, 200, 2, 3, },
    {   1,  192, 100, 100, 1, 3, },
    {   1,  192, 100, 100, 2, 3, },
    {   1,  384,  50,  50, 1, 3, },
    {   1,  576,  50,  50, 1, 3, },
  };
  // clang-format on

  // Depthwise is memory BW bound so we want to flush LLC.
  bool flush = true;
  std::vector<char> llc;
  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 16;

  for (auto shape : shapes) {
    int N = shape[0];
    int K = shape[1];
    int H = shape[2];
    int W = shape[3];
    int stride_h = shape[4];
    int stride_w = stride_h;
    int R = shape[5];
    int S = R;
    int PAD_T = (R - 1) / 2, PAD_B = (R - 1) / 2, PAD_L = (S - 1) / 2,
        PAD_R = (S - 1) / 2;

    conv_param_t<2> conv_p(
        N,
        K,
        K,
        {H, W},
        K,
        {R, S},
        {stride_h, stride_w},
        {PAD_T, PAD_L, PAD_B, PAD_R});
    int H_OUT = conv_p.OUT_DIM[0];
    int W_OUT = conv_p.OUT_DIM[1];

    int MDim = N * H_OUT * W_OUT;
    int KDim = R * S * K;
    int KDimPerGroup = KDim / conv_p.G;

    aligned_vector<uint8_t> A(N * H * W * K);
    aligned_vector<int8_t> B(KDim);
    aligned_vector<int32_t> C_ref(MDim * K), C(C_ref.size());
    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = 43;

    randFill<int8_t>(B, -16, 16);
    int32_t B_zero_point = 5;

    aligned_vector<float> C_multiplier(1);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col(MDim * KDim);
    im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());

    aligned_vector<int32_t> col_offsets(K);
    aligned_vector<int32_t> bias(K);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    conv_ref(conv_p, A.data(), A_zero_point, B.data(), C_ref.data());

    for (int g = 0; g < conv_p.G; ++g) {
      // Compute row offset
      row_offsets_u8acc32_ref(
          MDim,
          KDimPerGroup,
          KDim,
          A_im2col.data() + g * KDimPerGroup,
          row_offsets.data());

      // Requantization
      requantize_u8acc32_ref(
          MDim,
          1,
          conv_p.G,
          C_ref.data() + g,
          C_uint8_ref.data() + g,
          C_multiplier.data(),
          C_zero_point,
          A_zero_point,
          &B_zero_point,
          row_offsets.data(),
          col_offsets.data() + g,
          bias.data() + g,
          K);
    }

    PackedDepthWiseConvMatrix Bp(K, R * S, B.data());

    double bytes =
        (K * (N * (2. * sizeof(int32_t) * H_OUT * W_OUT + H * W) + R * S));
    double ops = 2.0 * N * H_OUT * W_OUT * K * R * S;

    double ttot = measureWithWarmup(
        [&]() {
          int num_threads = fbgemm_get_num_threads();
          int tid = fbgemm_get_thread_num();

          depthwise_2d_same_pad(
              N,
              H,
              W,
              K,
              stride_h,
              stride_w,
              A_zero_point,
              A.data(),
              B_zero_point,
              Bp,
              C_multiplier[0],
              C_zero_point,
              C_uint8.data(),
              col_offsets.data(),
              bias.data(),
              false, /* fuse_relu */
              1.0f, /* act_scale * w_scale */
              tid,
              num_threads);
        },
        NWARMUP,
        NITER,
        [&]() {
          if (flush) {
            llc_flush(llc);
          }
        },
        true /*useOpenMP*/);

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H_OUT; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          for (int g = 0; g < K; ++g) {
            uint8_t expected =
                C_uint8_ref[((n * H_OUT + h) * W_OUT + w) * K + g];
            uint8_t actual = C_uint8[((n * H_OUT + h) * W_OUT + w) * K + g];
            if (expected != actual) {
              cerr << "Depthwise 3x3 results differ at (" << n << ", " << h
                   << ", " << w << ", " << g << "). expected " << (int)expected
                   << " actual " << (int)actual << endl;
              return -1;
            }
            assert(expected == actual);
          }
        }
      }
    }

    // Report performance
    printf(
        "N = %d K = %d H = %d W = %d stride = %d R = %d\n",
        N,
        K,
        H,
        W,
        stride_h,
        R);
    printf("GB/s = %f Gops/s = %f\n", bytes / ttot / 1e9, ops / ttot / 1e9);
  } // for each shape

  return 0;
}
