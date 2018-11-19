/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "I8DepthwiseTest.h"

#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

#include "TestUtils.h"
#include "bench/AlignedVec.h"
#include "bench/BenchUtils.h"
#include "src/FbgemmI8Depthwise.h"
#include "src/RefImplementations.h"

using namespace std;

namespace fbgemm {

// From Xray OCR
static vector<vector<int>> shapes = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  // N, K, H_in, W_in, stride
  {   1,  272,  47, 125, 1, },
//  {   1,  272,  64, 125, 1, },
//  {   1,  272,  66, 125, 1, },
//  {   1,  272,  67, 100, 1, },
//  {   1,  272,  75,  75, 1, },
  {   1,  272,  75,  76, 1, },
//  {   1,  272,  75, 100, 1, },
//  {   1,  272,  94,  75, 1, },
//  {   1,  272, 109,  75, 1, },
  {   1,  544,  24,  63, 1, },
//  {   1,  544,  33,  63, 1, },
//  {   1,  544,  34,  50, 1, },
//  {   1,  544,  36,  63, 1, },
//  {   1,  544,  38,  38, 1, },
//  {   1,  544,  38,  40, 1, },
  {   1,  544,  47,  38, 1, },
  {   1, 1088,   7,   7, 1, },
  {  51, 1088,   7,   7, 1, },
//  { 100, 1088,   7,   7, 1, },

  {   1,  248,  93, 250, 2, },
//  {   1,  248, 128, 250, 2, },
//  {   1,  248, 133, 200, 2, },
//  {   1,  248, 150, 150, 2, },
  {   1,  248, 150, 151, 2, },
//  {   1,  248, 150, 158, 2, },
//  {   1,  248, 188, 150, 2, },
//  {   1,  248, 225, 150, 2, },
  {   1,  272,  47, 125, 2, },
//  {   1,  272,  64, 125, 2, },
//  {   1,  272,  66, 125, 2, },
//  {   1,  272,  67, 100, 2, },
//  {   1,  272,  75,  75, 2, },
//  {   1,  272,  75,  76, 2, },
  {   1,  272,  94,  75, 2, },
  {   1,  544,  14,  14, 2, },
  {  51,  544,  14,  14, 2, },
//  { 100,  544,  14,  14, 2, },

  {   1,    8,   4,   4, 1, },
};

TEST(FBGemmDepthWiseTest, Test3x3) {
  for (auto shape : shapes) {
    int N = shape[0];
    int K = shape[1];
    int H = shape[2];
    int W = shape[3];
    int stride_h = shape[4];
    int stride_w = stride_h;
    constexpr int R = 3, S = 3;
    constexpr int PAD_T = 1, PAD_B = 1, PAD_L = 1, PAD_R = 1;
    int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
    int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;

    aligned_vector<uint8_t> A(N * H * W * K);
    aligned_vector<int8_t> B(K * R * S);
    aligned_vector<int32_t> C_ref(N * H_OUT * W_OUT * K), C(C_ref.size());

    randFill(A, 0, 86);
    int32_t A_zero_point = 43;

    randFill(B, -16, 16);
    int32_t B_zero_point = 5;

    depthwise_3x3_pad_1_ref(
        N,
        H,
        W,
        K,
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
    depthwise_3x3_pad_1_ref(
        N,
        H,
        W,
        K,
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

    Packed3x3ConvMatrix Bp(K, B.data());

    depthwise_3x3_pad_1(
        N, H, W, K, stride_h, stride_w, A_zero_point, A.data(), Bp, C.data());

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H_OUT; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          for (int k = 0; k < K; ++k) {
            int32_t expected = C_ref[((n * H_OUT + h) * W_OUT + w) * K + k];
            int32_t actual = C[((n * H_OUT + h) * W_OUT + w) * K + k];
            EXPECT_EQ(expected, actual)
                << "Depthwise 3x3 results differ at (" << n << ", " << h << ", "
                << w << ", " << k << ").";
          }
        }
      }
    }

    depthwise_3x3_pad_1(
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
        C_multiplier,
        C_zero_point,
        C_uint8.data(),
        col_offsets.data(),
        bias.data(),
        0,
        1);

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H_OUT; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          for (int k = 0; k < K; ++k) {
            int32_t expected =
                C_uint8_ref[((n * H_OUT + h) * W_OUT + w) * K + k];
            int32_t actual = C_uint8[((n * H_OUT + h) * W_OUT + w) * K + k];
            EXPECT_EQ(expected, actual)
                << "Depthwise 3x3 results differ at (" << n << ", " << h << ", "
                << w << ", " << k << ").";
          }
        }
      }
    }
  } // for each shape
} // Test3x3

TEST(FBGemmDepthWiseTest, Test3x3x3) {
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
        C.data());

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int t = 0; t < T_OUT; ++t) {
        for (int h = 0; h < H_OUT; ++h) {
          for (int w = 0; w < W_OUT; ++w) {
            for (int k = 0; k < K; ++k) {
              int32_t expected =
                  C_ref[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + k];
              int32_t actual =
                  C[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + k];
              ASSERT_EQ(expected, actual)
                  << "Depthwise 3x3 results differ at (" << n << ", " << t
                  << ", " << h << ", " << w << ", " << k << ") " << shape[0]
                  << " " << shape[1] << " " << shape[2] << " " << shape[3]
                  << " " << shape[4] << " " << shape[5];
            }
          } // w
        } // h
      } // t
    } // n

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
        0,
        1);

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int t = 0; t < T_OUT; ++t) {
        for (int h = 0; h < H_OUT; ++h) {
          for (int w = 0; w < W_OUT; ++w) {
            for (int k = 0; k < K; ++k) {
              int32_t expected = C_uint8_ref
                  [(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + k];
              int32_t actual =
                  C_uint8[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + k];
              EXPECT_EQ(expected, actual)
                  << "Depthwise 3x3 results differ at (" << n << ", " << t
                  << ", " << h << ", " << w << ", " << k << ").";
            }
          } // w
        } // h
      } // t
    } // n
  } // for each shape
} // Test3x3x3

TEST(FBGemmDepthWiseTest, Test3x3PerChannelQuantization) {
  for (auto shape : shapes) {
    int N = shape[0];
    int K = shape[1];
    int H = shape[2];
    int W = shape[3];
    int stride_h = shape[4];
    int stride_w = stride_h;
    constexpr int R = 3, S = 3;
    constexpr int PAD_T = 1, PAD_B = 1, PAD_L = 1, PAD_R = 1;
    int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
    int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;

    aligned_vector<uint8_t> A(N * H * W * K);
    aligned_vector<int8_t> B(K * R * S);
    int32_t C_num_rows = N * H_OUT * W_OUT;
    aligned_vector<int32_t> C_ref(C_num_rows * K), C(C_ref.size());

    randFill(A, 0, 86);
    int32_t A_zero_point = 43;

    // Each row of G has a different range to really test per-channel
    // quantization.
    vector<int32_t> B_zero_point(K);
    for (auto k = 0; k < K; ++k) {
      aligned_vector<int8_t> Bk(R * S);
      randFill(Bk, -16 + k, 16 + k);
      copy(Bk.begin(), Bk.end(), B.begin() + k * R * S);

      B_zero_point[k] = 5 + k;
    }

    depthwise_3x3_pad_1_ref(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A.data(),
        B.data(),
        C_ref.data());

    aligned_vector<int32_t> C_ref_transpose(C_ref);
    transpose_matrix(C_ref.data(), C_num_rows, K);
    vector<float> C_multiplier(K);
    for (auto k = 0; k < K; ++k) {
      auto C_ref_k_begin = C_ref_transpose.begin() + k * C_num_rows;
      auto C_ref_k_end = C_ref_k_begin + C_num_rows;
      int32_t minimum = *min_element(C_ref_k_begin, C_ref_k_end);
      int32_t maximum = *max_element(C_ref_k_begin, C_ref_k_end);
      C_multiplier[k] = 255. / (maximum - minimum);
      cerr << "k " << k << " minimum " << minimum << " maximum " << maximum
           << " multiplier " << C_multiplier[k] << endl;
    }
    int32_t C_zero_point = 5;

    aligned_vector<int32_t> col_offsets(K);
    aligned_vector<int32_t> bias(K);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());
    depthwise_3x3_per_channel_quantization_pad_1_ref(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A.data(),
        B_zero_point.data(),
        B.data(),
        C_multiplier.data(),
        C_zero_point,
        C_uint8_ref.data(),
        col_offsets.data(),
        bias.data());

    Packed3x3ConvMatrix Bp(K, B.data());

    depthwise_3x3_per_channel_quantization_pad_1(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A.data(),
        B_zero_point.data(),
        Bp,
        C_multiplier.data(),
        C_zero_point,
        C_uint8.data(),
        col_offsets.data(),
        bias.data(),
        0,
        1);

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H_OUT; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          for (int k = 0; k < K; ++k) {
            int32_t expected =
                C_uint8_ref[((n * H_OUT + h) * W_OUT + w) * K + k];
            int32_t actual = C_uint8[((n * H_OUT + h) * W_OUT + w) * K + k];
            EXPECT_EQ(expected, actual)
                << "Depthwise 3x3 results differ at (" << n << ", " << h << ", "
                << w << ", " << k << ").";
          }
        }
      }
    }
  } // for each shape
} // Test3x3PerChannelQuantization

} // namespace fbgemm
