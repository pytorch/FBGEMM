/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

#include "bench/AlignedVec.h"
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"
#include "src/RefImplementations.h"

using namespace std;

namespace fbgemm {

// From Xray OCR
// clang-format off
static vector<vector<int>> shapes = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  // N, K, H_in, W_in, stride, kernel
  {   1,  272,  47, 125, 1, 3 },
  {   1,  272,  47, 125, 1, 5 },
//  {   1,  272,  64, 125, 1, 3 },
//  {   1,  272,  66, 125, 1, 3 },
//  {   1,  272,  67, 100, 1, 3 },
//  {   1,  272,  75,  75, 1, 3 },
//   {   1,  272,  75,  76, 1, 3 },
//  {   1,  272,  75, 100, 1, 3 },
//  {   1,  272,  94,  75, 1, 3 },
//  {   1,  272, 109,  75, 1, 3 },
  {   1,  544,  24,  63, 1, 3 },
//  {   1,  544,  33,  63, 1, 3 },
//  {   1,  544,  34,  50, 1, 3 },
//  {   1,  544,  36,  63, 1, 3 },
//  {   1,  544,  38,  38, 1, 3 },
//  {   1,  544,  38,  40, 1, 3 },
  {   1,  544,  47,  38, 1, 3 },
  {   1, 1088,   7,   7, 1, 3 },
  {  51, 1088,   7,   7, 1, 3 },
  {   2, 1088,   7,   7, 1, 5 },
//  { 100, 1088,   7,   7, 1, 3 },

  {   1,  248,  93, 250, 2, 3 },
  {   1,  248,  93, 250, 2, 5 },
//  {   1,  248, 128, 250, 2, 3 },
//  {   1,  248, 133, 200, 2, 3 },
//  {   1,  248, 150, 150, 2, 3 },
  {   1,  248, 150, 151, 2, 3 },
//  {   1,  248, 150, 158, 2, 3 },
//  {   1,  248, 188, 150, 2, 3 },
//  {   1,  248, 225, 150, 2, 3 },
  {   1,  272,  47, 125, 2, 3 },
//  {   1,  272,  64, 125, 2, 3 },
//  {   1,  272,  66, 125, 2, 3 },
//  {   1,  272,  67, 100, 2, 3 },
//  {   1,  272,  75,  75, 2, 3 },
//  {   1,  272,  75,  76, 2, 3 },
  {   1,  272,  94,  75, 2, 3 },
  {   1,  544,  14,  14, 2, 3 },
  {  51,  544,  14,  14, 2, 3 },
//  { 100,  544,  14,  14, 2, 3 },

  {   1,    8,   4,   4, 1, 3 },
};

static vector<vector<int>> shapes_3d = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  // N, K, T_in, H_in, W_in, stride
  {   1,  32,   16,  28, 28, 1, },
  {   1, 128,    8,  14, 14, 2, },
  {   5,  16,   32,  56, 56, 1, },
  {   1,   8,    4,   4,  4, 1, },
};
// clang-format on

namespace {

class FBGemmDepthWiseTest : public testing::TestWithParam<tuple<bool, bool>> {};

// Two parameters are K (or Groups) and kernel_prod, i.e.,
// (output_channels)(kernel_prod)
// output_channels == Groups.
// For example, kernel_prod for 3x3 convolution is 9
class FBGemmDepthWisePackUnpackTest
    : public testing::TestWithParam<tuple<int, int>> {};

} // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDepthWiseTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDepthWisePackUnpackTest,
    ::testing::Combine(
        ::testing::ValuesIn({8, 16, 24, 32, 40, 64, 72}),
        ::testing::ValuesIn({1, 2, 3, 4, 5, 9, 10, 11, 27})));

TEST_P(FBGemmDepthWiseTest, Test3x3) {
  bool a_symmetric, b_symmetric;
  tie(a_symmetric, b_symmetric) = GetParam();

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
    int32_t A_zero_point = a_symmetric ? 0 : 43;

    randFill<int8_t>(B, -16, 16);
    int32_t B_zero_point = b_symmetric ? 0 : 5;

    aligned_vector<float> C_multiplier(1);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    aligned_vector<int32_t> col_offsets(K);
    aligned_vector<int32_t> bias(K);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col;
    if (!b_symmetric) {
      A_im2col.resize(MDim * KDim);
      im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());
    }

    conv_ref(conv_p, A.data(), A_zero_point, B.data(), C_ref.data());

    for (int g = 0; g < conv_p.G; ++g) {
      // Compute row offset
      if (!b_symmetric) {
        row_offsets_u8acc32_ref(
            MDim,
            KDimPerGroup,
            KDim,
            A_im2col.data() + g * KDimPerGroup,
            row_offsets.data());
      }

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
        a_symmetric ? nullptr : col_offsets.data(),
        bias.data(),
        false, /* fuse_relu */
        1.0f, /* act_scale * w_scale */
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
                << "Depthwise " << R << "x" << S << " results differ at (" << n
                << ", " << h << ", " << w << ", " << k << ").";
          }
        }
      }
    }
  } // for each shape
} // Test3x3

TEST_P(FBGemmDepthWiseTest, Test3x3x3) {
  bool a_symmetric, b_symmetric;
  tie(a_symmetric, b_symmetric) = GetParam();

  // 3x3x3 tests take a long time so for a symmetric quantization, we only
  // test with 2 shapes.
  for (auto shape : a_symmetric || b_symmetric
           ? vector<vector<int>>(shapes_3d.cbegin(), shapes_3d.cbegin() + 2)
           : shapes_3d) {
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

    conv_param_t<3> conv_p(
        N,
        K,
        K,
        {T, H, W},
        K,
        {K_T, K_H, K_W},
        {stride_t, stride_h, stride_w},
        {PAD_P, PAD_T, PAD_L, PAD_N, PAD_B, PAD_R});
    int T_OUT = conv_p.OUT_DIM[0];
    int H_OUT = conv_p.OUT_DIM[1];
    int W_OUT = conv_p.OUT_DIM[2];

    int MDim = N * T_OUT * H_OUT * W_OUT;
    int KDim = K_T * K_H * K_W * K;
    int KDimPerGroup = KDim / conv_p.G;

    aligned_vector<uint8_t> A(N * T * H * W * K);
    aligned_vector<int8_t> B(KDim);
    aligned_vector<int32_t> C_ref(MDim * K), C(C_ref.size());
    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = a_symmetric ? 0 : 43;

    randFill<int8_t>(B, -16, 16);
    int32_t B_zero_point = b_symmetric ? 0 : 5;

    aligned_vector<float> C_multiplier(1);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    aligned_vector<int32_t> col_offsets(K);
    aligned_vector<int32_t> bias(K);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col;
    if (!b_symmetric) {
      A_im2col.resize(MDim * KDim);
      im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());
    }

    conv_ref(conv_p, A.data(), A_zero_point, B.data(), C_ref.data());

    for (int g = 0; g < conv_p.G; ++g) {
      // Compute row offset
      if (!b_symmetric) {
        row_offsets_u8acc32_ref(
            MDim,
            KDimPerGroup,
            KDim,
            A_im2col.data() + g * KDimPerGroup,
            row_offsets.data());
      }

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

    PackedDepthWiseConvMatrix Bp(K, 3 * 3 * 3, B.data());

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
        C_multiplier[0],
        C_zero_point,
        C_uint8.data(),
        a_symmetric ? nullptr : col_offsets.data(),
        bias.data(),
        false, /* fuse_relu */
        1.0f, /* act_scale * w_scale */
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

    // Each row of G has a different range to really test per-channel
    // quantization.
    vector<int32_t> B_zero_point(K);
    for (auto k = 0; k < K; ++k) {
      aligned_vector<int8_t> Bk(R * S);
      // limit min, max to int8_t range
      randFill<int8_t>(Bk, -16 + k % 112, 16 + k % 112);
      copy(Bk.begin(), Bk.end(), B.begin() + k * R * S);

      B_zero_point[k] = 5 + k;
    }

    aligned_vector<float> C_multiplier(K);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    aligned_vector<int32_t> col_offsets(K);
    aligned_vector<int32_t> bias(K);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    // im2col to compute row offset later
    vector<int32_t> row_offsets(MDim);
    vector<uint8_t> A_im2col(MDim * KDim);
    im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());

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
          C_multiplier.data() + g,
          C_zero_point,
          A_zero_point,
          B_zero_point.data() + g,
          row_offsets.data(),
          col_offsets.data() + g,
          bias.data() + g,
          K);
    }

    PackedDepthWiseConvMatrix Bp(K, R * S, B.data());
    depthwise_2d_per_channel_quantization_same_pad(
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
        false, /* fuse_relu */
        nullptr, /* act_scale * w_scale */
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
                << "Depthwise " << R << "x" << S << " results differ at (" << n
                << ", " << h << ", " << w << ", " << k << ").";
          }
        }
      }
    }
  } // for each shape
} // Test3x3PerChannelQuantization

TEST(FBGemmDepthWiseTest, Test3x3x3PerChannelQuantization) {
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

    conv_param_t<3> conv_p(
        N,
        K,
        K,
        {T, H, W},
        K,
        {K_T, K_H, K_W},
        {stride_t, stride_h, stride_w},
        {PAD_P, PAD_T, PAD_L, PAD_N, PAD_B, PAD_R});
    int T_OUT = conv_p.OUT_DIM[0];
    int H_OUT = conv_p.OUT_DIM[1];
    int W_OUT = conv_p.OUT_DIM[2];

    int MDim = N * T_OUT * H_OUT * W_OUT;
    int KDim = K_T * K_H * K_W * K;
    int KDimPerGroup = KDim / conv_p.G;

    aligned_vector<uint8_t> A(N * T * H * W * K);
    aligned_vector<int8_t> B(KDim);
    aligned_vector<int32_t> C_ref(MDim * K), C(C_ref.size());
    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = 43;

    // Each row of G has a different range to really test per-channel
    // quantization.
    vector<int32_t> B_zero_point(K);
    for (auto k = 0; k < K; ++k) {
      aligned_vector<int8_t> Bk(K_T * K_H * K_W);
      // limit min, max to int8_t range
      randFill<int8_t>(Bk, -16 + k % 112, 16 + k % 112);
      copy(Bk.begin(), Bk.end(), B.begin() + k * K_T * K_H * K_W);

      B_zero_point[k] = 5 + k;
    }

    aligned_vector<float> C_multiplier(K);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    aligned_vector<int32_t> col_offsets(K);
    aligned_vector<int32_t> bias(K);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col(MDim * KDim);
    im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());

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
          C_multiplier.data() + g,
          C_zero_point,
          A_zero_point,
          B_zero_point.data() + g,
          row_offsets.data(),
          col_offsets.data() + g,
          bias.data() + g,
          K);
    }

    PackedDepthWiseConvMatrix Bp(K, 3 * 3 * 3, B.data());

    depthwise_3x3x3_per_channel_quantization_pad_1(
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
        B_zero_point.data(),
        Bp,
        C_multiplier.data(),
        C_zero_point,
        C_uint8.data(),
        col_offsets.data(),
        bias.data(),
        false, /* fuse_relu */
        nullptr, /* act_scale * w_scale */
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
              ASSERT_EQ(expected, actual)
                  << "Depthwise 3x3 results differ at (" << n << ", " << t
                  << ", " << h << ", " << w << ", " << k << ").";
            }
          } // w
        } // h
      } // t
    } // n
  } // for each shape
} // Test3x3PerChannelQuantization

TEST_P(FBGemmDepthWisePackUnpackTest, TestPackUnpack) {
  int K, kernel_prod;
  tie(K, kernel_prod) = GetParam();

  ASSERT_EQ(K % 8, 0)
      << "output channels (== groups) should be a multiple of 8";
  aligned_vector<int8_t> B(K * kernel_prod);
  randFill<int8_t>(B, -16, 16);

  aligned_vector<int8_t> BUnpacked(K * kernel_prod);

  PackedDepthWiseConvMatrix BPacked(K, kernel_prod, B.data());
  BPacked.unpack(BUnpacked.data());

  ASSERT_EQ(B, BUnpacked)
      << "Original and unpacked data elements are not the same";
} // TestPackUnpack

} // namespace fbgemm
