/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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
  // N, G, H_in, W_in, stride, kernel
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
  {  2, 1088,   7,   7, 1, 3 },
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
  // {  51,  544,  14,  14, 2, 3 },
//  { 100,  544,  14,  14, 2, 3 },

  {   1,    8,   4,   4, 1, 3 },
  // Tests for the shapes when OH/OW is less than padding
  {   1,  72,  1, 1, 2, 5 },
  {   1,  72,  7, 1, 2, 5 },
  {   1,  72,  1, 7, 2, 5 },
};

static vector<vector<int>> shapes_3d = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  // N, K, T_in, H_in, W_in, stride_t, stride_h, stride_w, K_T, K_H, K_W
  {   1,  32,   16,  28, 28, 1, 1, 1, 3, 3, 3, },
  {   1, 128,    8,  14, 14, 2, 2, 2, 3, 3, 3, },
  {   5,  16,   32,  56, 56, 1, 1, 1, 3, 3, 3, },
  {   1,   8,    4,   4,  4, 1, 1, 1, 3, 3, 3, },
  {   1,  32,   16,  28, 28, 1, 1, 1, 3, 5, 5, },
  {   1,  32,   16,  28, 28, 1, 2, 2, 3, 5, 5, },
  {   1,  32,   16,  28, 28, 1, 1, 1, 5, 5, 5, },
};
// clang-format on

namespace {

class FBGemmDepthWiseTest
    : public testing::TestWithParam<tuple<bool, bool, int>> {};

class FBGemmDepthWisePerChannelQuantizationTest
    : public testing::TestWithParam<int> {};

} // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDepthWiseTest,
    ::testing::Combine(
        ::testing::Bool(), // a_symmetric
        ::testing::Bool(), // b_symmetric
        ::testing::Values(1, 2))); // oc_per_g

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDepthWisePerChannelQuantizationTest,
    ::testing::Values(1, 2));

TEST_P(FBGemmDepthWiseTest, Test3DFloatOutput) {
  bool a_symmetric, b_symmetric;
  int oc_per_g;
  tie(a_symmetric, b_symmetric, oc_per_g) = GetParam();

  // 3D tests take a long time so for a symmetric quantization, we only
  // test with 2 shapes.
  for (auto shape : shapes_3d) {
    int N = shape[0];
    int G = shape[1];
    int T = shape[2];
    int H = shape[3];
    int W = shape[4];
    int stride_t = shape[5];
    int stride_h = shape[6];
    int stride_w = shape[7];
    int K_T = shape[8];
    int K_H = shape[9];
    int K_W = shape[10];
    int PAD_P = (K_T - 1) / 2, PAD_N = PAD_P, PAD_T = (K_H - 1) / 2,
        PAD_B = PAD_T, PAD_L = (K_W - 1) / 2, PAD_R = PAD_L;
    int OC = G * oc_per_g;

    conv_param_t<3> conv_p(
        N,
        G,
        OC,
        {T, H, W},
        G,
        {K_T, K_H, K_W},
        {stride_t, stride_h, stride_w},
        {PAD_P, PAD_T, PAD_L, PAD_N, PAD_B, PAD_R});
    int T_OUT = conv_p.OUT_DIM[0];
    int H_OUT = conv_p.OUT_DIM[1];
    int W_OUT = conv_p.OUT_DIM[2];

    int MDim = N * T_OUT * H_OUT * W_OUT;
    int KDim = K_T * K_H * K_W * G;
    int KDimPerGroup = KDim / G;

    aligned_vector<uint8_t> A(N * T * H * W * G);
    aligned_vector<int8_t> B(KDim * oc_per_g);
    aligned_vector<int8_t> B_tr(B.size());
    aligned_vector<int32_t> C_ref(MDim * OC), C(C_ref.size());
    aligned_vector<float> C_float_ref(C_ref.size()), C_float(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = a_symmetric ? 0 : 43;

    randFill<int8_t>(B, -16, 16);
    int32_t B_zero_point = b_symmetric ? 0 : 5;

    float Aint8_scale = 0.11;
    aligned_vector<float> Bint8_scale(1);
    aligned_vector<float> C_multiplier(1);
    randFill(Bint8_scale, 0.001234f / 2, 0.001234f * 3 / 2);
    for (int i = 0; i < Bint8_scale.size(); ++i) {
      C_multiplier[i] = Aint8_scale * Bint8_scale[i];
    }

    aligned_vector<int32_t> col_offsets(OC);
    aligned_vector<float> bias(OC);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40.f, 40.f);

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col;
    if (!b_symmetric) {
      A_im2col.resize(MDim * KDim);
      im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());
    }

    // reference implementation conv_ref expects weights to be in G (T R S C/G)
    // K/G
    transposeConvWeights(conv_p, B.data(), B_tr.data());
    conv_ref(conv_p, A.data(), A_zero_point, B_tr.data(), C_ref.data());

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
      requantize_u8acc32_float_output_ref(
          MDim,
          oc_per_g,
          OC,
          C_ref.data() + g * oc_per_g,
          C_float_ref.data() + g * oc_per_g,
          Aint8_scale,
          Bint8_scale.data(),
          A_zero_point,
          &B_zero_point,
          row_offsets.data(),
          col_offsets.data() + g * oc_per_g,
          bias.data() + g * oc_per_g,
          OC);
    }

    PackedDepthWiseConvMatrix Bp(OC, K_T * K_H * K_W, B.data());

    depthwise_3d_same_pad<QuantizationGranularity::TENSOR>(
        conv_p,
        A_zero_point,
        A.data(),
        &B_zero_point,
        Bp,
        C_multiplier.data(),
        0, // dummy
        C_float.data(),
        a_symmetric ? nullptr : col_offsets.data(),
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
            for (int k = 0; k < OC; ++k) {
              float expected = C_float_ref
                  [(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * OC + k];
              float actual =
                  C_float[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * OC + k];
              EXPECT_NEAR(actual, expected, 1e-5 * std::abs(expected) + 1e-5)
                  << "Depthwise 3D results differ at (" << n << ", " << t
                  << ", " << h << ", " << w << ", " << k << ").";
            }
          } // w
        } // h
      } // t
    } // n
  } // for each shape
} // Test3D

TEST_P(
    FBGemmDepthWisePerChannelQuantizationTest,
    Test3DPerChannelQuantizationFloatOutput) {
  int oc_per_g = GetParam();

  for (auto shape : shapes_3d) {
    int N = shape[0];
    int G = shape[1];
    int T = shape[2];
    int H = shape[3];
    int W = shape[4];
    int stride_t = shape[5];
    int stride_h = shape[6];
    int stride_w = shape[7];
    int K_T = shape[8];
    int K_H = shape[9];
    int K_W = shape[10];
    int PAD_P = (K_T - 1) / 2, PAD_N = PAD_P, PAD_T = (K_H - 1) / 2,
        PAD_B = PAD_T, PAD_L = (K_W - 1) / 2, PAD_R = PAD_L;
    int OC = G * oc_per_g;

    conv_param_t<3> conv_p(
        N,
        G,
        OC,
        {T, H, W},
        G,
        {K_T, K_H, K_W},
        {stride_t, stride_h, stride_w},
        {PAD_P, PAD_T, PAD_L, PAD_N, PAD_B, PAD_R});
    int T_OUT = conv_p.OUT_DIM[0];
    int H_OUT = conv_p.OUT_DIM[1];
    int W_OUT = conv_p.OUT_DIM[2];

    int MDim = N * T_OUT * H_OUT * W_OUT;
    int KDim = K_T * K_H * K_W * G;
    int KDimPerGroup = KDim / G;

    aligned_vector<uint8_t> A(N * T * H * W * G);
    aligned_vector<int8_t> B(KDim * oc_per_g);
    aligned_vector<int8_t> B_tr(B.size());
    aligned_vector<int32_t> C_ref(MDim * OC), C(C_ref.size());
    aligned_vector<float> C_float_ref(C_ref.size()), C_float(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = 43;

    // Each row of G has a different range to really test per-channel
    // quantization.
    vector<int32_t> B_zero_point(OC);
    for (auto k = 0; k < OC; ++k) {
      aligned_vector<int8_t> Bk(K_T * K_H * K_W);
      // limit min, max to int8_t range
      randFill<int8_t>(Bk, -16 + k % 112, 16 + k % 112);
      copy(Bk.begin(), Bk.end(), B.begin() + k * K_T * K_H * K_W);

      B_zero_point[k] = 5 + k;
    }

    float Aint8_scale = 0.11;
    aligned_vector<float> Bint8_scale(OC);
    aligned_vector<float> C_multiplier(OC);
    randFill(Bint8_scale, 0.001234f / 2, 0.001234f * 3 / 2);
    for (int i = 0; i < Bint8_scale.size(); ++i) {
      C_multiplier[i] = Aint8_scale * Bint8_scale[i];
    }

    aligned_vector<int32_t> col_offsets(OC);
    aligned_vector<float> bias(OC);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40.f, 40.f);

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col(MDim * KDim);
    im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());

    // reference implementation conv_ref expects weights to be in G (T R S C/G)
    // K/G
    transposeConvWeights(conv_p, B.data(), B_tr.data());
    conv_ref(conv_p, A.data(), A_zero_point, B_tr.data(), C_ref.data());

    for (int g = 0; g < conv_p.G; ++g) {
      // Compute row offset
      row_offsets_u8acc32_ref(
          MDim,
          KDimPerGroup,
          KDim,
          A_im2col.data() + g * KDimPerGroup,
          row_offsets.data());

      // Requantization
      requantize_u8acc32_float_output_ref(
          MDim,
          oc_per_g,
          OC,
          C_ref.data() + g * oc_per_g,
          C_float_ref.data() + g * oc_per_g,
          Aint8_scale,
          Bint8_scale.data() + g * oc_per_g,
          A_zero_point,
          B_zero_point.data() + g * oc_per_g,
          row_offsets.data(),
          col_offsets.data() + g * oc_per_g,
          bias.data() + g * oc_per_g,
          1);
    }

    PackedDepthWiseConvMatrix Bp(OC, K_T * K_H * K_W, B.data());

    depthwise_3d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
        conv_p,
        A_zero_point,
        A.data(),
        B_zero_point.data(),
        Bp,
        C_multiplier.data(),
        0, // dummy
        C_float.data(),
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
            for (int k = 0; k < OC; ++k) {
              float expected = C_float_ref
                  [(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * OC + k];
              float actual =
                  C_float[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * OC + k];
              EXPECT_NEAR(actual, expected, 1e-5 * std::abs(expected) + 1e-5)
                  << "Depthwise 3D results differ at (" << n << ", " << t
                  << ", " << h << ", " << w << ", " << k << ").";
            }
          } // w
        } // h
      } // t
    } // n
  } // for each shape
} // Test3DPerChannelQuantization

} // namespace fbgemm
