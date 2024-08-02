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
#include <cstdio>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "bench/AlignedVec.h"
#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"
#include "src/DirectConv.h"
#include "src/OptimizedKernelsAvx2.h"
#include "src/RefImplementations.h"

using namespace std;

namespace fbgemm {

// From Xray OCR
// clang-format off
//  conv_param_t<>(N, IC, OC, H, W, G,
//		/* kern */ {kernel1, kernel2}, /* stride */ {stride1, stride2}, /*
//padding */ {pad, pad, pad, pad},
//    /* dialation */ {1, 1}, /* otpt_pad */ {0,0}, /* trans */ transpose),
// 2D conv shapes
  vector<conv_param_t<2>> shapes = {
    // MB, IC, OC, IH, IW, G, KH, KW, stride_h, stride_w,
    // pad_h_top, pad_w_left, pad_h_bottom, pad_w_right,
    // (dilation_h, dilation_w, output_padding_h, output_padding_w, tranpose)
    // 2D convolutions
    // regular

    //  Ferraris Model
    //  Data from -
    //  https://docs.google.com/spreadsheets/d/1VM-nglZl-pSwBdgYm3VbeLRcORc5y_vTRl9anRCUSDQ/edit#gid=1776750723
    //  conv_param_t<>(N, IC, OC, H, W, G,
    //		/* kern */ {kernel1, kernel2}, /* stride */ {stride1, stride2}, /*
    //padding */ {pad, pad, pad, pad},
    //    /* dialation */ {1, 1}, /* otpt_pad */ {0,0}, /* trans */ transpose),

    conv_param_t<>(1, 128, 128,     {2, 257}, 1, {2, 6}, {1, 2}, {0, 0, 0, 0},     {1, 1}, {0, 0}, false),
    conv_param_t<>(1, 16, 16,     {2, 126}, 1, {2, 6}, {1, 2}, {0, 0, 0, 0},     {1, 1}, {0, 0}, false),
    conv_param_t<>(1, 64, 64,     {2, 257}, 1, {2, 6}, {1, 2}, {0, 0, 0, 0},     {1, 1}, {0, 0}, false),
  };

vector<conv_param_t<2>> shapes_trans = {
    conv_param_t<>(1, 256, 176, {2, 4}, 1,   {2, 6}, {1, 2}, {0, 0, 0, 0},
    {1, 1}, {0, 0}, true),
    conv_param_t<>(1, 128, 128, {4, 12}, 1,   {2, 6}, {1, 1}, {0, 0, 0, 0},
    {1, 1}, {0, 0}, true),
    conv_param_t<>(1, 512, 64, {4, 50}, 1,   {2, 6}, {1, 1}, {0, 0, 0, 0},
    {1, 1}, {0, 0}, true),

};

namespace {
/*
class FBGemmDirectConvTest
    : public testing::TestWithParam<tuple<bool, bool, int>> {};
*/
class FBGemmDirectConvTransTest
    : public testing::TestWithParam<tuple<bool, bool, int>> {};

class FBGemmDirectConvTransFbgemmTest
    : public testing::TestWithParam<tuple<bool, bool, int>> {};

} // namespace

template <int SPATIAL_DIM>
void transposeConvWeights_KwIchO8I4(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest) {
  int G = conv_p.G;
  int IC_per_G = conv_p.IC / conv_p.G;
  int OC_per_G = conv_p.OC / conv_p.G;

  int filter_prod = std::accumulate(
      conv_p.K.begin(),
      conv_p.K.begin() + SPATIAL_DIM,
      1,
      std::multiplies<int>());
  // Transforms weights from  G K/G (T R S C/G) to G (T R S C/G) K/G format.
  // the transposed weight layout: W[oc/8][h][w][ic/4][8][4]
  for (int g = 0; g < G; ++g) {
    for (int k = 0; k < OC_per_G; ++k) {
      for (int f = 0; f < filter_prod; ++f) {
        for (int c = 0; c < IC_per_G; ++c) {
          int ocB = k / 8;
          int ocb = k % 8;
          int icB = c / 4;
          int icb = c % 4;
          dest
              [(((ocB * filter_prod + f) * (IC_per_G / 4) + icB) * 8 + ocb) *
                   4 +
               icb] =
                  src[((g * OC_per_G + k) * filter_prod + f) * IC_per_G + c];
        }
      }
    }
  }
}

void directConvRowSum(
    const conv_param_t<2>& conv_p,
    uint8_t* A,
    int32_t* inSum,
    int32_t* rowSum) {
  int IN0 = conv_p.IN_DIM[0];
  int IN1 = conv_p.IN_DIM[1];
  int IC = conv_p.IC;
  int K0 = conv_p.K[0];
  int K1 = conv_p.K[1];
  int OUT0 = conv_p.OUT_DIM[0];
  int OUT1 = conv_p.OUT_DIM[1];
  int stride = conv_p.stride[1];

  memset(rowSum, 0, sizeof(int32_t) * OUT0 * OUT1);
  for (int ih = 0; ih < IN0; ++ih)
    for (int iw = 0; iw < IN1; ++iw) {
      inSum[ih * IN1 + iw] = reduceAvx2(A + ih * IN1 * IC + iw * IC, IC);
  }


  for (int ih = 0; ih < IN0; ++ih)
    for (int iw = 0; iw < IN1; iw++) {
      for (int r = 0; r < K0; ++r) {
        for (int s = 0; s < K1; ++s) {
          rowSum[(ih + r) * OUT1 + iw * stride + s] += inSum[ih * IN1 + iw];
        }
      }
    }
  /*
    compare_buffers(
        rowSum,
        rowoffsets,
        OUT0,
        OUT1,
        OUT1,
        5);
  */
}


void col_offsets_with_zero_pt_s8acc32_DirectConvT_ref(
    const conv_param_t<2>& conv_p,
    const int8_t* Bint8,
    const int32_t* B_zero_point,
    int32_t* col_offsets,
    int ncols_per_quant_group) {
  int IC = conv_p.IC;
  int OC = conv_p.OC;
  array<int, 2> IN_DIM = conv_p.IN_DIM;
  array<int, 2> OUT_DIM = conv_p.OUT_DIM;
  array<int, 2> K = conv_p.K;
  array<int, 2> stride = conv_p.stride;

  int MDim = conv_p.MB * OUT_DIM[0] * OUT_DIM[1];
  int NDim = conv_p.OC / conv_p.G;
  // int KDim = K[0] * K[1] * conv_p.IC;

  std::memset(col_offsets, 0, MDim * NDim);
  vector<int> count(MDim * NDim, 0);
  for (int oc = 0; oc < OC; oc++) {
    for (int ih = 0; ih < IN_DIM[0]; ih++) {
      for (int iw = 0; iw < IN_DIM[1]; iw++) {
        for (int kh = 0; kh < K[0]; kh++) {
          for (int kw = 0; kw < K[1]; kw++) {
            for (int ic = 0; ic < IC; ic++) {
              int oh = ih * stride[0] + kh;
              int ow = iw * stride[1] + kw;
              col_offsets[(oh * OUT_DIM[1] + ow) * OC + oc] += Bint8
                  [(((((oc / 8) * K[0] + kh) * K[1] + kw) * (IC / 4) + ic / 4) *
                        8 +
                    (oc % 8)) *
                       4 +
                   (ic % 4)];
              count[(oh * OUT_DIM[1] + ow) * OC + oc]++;
            }
          }
        }
      }
    }
  }

  for (int oc = 0; oc < OC; oc++) {
    for (int oh = 0; oh < OUT_DIM[0]; oh++) {
      for (int ow = 0; ow < OUT_DIM[1]; ow++) {
        col_offsets[(oh * OUT_DIM[1] + ow) * OC + oc] -=
            B_zero_point[oc / ncols_per_quant_group] *
            count[(oh * OUT_DIM[1] + ow) * OC + oc];
      }
    }
  }
}


void QuantizeDirectConv_ref(
    const conv_param_t<2>& conv_p,
    aligned_vector<uint8_t> Aint8,
    aligned_vector<int8_t> Bint8,
    aligned_vector<int32_t>& Cint32_ref,
    aligned_vector<uint8_t>& Cint8_ref,
    int32_t Aint8_zero_point,
    aligned_vector<float> C_multiplier,
    int32_t C_zero_point,
    aligned_vector<int32_t> Bint8_zero_point) {
  int im_out_dim = accumulate(
      conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());
  int kernel_dim =
      accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());

  aligned_vector<int8_t> Bint8_tr(
      kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

  transposeConvWeights<2>(conv_p, Bint8.data(), Bint8_tr.data());
  conv_ref(
      conv_p,
      Aint8.data(),
      Aint8_zero_point,
      Bint8_tr.data(),
      Cint32_ref.data());

  // matrix dimensions after im2col
  int MDim = conv_p.MB * im_out_dim;
  int NDim = conv_p.OC / conv_p.G;
  int KDim = kernel_dim * conv_p.IC;
  int KDimPerGroup = KDim / conv_p.G;

  int OC_per_G = conv_p.OC / conv_p.G;

  // computing row offset
  vector<int32_t> row_offsets(MDim);
  vector<uint8_t> Aint8_im2col(MDim * KDim);
  im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

  vector<int32_t> row_offsets_sum(MDim, 0);
  vector<int32_t> in_row_offsets_sum(conv_p.IN_DIM[0] * conv_p.IN_DIM[1], 0);

  // computing column offset
  vector<int32_t> col_offsets(conv_p.OC);
  for (int g = 0; g < conv_p.G; ++g) {
    col_offsets_with_zero_pt_s8acc32_ref(
        KDimPerGroup,
        OC_per_G,
        OC_per_G,
        Bint8_tr.data() + g * KDimPerGroup * OC_per_G,
        Bint8_zero_point.data(),
        col_offsets.data() + g * OC_per_G,
        conv_p.OC);
  }

  for (int g = 0; g < conv_p.G; ++g) {
    row_offsets_u8acc32_ref(
        MDim,
        KDimPerGroup,
        KDim,
        Aint8_im2col.data() + g * KDimPerGroup,
        row_offsets.data());

    requantize_u8acc32_ref(
        MDim,
        NDim,
        conv_p.G * NDim,
        Cint32_ref.data() + g * NDim,
        Cint8_ref.data() + g * NDim,
        C_multiplier.data() + g * NDim / conv_p.OC,
        C_zero_point,
        Aint8_zero_point,
        Bint8_zero_point.data() + g * NDim / conv_p.OC,
        row_offsets.data(),
        col_offsets.data() + g * NDim,
        nullptr,
        conv_p.OC);
  }
}

/*
INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDirectConvTest,
    ::testing::Combine(
        ::testing::Bool(), // a_symmetric
        ::testing::Bool(), // b_symmetric
        ::testing::Values(1, 2))); // oc_per_g

TEST_P(FBGemmDirectConvTest, Test2D) {
  bool a_symmetric, b_symmetric;
  int oc_per_g;
  tie(a_symmetric, b_symmetric, oc_per_g) = GetParam();

  for (auto conv_p : shapes) {
    int im_in_dim = accumulate(
        conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());
    aligned_vector<uint8_t> aBuf(conv_p.MB * im_in_dim * conv_p.IC);

    int kernel_dim =
        accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());

    aligned_vector<int8_t> bBuf(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));


    aligned_vector<int8_t> bBuf_pf(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<int8_t> Bint8_tr(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<int8_t> Bint8_tr_vec(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<float> C_multiplier(1);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    int im_out_dim = accumulate(
        conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());
    // matrix dimensions after im2col
    int MDim = conv_p.MB * im_out_dim;
    int NDim = conv_p.OC / conv_p.G;
    int KDim = kernel_dim * conv_p.IC;
    int KDimPerGroup = KDim / conv_p.G;

    int OC_per_G = conv_p.OC / conv_p.G;
    aligned_vector<int32_t> Cint32_ref(conv_p.MB * im_out_dim * conv_p.OC);
    aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
    aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_fb2(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb2(Cint32_ref.size());

    DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp fn;
    // fn = GemmGetOrCreate<inst_set_t::avx2>(
    //    true, _MB, _NB, _KB);
    DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;

    fn = codeObj.getOrCreateDirectConv<inst_set_t::avx2>(
        true,
        conv_p.OUT_DIM[1],
        conv_p.IN_DIM[1] * conv_p.IC,
        conv_p.stride[1] * conv_p.IC);

    randFill<uint8_t>(aBuf, 0, 5);
    randFill<int8_t>(bBuf, -4, 4);
    randFill<int8_t>(bBuf_pf, -4, 4);

    int32_t Aint8_zero_point = 4;
    aligned_vector<int32_t> Bint8_zero_point(1);
    randFill(Bint8_zero_point, -3, -1);


    aligned_vector<int8_t> bBuf_tr(bBuf.size());
    transposeConvWeights_KwIchO8I4<2>(conv_p, bBuf.data(), bBuf_tr.data());

    for (int i = 0; i < conv_p.OC; i += 8) {
      fn(aBuf.data(),
         bBuf_tr.data() + i * kernel_dim * conv_p.IC,
         bBuf_pf.data(),
         Cint32_fb.data() + i,
         conv_p.IC * conv_p.K[1],
         conv_p.OC);
    }

    // reference quantized int8 convolution implementation
    QuantizeDirectConv_ref(
        conv_p,
        aBuf,
        bBuf,
        Cint32_ref,
        Cint8_ref,
        Aint8_zero_point,
        C_multiplier,
        C_zero_point,
        Bint8_zero_point);

    compare_buffers(
        Cint32_fb.data(),
        Cint32_ref.data(),
        conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1],
        conv_p.OC,
        conv_p.OC,
        5);


  // computing column offset
  vector<int32_t> col_offsets(conv_p.OC);
  transposeConvWeights<2>(conv_p, bBuf.data(), Bint8_tr.data());
  for (int g = 0; g < conv_p.G; ++g) {
    col_offsets_with_zero_pt_s8acc32_ref(
        KDimPerGroup,
        OC_per_G,
        OC_per_G,
        Bint8_tr.data() + g * KDimPerGroup * OC_per_G,
        Bint8_zero_point.data(),
        col_offsets.data() + g * OC_per_G,
        conv_p.OC);
  }

  vector<int32_t> row_offsets(MDim);
  vector<uint8_t> Aint8_im2col(MDim * KDim);
  im2col_ref(conv_p, aBuf.data(), Aint8_zero_point, Aint8_im2col.data());
  for (int g = 0; g < conv_p.G; ++g) {
    row_offsets_u8acc32_ref(
        MDim,
        KDimPerGroup,
        KDim,
        Aint8_im2col.data() + g * KDimPerGroup,
        row_offsets.data());

    requantize_u8acc32_ref(
        MDim,
        NDim,
        conv_p.G * NDim,
        Cint32_fb.data() + g * NDim,
        Cint8_fb.data() + g * NDim,
        C_multiplier.data() + g * NDim / conv_p.OC,
        C_zero_point,
        Aint8_zero_point,
        Bint8_zero_point.data() + g * NDim / conv_p.OC,
        row_offsets.data(),
        col_offsets.data() + g * NDim,
        nullptr,
        conv_p.OC);
  }

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int h = 0; h < conv_p.OUT_DIM[0]; ++h) {
        for (int w = 0; w < conv_p.OUT_DIM[1]; ++w) {
          for (int k = 0; k < conv_p.OC; ++k) {
            int H_OUT = conv_p.OUT_DIM[0];
            int W_OUT = conv_p.OUT_DIM[1];
            int OC = conv_p.OC;
            int32_t expected =
                Cint8_ref[((n * H_OUT + h) * W_OUT + w) * OC + k];
            int32_t actual = Cint8_fb[((n * H_OUT + h) * W_OUT + w) * OC + k];
            EXPECT_EQ(actual, expected)
                << "Directconv " << conv_p.K[0] << "x" << conv_p.K[1] << " results differ at (" << n
                << ", " << h << ", " << w << ", " << k << ").";
          }
        }
      }
    }

  } // for each shape
}
*/


INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDirectConvTransTest,
    ::testing::Combine(
        ::testing::Bool(), // a_symmetric
        ::testing::Bool(), // b_symmetric
        ::testing::Values(1, 2))); // oc_per_g

TEST_P(FBGemmDirectConvTransTest, Test2D) {
  bool a_symmetric, b_symmetric;
  int oc_per_g;
  tie(a_symmetric, b_symmetric, oc_per_g) = GetParam();

  for (auto conv_p : shapes_trans) {
    int im_in_dim = accumulate(
        conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());
    aligned_vector<uint8_t> aBuf(conv_p.MB * im_in_dim * conv_p.IC);

    int kernel_dim =
        accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());

    aligned_vector<int8_t> bBuf(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));


    aligned_vector<int8_t> bBuf_pf(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<int8_t> Bint8_tr(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<int8_t> Bint8_tr_vec(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<float> C_multiplier(1);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    int im_out_dim = accumulate(
        conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());
    // matrix dimensions after im2col
    int MDim = conv_p.MB * im_out_dim;
    int NDim = conv_p.OC / conv_p.G;
    int KDim = kernel_dim * conv_p.IC;
    int KDimPerGroup = KDim / conv_p.G;

    int OC_per_G = conv_p.OC / conv_p.G;
    aligned_vector<int32_t> Cint32_ref(conv_p.MB * im_out_dim * conv_p.OC);
    aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_fb2(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb2(Cint32_ref.size());

    randFill<uint8_t>(aBuf, 0, 5);
    randFill<int8_t>(bBuf, -4, 4);
    randFill<int8_t>(bBuf_pf, -4, 4);

    int32_t Aint8_zero_point = 4;
    aligned_vector<int32_t> Bint8_zero_point(1);
    randFill(Bint8_zero_point, -3, -1);

    aligned_vector<int8_t> &Bint8 = bBuf;
    aligned_vector<uint8_t> &Aint8 = aBuf;

    // reference implementation
    // conv_ref expects weights to be in G (R S C/G) K/G
    transposeConvWeights<2>(conv_p, Bint8.data(), Bint8_tr.data());
    transposeConvWeights_KwIchO8I4<2>(
        conv_p, Bint8.data(), Bint8_tr_vec.data());

    conv_ref(
        // DirectConvTrans_ref(
        conv_p,
        Aint8.data(),
        Aint8_zero_point,
        Bint8_tr.data(),
        Cint32_ref.data());


    // computing row offset
    vector<int32_t> row_offsets(MDim);
    vector<uint8_t> Aint8_im2col(MDim * KDim);
    im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

    // computing column offset
    vector<int32_t> col_offsets(conv_p.OC);
    for (int g = 0; g < conv_p.G; ++g) {
      col_offsets_with_zero_pt_s8acc32_ref(
          KDimPerGroup,
          OC_per_G,
          OC_per_G,
          Bint8_tr.data() + g * KDimPerGroup * OC_per_G,
          Bint8_zero_point.data(),
          col_offsets.data() + g * OC_per_G,
          conv_p.OC);
    }

    for (int g = 0; g < conv_p.G; ++g) {
      row_offsets_u8acc32_ref(
          MDim,
          KDimPerGroup,
          KDim,
          Aint8_im2col.data() + g * KDimPerGroup,
          row_offsets.data());

      requantize_u8acc32_ref(
          MDim,
          NDim,
          conv_p.G * NDim,
          Cint32_ref.data() + g * NDim,
          Cint8_ref.data() + g * NDim,
          C_multiplier.data() + g * NDim / conv_p.OC,
          C_zero_point,
          Aint8_zero_point,
          Bint8_zero_point.data() + g * NDim / conv_p.OC,
          row_offsets.data(),
          col_offsets.data() + g * NDim,
          nullptr,
          conv_p.OC);
    }

    // computing column offset
    vector<int32_t> col_offsetsT(conv_p.OC * MDim);
    for (int g = 0; g < conv_p.G; ++g) {
      col_offsets_with_zero_pt_s8acc32_DirectConvT_ref(
          conv_p,
          Bint8_tr_vec.data() + g * KDimPerGroup * OC_per_G,
          Bint8_zero_point.data(),
          col_offsetsT.data() + g * OC_per_G,
          conv_p.OC);
    }

    string runType;

    PackedDirectConvMatrix packedB(conv_p.IC, conv_p.OC, kernel_dim, Bint8.data());

    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
        doNothingObj,
        C_multiplier.data(),
        C_zero_point,
        Aint8_zero_point,
        Bint8_zero_point.data(),
        nullptr, // row offsets
        col_offsetsT.data(),
        nullptr, // bias
        conv_p.OC,
        conv_p.G);

    int32_t* bias_p = nullptr;
    fbgemmDirectConv(conv_p,
            Aint8.data(),
            packedB,
            Cint8_fb.data(),
            Cint32_fb.data(),
            outputProcObj,
            bias_p, //bias
            0,
            1);

    /*
    compare_buffers(
        Cint8_ref.data(),
        Cint8_fb.data(),
        MDim,
        NDim * conv_p.G,
        NDim * conv_p.G,
        5);
   */

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int h = 0; h < conv_p.OUT_DIM[0]; ++h) {
        for (int w = 0; w < conv_p.OUT_DIM[1]; ++w) {
          for (int k = 0; k < conv_p.OC; ++k) {
            int H_OUT = conv_p.OUT_DIM[0];
            int W_OUT = conv_p.OUT_DIM[1];
            int OC = conv_p.OC;
            int32_t expected =
                Cint8_ref[((n * H_OUT + h) * W_OUT + w) * OC + k];
            int32_t actual = Cint8_fb[((n * H_OUT + h) * W_OUT + w) * OC + k];
            EXPECT_EQ(actual, expected)
                << "DirectconvTrans " << conv_p.K[0] << "x" << conv_p.K[1] << " results differ at (" << n
                << ", " << h << ", " << w << ", " << k << ").";
          }
        }
      }
    }

  } // for each shape
}


INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmDirectConvTransFbgemmTest,
    ::testing::Combine(
        ::testing::Bool(), // a_symmetric
        ::testing::Bool(), // b_symmetric
        ::testing::Values(1, 2))); // oc_per_g


TEST_P(FBGemmDirectConvTransFbgemmTest, Test2D) {
  bool a_symmetric, b_symmetric;
  int oc_per_g;
  tie(a_symmetric, b_symmetric, oc_per_g) = GetParam();

  for (auto conv_p : shapes_trans) {
    int im_in_dim = accumulate(
        conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());
    aligned_vector<uint8_t> aBuf(conv_p.MB * im_in_dim * conv_p.IC);

    int kernel_dim =
        accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());

    aligned_vector<int8_t> bBuf(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));


    aligned_vector<int8_t> bBuf_pf(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<int8_t> Bint8_tr(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<int8_t> Bint8_tr_vec(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<float> C_multiplier(1);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    int im_out_dim = accumulate(
        conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());
    // matrix dimensions after im2col
    int MDim = conv_p.MB * im_out_dim;
    int NDim = conv_p.OC / conv_p.G;
    int KDim = kernel_dim * conv_p.IC;
    int KDimPerGroup = KDim / conv_p.G;

    int OC_per_G = conv_p.OC / conv_p.G;
    aligned_vector<int32_t> Cint32_ref(conv_p.MB * im_out_dim * conv_p.OC);
    aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_fb2(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb2(Cint32_ref.size());

    randFill<uint8_t>(aBuf, 0, 5);
    randFill<int8_t>(bBuf, -4, 4);
    randFill<int8_t>(bBuf_pf, -4, 4);

    int32_t Aint8_zero_point = 4;
    aligned_vector<int32_t> Bint8_zero_point(1);
    randFill(Bint8_zero_point, -3, -1);

    aligned_vector<int8_t> &Bint8 = bBuf;
    aligned_vector<uint8_t> &Aint8 = aBuf;

    // reference implementation
    // conv_ref expects weights to be in G (R S C/G) K/G
    transposeConvWeights<2>(conv_p, Bint8.data(), Bint8_tr.data());

    conv_ref(
        // DirectConvTrans_ref(
        conv_p,
        Aint8.data(),
        Aint8_zero_point,
        Bint8_tr.data(),
        Cint32_ref.data());


    // computing row offset
    vector<int32_t> row_offsets(MDim);
    vector<uint8_t> Aint8_im2col(MDim * KDim);
    im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

    // computing column offset
    vector<int32_t> col_offsets(conv_p.OC);
    for (int g = 0; g < conv_p.G; ++g) {
      col_offsets_with_zero_pt_s8acc32_ref(
          KDimPerGroup,
          OC_per_G,
          OC_per_G,
          Bint8_tr.data() + g * KDimPerGroup * OC_per_G,
          Bint8_zero_point.data(),
          col_offsets.data() + g * OC_per_G,
          conv_p.OC);
    }

    for (int g = 0; g < conv_p.G; ++g) {
      row_offsets_u8acc32_ref(
          MDim,
          KDimPerGroup,
          KDim,
          Aint8_im2col.data() + g * KDimPerGroup,
          row_offsets.data());

      requantize_u8acc32_ref(
          MDim,
          NDim,
          conv_p.G * NDim,
          Cint32_ref.data() + g * NDim,
          Cint8_ref.data() + g * NDim,
          C_multiplier.data() + g * NDim / conv_p.OC,
          C_zero_point,
          Aint8_zero_point,
          Bint8_zero_point.data() + g * NDim / conv_p.OC,
          row_offsets.data(),
          col_offsets.data() + g * NDim,
          nullptr,
          conv_p.OC);
    }

    // fbgemm top-level function for direct conv path
    PackWeightsForConv<2> packedB_2D(conv_p, Bint8.data());

    if (packedB_2D.getPackedWForDirectconv().get()) {
      packedB_2D.getPackedWForDirectconv().get()->col_offsets_with_zero_pt_s8acc32_DirectConvT(
          conv_p,
          Bint8_zero_point.data(),
          col_offsets,
          conv_p.OC);
    }

    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
        doNothingObj,
        C_multiplier.data(),
        C_zero_point,
        Aint8_zero_point,
        Bint8_zero_point.data(),
        nullptr, // row offsets
        col_offsets.data(),
        nullptr, // bias
        conv_p.OC,
        conv_p.G);

    fbgemmConv(
        conv_p,
        Aint8.data(),
        packedB_2D,
        Cint8_fb.data(),
        Cint32_fb.data(),
        outputProcObj,
        0,
        1);

    /*
    compare_buffers(
        Cint8_ref.data(),
        Cint8_fb.data(),
        MDim,
        NDim * conv_p.G,
        NDim * conv_p.G,
        5);
   */

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int h = 0; h < conv_p.OUT_DIM[0]; ++h) {
        for (int w = 0; w < conv_p.OUT_DIM[1]; ++w) {
          for (int k = 0; k < conv_p.OC; ++k) {
            int H_OUT = conv_p.OUT_DIM[0];
            int W_OUT = conv_p.OUT_DIM[1];
            int OC = conv_p.OC;
            int32_t expected =
                Cint8_ref[((n * H_OUT + h) * W_OUT + w) * OC + k];
            int32_t actual = Cint8_fb[((n * H_OUT + h) * W_OUT + w) * OC + k];
            EXPECT_EQ(actual, expected)
                << "DirectconvTrans " << conv_p.K[0] << "x" << conv_p.K[1] << " results differ at (" << n
                << ", " << h << ", " << w << ", " << k << ").";
          }
        }
      }
    }

  } // for each shape
}

} // fbgemm namespace
