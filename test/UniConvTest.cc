/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <random>
#include <iostream>
#include <stdexcept>


#include <gtest/gtest.h>

#include "QuantizationHelpers.h"
#include "TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

namespace {

// tuple represents MB, IC, OC, IT, IH, IW, KH/KW, stride, pad
class uniConvTest
    : public testing::TestWithParam<
          tuple<int, int, int, int, int, int, int, int, int, int>> {};

}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    uniConvTest,
    ::testing::Combine(
        ::testing::ValuesIn({1, 2}), // MB
        ::testing::ValuesIn({16, 32}), // IC
        ::testing::ValuesIn({16, 32}), // OC
        ::testing::ValuesIn({17}), // IT
        ::testing::ValuesIn({10, 30, 55}), // IH
        ::testing::ValuesIn({10, 30, 55}), // IW
        ::testing::ValuesIn({1, 4, 16}), // G
        ::testing::ValuesIn({1, 3, 7}), // kernel
        ::testing::ValuesIn({1, 2}), // stride
        ::testing::ValuesIn({0, 1, 2}))); // pad

/**
 * Test for conv packing
 */
TEST_P(uniConvTest, packingTest) {
  int MB, IC, OC, IT, IH, IW, G, kernel, stride, pad;
  tie(MB, IC, OC, IT, IH, IW, G, kernel, stride, pad) = GetParam();

  conv_param_t<2> conv_p_2d(
      MB,
      IC,
      OC,
      {IH, IW},
      G,
      {kernel, kernel},
      {stride, stride},
      {pad, pad, pad, pad});

  int kernel_dim_2d = kernel * kernel;
  aligned_vector<int8_t> Bint8_2d(
      kernel_dim_2d * conv_p_2d.IC * (conv_p_2d.OC / conv_p_2d.G));
  PackWeightsForConv<2> packedB_2D(conv_p_2d, Bint8_2d.data());

  switch (ConvFastPath<2, int32_t>(conv_p_2d)) {
    case optimized_conv_t::depthwise: {
      ASSERT_EQ(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_2D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      break;
    }
    case optimized_conv_t::groupwise: {
      ASSERT_EQ(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_2D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "Groupwise packed matrix is null";
      break;
    }
    case optimized_conv_t::pointwise: {
      ASSERT_EQ(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_2D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "Groupwise packed matrix should be null";
      ASSERT_NE(packedB_2D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix is null";
      break;
    }
    case optimized_conv_t::im2col: {
      ASSERT_EQ(packedB_2D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_2D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix is null";
      break;
    }
  }

  conv_param_t<3> conv_p_3d(
      MB,
      IC,
      OC,
      {IT, IH, IW},
      G,
      {kernel, kernel, kernel},
      {stride, stride, stride},
      {pad, pad, pad, pad, pad, pad});

  int kernel_dim_3d = kernel * kernel * kernel;
  aligned_vector<int8_t> Bint8_3d(
      kernel_dim_3d * conv_p_3d.IC * (conv_p_3d.OC / conv_p_3d.G));
  PackWeightsForConv<3> packedB_3D(conv_p_3d, Bint8_3d.data());

  switch (ConvFastPath<3, int32_t>(conv_p_3d)) {
    case optimized_conv_t::depthwise: {
      ASSERT_EQ(packedB_3D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_3D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_3D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      break;
    }
    case optimized_conv_t::groupwise: {
      ASSERT_TRUE(false) << "groupwise are not supported for 3D";
      break;
    }
    case optimized_conv_t::pointwise: {
      ASSERT_EQ(packedB_3D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_3D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_NE(packedB_3D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix is null";
      break;
    }
    case optimized_conv_t::im2col: {
      ASSERT_EQ(packedB_3D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_3D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForPointwise(), nullptr)
          << "pointwise packed matrix should be null";
      ASSERT_NE(packedB_3D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix is null";
      break;
    }
  }
}

/**
 * Test for packing/unpacking
 */
TEST_P(uniConvTest, packUnpackTest) {
  int MB, IC, OC, IT, IH, IW, G, kernel, stride, pad;
  tie(MB, IC, OC, IT, IH, IW, G, kernel, stride, pad) = GetParam();

  conv_param_t<2> conv_p_2d(
      MB,
      IC,
      OC,
      {IH, IW},
      G,
      {kernel, kernel},
      {stride, stride},
      {pad, pad, pad, pad});

  int kernel_dim_2d = kernel * kernel;

  aligned_vector<int8_t> Bint8_2d(
      kernel_dim_2d * conv_p_2d.IC * (conv_p_2d.OC / conv_p_2d.G));
  aligned_vector<int8_t> Bint8_2d_unpacked(
      kernel_dim_2d * conv_p_2d.IC * (conv_p_2d.OC / conv_p_2d.G));

  PackWeightsForConv<2> packedB_2D(conv_p_2d, Bint8_2d.data());

  packedB_2D.unpack(Bint8_2d_unpacked.data());

  ASSERT_EQ(Bint8_2d, Bint8_2d_unpacked)
      << "Original and unpacked data elements are not the same [2D]";

  conv_param_t<3> conv_p_3d(
      MB,
      IC,
      OC,
      {IT, IH, IW},
      G,
      {kernel, kernel, kernel},
      {stride, stride, stride},
      {pad, pad, pad, pad, pad, pad});

  int kernel_dim_3d = kernel * kernel * kernel;

  aligned_vector<int8_t> Bint8_3d(
      kernel_dim_3d * conv_p_3d.IC * (conv_p_3d.OC / conv_p_3d.G));

  aligned_vector<int8_t> Bint8_3d_unpacked(
      kernel_dim_3d * conv_p_3d.IC * (conv_p_3d.OC / conv_p_3d.G));

  PackWeightsForConv<3> packedB_3D(conv_p_3d, Bint8_3d.data());

  packedB_3D.unpack(Bint8_3d_unpacked.data());

  ASSERT_EQ(Bint8_3d, Bint8_3d_unpacked)
      << "Original and unpacked data elements are not the same [3D]";
}

TEST(uniConvTest, cornerCases) {
  int stride = 1;
  conv_param_t<2> conv_p_2d(
      1, // mini-batch
      16, // input channels
      32, // output channels
      {28, 28}, // input height/width
      4, // groups
      {3, 3}, // kernel height/width
      {stride, stride}, // strides
      {1, 1, 1, 1}); // padding

  int kernel_dim_2d = conv_p_2d.K[0] * conv_p_2d.K[1];

  aligned_vector<uint8_t> Aint8(
      conv_p_2d.MB * conv_p_2d.IN_DIM[0] * conv_p_2d.IN_DIM[1] * conv_p_2d.IC);
  aligned_vector<int8_t> Bint8_2d(
      kernel_dim_2d * conv_p_2d.IC * (conv_p_2d.OC / conv_p_2d.G));
  aligned_vector<int32_t> Cint32_fb(
      conv_p_2d.MB * conv_p_2d.OUT_DIM[0] * conv_p_2d.OUT_DIM[1] *
      conv_p_2d.OC);
  aligned_vector<uint8_t> Cint8_fb(Cint32_fb.size(), 0);

  // A matrix (input activations)
  randFill<uint8_t>(Aint8, 0, 5);
  int32_t Aint8_zero_point = 4;

  // B matrix (weights)
  randFill<int8_t>(Bint8_2d, -4, 4);
  aligned_vector<int32_t> Bint8_zero_point(1);
  randFill(Bint8_zero_point, -3, -1);

  aligned_vector<float> C_multiplier(Bint8_zero_point.size());
  randFill(C_multiplier, 0.1234f / 2, 0.1234f * 3 / 2);
  int32_t C_zero_point = 5;

  PackWeightsForConv<2> packedB_2D(conv_p_2d, Bint8_2d.data());

  vector<int32_t> col_offsets(conv_p_2d.OC);

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
      conv_p_2d.OC,
      conv_p_2d.G);

  try {
    conv_p_2d.stride[0] = 2;
    fbgemmConv(
        conv_p_2d,
        Aint8.data(),
        packedB_2D,
        Cint8_fb.data(),
        Cint32_fb.data(),
        outputProcObj,
        0,
        1);
  } catch (std::logic_error const& err) {
    std::string s(err.what());
    EXPECT_TRUE(s.rfind("[FBGEMM_CONV_ERROR]", 0) == 0);
  }

  // reset
  conv_p_2d.stride[0] = stride;
  // this should run fine
  fbgemmConv(
      conv_p_2d,
      Aint8.data(),
      packedB_2D,
      Cint8_fb.data(),
      Cint32_fb.data(),
      outputProcObj,
      0,
      1);
}
