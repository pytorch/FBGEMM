/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <random>
#include <iostream>


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
        ::testing::ValuesIn({3, 7}), // kernel
        ::testing::ValuesIn({1, 2}), // stride
        ::testing::ValuesIn({1, 2}))); // pad

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
      ASSERT_NE(packedB_2D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      break;
    }
    case optimized_conv_t::groupwise: {
      ASSERT_EQ(packedB_2D.getPackedWForIm2col(), nullptr)
          << "im2col packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_2D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_NE(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "Groupwise packed matrix is null";
      break;
    }
    case optimized_conv_t::im2col: {
      ASSERT_EQ(packedB_2D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_2D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_2D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
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
      ASSERT_NE(packedB_3D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
      break;
    }
    case optimized_conv_t::groupwise: {
      ASSERT_TRUE(false) << "groupwise are not supported for 3D";
      break;
    }
    case optimized_conv_t::im2col: {
      ASSERT_EQ(packedB_3D.getPackedWFor2DDW(), nullptr)
          << "2D depthwise packed matrix is null";
      ASSERT_EQ(packedB_3D.getPackedWFor3DDW(), nullptr)
          << "3D depthwise packed matrix should be null";
      ASSERT_EQ(packedB_3D.getPackedWForGroupwise(), nullptr)
          << "groupwise packed matrix should be null";
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
