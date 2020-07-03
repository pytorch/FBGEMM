/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <climits>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include "fbgemm/QuantUtils.h"
#include "fbgemm/Utils.h"

using namespace std;
using namespace fbgemm;

// tuple represents K, C, X, G, layout_t
// layout_t can be KCX or KXC
class QuantizeGroupwiseTest
    : public testing::TestWithParam<tuple<int, int, int, int, layout_t>> {};

class QuantizeTest : public testing::TestWithParam<int> {};
class FusedQuantizeDequantizeTest : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    QuantizeGroupwiseTest,
    ::testing::Combine(
        ::testing::ValuesIn({4, 12, 64}), // K
        ::testing::ValuesIn({12, 16, 32}), // C
        ::testing::ValuesIn({1, 10, 15, 30}), // X
        ::testing::ValuesIn({1, 4}), // G
        ::testing::ValuesIn({layout_t::KCX, layout_t::KXC})));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    QuantizeTest,
    ::testing::Values(1, 2, 5, 8, 9, 16, 20, 28, 32, 33));

template <typename T, layout_t LT>
void ref_impl(
    const vector<float>& src,
    int K,
    int C,
    int X,
    int G,
    const vector<float>& scales,
    const vector<int>& zero_points,
    vector<T>& dst) {
  int C_per_G = C / G;
  for (int i = 0; i < K; ++i) {
    for (int g = 0; g < G; ++g) {
      for (int c = 0; c < C / G; ++c) {
        for (int x = 0; x < X; ++x) {
          float num;
          if (LT == layout_t::KCX) {
            num = src[(i * C + g * C_per_G + c) * X + x];
          } else {
            num = src[(i * X + x) * C + g * C_per_G + c];
          }
          int res = nearbyint(zero_points[g] + num / scales[g]);
          T final_res = min<T>(
              max<T>(res, numeric_limits<T>::min()), numeric_limits<T>::max());
          if (LT == layout_t::KCX) {
            dst[(i * C + g * C_per_G + c) * X + x] = final_res;
          } else {
            dst[(i * X + x) * C + g * C_per_G + c] = final_res;
          }
        }
      }
    }
  }
}

template <typename T, layout_t LT>
void runTests(
    const vector<float>& src,
    int K,
    int C,
    int X,
    int G,
    const vector<float>& scales,
    const vector<int>& zero_points,
    vector<T>& dst,
    vector<T>& dst_ref) {
  QuantizeGroupwise<T, LT>(
      src.data(), K, C, X, G, scales.data(), zero_points.data(), dst.data());

  ref_impl<T, LT>(src, K, C, X, G, scales, zero_points, dst_ref);
}

/**
 * There can be off-by-one error in quantized values due to how the mid-point
 * cases are rounded-off in vectorized vs scalar codes and due to adding of
 * zero_point before rounding vs after rounding. We ignore such differences
 * while comparing results.
 */
template <typename T>
::testing::AssertionResult isNear(
    const vector<T>& res,
    const vector<T>& res_ref) {
  bool match = true;
  if (res.size() == res_ref.size()) {
    for (int i = 0; i < res.size(); ++i) {
      if (!(res[i] == res_ref[i] || res[i] == res_ref[i] + 1 ||
            res[i] == res_ref[i] - 1)) {
        match = false;
        break;
      }
    }
  }
  if (match)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << " Quantized results do not match";
}

bool floatEqualAll(vector<float>& a, vector<float>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < a.size(); i++) {
    if (fabs(a[i] - b[i]) > std::numeric_limits<float>::epsilon()) {
      return false;
    }
  }
  return true;
}

/**
 * Test for QuantizeGroupwise
 */
TEST_P(QuantizeGroupwiseTest, quantizeGTest) {
  int K, C, X, G;
  layout_t layout;
  tie(K, C, X, G, layout) = GetParam();

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> disFP(0.1, 1.1);

  vector<float> inp(K * C * X);
  generate(inp.begin(), inp.end(), [&, disFP]() mutable { return disFP(gen); });

  vector<float> scales(G);
  generate(scales.begin(), scales.end(), [&, disFP]() mutable {
    return disFP(gen);
  });

  uniform_int_distribution<> disUInt8(0, 8);
  vector<int> zero_points_uint8(G);
  generate(
      zero_points_uint8.begin(),
      zero_points_uint8.end(),
      [&, disUInt8]() mutable { return disUInt8(gen); });

  uniform_int_distribution<> disInt8(-64, 63);
  vector<int> zero_points_int8(G);
  generate(
      zero_points_int8.begin(), zero_points_int8.end(), [&, disInt8]() mutable {
        return disInt8(gen);
      });

  uniform_int_distribution<> disInt32(-512, 512);
  vector<int> zero_points_int32(G);
  generate(
      zero_points_int32.begin(),
      zero_points_int32.end(),
      [&, disInt32]() mutable { return disInt32(gen); });

  vector<uint8_t> dstuint8(K * C * X);
  vector<uint8_t> dstuint8_ref(K * C * X);

  vector<int8_t> dstint8(K * C * X);
  vector<int8_t> dstint8_ref(K * C * X);

  vector<int32_t> dstint32(K * C * X);
  vector<int32_t> dstint32_ref(K * C * X);

  if (layout == layout_t::KCX) {
    runTests<uint8_t, layout_t::KCX>(
        inp, K, C, X, G, scales, zero_points_uint8, dstuint8, dstuint8_ref);
    runTests<int8_t, layout_t::KCX>(
        inp, K, C, X, G, scales, zero_points_int8, dstint8, dstint8_ref);
    runTests<int32_t, layout_t::KCX>(
        inp, K, C, X, G, scales, zero_points_int32, dstint32, dstint32_ref);
  } else {
    runTests<uint8_t, layout_t::KXC>(
        inp, K, C, X, G, scales, zero_points_uint8, dstuint8, dstuint8_ref);
    runTests<int8_t, layout_t::KXC>(
        inp, K, C, X, G, scales, zero_points_int8, dstint8, dstint8_ref);
    runTests<int32_t, layout_t::KXC>(
        inp, K, C, X, G, scales, zero_points_int32, dstint32, dstint32_ref);
  }

  EXPECT_TRUE(isNear(dstuint8, dstuint8_ref));
  EXPECT_TRUE(isNear(dstint8, dstint8_ref));
  EXPECT_TRUE(isNear(dstint32, dstint32_ref));
}

template <typename T>
void runQuantizeTests(
    const vector<float>& src,
    float scale,
    int zero_point,
    vector<T>& dst,
    vector<T>& dst_ref) {
  // reference
  for (int i = 0; i < src.size(); ++i) {
    dst_ref[i] = Quantize<T>(src[i], zero_point, scale, CHAR_BIT * sizeof(T));
  }

  TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(T);

  Quantize<T>(src.data(), dst.data(), src.size(), qparams);
}

/**
 * Test for QuantizeGroupwise
 */
TEST_P(QuantizeTest, quantizeTest) {
  int len;
  len = GetParam();

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> disFP(-1.0e6, 1.0e6);

  vector<float> inp(len);
  generate(inp.begin(), inp.end(), [&, disFP]() mutable { return disFP(gen); });

  float scale = disFP(gen);

  // Generate a number between [0, 255] both inclusive
  uniform_int_distribution<> disUInt8(0, 255);
  int zero_point_uint8 = disUInt8(gen);

  uniform_int_distribution<> disInt8(-128, 127);
  int zero_point_int8 = disInt8(gen);

  vector<uint8_t> dstuint8(len);
  vector<uint8_t> dstuint8_ref(len);

  vector<int8_t> dstint8(len);
  vector<int8_t> dstint8_ref(len);

  runQuantizeTests<uint8_t>(
      inp, scale, zero_point_uint8, dstuint8, dstuint8_ref);
  runQuantizeTests<int8_t>(inp, scale, zero_point_int8, dstint8, dstint8_ref);

  EXPECT_TRUE(isNear(dstuint8, dstuint8_ref));
  EXPECT_TRUE(isNear(dstint8, dstint8_ref));
}

// vector and scalar code should have the same behavior
TEST(QuantizeTestSingle, vectorScalar) {
  // This length will exercise both the vector and scalar path
  int len = 33;
  vector<float> src(len);
  vector<uint8_t> dst(len);

  for (int i = 0; i < len; ++i) {
    src[i] = -2.9483526e-05;
  }
  float scale = 2.3124334356729307e-07;
  int zero_point = 128;

  TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(uint8_t);

  Quantize<uint8_t>(src.data(), dst.data(), len, qparams);

  // Check if all elements are equal
  EXPECT_TRUE(
      adjacent_find(dst.begin(), dst.end(), not_equal_to<int>()) == dst.end());
}

TEST(QuantizeTest, cornerCases) {
  TensorQuantizationParams qparams;
  qparams.scale = 1.19209e-07;
  qparams.zero_point = 0;
  qparams.precision = 8;
  std::vector<float> src1 = {3.40282e+38, -2.16845e+38};

  std::vector<int8_t> dst_int8(src1.size());
  Quantize<int8_t>(src1.data(), dst_int8.data(), dst_int8.size(), qparams);
  EXPECT_EQ(dst_int8[0], 127);
  EXPECT_EQ(dst_int8[1], -128);

  // Tests vectorized and remainder paths
  std::vector<float> src2 = {3.40282e+38,
                             -2.16845e+38,
                             3.40282e+38,
                             -2.16845e+38,
                             3.40282e+38,
                             -2.16845e+38,
                             3.40282e+38,
                             -2.16845e+38,
                             3.40282e+38};
  std::vector<uint8_t> dst_uint8(src2.size());
  Quantize<uint8_t>(src2.data(), dst_uint8.data(), dst_uint8.size(), qparams);
  EXPECT_EQ(dst_uint8[0], 255);
  EXPECT_EQ(dst_uint8[1], 0);
  EXPECT_EQ(dst_uint8[8], 255);

  qparams.precision = 16;
  std::vector<int16_t> dst_int16(src2.size());
  Quantize<int16_t>(src2.data(), dst_int16.data(), dst_int16.size(), qparams);
  EXPECT_EQ(dst_int16[0], 32767);
  EXPECT_EQ(dst_int16[1], -32768);
}

template <typename T>
void runFusedQuantizeDequantizeTests(
    const vector<float>& src,
    float scale,
    int zero_point,
    vector<float>& dst,
    vector<float>& dst_ref) {
  TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(T);
  // reference
  for (int i = 0; i < src.size(); ++i) {
    dst_ref[i] = FusedQuantizeDequantize<T>(src[i], qparams);
  }
  FusedQuantizeDequantize<T>(src.data(), dst.data(), src.size(), qparams);
}

TEST_P(FusedQuantizeDequantizeTest, fusedQuantizeDequantizeTest) {
  int len;
  len = GetParam();

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> disFP(-1.0e6, 1.0e6);

  vector<float> inp(len);
  generate(inp.begin(), inp.end(), [&, disFP]() mutable { return disFP(gen); });

  float scale = disFP(gen);

  // Generate a number between [0, 255] both inclusive
  uniform_int_distribution<> disUInt8(0, 255);
  int zero_point_uint8 = disUInt8(gen);

  uniform_int_distribution<> disInt8(-128, 127);
  int zero_point_int8 = disInt8(gen);

  vector<float> dstfloat(len);
  vector<float> dstfloat_ref(len);

  runFusedQuantizeDequantizeTests<uint8_t>(
      inp, scale, zero_point_uint8, dstfloat, dstfloat_ref);
  EXPECT_TRUE(floatEqualAll(dstfloat, dstfloat_ref));

  runFusedQuantizeDequantizeTests<int8_t>(
      inp, scale, zero_point_int8, dstfloat, dstfloat_ref);
  EXPECT_TRUE(floatEqualAll(dstfloat, dstfloat_ref));
}

// vector and scalar code should have the same behavior
TEST(FusedQuantizeDequantizeTestSingle, vectorScalar) {
  // This length will exercise both the vector and scalar path
  int len = 33;
  vector<float> src(len);
  vector<float> dst(len);

  for (int i = 0; i < len; ++i) {
    src[i] = -2.9483526e-05;
  }
  float scale = 2.3124334356729307e-07;
  int zero_point = 128;

  TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(uint8_t);

  FusedQuantizeDequantize<uint8_t>(src.data(), dst.data(), src.size(), qparams);
  // Check if all elements are equal
  EXPECT_TRUE(
      adjacent_find(dst.begin(), dst.end(), not_equal_to<float>()) ==
      dst.end());
}

TEST(FusedQuantizeDequantizeTest, cornerCases) {
  TensorQuantizationParams qparams;
  qparams.scale = 1.19209e-07;
  qparams.zero_point = 0;
  qparams.precision = 8;
  vector<float> src1 = {3.40282e+38, -2.16845e+38};
  vector<float> ref = {1.5139543e-05, -1.5258752e-05};

  vector<float> dst_int8(src1.size());

  FusedQuantizeDequantize<int8_t>(
      src1.data(), dst_int8.data(), src1.size(), qparams);
  EXPECT_TRUE(floatEqualAll(dst_int8, ref));


  // Tests vectorized and remainder paths
  vector<float> src2 = {3.40282e+38,
                        -2.16845e+38,
                        3.40282e+38,
                        -2.16845e+38,
                        3.40282e+38,
                        -2.16845e+38,
                        3.40282e+38,
                        -2.16845e+38,
                        3.40282e+38};

  vector<float> ref2 = {3.0398295e-05,
                        0,
                        3.0398295e-05,
                        0,
                        3.0398295e-05,
                        0,
                        3.0398295e-05,
                        0,
                        3.0398295e-05};

  std::vector<float> dst_uint8(src2.size(), 0);

  FusedQuantizeDequantize<uint8_t>(
      src2.data(), dst_uint8.data(), src2.size(), qparams);

  EXPECT_TRUE(floatEqualAll(dst_uint8, ref2));
}
