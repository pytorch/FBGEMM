/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
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

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    QuantizeGroupwiseTest,
    ::testing::Combine(
        ::testing::ValuesIn({4, 12, 64}), // K
        ::testing::ValuesIn({12, 16, 32}), // C
        ::testing::ValuesIn({1, 10, 15, 30}), // X
        ::testing::ValuesIn({1, 4}), // G
        ::testing::ValuesIn({layout_t::KCX, layout_t::KXC})));

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

/**
 * Test for QuantizeGroupwise
 */
TEST_P(QuantizeGroupwiseTest, quantizeTest) {
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
