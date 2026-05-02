/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <bit>
#include <cmath>
#include <random>

#include "bench/BenchUtils.h" // @manual
#include "fbgemm/FbgemmConvert.h"

using namespace std;
using namespace fbgemm;

TEST(FBGemmBfloat16Test, Conversion) {
  float a[100];
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b[100];
  float c[100];
  FloatToBfloat16_ref(a, b, 100);
  Bfloat16ToFloat_ref(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(b[i], cpu_float2bfloat16(a[i]))
        << "ref conversion differs from scalar at i=" << i;
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128);
  }
}

TEST(FBGemmBfloat16Test, Conversion_simd) {
  float a[100];
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b_ref[100];
  bfloat16 b_simd[100];
  float c[100];
  FloatToBfloat16_ref(a, b_ref, 100);
  FloatToBfloat16_simd(a, b_simd, 100);
  Bfloat16ToFloat_simd(b_simd, c, 100);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(b_simd[i], b_ref[i])
        << "simd and ref differ at i=" << i
        << " input=0x" << std::hex << std::bit_cast<uint32_t>(a[i]);
    EXPECT_EQ(b_simd[i], cpu_float2bfloat16(a[i]))
        << "simd differs from scalar at i=" << i;
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128);
  }
}

TEST(FBGemmBfloat16Test, Conversion_simd2) {
  vector<vector<int>> shapes;
  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<int> dm(1, 256);
  uniform_int_distribution<int> dn(1, 1024);

  for (int i = 0; i < 10; i++) {
    int m = dm(generator);
    int n = dn(generator);
    shapes.push_back({m, n});
  }

  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];

    aligned_vector<float> A_fp32_ref(m * n);
    aligned_vector<bfloat16> A_bf16_ref(m * n);
    aligned_vector<bfloat16> A_bf16_simd(m * n);
    aligned_vector<float> A_fp32_final(m * n);
    for (int i = 0; i < m * n; ++i) {
      A_fp32_ref[i] = i + 1.25;
    }

    FloatToBfloat16_ref(A_fp32_ref.data(), A_bf16_ref.data(), m * n);
    FloatToBfloat16_simd(A_fp32_ref.data(), A_bf16_simd.data(), m * n);
    Bfloat16ToFloat_simd(A_bf16_simd.data(), A_fp32_final.data(), m * n);
    for (int i = 0; i < m * n; ++i) {
      EXPECT_EQ(A_bf16_simd[i], A_bf16_ref[i])
          << "m=" << m << " n=" << n << " i=" << i;
      EXPECT_LE(
          fabs(A_fp32_final[i] - A_fp32_ref[i]) / A_fp32_ref[i], 1.0 / 128);
    }
  }
}

