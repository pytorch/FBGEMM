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

namespace {
void checkSpecialValues(void (*convert)(const float*, bfloat16*, size_t)) {
  constexpr uint32_t inputs[] = {
      0x7f800001u, 0x7fc00001u, 0x7fff8000u, 0x7fffffffu,
      0xff800001u, 0xffc00001u, 0xffff8000u, 0xffffffffu,
      0x7f800000u, 0xff800000u};
  constexpr uint16_t expected[] = {
      0x7fc0, 0x7fc0, 0x7fff, 0x7fff, 0xffc0, 0xffc0, 0xffff, 0xffff,
      0x7f80, 0xff80};
  float src[33];
  bfloat16 dst[33];
  for (size_t i = 0; i < 33; ++i) {
    src[i] = std::bit_cast<float>(inputs[i % 10]);
  }
  convert(src, dst, 33); // SIMD blocks followed by a scalar tail.
  for (size_t i = 0; i < 33; ++i) {
    EXPECT_EQ(dst[i].val, expected[i % 10]);
  }
}
} // namespace

TEST(FBGemmBfloat16Test, Conversion) {
  checkSpecialValues(FloatToBfloat16_ref);
  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    const uint32_t high = (0x3f00u + i) | (i & 2 ? 0x8000u : 0);
    a[i] = std::bit_cast<float>((high << 16) | 0x8000u);
  }
  bfloat16 b[100]; // bfloat16 type
  float c[100]; // fp32 type
  FloatToBfloat16_ref(a, b, 100);
  Bfloat16ToFloat_ref(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    const uint16_t high = std::bit_cast<uint32_t>(a[i]) >> 16;
    EXPECT_EQ(b[i].val, high + (high & 1));
    EXPECT_LE(fabs((c[i] - a[i]) / a[i]), 1.0 / 128);
  }
}

TEST(FBGemmBfloat16Test, Conversion_simd) {
  checkSpecialValues(FloatToBfloat16_simd);
  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    const uint32_t high = (0x3f00u + i) | (i & 2 ? 0x8000u : 0);
    a[i] = std::bit_cast<float>((high << 16) | 0x8000u);
  }
  bfloat16 b[100]; // bfloat16 type
  float c[100]; // fp32 type
  FloatToBfloat16_simd(a, b, 100);
  Bfloat16ToFloat_simd(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    const uint16_t high = std::bit_cast<uint32_t>(a[i]) >> 16;
    EXPECT_EQ(b[i].val, high + (high & 1));
    EXPECT_LE(fabs((c[i] - a[i]) / a[i]), 1.0 / 128)
        << "Conversion results differ at (" << i << " ). ref: " << a[i]
        << " conversion: " << c[i];
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

    cerr << "m = " << m << " n = " << n << '\n';
    aligned_vector<float> A_fp32_ref(m * n); // fp32 type
    aligned_vector<bfloat16> A_bfloat16(m * n); // bfloat16 type
    aligned_vector<float> A_fp32_final(m * n); // fp32 type
    // randFill(A_fp32_ref, 0.0f, 4.0f);
    for (int i = 0; i < m * n; ++i) {
      A_fp32_ref[i] = i + 1.25;
    }

    FloatToBfloat16_simd(A_fp32_ref.data(), A_bfloat16.data(), m * n);
    Bfloat16ToFloat_simd(A_bfloat16.data(), A_fp32_final.data(), m * n);
    for (int i = 0; i < m * n; ++i) {
      // The relative error should be less than 1/(2^7) since bfloat16
      // has 7 bits mantissa.
      // printf( "A_fp32_final[%d]: %f; A_fp32_ref[%d]: %f\n", i,
      // A_fp32_final[i], i, A_fp32_ref[i]);
      EXPECT_LE(
          fabs(A_fp32_final[i] - A_fp32_ref[i]) / A_fp32_ref[i], 1.0 / 128);
    }
  }
}
