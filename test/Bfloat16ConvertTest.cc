/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cpuinfo.h>
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <random>

#include "bench/BenchUtils.h" // @manual
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/FloatConversion.h"
#include "fbgemm/Utils.h"

using namespace std;
using namespace fbgemm;

namespace {

void checkLegacyFiniteRounding(
    void (*convert)(const float*, bfloat16*, size_t)) {
  constexpr std::array<uint32_t, 16> bits{
      0x3f808000,
      0x3f818000,
      0xbf808000,
      0xbf818000,
      0x3f807fff,
      0x3f808001,
      0x00008000,
      0x00018000,
      0x80008000,
      0x80018000,
      0x00000000,
      0x80000000,
      0x7f800000,
      0xff800000,
      0x7f7fffff,
      0xff7fffff};
  constexpr std::array<uint16_t, 16> words{
      0x3f81,
      0x3f82,
      0xbf81,
      0xbf82,
      0x3f80,
      0x3f81,
      0x0001,
      0x0002,
      0x8001,
      0x8002,
      0x0000,
      0x8000,
      0x7f80,
      0xff80,
      0x7f80,
      0xff80};
  // Normal-only batches avoid the native path's subnormal fallback.
  for (size_t pattern_size : {size_t{6}, bits.size()}) {
    SCOPED_TRACE(pattern_size);
    std::array<float, 33> input{};
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = std::bit_cast<float>(bits[i % pattern_size]);
      EXPECT_EQ(cpu_float2bfloat16(input[i]).val, words[i % pattern_size]);
    }
    for (size_t count : {1, 7, 8, 15, 16, 17, 31, 32, 33}) {
      SCOPED_TRACE(count);
      std::array<bfloat16, 35> output{};
      std::array<bfloat16, 35> expected{};
      output.fill(bfloat16{0xbeef});
      expected.fill(bfloat16{0xbeef});
      for (size_t i = 0; i < count; ++i) {
        expected[i + 1] = bfloat16{words[i % pattern_size]};
      }
      convert(input.data(), output.data() + 1, count);
      EXPECT_EQ(output, expected);
    }
  }
}

} // namespace

TEST(FBGemmBfloat16Test, LegacyFiniteRounding) {
  checkLegacyFiniteRounding(FloatToBfloat16_ref);
  checkLegacyFiniteRounding(FloatToBfloat16_simd);
}

#if defined(FBGEMM_HAVE_NATIVE_BF16) && \
    (defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64))
TEST(FBGemmBfloat16Test, LegacyFiniteRoundingNative) {
  ASSERT_TRUE(cpuinfo_initialize());
#if defined(__aarch64__)
  const bool native_bf16 = cpuinfo_has_arm_bf16();
#else
  const bool native_bf16 =
      fbgemmHasAvx512Support() && fbgemmHasAvx512Bf16Support();
#endif
  if (!native_bf16) {
    GTEST_SKIP() << "Native BF16 is unavailable";
  }
  checkLegacyFiniteRounding(FloatToBfloat16_simd);
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
TEST(FBGemmBfloat16Test, LegacyFiniteRoundingAvx2) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx2Support()) {
    GTEST_SKIP() << "AVX2 is unavailable";
  }
  checkLegacyFiniteRounding(FloatToBfloat16_avx2);
}

TEST(FBGemmBfloat16Test, LegacyFiniteRoundingAvx512) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 is unavailable";
  }
  checkLegacyFiniteRounding(FloatToBfloat16_avx512);
}
#endif

TEST(FBGemmBfloat16Test, Conversion) {
  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b[100]; // bfloat16 type
  float c[100]; // fp32 type
  FloatToBfloat16_ref(a, b, 100);
  Bfloat16ToFloat_ref(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128);
  }
}

TEST(FBGemmBfloat16Test, Conversion_simd) {
  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b[100]; // bfloat16 type
  float c[100]; // fp32 type
  FloatToBfloat16_simd(a, b, 100);
  Bfloat16ToFloat_simd(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128)
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
