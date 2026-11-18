/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "src/isa_specific/PlatformSpecificKernelsMacros.h" // @manual

TEST(PlatformSpecificKernelsMacrosTest, RefIsAlwaysIncluded) {
#define FBGEMM_SIMD_NAMESPACE ref
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx2));
  EXPECT_TRUE(FBGEMM_SIMD_IN(sve128));
  EXPECT_TRUE(FBGEMM_SIMD_IN(ref));
#undef FBGEMM_SIMD_NAMESPACE
}

TEST(PlatformSpecificKernelsMacrosTest, ListedPlatform) {
#define FBGEMM_SIMD_NAMESPACE avx512
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx512));
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx512, neon));
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx2, avx512, sve128));
#undef FBGEMM_SIMD_NAMESPACE
}

TEST(PlatformSpecificKernelsMacrosTest, UnlistedPlatform) {
#define FBGEMM_SIMD_NAMESPACE avx512
  EXPECT_FALSE(FBGEMM_SIMD_IN(avx2));
  EXPECT_FALSE(FBGEMM_SIMD_IN(avx512_bf16));
  EXPECT_FALSE(FBGEMM_SIMD_IN(avx2, avx512_bf16, neon, sve128));
#undef FBGEMM_SIMD_NAMESPACE
}

TEST(PlatformSpecificKernelsMacrosTest, EverySlot) {
#define FBGEMM_SIMD_NAMESPACE sve128
  EXPECT_TRUE(FBGEMM_SIMD_IN(sve128, avx2, avx2, avx2, avx2, avx2, avx2));
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx2, avx2, avx2, sve128, avx2, avx2, avx2));
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx2, avx2, avx2, avx2, avx2, avx2, sve128));
  EXPECT_FALSE(FBGEMM_SIMD_IN(avx2, avx2, avx2, avx2, avx2, avx2, avx2));
#undef FBGEMM_SIMD_NAMESPACE
}

TEST(PlatformSpecificKernelsMacrosTest, PreprocessorCondition) {
#define FBGEMM_SIMD_NAMESPACE neon
#if !FBGEMM_SIMD_IN(neon, sve128)
  ADD_FAILURE() << "neon is listed";
#endif
#if FBGEMM_SIMD_IN(avx2, sve128)
  ADD_FAILURE() << "neon is not listed";
#endif
#undef FBGEMM_SIMD_NAMESPACE
}

TEST(PlatformSpecificKernelsMacrosTest, EighthPlatformHitsTooManyCheck) {
#define FBGEMM_SIMD_NAMESPACE sve128
#define FBGEMM_SIMD_TOO_MANY_PLATFORMS_neon() 1 ||
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx2, avx2, avx2, avx2, avx2, avx2, avx2, neon));
#undef FBGEMM_SIMD_TOO_MANY_PLATFORMS_neon
#undef FBGEMM_SIMD_NAMESPACE
}

TEST(PlatformSpecificKernelsMacrosTest, Avx512AndAvx512_256Bf16NeedAvx512Bf16) {
#pragma push_macro("FBGEMM_SIMD_avx512_AND_avx512_256_bf16_REQUIRE_avx512_bf16")
#undef FBGEMM_SIMD_avx512_AND_avx512_256_bf16_REQUIRE_avx512_bf16
#define FBGEMM_SIMD_avx512_AND_avx512_256_bf16_REQUIRE_avx512_bf16(ok) (ok)
#define FBGEMM_SIMD_NAMESPACE avx512
  EXPECT_FALSE(FBGEMM_SIMD_IN(avx512, avx512_256_bf16));
  EXPECT_FALSE(
      FBGEMM_SIMD_IN(avx2, avx2, avx2, avx2, avx2, avx512_256_bf16, avx512));
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx512, avx512_256_bf16, avx512_bf16));
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx2, avx512));
#undef FBGEMM_SIMD_NAMESPACE
#define FBGEMM_SIMD_NAMESPACE avx512_256_bf16
  EXPECT_FALSE(FBGEMM_SIMD_IN(avx512, avx512_256_bf16));
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx512_256_bf16));
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx2, avx512_256_bf16));
  EXPECT_TRUE(FBGEMM_SIMD_IN(avx512_256_bf16, avx512_bf16, avx512));
  EXPECT_TRUE(FBGEMM_SIMD_IN(
      avx512, avx2, avx512_256, avx512_256_bf16, neon, sve128, avx512_bf16));
#undef FBGEMM_SIMD_NAMESPACE
#pragma pop_macro("FBGEMM_SIMD_avx512_AND_avx512_256_bf16_REQUIRE_avx512_bf16")
}
