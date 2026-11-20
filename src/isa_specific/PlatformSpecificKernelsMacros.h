/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define FBGEMM_SIMD_ID_ref() 0
#define FBGEMM_SIMD_ID_avx2() 1
#define FBGEMM_SIMD_ID_avx512_256() 2
#define FBGEMM_SIMD_ID_avx512_256_bf16() 3
#define FBGEMM_SIMD_ID_avx512() 4
#define FBGEMM_SIMD_ID_avx512_bf16() 5
#define FBGEMM_SIMD_ID_neon() 6
#define FBGEMM_SIMD_ID_sve128() 7
#define FBGEMM_SIMD_ID_none() 8

#define FBGEMM_SIMD_EXPAND(x) x
#define FBGEMM_SIMD_ID_IMPL(isa) FBGEMM_SIMD_ID_##isa()
#define FBGEMM_SIMD_ID(isa) FBGEMM_SIMD_ID_IMPL(isa)
#define FBGEMM_SIMD_BIT(isa) (1 << FBGEMM_SIMD_ID(isa))
#define FBGEMM_SIMD_HAS(mask, isa) (((mask) >> FBGEMM_SIMD_ID(isa)) & 1)
#define FBGEMM_SIMD_avx512_AND_avx512_256_bf16_REQUIRE_avx512_bf16(ok) \
  (1 / (ok))
#define FBGEMM_SIMD_CHECK_OVERRIDE(mask)                      \
  FBGEMM_SIMD_avx512_AND_avx512_256_bf16_REQUIRE_avx512_bf16( \
      !(FBGEMM_SIMD_HAS(mask, avx512) &                       \
        FBGEMM_SIMD_HAS(mask, avx512_256_bf16)) |             \
      FBGEMM_SIMD_HAS(mask, avx512_bf16))
#ifdef FBGEMM_SIMD_DISABLED
#define FBGEMM_SIMD_DISABLED_MASK FBGEMM_SIMD_BIT(FBGEMM_SIMD_DISABLED)
#else
#define FBGEMM_SIMD_DISABLED_MASK 0
#endif
#define FBGEMM_SIMD_IN_MASK(mask) \
  FBGEMM_SIMD_HAS((mask) & ~FBGEMM_SIMD_DISABLED_MASK, FBGEMM_SIMD_NAMESPACE)
#define FBGEMM_SIMD_OVERRIDE_IN_MASK(mask) \
  (FBGEMM_SIMD_CHECK_OVERRIDE(mask) *      \
   FBGEMM_SIMD_IN_MASK((mask) | FBGEMM_SIMD_BIT(ref)))
#define FBGEMM_SIMD_TOO_MANY_PLATFORMS_none()
#define FBGEMM_SIMD_LIST_IMPL(check, a, b, c, d, e, f, g, h, last, ...) \
  (FBGEMM_SIMD_TOO_MANY_PLATFORMS_##last() check(                       \
      FBGEMM_SIMD_BIT(a) | FBGEMM_SIMD_BIT(b) | FBGEMM_SIMD_BIT(c) |    \
      FBGEMM_SIMD_BIT(d) | FBGEMM_SIMD_BIT(e) | FBGEMM_SIMD_BIT(f) |    \
      FBGEMM_SIMD_BIT(g) | FBGEMM_SIMD_BIT(h)))
#define FBGEMM_SIMD_IN(...)                 \
  FBGEMM_SIMD_EXPAND(FBGEMM_SIMD_LIST_IMPL( \
      FBGEMM_SIMD_IN_MASK,                  \
      __VA_ARGS__,                          \
      none,                                 \
      none,                                 \
      none,                                 \
      none,                                 \
      none,                                 \
      none,                                 \
      none,                                 \
      none))
#define FBGEMM_OVERRIDE_IN(...)             \
  FBGEMM_SIMD_EXPAND(FBGEMM_SIMD_LIST_IMPL( \
      FBGEMM_SIMD_OVERRIDE_IN_MASK,         \
      __VA_ARGS__,                          \
      none,                                 \
      none,                                 \
      none,                                 \
      none,                                 \
      none,                                 \
      none,                                 \
      none,                                 \
      none))
