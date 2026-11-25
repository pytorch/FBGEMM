/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && \
    defined(__AVX512VL__)
#define FBGEMM_SIMD_HAS_AVX512
#endif

// MSVC (not clang-cl) has no AVX-512 BF16 flag and never defines
// __AVX512BF16__, so we skip this check there.
#if defined(__AVX512BF16__) || (defined(_MSC_VER) && !defined(__clang__))
#define FBGEMM_SIMD_HAS_AVX512_BF16
#endif

#if defined(FBGEMM_SIMD_AVX512_BF16) || defined(FBGEMM_SIMD_AVX512) || \
    defined(FBGEMM_SIMD_AVX512_256) || defined(FBGEMM_SIMD_AVX512_256_BF16)
#ifndef FBGEMM_SIMD_HAS_AVX512
#error "Compile with AVX-512 flags"
#endif
#endif

#if defined(FBGEMM_SIMD_AVX512_BF16) || defined(FBGEMM_SIMD_AVX512_256_BF16)
#ifndef FBGEMM_SIMD_HAS_AVX512_BF16
#error "Compile with AVX-512 BF16 flags"
#endif
#endif

#if defined(FBGEMM_SIMD_AVX512_256) || defined(FBGEMM_SIMD_AVX512_256_BF16)
#if !defined(FBGEMM_PREFERRED_VECTOR_WIDTH) || \
    FBGEMM_PREFERRED_VECTOR_WIDTH != 256
#error \
    "Compile with -mprefer-vector-width=256 -DFBGEMM_PREFERRED_VECTOR_WIDTH=256"
#endif
#endif

#if defined(FBGEMM_SIMD_AVX2) && !defined(__AVX2__)
#error "Compile with AVX2 flags"
#endif

#if defined(FBGEMM_SIMD_SVE128) &&                                       \
    (!defined(__ARM_FEATURE_SVE2) || !defined(__ARM_FEATURE_SVE_BITS) || \
     __ARM_FEATURE_SVE_BITS != 128)
#error "Compile with SVE2 and -msve-vector-bits=128"
#endif

#if defined(FBGEMM_SIMD_NEON) && (!defined(__aarch64__) || !defined(__ARM_NEON))
#error "Compile for AArch64 with Advanced SIMD (ASIMD)"
#endif
