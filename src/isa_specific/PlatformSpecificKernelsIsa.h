/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../PlatformSpecificKernels.h"
#include "./PlatformSpecificKernelsMacros.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && \
    defined(__AVX512VL__)
#define FBGEMM_SIMD_HAS_AVX512
#endif

// MSVC (not clang-cl) has no AVX-512 BF16 flag and never defines
// __AVX512BF16__, so we skip this check there.
#if defined(__AVX512BF16__) || (defined(_MSC_VER) && !defined(__clang__))
#define FBGEMM_SIMD_HAS_AVX512_BF16
#endif

#ifdef FBGEMM_SIMD_AVX512_BF16
#ifndef FBGEMM_SIMD_HAS_AVX512
#error "Compile with AVX-512 flags"
#endif
#ifndef FBGEMM_SIMD_HAS_AVX512_BF16
#error "Compile with AVX-512 BF16 flags"
#endif
#define FBGEMM_SIMD_NAMESPACE avx512_bf16
#elif defined(FBGEMM_SIMD_AVX512)
#ifndef FBGEMM_SIMD_HAS_AVX512
#error "Compile with AVX-512 flags"
#endif
#define FBGEMM_SIMD_NAMESPACE avx512
#elif defined(FBGEMM_SIMD_AVX512_256)
#ifndef FBGEMM_SIMD_HAS_AVX512
#error "Compile with AVX-512 flags"
#endif
#if !defined(FBGEMM_PREFERRED_VECTOR_WIDTH) || \
    FBGEMM_PREFERRED_VECTOR_WIDTH != 256
#error \
    "Compile with -mprefer-vector-width=256 -DFBGEMM_PREFERRED_VECTOR_WIDTH=256"
#endif
#define FBGEMM_SIMD_NAMESPACE avx512_256
#elif defined(FBGEMM_SIMD_AVX512_256_BF16)
#ifndef FBGEMM_SIMD_HAS_AVX512
#error "Compile with AVX-512 flags"
#endif
#ifndef FBGEMM_SIMD_HAS_AVX512_BF16
#error "Compile with AVX-512 BF16 flags"
#endif
#if !defined(FBGEMM_PREFERRED_VECTOR_WIDTH) || \
    FBGEMM_PREFERRED_VECTOR_WIDTH != 256
#error \
    "Compile with -mprefer-vector-width=256 -DFBGEMM_PREFERRED_VECTOR_WIDTH=256"
#endif
#define FBGEMM_SIMD_NAMESPACE avx512_256_bf16
#elif defined(FBGEMM_SIMD_AVX2)
#ifndef __AVX2__
#error "Compile with AVX2 flags"
#endif
#define FBGEMM_SIMD_NAMESPACE avx2
#elif defined(FBGEMM_SIMD_SVE128)
#if !defined(__ARM_FEATURE_SVE2) || !defined(__ARM_FEATURE_SVE_BITS) || \
    __ARM_FEATURE_SVE_BITS != 128
#error "Compile with SVE2 and -msve-vector-bits=128"
#endif
#define FBGEMM_SIMD_NAMESPACE sve128
#elif defined(FBGEMM_SIMD_NEON)
#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Compile for AArch64 with Advanced SIMD (ASIMD)"
#endif
#define FBGEMM_SIMD_NAMESPACE neon
#elif defined(FBGEMM_SIMD_REF)
#define FBGEMM_SIMD_NAMESPACE ref
#else
#error "Define FBGEMM_SIMD_<ISA> before including"
#endif

#pragma push_macro("FBGEMM_SIMD_NAMESPACE")
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE ref
#define FBGEMM_SIMD_BASE ::fbgemm::PlatformSpecificKernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx2
#define FBGEMM_SIMD_BASE ::fbgemm::ref::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx512_256
#define FBGEMM_SIMD_BASE ::fbgemm::avx2::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx512_256_bf16
#define FBGEMM_SIMD_BASE ::fbgemm::avx512_256::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx512
#define FBGEMM_SIMD_BASE ::fbgemm::avx512_256::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx512_bf16
#define FBGEMM_SIMD_BASE ::fbgemm::avx512::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE neon
#define FBGEMM_SIMD_BASE ::fbgemm::ref::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE sve128
#define FBGEMM_SIMD_BASE ::fbgemm::neon::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#pragma pop_macro("FBGEMM_SIMD_NAMESPACE")
