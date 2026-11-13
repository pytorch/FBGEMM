/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../PlatformSpecificKernels.h"

#ifdef FBGEMM_SIMD_AVX512
#if !defined(__AVX512F__) || !defined(__AVX512BW__) || \
    !defined(__AVX512DQ__) || !defined(__AVX512VL__)
#error "Compile with AVX-512 flags"
#endif
#define FBGEMM_SIMD_NAMESPACE avx512
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

namespace fbgemm::FBGEMM_SIMD_NAMESPACE {

struct Kernels final : PlatformSpecificKernels {
  std::string_view name() const noexcept override;

  void Bfloat16ToFloat(const bfloat16* src, float* dst, size_t size)
      const noexcept override;
};

} // namespace fbgemm::FBGEMM_SIMD_NAMESPACE
