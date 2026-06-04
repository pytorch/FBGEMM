/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>
#endif
#define FBGEMM_EXPORTS
#include "./Bf16ConvertAvx2.h"
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {

namespace {

inline void FloatToBfloat16KernelAvx2(const float* src, bfloat16* dst) {
  // Two float m256i -> One bfloat16 m256i
  const __m256 src_reg0 = _mm256_loadu_ps(src);
  const __m256 src_reg1 = _mm256_loadu_ps(src + 8);
  __m256i dst_reg = internal::cvt_fp32x16_bf16x16(src_reg0, src_reg1);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), dst_reg);
}

inline void Bfloat16ToFloatKernelAvx2(const bfloat16* src, float* dst) {
  // One bfloat16 m128i -> One float m256i
  const __m128i src_reg =
      _mm_lddqu_si128(reinterpret_cast<const __m128i*>(src));
  __m256i dst_reg_bf16 = _mm256_cvtepu16_epi32(src_reg);
  __m256i dst_reg = _mm256_slli_epi32(dst_reg_bf16, 16);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), dst_reg);
}

} // namespace

void FloatToBfloat16_avx2(const float* src, bfloat16* dst, size_t size) {
  size_t i = 0;
  for (i = 0; i + 8 * 2 <= size; i += 8 * 2) {
    FloatToBfloat16KernelAvx2(src + i, dst + i);
  }
  FloatToBfloat16_ref(src + i, dst + i, size - i);
}

void Bfloat16ToFloat_avx2(const bfloat16* src, float* dst, size_t size) {
  size_t i = 0;
  for (i = 0; i + 8 <= size; i += 8) {
    Bfloat16ToFloatKernelAvx2(src + i, dst + i);
  }
  Bfloat16ToFloat_ref(src + i, dst + i, size - i);
}

} // namespace fbgemm
