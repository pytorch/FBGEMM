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
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace {

// Emulated ties-to-even; used when AVX-512-BF16 is not present.
inline __m256i QuantizeBfloat16Avx512(const __m512& x0) {
  __m512i val = _mm512_castps_si512(x0);
  __m512i lsb =
      _mm512_and_si512(_mm512_srli_epi32(val, 16), _mm512_set1_epi32(1));
  __m512i rnd =
      _mm512_add_epi32(val, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7FFF)));
  return _mm512_cvtepi32_epi16(_mm512_srli_epi32(rnd, 16));
}

// Hardware vcvtneps2bf16; requires AVX-512-BF16.
inline __m256i QuantizeBfloat16Avx512Bf16(const __m512& x0) {
  return (__m256i)_mm512_cvtneps_pbh(x0);
}

inline void FloatToBfloat16KernelAvx512(const float* src, bfloat16* dst) {
  // One float m512i -> One bfloat16 m256i
  const __m512 src_reg0 = _mm512_loadu_ps(src);
  __m256i dst_reg0 = QuantizeBfloat16Avx512(src_reg0);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), dst_reg0);
}

inline void FloatToBfloat16KernelAvx512Bf16(const float* src, bfloat16* dst) {
  const __m512 src_reg0 = _mm512_loadu_ps(src);
  __m256i dst_reg0 = QuantizeBfloat16Avx512Bf16(src_reg0);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), dst_reg0);
}

inline void Bfloat16ToFloatKernelAvx512(const bfloat16* src, float* dst) {
  // One bfloat16 m256i -> One float m512i
  const __m256i src_reg =
      _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(src));
  __m512i dst_reg_bf16 = _mm512_cvtepu16_epi32(src_reg);
  __m512i dst_reg = _mm512_slli_epi32(dst_reg_bf16, 16);
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), dst_reg);
}

} // namespace

void FloatToBfloat16_avx512(const float* src, bfloat16* dst, size_t size) {
  static const bool has_bf16 = fbgemmHasAvx512Bf16Support();
  size_t i = 0;
  if (has_bf16) {
    for (; i + 16 <= size; i += 16) {
      FloatToBfloat16KernelAvx512Bf16(src + i, dst + i);
    }
  } else {
    for (; i + 16 <= size; i += 16) {
      FloatToBfloat16KernelAvx512(src + i, dst + i);
    }
  }
  FloatToBfloat16_avx2(src + i, dst + i, size - i);
}

void Bfloat16ToFloat_avx512(const bfloat16* src, float* dst, size_t size) {
  size_t i = 0;
  for (i = 0; i + 16 <= size; i += 16) {
    Bfloat16ToFloatKernelAvx512(src + i, dst + i);
  }
  Bfloat16ToFloat_avx2(src + i, dst + i, size - i);
}

} // namespace fbgemm
