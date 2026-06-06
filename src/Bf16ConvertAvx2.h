/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __AVX2__
#include <immintrin.h>

namespace fbgemm::internal {

// fp32->bf16 conversion via the "val + 0x8000, take high 16 bits" trick.
//
// NOTE: This rounds half values *away from zero*, NOT round-to-nearest-even
// (RNE). It therefore differs from:
//   - the scalar reference cpu_float2bfloat16(), which uses RNE, and
//   - the AVX-512 path (Fused8BitRowwiseQuantizedSBFloatToBfloat16Avx512),
//     which uses the hardware vcvtneps2bf16 (RNE).
// As a result, the same logical dequant can produce bf16 bit patterns that
// differ by up to 1 ULP depending on which backend runs (AVX-512-BF16 vs AVX2
// vs scalar Ref). Callers that need bit-exact agreement with
// cpu_float2bfloat16() must not use these helpers. Additionally, a NaN whose
// upper 7 mantissa bits are all 1 can carry into the exponent and become
// +/-inf.
inline __m256i cvt_fp32_to_bf16x8(__m256i val) {
  return _mm256_srli_epi32(
      _mm256_add_epi32(val, _mm256_set1_epi32(0x8000)), 16);
}

// Convert 2x8 fp32 to 16 packed bf16
inline __m256i cvt_fp32x16_bf16x16(__m256 a, __m256 b) {
  __m256i y0 = cvt_fp32_to_bf16x8(_mm256_castps_si256(a));
  __m256i y1 = cvt_fp32_to_bf16x8(_mm256_castps_si256(b));
  return _mm256_permute4x64_epi64(_mm256_packus_epi32(y0, y1), 0xd8);
}

// Convert 8 fp32 to 8 packed bf16 (128-bit result)
inline __m128i cvt_fp32x8_bf16x8(__m256 src) {
  __m256i r = cvt_fp32_to_bf16x8(_mm256_castps_si256(src));
  return _mm_packus_epi32(
      _mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
}

} // namespace fbgemm::internal

#endif
