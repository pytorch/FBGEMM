/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "OptimizedKernelsAvx2.h"
#include <immintrin.h>

namespace fbgemm {

std::int32_t reduceAvx2(const std::uint8_t* A, int len) {
  std::int32_t row_sum = 0;
#if defined(__AVX2__)
  __m256i sum_v = _mm256_setzero_si256();
  __m256i one_epi16_v = _mm256_set1_epi16(1);
  __m256i one_epi8_v = _mm256_set1_epi8(1);

  int i;
  // vectorized
  for (i = 0; i < len / 32 * 32; i += 32) {
    __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    sum_v = _mm256_add_epi32(
        sum_v,
        _mm256_madd_epi16(
            _mm256_maddubs_epi16(src_v, one_epi8_v), one_epi16_v));
  }

  alignas(64) std::int32_t temp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);
  for (int k = 0; k < 8; ++k) {
    row_sum += temp[k];
  }

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i];
  }

#else
  for (int i = 0; i < len; ++i) {
    row_sum += A[i];
  }
#endif
  return row_sum;
}

} // namespace fbgemm
