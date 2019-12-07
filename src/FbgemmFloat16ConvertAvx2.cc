/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <immintrin.h>
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {

namespace {

inline void FloatToFloat16KernelAvx2(const float* src, float16* dst) {
  __m256 float_vector = _mm256_loadu_ps(src);
  __m128i half_vector = _mm256_cvtps_ph(
      float_vector, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  _mm_storeu_si128((__m128i*)dst, half_vector);
}

inline void Float16ToFloatKernelAvx2(const float16* src, float* dst) {
  __m128i half_vector = _mm_loadu_si128((__m128i*)src);
  __m256 float_vector = _mm256_cvtph_ps(half_vector);
  _mm256_storeu_ps(dst, float_vector);
}

} // namespace

void FloatToFloat16_avx2(const float* src, float16* dst, int size) {
  int i = 0;
  for (i = 0; i + 8 <= size; i += 8) {
    FloatToFloat16KernelAvx2(src + i, dst + i);
  }
  FloatToFloat16_ref(src + i, dst + i, size - i);
}

void Float16ToFloat_avx2(const float16* src, float* dst, int size) {
  int i = 0;
  for (i = 0; i + 8 <= size; i += 8) {
    Float16ToFloatKernelAvx2(src + i, dst + i);
  }
  Float16ToFloat_ref(src + i, dst + i, size - i);
}

} // namespace fbgemm
