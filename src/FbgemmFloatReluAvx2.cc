/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <immintrin.h>
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {
void FloatRelu_avx2(const float* src, float* dst, size_t size) {
  size_t i = 0;
  __m256 zero_vector = _mm256_set1_ps(0.0f);
  for (i = 0; i + 8 <= size; i += 8) {
    __m256 float_vector = _mm256_loadu_ps(src + i);
    float_vector = _mm256_max_ps(zero_vector, float_vector);
    _mm256_storeu_ps(dst + i, float_vector);
  }
  FloatRelu_ref(src + i, dst + i, size - i);
}
} // namespace fbgemm
