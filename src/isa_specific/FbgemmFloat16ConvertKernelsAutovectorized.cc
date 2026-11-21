/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./PlatformSpecificKernelsIsa.h"

#if FBGEMM_SIMD_IN(ref) || \
    (defined(FBGEMM_HAS_FLOAT16) && FBGEMM_SIMD_IN(avx2, avx512, neon))

#include "fbgemm/FloatConversion.h"

namespace fbgemm::FBGEMM_SIMD_NAMESPACE {

#if FBGEMM_SIMD_IN(ref, avx2, avx512, neon)
void Kernels::Float16ToFloat32(const float16* src, float* dst, size_t size)
    const noexcept {
  for (size_t i = 0; i != size; ++i) {
    dst[i] = cpu_half2float(src[i]);
  }
}
#endif

#if FBGEMM_SIMD_IN(ref, avx2, avx512, neon)
void Kernels::Float32ToFloat16(const float* src, float16* dst, size_t size)
    const noexcept {
  for (size_t i = 0; i != size; ++i) {
    dst[i] = cpu_float2half(src[i]);
  }
}
#endif

#if FBGEMM_SIMD_IN(ref, avx2, avx512, neon)
void Kernels::Float32ToFloat16Clipping(
    const float* src,
    float16* dst,
    size_t size) const noexcept {
  for (size_t i = 0; i != size; ++i) {
    dst[i] = cpu_float2half(clipToFloat16Range(src[i]));
  }
}
#endif

} // namespace fbgemm::FBGEMM_SIMD_NAMESPACE

#endif
