/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./PlatformSpecificKernelsIsa.h"

// We use autovectorization if the compiler can, see
// FbgemmFloat16ConvertKernelsAutovectorized.cc.
#ifndef FBGEMM_HAS_FLOAT16

#include <cstring>

#if FBGEMM_SIMD_IN(avx2, avx512)
#include <immintrin.h>
#endif

#if FBGEMM_SIMD_IN(neon)
#include <arm_neon.h>
#endif

#include "fbgemm/FloatConversion.h"

namespace fbgemm::FBGEMM_SIMD_NAMESPACE {

#if FBGEMM_SIMD_IN(avx2, avx512, neon)
namespace {

#if FBGEMM_SIMD_IN(avx2)
using RegisterF32 = __m256;
using RegisterF16 = __m256i;
using HalfRegisterF16 = __m128i;
#endif

#if FBGEMM_SIMD_IN(avx512)
using RegisterF32 = __m512;
using RegisterF16 = __m512i;
using HalfRegisterF16 = __m256i;
#endif

#if FBGEMM_SIMD_IN(neon)
using RegisterF32 = float32x4_t;
using RegisterF16 = float16x8_t;
using HalfRegisterF16 = float16x4_t;
#endif

constexpr size_t kRegisterBytes = sizeof(RegisterF32);
constexpr size_t kF32s = kRegisterBytes / sizeof(float);
constexpr size_t kF16s = kRegisterBytes / sizeof(float16);

struct RegisterF32Pair {
  RegisterF32 lo;
  RegisterF32 hi;
};

template <typename Register>
Register load(const void* src) {
  Register res;
  // NOLINTNEXTLINE(facebook-security-vulnerable-memcpy)
  std::memcpy(&res, src, sizeof(Register));
  return res;
}

template <typename Register>
void store(void* dst, Register x) {
  // NOLINTNEXTLINE(facebook-security-vulnerable-memcpy)
  std::memcpy(dst, &x, sizeof(Register));
}

#if FBGEMM_SIMD_IN(avx2)
RegisterF32Pair convertF16ToF32(RegisterF16 x) {
  return {
      _mm256_cvtph_ps(_mm256_castsi256_si128(x)),
      _mm256_cvtph_ps(_mm256_extracti128_si256(x, 1))};
}

HalfRegisterF16 convertF32ToF16(RegisterF32 x) {
  return _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

RegisterF32 clampf32(RegisterF32 x, float lo, float hi) {
  x = _mm256_min_ps(_mm256_set1_ps(hi), x);
  return _mm256_max_ps(x, _mm256_set1_ps(lo));
}
#endif

#if FBGEMM_SIMD_IN(avx512)
RegisterF32Pair convertF16ToF32(RegisterF16 x) {
  return {
      _mm512_cvtph_ps(_mm512_castsi512_si256(x)),
      _mm512_cvtph_ps(_mm512_extracti64x4_epi64(x, 1))};
}

HalfRegisterF16 convertF32ToF16(RegisterF32 x) {
  return _mm512_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

RegisterF32 clampf32(RegisterF32 x, float lo, float hi) {
  x = _mm512_min_ps(_mm512_set1_ps(hi), x);
  return _mm512_max_ps(x, _mm512_set1_ps(lo));
}
#endif

#if FBGEMM_SIMD_IN(neon)
RegisterF32Pair convertF16ToF32(RegisterF16 x) {
  return {vcvt_f32_f16(vget_low_f16(x)), vcvt_high_f32_f16(x)};
}

HalfRegisterF16 convertF32ToF16(RegisterF32 x) {
  return vcvt_f16_f32(x);
}

RegisterF32 clampf32(RegisterF32 x, float lo, float hi) {
  x = vminq_f32(x, vdupq_n_f32(hi));
  return vmaxnmq_f32(x, vdupq_n_f32(lo));
}
#endif

} // namespace
#endif

#if FBGEMM_SIMD_IN(avx2, avx512, neon)
void Kernels::Float16ToFloat32(const float16* src, float* dst, size_t size)
    const noexcept {
  size_t i = 0;
  for (; i + kF16s <= size; i += kF16s) {
    const auto [lo, hi] = convertF16ToF32(load<RegisterF16>(src + i));
    store(dst + i, lo);
    store(dst + i + kF32s, hi);
  }
  for (; i != size; ++i) {
    dst[i] = cpu_half2float(src[i]);
  }
}
#endif

#if FBGEMM_SIMD_IN(avx2, avx512, neon)
void Kernels::Float32ToFloat16(const float* src, float16* dst, size_t size)
    const noexcept {
  size_t i = 0;
  for (; i + kF16s <= size; i += kF16s) {
    store(dst + i, convertF32ToF16(load<RegisterF32>(src + i)));
    store(dst + i + kF32s, convertF32ToF16(load<RegisterF32>(src + i + kF32s)));
  }
  for (; i != size; ++i) {
    dst[i] = cpu_float2half(src[i]);
  }
}
#endif

#if FBGEMM_SIMD_IN(avx2, avx512, neon)
void Kernels::Float32ToFloat16Clipping(
    const float* src,
    float16* dst,
    size_t size) const noexcept {
  constexpr float kFloat16Max = 65504.f;
  size_t i = 0;
  for (; i + kF16s <= size; i += kF16s) {
    const RegisterF32 lo =
        clampf32(load<RegisterF32>(src + i), -kFloat16Max, kFloat16Max);
    const RegisterF32 hi =
        clampf32(load<RegisterF32>(src + i + kF32s), -kFloat16Max, kFloat16Max);
    store(dst + i, convertF32ToF16(lo));
    store(dst + i + kF32s, convertF32ToF16(hi));
  }
  for (; i != size; ++i) {
    dst[i] = cpu_float2half(clipToFloat16Range(src[i]));
  }
}
#endif

} // namespace fbgemm::FBGEMM_SIMD_NAMESPACE

#endif
