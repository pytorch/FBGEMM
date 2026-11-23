/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./PlatformSpecificKernelsIsa.h"

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
using RegisterBF16 = __m256i;
#endif

#if FBGEMM_SIMD_IN(avx512)
using RegisterF32 = __m512;
using RegisterBF16 = __m512i;
#endif

#if FBGEMM_SIMD_IN(neon)
using RegisterF32 = float32x4_t;
using RegisterBF16 = uint16x8_t;
#endif

constexpr size_t kRegisterBytes = sizeof(RegisterF32);
constexpr size_t kF32s = kRegisterBytes / sizeof(float);
constexpr size_t kBF16s = kRegisterBytes / sizeof(bfloat16);

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
__m256i roundF32ToBF16Bits(RegisterF32 x) {
  const __m256i bits = _mm256_castps_si256(x);
  const __m256i isNan = _mm256_cmpgt_epi32(
      _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)),
      _mm256_set1_epi32(0x7f800000));
  const __m256i nan =
      _mm256_or_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(0x0040));
  const __m256i rounded =
      _mm256_srli_epi32(_mm256_add_epi32(bits, _mm256_set1_epi32(1 << 15)), 16);
  return _mm256_blendv_epi8(rounded, nan, isNan);
}

RegisterBF16 convertF32ToBF16(RegisterF32Pair x) {
  return _mm256_permute4x64_epi64(
      _mm256_packus_epi32(roundF32ToBF16Bits(x.lo), roundF32ToBF16Bits(x.hi)),
      0xd8);
}
#endif

#if FBGEMM_SIMD_IN(avx512)
__m512i roundF32ToBF16Bits(RegisterF32 x) {
  const __m512i bits = _mm512_castps_si512(x);
  const __mmask16 isNan = _mm512_cmpgt_epi32_mask(
      _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
      _mm512_set1_epi32(0x7f800000));
  const __m512i nan =
      _mm512_or_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(0x0040));
  const __m512i rounded =
      _mm512_srli_epi32(_mm512_add_epi32(bits, _mm512_set1_epi32(1 << 15)), 16);
  return _mm512_mask_mov_epi32(rounded, isNan, nan);
}

RegisterBF16 convertF32ToBF16(RegisterF32Pair x) {
  return _mm512_permutexvar_epi64(
      _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7),
      _mm512_packus_epi32(roundF32ToBF16Bits(x.lo), roundF32ToBF16Bits(x.hi)));
}
#endif

#if FBGEMM_SIMD_IN(neon)
uint32x4_t isNan(RegisterF32 x) {
  return vcgtq_u32(
      vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x7fffffff)),
      vdupq_n_u32(0x7f800000));
}

RegisterBF16 convertF32ToBF16(RegisterF32Pair x) {
  const uint32x4_t lo = vreinterpretq_u32_f32(x.lo);
  const uint32x4_t hi = vreinterpretq_u32_f32(x.hi);
  const uint16x8_t nanMask = vuzp1q_u16(
      vreinterpretq_u16_u32(isNan(x.lo)), vreinterpretq_u16_u32(isNan(x.hi)));
  const uint16x8_t nan = vorrq_u16(
      vshrn_high_n_u32(vshrn_n_u32(lo, 16), hi, 16), vdupq_n_u16(0x0040));
  const uint32x4_t half = vdupq_n_u32(1 << 15);
  const uint16x8_t rounded = vaddhn_high_u32(vaddhn_u32(lo, half), hi, half);
  return vbslq_u16(nanMask, nan, rounded);
}
#endif

} // namespace
#endif

#if FBGEMM_SIMD_IN(avx2, avx512, neon)
void Kernels::Float32ToBFloat16(const float* src, bfloat16* dst, size_t size)
    const noexcept {
  size_t i = 0;
  for (; i + kBF16s <= size; i += kBF16s) {
    store(
        dst + i,
        convertF32ToBF16(
            {load<RegisterF32>(src + i), load<RegisterF32>(src + i + kF32s)}));
  }
  for (; i != size; ++i) {
    dst[i] = cpu_float2bfloat16(src[i]);
  }
}
#endif

} // namespace fbgemm::FBGEMM_SIMD_NAMESPACE
