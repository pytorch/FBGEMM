/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm/QuantUtilsAvx2.h"
#include <immintrin.h>
#include <algorithm> //for std::min/std::max
#include <cmath> //for nearbyint
#include <limits> //for numeric_limits
#include "fbgemm/Fbgemm.h" //for ReQuantizeOutput

namespace fbgemm {

using namespace std;
////////////////////////////////////////////////////////////////////////////////
// Utility functions

// FIXME: code duplication with PackAWithQuantRowOffset
void QuantizeAvx2(
    const float* src,
    uint8_t* dst,
    int len,
    const TensorQuantizationParams& qparams) {
#if defined(__AVX2__) && defined(__FMA__)
  constexpr int VLEN = 8;
  std::size_t i = 0;
  __m256 inverse_scale_v = _mm256_set1_ps(1.f / qparams.scale);
  for (; i < len / VLEN * VLEN; i += VLEN) {
    __m256 src_v = _mm256_loadu_ps(src + i);
    __m256 transformed_v = _mm256_fmadd_ps(
        src_v, inverse_scale_v, _mm256_set1_ps(qparams.zero_point));
    __m256 clipped_v = _mm256_min_ps(
        _mm256_max_ps(transformed_v, _mm256_set1_ps(0.f)),
        _mm256_set1_ps(255.f));
    __m256i rounded_v = _mm256_cvtps_epi32(clipped_v);
    alignas(64) std::int32_t temp_int32[VLEN];
    _mm256_store_si256((__m256i*)temp_int32, rounded_v);
    for (int j = 0; j < VLEN; ++j) {
      dst[i + j] = temp_int32[j];
    }
  }

  for (; i < len; ++i) {
    float transformed = qparams.zero_point + src[i] / qparams.scale;
    float clipped = std::min(std::max(transformed, 0.f), 255.f);
    // Not exactly the same behavior as the vectorized code.
    // The vectorized code above always rounds to even in halfway cases
    // (https://software.intel.com/en-us/node/523819), but std::nearbyint
    // does the same only when the current rounding mode is FE_TONEAREST.
    // However, in practice, this should not be a problem because most cases
    // use the default rounding mode FE_TONEAREST.
    // Note that we cannot implement the same behavior as the vectorized code
    // using std::round because it does rounding away from zero in halfway
    // cases.
    dst[i] = nearbyint(clipped);
  }
#endif
}

void FindMinMax(const float* a, float* min, float* max, int len) {
  if (len <= 0) {
    *min = 0.0f;
    *max = 0.0f;
    return;
  }

  float temp_min = *a, temp_max = *a;
  int i = 0;

#ifdef __AVX__
  __m256 min_v = _mm256_set1_ps(*a), max_v = _mm256_set1_ps(*a);
  constexpr int VLEN = 8;
  if (len >= VLEN) {
    for (; i < len / VLEN * VLEN; i += VLEN) {
      min_v = _mm256_min_ps(min_v, _mm256_loadu_ps(a + i));
      max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(a + i));
    }

    float min_buf[VLEN], max_buf[VLEN];
    _mm256_storeu_ps(min_buf, min_v);
    _mm256_storeu_ps(max_buf, max_v);
    for (int j = 0; j < VLEN; ++j) {
      temp_min = std::min(temp_min, min_buf[j]);
      temp_max = std::max(temp_max, max_buf[j]);
    }
  }
#endif

  for (; i < len; i++) {
    temp_min = std::min(temp_min, a[i]);
    temp_max = std::max(temp_max, a[i]);
  }
  *min = temp_min;
  *max = temp_max;
}

////////////////////////////////////////////////////////////////////////////////
// Requantization (with floats)

#ifdef __AVX2__
void RequantizeAvx2(
    const int32_t* src,
    uint8_t* dst,
    int len,
    const RequantizationParams& params) {
  DoNothing<> doNothingObj{};
  ReQuantizeOutput<false /* FUSE_RELU */> requantizeObj(
      doNothingObj,
      &params.real_multiplier,
      params.target_qparams.zero_point,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      len);
  requantizeObj.f<inst_set_t::avx2>(dst, src, {0, 1, 0, len}, 0, 0);
}

void RequantizeFixedPointAvx2(
    const int32_t* src,
    uint8_t* dst,
    int len,
    const RequantizationParams& params) {
  constexpr int VLEN = 8;

  __m256i b = _mm256_set1_epi32(params.multiplier);

  // AVX2 doesn't support arithmetic right shift.
  // As a work around, we convert 64-bit multiplied results to uint64_t by
  // adding 0x8000000000000000ULL, logical right shift, and subtract by
  // (0x8000000000000000ULL >> right_shift).
  __m256i pre_shift_nudge = _mm256_set1_epi64x(
      (1ll << (params.right_shift - 1)) + 0x8000000000000000ULL);
  __m256i post_shift_nudge = _mm256_set1_epi64x(
      params.target_qparams.zero_point -
      (0x8000000000000000ULL >> params.right_shift));

  __m256i min_v = _mm256_set1_epi32(numeric_limits<uint8_t>::min());
  __m256i max_v = _mm256_set1_epi32(numeric_limits<uint8_t>::max());

  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  int i = 0;
  for (; i < len / VLEN * VLEN; i += VLEN) {
    __m256i a_v = _mm256_loadu_si256((const __m256i*)(src + i));

    // a = a0 | a1 | a2 | a3 | a4 | a5 | a6 | a7
    // b = b0 | b1 | b3 | b3 | b4 | b5 | b6 | b7
    __m256i a_even_v = a_v;
    __m256i a_odd_v = _mm256_srli_si256(a_v, 4);

    __m256i ab_even_v = _mm256_mul_epi32(a_even_v, b);
    __m256i ab_odd_v = _mm256_mul_epi32(a_odd_v, b);

    __m256i even_rounded_v = _mm256_add_epi64(ab_even_v, pre_shift_nudge);
    __m256i odd_rounded_v = _mm256_add_epi64(ab_odd_v, pre_shift_nudge);

    __m256i even_result_v = _mm256_add_epi64(
        _mm256_srli_epi64(even_rounded_v, params.right_shift),
        post_shift_nudge);
    __m256i odd_result_v = _mm256_add_epi64(
        _mm256_srli_epi64(odd_rounded_v, params.right_shift), post_shift_nudge);
    odd_result_v = _mm256_slli_si256(odd_result_v, 4);

    // even_result_v has numbers we want in its even 32-bit SIMD lanes, and
    // odd_result_v has numbers we want in its odd 32-bit SIMD lanes.
    // Use blend to combine them.
    __m256i result_v = _mm256_blend_epi32(even_result_v, odd_result_v, 0xaa);
    __m256i clipped_v =
        _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, result_v));

    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);
    *(int64_t*)(dst + i) = _mm256_extract_epi64(clipped_v, 0);
  }

  for (; i < len; ++i) {
    int64_t ab_64 =
        static_cast<int64_t>(src[i]) * static_cast<int64_t>(params.multiplier);
    int64_t nudge = 1ll << std::max(0, params.right_shift - 1);
    int64_t quantized_down = params.target_qparams.zero_point +
        ((ab_64 + nudge) >> params.right_shift);
    dst[i] = std::min<int64_t>(std::max<int64_t>(quantized_down, 0l), 255l);
  }
}
#endif

} // namespace fbgemm
