/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm> // for min and max
#include <cassert>
#include <cmath> // for lrintf and sqrt
#include <cstdint>
#include <type_traits> // for is_same

#include <immintrin.h>

namespace fbgemm {

// c = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
// A is in uint8_t
// B is in int8_t and pre-interleaved
// C is in int32_t and 4 registers have results in the following layout:
// c0_v:   c[0:4], c[16:20]
// c1_v:   c[4:8], c[20:24]
// c2_v:  c[8:12], c[24:28]
// c3_v: c[12:16], c[28:32]
template <bool SUM_A = false>
static ALWAYS_INLINE void madd_epi16x4_packed(
    __m256i a0_v,
    __m256i a1_v,
    __m256i a2_v,
    __m256i a3_v,
    const __m256i* b,
    __m256i* c0_v,
    __m256i* c1_v,
    __m256i* c2_v,
    __m256i* c3_v,
    __m256i* a_sum = nullptr) {
  __m256i a01_lo_v = _mm256_unpacklo_epi8(a0_v, a1_v);
  __m256i a01_hi_v = _mm256_unpackhi_epi8(a0_v, a1_v);
  __m256i a23_lo_v = _mm256_unpacklo_epi8(a2_v, a3_v);
  __m256i a23_hi_v = _mm256_unpackhi_epi8(a2_v, a3_v);

  if (SUM_A) {
    __m256i one_epi8_v = _mm256_set1_epi8(1);
    a_sum[0] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a01_lo_v, one_epi8_v), a_sum[0]);
    a_sum[1] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a01_hi_v, one_epi8_v), a_sum[1]);
    a_sum[0] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a23_lo_v, one_epi8_v), a_sum[0]);
    a_sum[1] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a23_hi_v, one_epi8_v), a_sum[1]);
  }

  __m256i a0_interleaved_v = _mm256_unpacklo_epi16(a01_lo_v, a23_lo_v);
  __m256i a1_interleaved_v = _mm256_unpackhi_epi16(a01_lo_v, a23_lo_v);
  __m256i a2_interleaved_v = _mm256_unpacklo_epi16(a01_hi_v, a23_hi_v);
  __m256i a3_interleaved_v = _mm256_unpackhi_epi16(a01_hi_v, a23_hi_v);

  __m256i b0_v = _mm256_load_si256(b + 0);
  __m256i b1_v = _mm256_load_si256(b + 1);
  __m256i b2_v = _mm256_load_si256(b + 2);
  __m256i b3_v = _mm256_load_si256(b + 3);

  __m256i ab0 = _mm256_maddubs_epi16(a0_interleaved_v, b0_v);
  __m256i ab1 = _mm256_maddubs_epi16(a1_interleaved_v, b1_v);
  __m256i ab2 = _mm256_maddubs_epi16(a2_interleaved_v, b2_v);
  __m256i ab3 = _mm256_maddubs_epi16(a3_interleaved_v, b3_v);

  __m256i one_v = _mm256_set1_epi16(1);
  *c0_v = _mm256_madd_epi16(ab0, one_v);
  *c1_v = _mm256_madd_epi16(ab1, one_v);
  *c2_v = _mm256_madd_epi16(ab2, one_v);
  *c3_v = _mm256_madd_epi16(ab3, one_v);
}

// c = a0 * b0 + a1 * b1 + a2 * b2
// A is in uint8_t
// B is in int8_t and pre-interleaved
// C is in int32_t and 4 registers have results in the following layout:
// c0_v:   c[0:4], c[16:20]
// c1_v:   c[4:8], c[20:24]
// c2_v:  c[8:12], c[24:28]
// c3_v: c[12:16], c[28:32]
template <bool SUM_A = false>
static ALWAYS_INLINE void madd_epi16x3_packed(
    __m256i a0_v,
    __m256i a1_v,
    __m256i a2_v,
    const __m256i* b,
    __m256i* c0_v,
    __m256i* c1_v,
    __m256i* c2_v,
    __m256i* c3_v,
    __m256i* a_sum = nullptr) {
  __m256i zero_v = _mm256_setzero_si256();

  __m256i a01_lo_v = _mm256_unpacklo_epi8(a0_v, a1_v);
  __m256i a01_hi_v = _mm256_unpackhi_epi8(a0_v, a1_v);
  __m256i a23_lo_v = _mm256_unpacklo_epi8(a2_v, zero_v);
  __m256i a23_hi_v = _mm256_unpackhi_epi8(a2_v, zero_v);

  if (SUM_A) {
    __m256i one_epi8_v = _mm256_set1_epi8(1);
    a_sum[0] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a01_lo_v, one_epi8_v), a_sum[0]);
    a_sum[1] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a01_hi_v, one_epi8_v), a_sum[1]);
    a_sum[0] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a23_lo_v, one_epi8_v), a_sum[0]);
    a_sum[1] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a23_hi_v, one_epi8_v), a_sum[1]);
  }

  __m256i a0_interleaved_v = _mm256_unpacklo_epi16(a01_lo_v, a23_lo_v);
  __m256i a1_interleaved_v = _mm256_unpackhi_epi16(a01_lo_v, a23_lo_v);
  __m256i a2_interleaved_v = _mm256_unpacklo_epi16(a01_hi_v, a23_hi_v);
  __m256i a3_interleaved_v = _mm256_unpackhi_epi16(a01_hi_v, a23_hi_v);

  __m256i b0_v = _mm256_load_si256(b + 0);
  __m256i b1_v = _mm256_load_si256(b + 1);
  __m256i b2_v = _mm256_load_si256(b + 2);
  __m256i b3_v = _mm256_load_si256(b + 3);

  __m256i ab0 = _mm256_maddubs_epi16(a0_interleaved_v, b0_v);
  __m256i ab1 = _mm256_maddubs_epi16(a1_interleaved_v, b1_v);
  __m256i ab2 = _mm256_maddubs_epi16(a2_interleaved_v, b2_v);
  __m256i ab3 = _mm256_maddubs_epi16(a3_interleaved_v, b3_v);

  __m256i one_v = _mm256_set1_epi16(1);
  *c0_v = _mm256_madd_epi16(ab0, one_v);
  *c1_v = _mm256_madd_epi16(ab1, one_v);
  *c2_v = _mm256_madd_epi16(ab2, one_v);
  *c3_v = _mm256_madd_epi16(ab3, one_v);
}

// c = a0 * b0 + a1 * b1
// A is in uint8_t
// B is in int8_t and pre-interleaved
// C is in int32_t and 4 registers have results in the following layout:
// c0_v:   c[0:4],   c[4:8]
// c1_v:  c[8:12], c[12:16]
// c2_v: c[16:20], c[20:24]
// c3_v: c[24:28], c[28:32]
template <bool SUM_A = false>
static ALWAYS_INLINE void madd_epi16x2_packed(
    __m256i a0_v,
    __m256i a1_v,
    const __m256i* b,
    __m256i* c0_v,
    __m256i* c1_v,
    __m256i* c2_v,
    __m256i* c3_v,
    __m256i* a_sum = nullptr) {
  __m256i a_lo_v = _mm256_unpacklo_epi8(a0_v, a1_v);
  __m256i a_hi_v = _mm256_unpackhi_epi8(a0_v, a1_v);

  if (SUM_A) {
    __m256i one_epi8_v = _mm256_set1_epi8(1);
    a_sum[0] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a_lo_v, one_epi8_v), a_sum[0]);
    a_sum[1] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a_hi_v, one_epi8_v), a_sum[1]);
  }

  __m256i b0_v = _mm256_load_si256(b + 0);
  __m256i b1_v = _mm256_load_si256(b + 1);

  __m256i ab_lo_v = _mm256_maddubs_epi16(a_lo_v, b0_v);
  __m256i ab_hi_v = _mm256_maddubs_epi16(a_hi_v, b1_v);

  *c0_v = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ab_lo_v));
  *c1_v = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ab_hi_v));
  *c2_v = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ab_lo_v, 1));
  *c3_v = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ab_hi_v, 1));
}

// c = a0 * b0
// A is in uint8_t
// B is in int8_t and pre-interleaved
// C is in int32_t and 4 registers have results in the following layout:
// c0_v:   c[0:4],   c[4:8]
// c1_v:  c[8:12], c[12:16]
// c2_v: c[16:20], c[20:24]
// c3_v: c[24:28], c[28:32]
template <bool SUM_A = false>
static ALWAYS_INLINE void madd_epi16_packed(
    __m256i a_v,
    const __m256i* b,
    __m256i* c0_v,
    __m256i* c1_v,
    __m256i* c2_v,
    __m256i* c3_v,
    __m256i* a_sum = nullptr) {
  __m256i zero_v = _mm256_setzero_si256();

  __m256i a_lo_v = _mm256_unpacklo_epi8(a_v, zero_v);
  __m256i a_hi_v = _mm256_unpackhi_epi8(a_v, zero_v);

  if (SUM_A) {
    __m256i one_epi8_v = _mm256_set1_epi8(1);
    a_sum[0] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a_lo_v, one_epi8_v), a_sum[0]);
    a_sum[1] =
        _mm256_adds_epi16(_mm256_maddubs_epi16(a_hi_v, one_epi8_v), a_sum[1]);
  }

  __m256i b0_v = _mm256_load_si256(b + 0);
  __m256i b1_v = _mm256_load_si256(b + 1);

  __m256i ab_lo_v = _mm256_maddubs_epi16(a_lo_v, b0_v);
  __m256i ab_hi_v = _mm256_maddubs_epi16(a_hi_v, b1_v);

  *c0_v = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ab_lo_v));
  *c1_v = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ab_hi_v));
  *c2_v = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ab_lo_v, 1));
  *c3_v = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ab_hi_v, 1));
}

// K is the number of accumulations we're doing
template <int K, bool SUM_A = false, bool REMAINDER = false, bool ACC = false>
static ALWAYS_INLINE void inner_prod_packed_(
    const __m256i* a_v,
    const __m256i* Bp,
    std::int32_t* C,
    int remainder,
    __m256i* a_sum = nullptr) {
  __m256i c[4], c_temp[4];
  __m256i a_sum_temp[2] = {0, 0};

  int k = 0;
  if (K >= 4) {
    madd_epi16x4_packed<SUM_A>(
        a_v[0],
        a_v[1],
        a_v[2],
        a_v[3],
        Bp,
        &c[0],
        &c[1],
        &c[2],
        &c[3],
        a_sum_temp);

    for (k = 4; k < K / 4 * 4; k += 4) {
      madd_epi16x4_packed<SUM_A>(
          a_v[k + 0],
          a_v[k + 1],
          a_v[k + 2],
          a_v[k + 3],
          Bp + k,
          &c_temp[0],
          &c_temp[1],
          &c_temp[2],
          &c_temp[3],
          a_sum_temp);

      c[0] = _mm256_add_epi32(c[0], c_temp[0]);
      c[1] = _mm256_add_epi32(c[1], c_temp[1]);
      c[2] = _mm256_add_epi32(c[2], c_temp[2]);
      c[3] = _mm256_add_epi32(c[3], c_temp[3]);
    }
  } else {
    c[0] = _mm256_setzero_si256();
    c[1] = _mm256_setzero_si256();
    c[2] = _mm256_setzero_si256();
    c[3] = _mm256_setzero_si256();
  }

  if (K - k == 3) {
    madd_epi16x3_packed<SUM_A>(
        a_v[k],
        a_v[k + 1],
        a_v[k + 2],
        Bp + k,
        &c_temp[0],
        &c_temp[1],
        &c_temp[2],
        &c_temp[3],
        a_sum_temp);

    c[0] = _mm256_add_epi32(c[0], c_temp[0]);
    c[1] = _mm256_add_epi32(c[1], c_temp[1]);
    c[2] = _mm256_add_epi32(c[2], c_temp[2]);
    c[3] = _mm256_add_epi32(c[3], c_temp[3]);
  }

  c_temp[0] = _mm256_permute2f128_si256(c[0], c[1], 0x20);
  c_temp[1] = _mm256_permute2f128_si256(c[2], c[3], 0x20);
  c_temp[2] = _mm256_permute2f128_si256(c[0], c[1], 0x31);
  c_temp[3] = _mm256_permute2f128_si256(c[2], c[3], 0x31);

  if (K - k == 0 || K - k == 3) {
    c[0] = c_temp[0];
    c[1] = c_temp[1];
    c[2] = c_temp[2];
    c[3] = c_temp[3];
  } else {
    if (K - k == 1) {
      madd_epi16_packed<SUM_A>(
          a_v[k], Bp + k, &c[0], &c[1], &c[2], &c[3], a_sum_temp);
    } else if (K - k == 2) {
      madd_epi16x2_packed<SUM_A>(
          a_v[k], a_v[k + 1], Bp + k, &c[0], &c[1], &c[2], &c[3], a_sum_temp);
    }

    c[0] = _mm256_add_epi32(c[0], c_temp[0]);
    c[1] = _mm256_add_epi32(c[1], c_temp[1]);
    c[2] = _mm256_add_epi32(c[2], c_temp[2]);
    c[3] = _mm256_add_epi32(c[3], c_temp[3]);
  }

  if (REMAINDER) {
    for (int r = 0; r < remainder / 8; ++r) {
      if (ACC) {
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(C + r * 8),
            _mm256_add_epi32(
                _mm256_loadu_si256(reinterpret_cast<__m256i*>(C + r * 8)),
                c[r]));
      } else {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(C + r * 8), c[r]);
      }
    }
  } else {
    if (ACC) {
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(C),
          _mm256_add_epi32(
              _mm256_loadu_si256(reinterpret_cast<__m256i*>(C)), c[0]));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(C + 8),
          _mm256_add_epi32(
              _mm256_loadu_si256(reinterpret_cast<__m256i*>(C + 8)), c[1]));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(C + 16),
          _mm256_add_epi32(
              _mm256_loadu_si256(reinterpret_cast<__m256i*>(C + 16)), c[2]));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(C + 24),
          _mm256_add_epi32(
              _mm256_loadu_si256(reinterpret_cast<__m256i*>(C + 24)), c[3]));
    } else {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(C), c[0]);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(C + 8), c[1]);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(C + 16), c[2]);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(C + 24), c[3]);
    }
  }

  if (SUM_A) {
    a_sum[0] = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_sum_temp[0]));
    a_sum[1] = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_sum_temp[1]));
    a_sum[2] =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a_sum_temp[0], 1));
    a_sum[3] =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a_sum_temp[1], 1));
  }
}

// Almost same as ReQuantizeOutput in OutputProcessing-inh.h but different
// row_offsets for each row because of depth-wise convolution
template <
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool PER_CHANNEL_QUANTIZATION,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    typename BIAS_TYPE>
static ALWAYS_INLINE void requantize_(
    std::int32_t A_zero_point,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    const std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    int n,
    const std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale = nullptr) {
  __m256 multiplier_v = _mm256_setzero_ps();
  // Broadcasted reciprocal of act_times_w_scale
  __m256 act_times_w_rcp_v = _mm256_setzero_ps();
  if (!PER_CHANNEL_QUANTIZATION) {
    multiplier_v = _mm256_set1_ps(*C_multiplier);
    if (std::is_same<BIAS_TYPE, float>::value) {
      act_times_w_rcp_v = _mm256_set1_ps(1.0f / (*act_times_w_scale));
    }
  }

  __m256i min_v = _mm256_set1_epi8(static_cast<std::uint8_t>(0));
  __m256i max_v = _mm256_set1_epi8(static_cast<std::uint8_t>(255));

  if (A_SYMMETRIC) {
    assert(A_zero_point == 0 || col_offsets == nullptr);
  }
  __m256i A_zero_point_v = _mm256_set1_epi32(A_zero_point);
  __m256i C_zero_point_epi16_v = _mm256_set1_epi16(C_zero_point);
  __m256i C_zero_point_epi8_v = _mm256_set1_epi8(C_zero_point);

  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  constexpr int VLEN = 8;
  int j = 0;
  for (; j < n / (VLEN * 4) * (VLEN * 4); j += (VLEN * 4)) {
    __m256i x_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C_int32 + j));
    __m256i y_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(C_int32 + j + VLEN));
    __m256i z_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(C_int32 + j + 2 * VLEN));
    __m256i w_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(C_int32 + j + 3 * VLEN));

    __m256i row_offset_v;
    if (!B_SYMMETRIC) {
      row_offset_v =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_offsets + j));
      x_v = _mm256_sub_epi32(x_v, row_offset_v);
    }
    __m256i col_off_v;
    if (!A_SYMMETRIC) {
      col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j)));
      x_v = _mm256_sub_epi32(x_v, col_off_v);
    }

    if (!B_SYMMETRIC) {
      row_offset_v = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(row_offsets + j + VLEN));
      y_v = _mm256_sub_epi32(y_v, row_offset_v);
    }
    if (!A_SYMMETRIC) {
      col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j + VLEN)));
      y_v = _mm256_sub_epi32(y_v, col_off_v);
    }

    if (!B_SYMMETRIC) {
      row_offset_v = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(row_offsets + j + 2 * VLEN));
      z_v = _mm256_sub_epi32(z_v, row_offset_v);
    }
    if (!A_SYMMETRIC) {
      col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j + 2 * VLEN)));
      z_v = _mm256_sub_epi32(z_v, col_off_v);
    }

    if (!B_SYMMETRIC) {
      row_offset_v = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(row_offsets + j + 3 * VLEN));
      w_v = _mm256_sub_epi32(w_v, row_offset_v);
    }
    if (!A_SYMMETRIC) {
      col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j + 3 * VLEN)));
      w_v = _mm256_sub_epi32(w_v, col_off_v);
    }

    // convert to float
    __m256 xf_v, yf_v, zf_v, wf_v;
    if (HAS_BIAS) { // static if
      if (std::is_same<BIAS_TYPE, float>::value) {
        __m256 x_bias_v, y_bias_v, z_bias_v, w_bias_v;
        if (PER_CHANNEL_QUANTIZATION) {
          x_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 0 * VLEN)),
              _mm256_loadu_ps(act_times_w_scale + j + 0 * VLEN));
          y_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 1 * VLEN)),
              _mm256_loadu_ps(act_times_w_scale + j + 1 * VLEN));
          z_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 2 * VLEN)),
              _mm256_loadu_ps(act_times_w_scale + j + 2 * VLEN));
          w_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 3 * VLEN)),
              _mm256_loadu_ps(act_times_w_scale + j + 3 * VLEN));
        } else {
          x_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 0 * VLEN)),
              act_times_w_rcp_v);
          y_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 1 * VLEN)),
              act_times_w_rcp_v);
          z_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 2 * VLEN)),
              act_times_w_rcp_v);
          w_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(
                  reinterpret_cast<const float*>(bias + j + 3 * VLEN)),
              act_times_w_rcp_v);
        }
        xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
        yf_v = _mm256_add_ps(_mm256_cvtepi32_ps(y_v), y_bias_v);
        zf_v = _mm256_add_ps(_mm256_cvtepi32_ps(z_v), z_bias_v);
        wf_v = _mm256_add_ps(_mm256_cvtepi32_ps(w_v), w_bias_v);
      } else {
        x_v = _mm256_add_epi32(
            x_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(bias + j + 0 * VLEN)));
        y_v = _mm256_add_epi32(
            y_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(bias + j + 1 * VLEN)));
        z_v = _mm256_add_epi32(
            z_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(bias + j + 2 * VLEN)));
        w_v = _mm256_add_epi32(
            w_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(bias + j + 3 * VLEN)));
        xf_v = _mm256_cvtepi32_ps(x_v);
        yf_v = _mm256_cvtepi32_ps(y_v);
        zf_v = _mm256_cvtepi32_ps(z_v);
        wf_v = _mm256_cvtepi32_ps(w_v);
      }
    } else {
      xf_v = _mm256_cvtepi32_ps(x_v);
      yf_v = _mm256_cvtepi32_ps(y_v);
      zf_v = _mm256_cvtepi32_ps(z_v);
      wf_v = _mm256_cvtepi32_ps(w_v);
    }

    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 0 * VLEN);
    }
    __m256 x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 1 * VLEN);
    }
    __m256 y_scaled_v = _mm256_mul_ps(yf_v, multiplier_v);
    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 2 * VLEN);
    }
    __m256 z_scaled_v = _mm256_mul_ps(zf_v, multiplier_v);
    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 3 * VLEN);
    }
    __m256 w_scaled_v = _mm256_mul_ps(wf_v, multiplier_v);

    __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);
    __m256i y_rounded_v = _mm256_cvtps_epi32(y_scaled_v);
    __m256i z_rounded_v = _mm256_cvtps_epi32(z_scaled_v);
    __m256i w_rounded_v = _mm256_cvtps_epi32(w_scaled_v);

    __m256i xy_packed_v = _mm256_adds_epi16(
        _mm256_packs_epi32(x_rounded_v, y_rounded_v), C_zero_point_epi16_v);
    __m256i zw_packed_v = _mm256_adds_epi16(
        _mm256_packs_epi32(z_rounded_v, w_rounded_v), C_zero_point_epi16_v);
    __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
    __m256i xyzw_clamped_v = _mm256_max_epu8(
        FUSE_RELU ? C_zero_point_epi8_v : min_v,
        _mm256_min_epu8(xyzw_packed_v, max_v));

    xyzw_clamped_v =
        _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);

    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(C_uint8 + j), xyzw_clamped_v);
  } // j loop vectorized and unrolled 4x

  for (; j < n / VLEN * VLEN; j += VLEN) {
    __m256i x_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C_int32 + j));

    if (!B_SYMMETRIC) {
      __m256i row_offset_v =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_offsets + j));
      x_v = _mm256_sub_epi32(x_v, row_offset_v);
    }
    if (!A_SYMMETRIC) {
      __m256i col_off_v = _mm256_mullo_epi32(
          A_zero_point_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(col_offsets + j)));
      x_v = _mm256_sub_epi32(x_v, col_off_v);
    }

    // Convert to float
    __m256 xf_v;
    if (HAS_BIAS) { // static if
      if (std::is_same<BIAS_TYPE, float>::value) {
        __m256 x_bias_v;
        if (PER_CHANNEL_QUANTIZATION) {
          x_bias_v = _mm256_div_ps(
              _mm256_loadu_ps(reinterpret_cast<const float*>(bias + j)),
              _mm256_loadu_ps(act_times_w_scale + j));
        } else {
          x_bias_v = _mm256_mul_ps(
              _mm256_loadu_ps(reinterpret_cast<const float*>(bias + j)),
              act_times_w_rcp_v);
        }
        xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
      } else {
        x_v = _mm256_add_epi32(
            x_v,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + j)));
        xf_v = _mm256_cvtepi32_ps(x_v);
      }
    } else {
      xf_v = _mm256_cvtepi32_ps(x_v);
    }

    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j);
    }
    __m256 x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
    __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);

    __m256i x_packed_v = _mm256_adds_epi16(
        _mm256_packs_epi32(x_rounded_v, _mm256_setzero_si256()),
        C_zero_point_epi16_v);
    x_packed_v = _mm256_packus_epi16(x_packed_v, _mm256_setzero_si256());
    __m256i x_clamped_v = _mm256_max_epu8(
        FUSE_RELU ? C_zero_point_epi8_v : min_v,
        _mm256_min_epu8(x_packed_v, max_v));

    x_clamped_v = _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(C_uint8 + j),
        _mm256_castsi256_si128(x_clamped_v));
  } // j loop vectorized

  for (; j < n; ++j) {
    std::int32_t raw = C_int32[j];
    if (!B_SYMMETRIC) {
      raw -= row_offsets[j];
    }
    if (!A_SYMMETRIC) {
      raw -= A_zero_point * col_offsets[j];
    }
    float raw_f;
    if (HAS_BIAS) { // static if
      if (std::is_same<BIAS_TYPE, float>::value) {
        raw_f = raw;
        raw_f += bias[j] / act_times_w_scale[PER_CHANNEL_QUANTIZATION ? j : 0];
      } else {
        raw += bias[j];
        raw_f = raw;
      }
    } else {
      raw_f = raw;
    }

    float ab = raw_f * C_multiplier[PER_CHANNEL_QUANTIZATION ? j : 0];
    long rounded = lrintf(ab) + C_zero_point;

    C_uint8[j] = std::max(
        FUSE_RELU ? static_cast<long>(C_zero_point) : 0l,
        std::min(255l, rounded));
  }
}

template <bool REMAINDER>
static ALWAYS_INLINE __m256i load_a(const std::uint8_t* A, __m256i mask_v) {
  if (REMAINDER) {
    return _mm256_maskload_epi32(reinterpret_cast<const int*>(A), mask_v);
  } else {
    return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(A));
  }
}

static inline std::pair<int, int> closest_factors_(int n) {
  int a = static_cast<int>(std::sqrt(n));
  while (n % a != 0) {
    a--;
  }
  return {a, n / a}; // a <= n / a
}

} // namespace fbgemm
