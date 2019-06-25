/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "GroupwiseConv.h"

#include <immintrin.h>
#include <cassert>

namespace fbgemm {

namespace {

// define constants and macros to make code semi-portable between AVX and AVX512
#ifdef __AVX512F__
constexpr int SIMD_WIDTH = 512;
typedef __m512i SIMD_TYPE_I32;

constexpr int K_UNROLL = 2;

#define _MM_MADDUBS_EPI16 _mm512_maddubs_epi16
#define _MM_MADD_EPI16 _mm512_madd_epi16
#define _MM_ADD_EPI32 _mm512_add_epi32
#define _MM_SAD_EPU8 _mm512_sad_epu8
#define _MM_MULLO_EPI32 _mm512_mullo_epi32

#define _MM_SETZERO _mm512_setzero_si512
#define _MM_SET1_EPI8 _mm512_set1_epi8
#define _MM_SET1_EPI16 _mm512_set1_epi16
#define _MM_SET1_EPI32 _mm512_set1_epi32

#define _MM_LOADU _mm512_loadu_si512
#define _MM_STOREU _mm512_storeu_si512

#else
constexpr int SIMD_WIDTH = 256;
typedef __m256i SIMD_TYPE_I32;

constexpr int K_UNROLL = 1;

#define _MM_MADDUBS_EPI16 _mm256_maddubs_epi16
#define _MM_MADD_EPI16 _mm256_madd_epi16
#define _MM_ADD_EPI32 _mm256_add_epi32
#define _MM_SAD_EPU8 _mm256_sad_epu8
#define _MM_MULLO_EPI32 _mm256_mullo_epi32

#define _MM_SETZERO _mm256_setzero_si256
#define _MM_SET1_EPI8 _mm256_set1_epi8
#define _MM_SET1_EPI16 _mm256_set1_epi16
#define _MM_SET1_EPI32 _mm256_set1_epi32

#define _MM_LOADU(a) _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(a))
#define _MM_STOREU(addr, data) \
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(addr), data)
#endif

constexpr int SIMD_WIDTH_I8 = SIMD_WIDTH / 8;
constexpr int SIMD_WIDTH_I32 = SIMD_WIDTH / 32;

using namespace std;

// Multiply A[h,w,c:c+32] with b_v[r,s] and accumulate across 4 input channels.
//
// Inside each b_v[r,s] register has layout
// [SIMD_WIDTH_I32/G_VEC_BLOCK_IC_PER_G][G_VEC_BLOCK][IC_PER_G] , where
// G_VEC_BLOCK is the number of groups we have per SIMD.
template <int G, int C_PER_G, int STRIDE, int r, int s>
inline __attribute__((always_inline)) void gconv_inner_kernel_(
    const uint8_t* A,
    int W_IN,
    int h,
    int w,
    const SIMD_TYPE_I32* b_v,
    SIMD_TYPE_I32* c_v,
    const SIMD_TYPE_I32& one_epi16_v) {
  constexpr int R = 3;
  constexpr int S = 3;
  constexpr int H_PAD = 1;
  constexpr int W_PAD = 1;
  constexpr int STRIDE_H = STRIDE;
  constexpr int STRIDE_W = STRIDE;

  int h_in = -H_PAD + h * STRIDE_H + r;
  int w_in = -W_PAD + w * STRIDE_W + s;

#ifdef __AVX512F__
  // Read 32 input channels and broadcast twice.
  __m512i a_v =
      _mm512_broadcast_i32x4(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(
          A + (h_in * W_IN + w_in) * G * C_PER_G)));
#else
  __m256i a_v;
  if (C_PER_G <= 8) {
    // Read from at least 8 input channels and broadcast 4 times
    a_v =
        _mm256_castpd_si256(_mm256_broadcast_sd(reinterpret_cast<const double*>(
            A + (h_in * W_IN + w_in) * G * C_PER_G)));
  } else {
    assert(C_PER_G == 16);
    // Read 16 input channels and broadcast twice.
    a_v = _mm256_castpd_si256(
        _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(
            A + (h_in * W_IN + w_in) * G * C_PER_G)));
  }
#endif

  SIMD_TYPE_I32 temp_v = _MM_MADDUBS_EPI16(a_v, b_v[r * S + s]);
  temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
  c_v[0] = _MM_ADD_EPI32(c_v[0], temp_v);

  if (C_PER_G >= 8 && K_UNROLL > 1) {
    temp_v = _MM_MADDUBS_EPI16(a_v, b_v[(R + r) * S + s]);
    temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
    c_v[1] = _MM_ADD_EPI32(c_v[1], temp_v);
  }
}

// sum 8-bit numbers across input-channel dimension
template <int G, int C_PER_G, int STRIDE, int r, int s>
inline __attribute__((always_inline)) void gconv_sum_a_inner_(
    const uint8_t* A,
    int W_IN,
    int h,
    int w,
    SIMD_TYPE_I32& rowoffset_v) {
  constexpr int H_PAD = 1;
  constexpr int W_PAD = 1;
  constexpr int STRIDE_H = STRIDE;
  constexpr int STRIDE_W = STRIDE;

  int h_in = -H_PAD + h * STRIDE_H + r;
  int w_in = -W_PAD + w * STRIDE_W + s;

  SIMD_TYPE_I32 a0_v = _MM_LOADU(A + (h_in * W_IN + w_in) * G * C_PER_G);

  SIMD_TYPE_I32 temp_v;
  if (C_PER_G == 4) {
    SIMD_TYPE_I32 one_epi16_v = _MM_SET1_EPI16(1);
    SIMD_TYPE_I32 one_epi8_v = _MM_SET1_EPI8(1);
    temp_v = _MM_MADDUBS_EPI16(a0_v, one_epi8_v);
    temp_v = _MM_MADD_EPI16(temp_v, one_epi16_v);
  } else {
    SIMD_TYPE_I32 a1_v =
        _MM_LOADU(A + (h_in * W_IN + w_in) * G * C_PER_G + SIMD_WIDTH_I8);
    temp_v = _MM_SETZERO();

    // Let a0[0] denote 0th (LSB) 8-bit of a0_v
    // After _mm256_sad_epu8, a0[0:2] = a0[0] + ... + a0[7]
    // a0[8:10] = a0[8] + ... + a0[15]
    // a0[16:18] = a0[16] + ... + a0[23]
    // a0[24:26] = a0[24] + ... + a0[31]
    a0_v = _MM_SAD_EPU8(a0_v, temp_v);
    a1_v = _MM_SAD_EPU8(a1_v, temp_v);

#ifdef __AVX512F__
    __m256i a0_low_v = _mm512_castsi512_si256(a0_v);
    __m256i a0_high_v = _mm512_extracti64x4_epi64(a0_v, 1);

    __m256i a1_low_v = _mm512_castsi512_si256(a1_v);
    __m256i a1_high_v = _mm512_extracti64x4_epi64(a1_v, 1);

    // After _mm256_hadd_epi32,
    // a0[0:4] = a0[0] + ... + a0[7], a0[4:8] = a0[8] + ... + a0[15]
    // a0[8:12] = a1[0] + ... + a1[7], a0[12:16] = a1[8] + ... + a1[15]
    // ...
    a0_low_v = _mm256_hadd_epi32(a0_low_v, a1_low_v);
    a0_high_v = _mm256_hadd_epi32(a0_high_v, a1_high_v);
#else
    a0_v = _mm256_hadd_epi32(a0_v, a1_v);
#endif

    if (C_PER_G == 8) {
#ifndef __AVX512F__
      temp_v = a0_v;
#endif
    } else {
      assert(C_PER_G == 16);
      SIMD_TYPE_I32 a2_v =
          _MM_LOADU(A + (h_in * W_IN + w_in) * G * C_PER_G + 2 * SIMD_WIDTH_I8);
      SIMD_TYPE_I32 a3_v =
          _MM_LOADU(A + (h_in * W_IN + w_in) * G * C_PER_G + 3 * SIMD_WIDTH_I8);

      // After _mm256_sad_epu8, a2[0:4] = a2[0] + ... + a2[7]
      // a2[8:12] = a2[8] + ... + a2[15]
      // a2[16:20] = a2[16] + ... + a2[23]
      // a2[24:28] = a2[24] + ... + a2[31]
      a2_v = _MM_SAD_EPU8(a2_v, temp_v);
      a3_v = _MM_SAD_EPU8(a3_v, temp_v);

#ifdef __AVX512F__
      __m256i a2_low_v = _mm512_castsi512_si256(a2_v);
      __m256i a2_high_v = _mm512_extracti64x4_epi64(a2_v, 1);

      __m256i a3_low_v = _mm512_castsi512_si256(a3_v);
      __m256i a3_high_v = _mm512_extracti64x4_epi64(a3_v, 1);

      a2_low_v = _mm256_hadd_epi32(a2_low_v, a3_low_v);
      a2_high_v = _mm256_hadd_epi32(a2_high_v, a3_high_v);

      a0_low_v = _mm256_hadd_epi32(a0_low_v, a2_low_v);
      a0_high_v = _mm256_hadd_epi32(a0_high_v, a2_high_v);
#else
      // a2[0:4] = a2[0] + ... + a2[7], a2[4:8] = a2[8] + ... + a2[15]
      // a2[8:12] = a3[0] + ... + a3[7], a2[12:16] = a3[8] + ... + a3[15]
      a2_v = _mm256_hadd_epi32(a2_v, a3_v);

      // temp[0:4] = a0[0] + ... + a0[15], temp[4:8] = a1[0] + ... + a1[15]
      // temp[8:12] = a2[0] + ... + a2[15], temp[12:16] = a3[0] + ... + a3[15]
      temp_v = _mm256_hadd_epi32(a0_v, a2_v);
#endif
    }
#ifdef __AVX512F__
    temp_v = _mm512_castsi256_si512(a0_low_v);
    temp_v = _mm512_inserti32x8(temp_v, a0_high_v, 1);
#endif
  }
  rowoffset_v = _MM_ADD_EPI32(temp_v, rowoffset_v);
}

// Compute 3x3 inner-prod at (h, w) output position for G_VEC_BLOCK groups,
// IC_PER_G input channels, and SIMD_WIDTH_I8 / G_VEC_BLOCK / IC_PER_G output
// channels.
// Also, optionally compute row offsets when SUM_A = true.
// Manually unroll for 3x3 times
template <
    int G,
    int C_PER_G,
    int STRIDE,
    bool A_ZP_ZERO,
    bool TOP,
    bool BOTTOM,
    bool LEFT,
    bool RIGHT,
    bool SUM_A>
inline __attribute__((always_inline)) void gconv_inner_prod_kernel_(
    const uint8_t* A,
    int32_t A_zero_point,
    int W_IN,
    int W_OUT,
    int h,
    int w,
    int k,
    const SIMD_TYPE_I32* b_v,
    int32_t* C,
    int32_t* rowOffsetBuf,
    __m256i* c_temp) {
  // TODO: handle case when K_per_G != C_PER_G
  constexpr int K_per_G = C_PER_G;
  constexpr int R = 3;
  constexpr int S = 3;

  SIMD_TYPE_I32 one_epi16_v = _MM_SET1_EPI16(1);
  SIMD_TYPE_I32 A_zp_epu8_v = _MM_SET1_EPI8(A_zero_point);

  SIMD_TYPE_I32 A_zp_epi32_v;
  SIMD_TYPE_I32 rowoffset_v;
  if (SUM_A) {
    A_zp_epi32_v = _MM_SET1_EPI32(A_zero_point);
    A_zp_epi32_v = _MM_MULLO_EPI32(A_zp_epi32_v, _MM_SET1_EPI32(C_PER_G));
    rowoffset_v = _MM_SETZERO();
  }

  SIMD_TYPE_I32 c_v[2];
  c_v[0] = _MM_SETZERO();
#ifdef __AVX512F__
  c_v[1] = _MM_SETZERO();
#endif

  SIMD_TYPE_I32 temp_v;
  int r, s;
  if (TOP || LEFT) {
    if (!A_ZP_ZERO) {
      r = 0;
      s = 0;
      temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[r * S + s]);
      temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
      c_v[0] = _MM_ADD_EPI32(c_v[0], temp_v);
      if (C_PER_G >= 8 && K_UNROLL > 1) {
        temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[(R + r) * S + s]);
        temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
        c_v[1] = _MM_ADD_EPI32(c_v[1], temp_v);
      }
      if (SUM_A) {
        rowoffset_v = _MM_ADD_EPI32(rowoffset_v, A_zp_epi32_v);
      }
    }
  } else {
    if (SUM_A) {
      gconv_sum_a_inner_<G, C_PER_G, STRIDE, 0, 0>(A, W_IN, h, w, rowoffset_v);
    }
    gconv_inner_kernel_<G, C_PER_G, STRIDE, 0, 0>(
        A, W_IN, h, w, b_v, c_v, one_epi16_v);
  }

  if (TOP) {
    if (!A_ZP_ZERO) {
      r = 0;
      s = 1;
      temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[r * S + s]);
      temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
      c_v[0] = _MM_ADD_EPI32(c_v[0], temp_v);
      if (C_PER_G >= 8 && K_UNROLL > 1) {
        temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[(R + r) * S + s]);
        temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
        c_v[1] = _MM_ADD_EPI32(c_v[1], temp_v);
      }
      if (SUM_A) {
        rowoffset_v = _MM_ADD_EPI32(rowoffset_v, A_zp_epi32_v);
      }
    }
  } else {
    if (SUM_A) {
      gconv_sum_a_inner_<G, C_PER_G, STRIDE, 0, 1>(A, W_IN, h, w, rowoffset_v);
    }
    gconv_inner_kernel_<G, C_PER_G, STRIDE, 0, 1>(
        A, W_IN, h, w, b_v, c_v, one_epi16_v);
  }

  if (TOP || RIGHT) {
    if (!A_ZP_ZERO) {
      r = 0;
      s = 2;
      temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[r * S + s]);
      temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
      c_v[0] = _MM_ADD_EPI32(c_v[0], temp_v);
      if (C_PER_G >= 8 && K_UNROLL > 1) {
        temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[(R + r) * S + s]);
        temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
        c_v[1] = _MM_ADD_EPI32(c_v[1], temp_v);
      }
      if (SUM_A) {
        rowoffset_v = _MM_ADD_EPI32(rowoffset_v, A_zp_epi32_v);
      }
    }
  } else {
    if (SUM_A) {
      gconv_sum_a_inner_<G, C_PER_G, STRIDE, 0, 2>(A, W_IN, h, w, rowoffset_v);
    }
    gconv_inner_kernel_<G, C_PER_G, STRIDE, 0, 2>(
        A, W_IN, h, w, b_v, c_v, one_epi16_v);
  }

  if (LEFT) {
    if (!A_ZP_ZERO) {
      r = 1;
      s = 0;
      temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[r * S + s]);
      temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
      c_v[0] = _MM_ADD_EPI32(c_v[0], temp_v);
      if (C_PER_G >= 8 && K_UNROLL > 1) {
        temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[(R + r) * S + s]);
        temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
        c_v[1] = _MM_ADD_EPI32(c_v[1], temp_v);
      }
      if (SUM_A) {
        rowoffset_v = _MM_ADD_EPI32(rowoffset_v, A_zp_epi32_v);
      }
    }
  } else {
    if (SUM_A) {
      gconv_sum_a_inner_<G, C_PER_G, STRIDE, 1, 0>(A, W_IN, h, w, rowoffset_v);
    }
    gconv_inner_kernel_<G, C_PER_G, STRIDE, 1, 0>(
        A, W_IN, h, w, b_v, c_v, one_epi16_v);
  }

  if (SUM_A) {
    gconv_sum_a_inner_<G, C_PER_G, STRIDE, 1, 1>(A, W_IN, h, w, rowoffset_v);
  }
  gconv_inner_kernel_<G, C_PER_G, STRIDE, 1, 1>(
      A, W_IN, h, w, b_v, c_v, one_epi16_v);

  if (RIGHT) {
    if (!A_ZP_ZERO) {
      r = 1;
      s = 2;
      temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[r * S + s]);
      temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
      c_v[0] = _MM_ADD_EPI32(c_v[0], temp_v);
      if (C_PER_G >= 8 && K_UNROLL > 1) {
        temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[(R + r) * S + s]);
        temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
        c_v[1] = _MM_ADD_EPI32(c_v[1], temp_v);
      }
      if (SUM_A) {
        rowoffset_v = _MM_ADD_EPI32(rowoffset_v, A_zp_epi32_v);
      }
    }
  } else {
    if (SUM_A) {
      gconv_sum_a_inner_<G, C_PER_G, STRIDE, 1, 2>(A, W_IN, h, w, rowoffset_v);
    }
    gconv_inner_kernel_<G, C_PER_G, STRIDE, 1, 2>(
        A, W_IN, h, w, b_v, c_v, one_epi16_v);
  }

  if (BOTTOM || LEFT) {
    if (!A_ZP_ZERO) {
      r = 2;
      s = 0;
      temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[r * S + s]);
      temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
      c_v[0] = _MM_ADD_EPI32(c_v[0], temp_v);
      if (C_PER_G >= 8 && K_UNROLL > 1) {
        temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[(R + r) * S + s]);
        temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
        c_v[1] = _MM_ADD_EPI32(c_v[1], temp_v);
      }
      if (SUM_A) {
        rowoffset_v = _MM_ADD_EPI32(rowoffset_v, A_zp_epi32_v);
      }
    }
  } else {
    if (SUM_A) {
      gconv_sum_a_inner_<G, C_PER_G, STRIDE, 2, 0>(A, W_IN, h, w, rowoffset_v);
    }
    gconv_inner_kernel_<G, C_PER_G, STRIDE, 2, 0>(
        A, W_IN, h, w, b_v, c_v, one_epi16_v);
  }

  if (BOTTOM) {
    if (!A_ZP_ZERO) {
      r = 2;
      s = 1;
      temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[r * S + s]);
      temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
      c_v[0] = _MM_ADD_EPI32(c_v[0], temp_v);
      if (C_PER_G >= 8 && K_UNROLL > 1) {
        temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[(R + r) * S + s]);
        temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
        c_v[1] = _MM_ADD_EPI32(c_v[1], temp_v);
      }
      if (SUM_A) {
        rowoffset_v = _MM_ADD_EPI32(rowoffset_v, A_zp_epi32_v);
      }
    }
  } else {
    if (SUM_A) {
      gconv_sum_a_inner_<G, C_PER_G, STRIDE, 2, 1>(A, W_IN, h, w, rowoffset_v);
    }
    gconv_inner_kernel_<G, C_PER_G, STRIDE, 2, 1>(
        A, W_IN, h, w, b_v, c_v, one_epi16_v);
  }

  if (BOTTOM || RIGHT) {
    if (!A_ZP_ZERO) {
      r = 2;
      s = 2;
      temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[r * S + s]);
      temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
      c_v[0] = _MM_ADD_EPI32(c_v[0], temp_v);
      if (C_PER_G >= 8 && K_UNROLL > 1) {
        temp_v = _MM_MADDUBS_EPI16(A_zp_epu8_v, b_v[(R + r) * S + s]);
        temp_v = _MM_MADD_EPI16(one_epi16_v, temp_v);
        c_v[1] = _MM_ADD_EPI32(c_v[1], temp_v);
      }
      if (SUM_A) {
        rowoffset_v = _MM_ADD_EPI32(rowoffset_v, A_zp_epi32_v);
      }
    }
  } else {
    if (SUM_A) {
      gconv_sum_a_inner_<G, C_PER_G, STRIDE, 2, 2>(A, W_IN, h, w, rowoffset_v);
    }
    gconv_inner_kernel_<G, C_PER_G, STRIDE, 2, 2>(
        A, W_IN, h, w, b_v, c_v, one_epi16_v);
  }

#ifdef __AVX512F__
  if (C_PER_G == 4) {
    // We have 4 groups interleaved. Permute to put 1st group to lower 128-bit,
    // ..., 4th group to the upper 128-bit.
    __m512i perm_v =
        _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
    // TODO: use permutex2
    c_v[0] = _mm512_permutexvar_epi32(perm_v, c_v[0]);
    _mm512_storeu_si512(C + (h * W_OUT + w) * G * K_per_G, c_v[0]);
  } else if (C_PER_G == 8) {
    __m256i c_low_v[2], c_high_v[2];

    // cij refers to ith group and jth output channel
    // c00 c01 c10 c11 c02 c03 c12 c13 c04 c05 c14 c15 c06 c07 c16 c17
    c_low_v[0] = _mm512_castsi512_si256(c_v[0]);
    c_high_v[0] = _mm512_extracti64x4_epi64(c_v[0], 1);

    // c00+c01 c10+c11 c04+c05 c14+c15 c02+c03 c12+c13 c06+c07 c16+c17
    c_low_v[0] = _mm256_hadd_epi32(c_low_v[0], c_high_v[0]);

    // c00 c01 c10 c11 c02 c03 c12 c13 c04 c05 c14 c15 c06 c07 c16 c17
    c_low_v[1] = _mm512_castsi512_si256(c_v[1]);
    c_high_v[1] = _mm512_extracti64x4_epi64(c_v[1], 1);

    // c00+c01 c10+c11 c04+c05 c14+c15 c02+c03 c12+c13 c06+c07 c16+c17
    c_low_v[1] = _mm256_hadd_epi32(c_low_v[1], c_high_v[1]);

    c_v[0] =
        _mm512_inserti32x8(_mm512_castsi256_si512(c_low_v[0]), c_low_v[1], 1);
    __m512i perm_v =
        _mm512_set_epi32(15, 11, 13, 9, 7, 3, 5, 1, 14, 10, 12, 8, 6, 2, 4, 0);
    c_v[0] = _mm512_permutexvar_epi32(perm_v, c_v[0]);
    _mm512_storeu_si512(C + (h * W_OUT + w) * G * K_per_G, c_v[0]);
  } else {
    assert(C_PER_G == 16);
    __m256i c_low_v[2], c_high_v[2];

    c_low_v[0] = _mm512_castsi512_si256(c_v[0]);
    c_high_v[0] = _mm512_extracti64x4_epi64(c_v[0], 1);

    // c0+c1 c2+c3 c8+c9 c10+c11 c4+c5 c6+c7 c12+c13 c14+c15
    c_low_v[0] = _mm256_hadd_epi32(c_low_v[0], c_high_v[0]);

    c_low_v[1] = _mm512_castsi512_si256(c_v[1]);
    c_high_v[1] = _mm512_extracti64x4_epi64(c_v[1], 1);

    // c16+c17 c18+c19 c24+c25 c26+c27 c20+c21 c22+c23 c28+c29 c30+c31
    c_low_v[1] = _mm256_hadd_epi32(c_low_v[1], c_high_v[1]);

    // c0+c1+c2+c3 c8+c9+c10+c11 c16+c17+c18+c19 c24+c25+c26+c27
    // c4+c5+c6+c7 c12+c13+c14+c15 c20+c21+c22+c23 c28+c29+c30+c31
    c_low_v[0] = _mm256_hadd_epi32(c_low_v[0], c_low_v[1]);
    __m256i perm_v = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    c_low_v[0] = _mm256_permutevar8x32_epi32(c_low_v[0], perm_v);

    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(C + (h * W_OUT + w) * G * K_per_G + k),
        c_low_v[0]);
  }
#else
  if (C_PER_G == 4) {
    // We have 1st group in even lanes and 2nd group in odd lanes.
    // Permute to put 1st group to lower 128-bit and 2nd group in upper
    // 128-bit.
    __m256i perm_v = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    c_v[0] = _mm256_permutevar8x32_epi32(c_v[0], perm_v);
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(C + (h * W_OUT + w) * G * K_per_G), c_v[0]);
  } else if (C_PER_G == 8) {
    if (k % 8 != 0) {
      assert(k % 8 == 4);
      c_v[0] = _mm256_hadd_epi32(
          _mm256_loadu_si256(
              reinterpret_cast<__m256i*>(C + (h * W_OUT + w) * G * K_per_G)),
          c_v[0]);
      c_v[0] = _mm256_permute4x64_epi64(c_v[0], 0xd8);
    }
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(C + (h * W_OUT + w) * G * K_per_G), c_v[0]);
  } else {
    assert(C_PER_G == 16);
    // 3 hadd and 3 permute per 4 calls
    if (k % 8 == 0) {
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(C + (h * W_OUT + w) * G * K_per_G + k),
          c_v[0]);
    } else if (k % 8 == 2) {
      c_v[0] = _mm256_hadd_epi32(
          _mm256_loadu_si256(reinterpret_cast<__m256i*>(
              C + (h * W_OUT + w) * G * K_per_G + k - 2)),
          c_v[0]);
      c_v[0] = _mm256_permute4x64_epi64(c_v[0], 0xd8);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(C + (h * W_OUT + w) * G * K_per_G + k - 2),
          c_v[0]);
    } else if (k % 8 == 4) {
      c_temp[w] = c_v[0];
    } else {
      assert(k % 8 == 6);
      c_v[0] = _mm256_hadd_epi32(c_temp[w], c_v[0]);
      c_v[0] = _mm256_permute4x64_epi64(c_v[0], 0xd8);
      c_v[0] = _mm256_hadd_epi32(
          _mm256_loadu_si256(reinterpret_cast<__m256i*>(
              C + (h * W_OUT + w) * G * K_per_G + k - 6)),
          c_v[0]);
      c_v[0] = _mm256_permute4x64_epi64(c_v[0], 0xd8);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(C + (h * W_OUT + w) * G * K_per_G + k - 6),
          c_v[0]);
    }
  }
#endif

  if (SUM_A) {
    // Need to permute because _mm256_hadd_epi32 is used in gconv_sum_a_inner_
    if (C_PER_G == 8) {
      // 11 01 10 00 = 0xd8
      // Before permute,
      // v[0:4] = a0[0] + ... + a0[7], v[4:8] = a0[8] + ... + a0[15]
      // v[8:12] = a1[0] + ... + a1[7], v[12:16] = a1[8] + ... + a1[15]
      // v[16:20] = a0[16] + ... + a0[23], v[20:24] = a0[24] + ... + a0[31]
      // ...
      // After permute,
      // v[0:4] = a0[0] + ... + a0[7], v[4:8] = a0[8] + ... + a0[15]
      // v[8:12] = a0[16] + ... + a1[23], v[12:16] = a0[24] + ... + a1[31]
      // ...
#ifdef __AVX512F__
      __m512i perm_v = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
      rowoffset_v = _mm512_permutexvar_epi64(perm_v, rowoffset_v);
#else
      rowoffset_v = _mm256_permute4x64_epi64(rowoffset_v, 0xd8);
#endif
    } else if (C_PER_G == 16) {
      // Before permute,
      // v[0:4] = a0[0] + ... + a0[15], v[4:8] = a1[0] + ... + a1[15]
      // v[8:12] = a2[0] + ... + a2[15], v[12:16] = a3[0] + ... + a3[15]
      // v[16:20] = a0[16] + ... + a0[31], v[20:24] = a1[16] + ... + a1[31]
      // After permute,
      // v[0:4] = a0[0] + ... + a0[15], v[4:8] = a0[16] + ... + a0[31]
      // v[8:12] = a1[0] + ... + a1[15], v[12:16] = a1[16] + ... + a1[31]
      // ...
#ifdef __AVX512F__
      __m512i perm_v = _mm512_set_epi32(
          15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
      rowoffset_v = _mm512_permutexvar_epi32(perm_v, rowoffset_v);
#else
      __m256i perm_v = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
      rowoffset_v = _mm256_permutevar8x32_epi32(rowoffset_v, perm_v);
#endif
    }

    // Note we're computing row_offset only for group a multiple of
    // SIMD_WIDTH_I32 (see row_buf is set only when g == gOuter in
    // fbgemmGroupwiseConv function of GroupwiseConv.cc)
    _MM_STOREU(rowOffsetBuf + (h * W_OUT + w) * SIMD_WIDTH_I32, rowoffset_v);
  }
}

// Compute convolution for one row across G_VEC_BLOCK groups, IC_PER_G input
// channels, and SIMD_WIDTH_I8 / G_VEC_BLOCK / IC_PER_G output channels.
// Specialize for left/right edge
template <
    int G,
    int C_PER_G,
    int STRIDE,
    bool A_ZP_ZERO,
    bool TOP,
    bool BOTTOM,
    bool SUM_A>
inline __attribute__((always_inline)) void gconv_kernel_(
    int W_IN,
    int W_OUT,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    int k,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf,
    __m256i* c_temp) {
  // TODO: handle case when K_per_G != C_PER_G
  constexpr int K_PER_G = C_PER_G;
  constexpr int W_PAD = 1;
  constexpr int R = 3;
  constexpr int S = 3;
  constexpr int G_VEC_BLOCK = std::max(SIMD_WIDTH_I32 / K_PER_G, 1);

  // Load weights
  SIMD_TYPE_I32 b_v[R * S * (K_PER_G >= 8 ? K_UNROLL : 1)];
  for (int r = 0; r < R; ++r) {
    for (int s = 0; s < S; ++s) {
      const int8_t* B_ptr =
          B + ((r * S + s) * K_PER_G + k) * G_VEC_BLOCK * C_PER_G;
      b_v[r * S + s] = _MM_LOADU(B_ptr);
      if (K_PER_G >= 8 && K_UNROLL > 1) {
        b_v[(R + r) * S + s] = _MM_LOADU(B_ptr + SIMD_WIDTH_I8);
      }
    }
  }

  int w = 0;
  gconv_inner_prod_kernel_<
      G,
      C_PER_G,
      STRIDE,
      A_ZP_ZERO,
      TOP,
      BOTTOM,
      true,
      false,
      SUM_A>(
      A, A_zero_point, W_IN, W_OUT, h, w, k, b_v, C, rowOffsetBuf, c_temp);

  // top edge excluding corners
  for (w = W_PAD; w <= (W_IN + W_PAD - S) / STRIDE; ++w) {
    gconv_inner_prod_kernel_<
        G,
        C_PER_G,
        STRIDE,
        A_ZP_ZERO,
        TOP,
        BOTTOM,
        false,
        false,
        SUM_A>(
        A, A_zero_point, W_IN, W_OUT, h, w, k, b_v, C, rowOffsetBuf, c_temp);
  } // for each w

  // top-right corner
  if (w < W_OUT) {
    gconv_inner_prod_kernel_<
        G,
        C_PER_G,
        STRIDE,
        A_ZP_ZERO,
        TOP,
        BOTTOM,
        false,
        true,
        SUM_A>(
        A, A_zero_point, W_IN, W_OUT, h, w, k, b_v, C, rowOffsetBuf, c_temp);
  } // w < W_OUT
}

// Compute convolution for one row across all input/output channels in
// G_VEC_BLOCK groups.
// Specialize for k
template <
    int G,
    int C_PER_G,
    int STRIDE,
    bool A_ZP_ZERO,
    bool TOP,
    bool BOTTOM,
    bool SUM_A>
inline __attribute__((always_inline)) void gconv_kernel_(
    int W_IN,
    int W_OUT,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf) {
  // TODO: handle case when K_per_G != C_PER_G
  constexpr int K_per_G = C_PER_G;

  if (K_per_G == 4) {
    // Each gconv_kernel_ call processes
    // 4 groups, 4 input channels, 4 output channels in AVX512
    // 2 groups, 4 input channels, 4 output channels in AVX2
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, SUM_A>(
        W_IN, W_OUT, A, A_zero_point, h, 0, B, C, rowOffsetBuf, nullptr);
  } else if (K_per_G == 8) {
    // Each gconv_kernel_ call processes
    // 2 groups, 8 input channels, 8 output channels in AVX512 (K_UNROLL = 2)
    // 1 group, 8 input channels, 4 output channels in AVX2
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, SUM_A>(
        W_IN, W_OUT, A, A_zero_point, h, 0, B, C, rowOffsetBuf, nullptr);
#ifndef __AVX512F__
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, 4, B, C, rowOffsetBuf, nullptr);
#endif
  } else {
    assert(K_per_G == 16);
#ifdef __AVX512F__
    // Each gconv_kernel_ call processes
    // 1 group, 16 input channels, 8 output channels (K_UNROLL = 2)
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, SUM_A>(
        W_IN, W_OUT, A, A_zero_point, h, 0, B, C, rowOffsetBuf, nullptr);
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, 8, B, C, rowOffsetBuf, nullptr);
#else
    __m256i c_temp[W_OUT];
    // Each gconv_kernel_ call processes
    // 1 group, 16 input channels, 2 output channels
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, SUM_A>(
        W_IN, W_OUT, A, A_zero_point, h, 0, B, C, rowOffsetBuf, c_temp);
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, 2, B, C, rowOffsetBuf, c_temp);
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, 4, B, C, rowOffsetBuf, c_temp);
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, 6, B, C, rowOffsetBuf, c_temp);
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, 8, B, C, rowOffsetBuf, c_temp);
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, 10, B, C, rowOffsetBuf, c_temp);
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, 12, B, C, rowOffsetBuf, c_temp);
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, 14, B, C, rowOffsetBuf, c_temp);
#endif
  }
}

// Specialize for rowOffsetBuf
template <int G, int C_PER_G, int STRIDE, bool A_ZP_ZERO, bool TOP, bool BOTTOM>
void gconv_kernel_(
    int W_IN,
    int W_OUT,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    int h,
    const std::int8_t* B,
    std::int32_t* C,
    std::int32_t* rowOffsetBuf) {
  if (rowOffsetBuf) {
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, true>(
        W_IN, W_OUT, A, A_zero_point, h, B, C, rowOffsetBuf);
  } else {
    gconv_kernel_<G, C_PER_G, STRIDE, A_ZP_ZERO, TOP, BOTTOM, false>(
        W_IN, W_OUT, A, A_zero_point, h, B, C, rowOffsetBuf);
  }
}

// Specialize for stride and A_zero_point
template <int G, int C_PER_G, bool TOP, bool BOTTOM, int SPATIAL_DIM>
void gconv_kernel_(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    int h,
    const std::int8_t* B,
    std::int32_t* C,
    std::int32_t* rowOffsetBuf) {
  assert(conv_p.stride[0] == conv_p.stride[1]);
  int W_IN = conv_p.IN_DIM[1];
  int W_OUT = conv_p.OUT_DIM[1];
  if (A_zero_point == 0) {
    if (conv_p.stride[0] == 1) {
      gconv_kernel_<G, C_PER_G, 1, true, TOP, BOTTOM>(
          W_IN, W_OUT, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else if (conv_p.stride[0] == 2) {
      gconv_kernel_<G, C_PER_G, 2, true, TOP, BOTTOM>(
          W_IN, W_OUT, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else {
      assert(false);
    }
  } else {
    if (conv_p.stride[0] == 1) {
      gconv_kernel_<G, C_PER_G, 1, false, TOP, BOTTOM>(
          W_IN, W_OUT, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else if (conv_p.stride[0] == 2) {
      gconv_kernel_<G, C_PER_G, 2, false, TOP, BOTTOM>(
          W_IN, W_OUT, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else {
      assert(false);
    }
  }
}

// Specialize for K, OC, pad, IC, and G
template <bool TOP, bool BOTTOM, int SPATIAL_DIM>
void gconv_kernel_(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    int h,
    const std::int8_t* B,
    std::int32_t* C,
    std::int32_t* rowOffsetBuf) {
  int IC = conv_p.IC;
  int G = conv_p.G;
  int IC_per_G = IC / G;
  assert(
      conv_p.K[0] == 3 && conv_p.K[1] == 3 && IC == conv_p.OC &&
      conv_p.pad[0] == 1 && conv_p.pad[1] == 1);
#ifndef __AVX512F__
  if (G == 8) {
    if (IC_per_G == 4) {
      gconv_kernel_<8, 4, TOP, BOTTOM, SPATIAL_DIM>(
          conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else if (IC_per_G == 8) {
      gconv_kernel_<8, 8, TOP, BOTTOM, SPATIAL_DIM>(
          conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else if (IC_per_G == 16) {
      gconv_kernel_<8, 16, TOP, BOTTOM, SPATIAL_DIM>(
          conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else {
      assert(false);
    }
  } else
#endif
  if (G == 16) {
    if (IC_per_G == 4) {
      gconv_kernel_<16, 4, TOP, BOTTOM, SPATIAL_DIM>(
          conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else if (IC_per_G == 8) {
      gconv_kernel_<16, 8, TOP, BOTTOM, SPATIAL_DIM>(
          conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else if (IC_per_G == 16) {
      gconv_kernel_<16, 16, TOP, BOTTOM, SPATIAL_DIM>(
          conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else {
      assert(false);
    }
  } else if (G == 32) {
    if (IC_per_G == 4) {
      gconv_kernel_<32, 4, TOP, BOTTOM, SPATIAL_DIM>(
          conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else if (IC_per_G == 8) {
      gconv_kernel_<32, 8, TOP, BOTTOM, SPATIAL_DIM>(
          conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else if (IC_per_G == 16) {
      gconv_kernel_<32, 16, TOP, BOTTOM, SPATIAL_DIM>(
          conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
    } else {
      assert(false);
    }
  } else {
    assert(false);
  }
}

} // anonymous namespace

} // namespace fbgemm
