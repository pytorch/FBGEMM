/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "FbgemmI8Depthwise.h"

#include <cassert>
#include <cmath> // for lrintf and sqrt
#include <tuple> // for tie

#include <immintrin.h>

using namespace std;

namespace fbgemm {

static int masks[8][8] = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  {  0,  0,  0,  0,  0,  0,  0,  0,  },
  { -1,  0,  0,  0,  0,  0,  0,  0,  },
  { -1, -1,  0,  0,  0,  0,  0,  0,  },
  { -1, -1, -1,  0,  0,  0,  0,  0,  },
  { -1, -1, -1, -1,  0,  0,  0,  0,  },
  { -1, -1, -1, -1, -1,  0,  0,  0,  },
  { -1, -1, -1, -1, -1, -1,  0,  0,  },
  { -1, -1, -1, -1, -1, -1, -1,  0,  },
};

template <int KERNEL_PROD>
PackedDepthWiseConvMatrix<KERNEL_PROD>::PackedDepthWiseConvMatrix(
    int K,
    const int8_t* smat)
    : K_(K) {
  // Transpose the input matrix to make packing faster.
  alignas(64) int8_t smat_transposed[K * KERNEL_PROD];
  for (int i = 0; i < KERNEL_PROD; ++i) {
    for (int j = 0; j < K; ++j) {
      smat_transposed[i * K + j] = smat[i + j * KERNEL_PROD];
    }
  }

  // Allocate packed arrays
  constexpr int KERNEL_PROD_ALIGNED = (KERNEL_PROD + 1) / 2 * 2;
  // pmat_ = static_cast<int8_t *>(fbgemmAlignedAlloc(
  //     64, ((K + 31) / 32) * KERNEL_PROD_ALIGNED * 32 * sizeof(int8_t)));
  posix_memalign(
      (void**)&pmat_,
      64,
      ((K + 31) / 32) * KERNEL_PROD_ALIGNED * 32 * sizeof(int8_t));

  // Pack input matrix
  // The layout is optimized to use vpmaddubsw efficiently (see
  // madd_epi16x4_packed function).
  // For a group of 32 channels, we have 10 32B SIMD registers.
  // Denote ith channel jth filter as (i, j)
  // 0th SIMD register:
  // (0, 0), (0, 1), (0, 2), (0, 3), ..., (3, 0), (3, 1), (3, 2), (3, 3)
  // (16, 0), (16, 1), (16, 2), (16, 3), ..., (19, 0), (19, 1), (19, 2), (19, 3)
  // 1st SIMD register:
  // (4, 0), (4, 1), (4, 2), (4, 3), ..., (7, 0), (7, 1), (7, 2), (7, 3)
  // (20, 0), (20, 1), (20, 2), (20, 3), ..., (23, 0), (23, 1), (23, 2), (23, 3)
  // 2nd SIMD register:
  // (8, 0), (8, 1), (8, 2), (8, 3), ..., (11, 0), (11, 1), (11, 2), (11, 3)
  // (24, 0), (24, 1), (24, 2), (24, 3), ..., (27, 0), (27, 1), (27, 2), (27, 3)
  // 3rd SIMD register:
  // (12, 0), (12, 1), (12, 2), (12, 3), ..., (15, 0), (15, 1), (15, 2), (15, 3)
  // (28, 0), (28, 1), (28, 2), (28, 3), ..., (31, 0), (31, 1), (31, 2), (31, 3)
  // 4-7th SIMD register: same as the previous 4 registers but for 4-7th filter
  // coefficients
  // ...
  //
  // REMAINDER
  // If KERNEL_PROD % 4 == 1 for example when KERNEL_PROD == 9
  // 8th SIMD register:
  // (0, 8), zero, ..., (7, 8), zero
  // (16, 8), zero, ..., (23, 8), zero
  // 9th SIMD register:
  // (8, 8), zero, ..., (15, 8), zero
  // (24, 8), zero, ..., (31, 8), zero
  // We use madd_epi16_packed for this case
  //
  // If KERNEL_PROD % 4 == 2 for example when KERNEL_PROD == 10
  // 8th SIMD register:
  // (0, 8), (0, 9), ..., (7, 8), (7, 9)
  // (16, 8), (16, 9), ..., (23, 8), (23, 9)
  // 9th SIMD register:
  // (8, 8), (8, 9), ..., (15, 8), (15, 9)
  // (24, 8), (24, 9), ..., (31, 8), (31, 9)
  //
  // If KERNEL_PROD % 4 == 3 for example when KERNEL_PROD == 11
  // 8th SIMD register:
  // (0, 8), (0, 9), (0, 10), zero, ..., (3, 8), (3, 9), (3, 10), zero
  // (16, 8), (16, 9), (16, 10), zero, ..., (19, 8), (19, 9), (19, 10), zero
  // 9th SIMD register:
  // (4, 8), (4, 9), (4, 10), zero, ..., (7, 8), (7, 9), (7, 10), zero
  // (20, 8), (20, 9), (20, 10), zero, ..., (23, 8), (23, 9), (23, 10), zero
  // 10th SIMD register:
  // (8, 8), (8, 9), (8, 10), zero, ..., (11, 8), (11, 9), (11, 10), zero
  // (24, 8), (24, 9), (24, 10), zero, ..., (27, 8), (27, 9), (27, 10), zero
  // 11th SIMD register:
  // (12, 8), (12, 9), (12, 10), zero, ..., (15, 8), (15, 9), (15, 10), zero
  // (28, 8), (28, 9), (28, 10), zero, ..., (31, 8), (31, 9), (31, 10), zero
  for (int k1 = 0; k1 < K; k1 += 32) {
    __m256i b_v[KERNEL_PROD];
    int remainder = K - k1;
    if (remainder < 32) {
      __m256i mask_v = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(masks[remainder / 4]));
      for (int i = 0; i < KERNEL_PROD; ++i) {
        b_v[i] = _mm256_maskload_epi32(
            reinterpret_cast<const int*>(smat_transposed + i * K + k1),
            mask_v);
      }
    } else {
      for (int i = 0; i < KERNEL_PROD; ++i) {
        b_v[i] = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(
            smat_transposed + i * K + k1));
      }
    }

    // Interleave 2 SIMD registers
    __m256i b_interleaved_epi16[KERNEL_PROD_ALIGNED];
    __m256i zero_v = _mm256_setzero_si256();
    for (int i = 0; i < KERNEL_PROD_ALIGNED / 2; ++i) {
      if (2 * i + 1 >= KERNEL_PROD) {
        b_interleaved_epi16[2 * i] = _mm256_unpacklo_epi8(b_v[2 * i], zero_v);
        b_interleaved_epi16[2 * i + 1] =
            _mm256_unpackhi_epi8(b_v[2 * i], zero_v);
      } else {
        b_interleaved_epi16[2 * i] =
            _mm256_unpacklo_epi8(b_v[2 * i], b_v[2 * i + 1]);
        b_interleaved_epi16[2 * i + 1] =
            _mm256_unpackhi_epi8(b_v[2 * i], b_v[2 * i + 1]);
      }
    }

    // Interleave 4 SIMD registers
    __m256i b_interleaved_epi32[KERNEL_PROD_ALIGNED];
    for (int i = 0; i < KERNEL_PROD_ALIGNED / 4; ++i) {
      b_interleaved_epi32[4 * i] = _mm256_unpacklo_epi16(
          b_interleaved_epi16[4 * i], b_interleaved_epi16[4 * i + 2]);
      b_interleaved_epi32[4 * i + 1] = _mm256_unpackhi_epi16(
          b_interleaved_epi16[4 * i], b_interleaved_epi16[4 * i + 2]);
      b_interleaved_epi32[4 * i + 2] = _mm256_unpacklo_epi16(
          b_interleaved_epi16[4 * i + 1], b_interleaved_epi16[4 * i + 3]);
      b_interleaved_epi32[4 * i + 3] = _mm256_unpackhi_epi16(
          b_interleaved_epi16[4 * i + 1], b_interleaved_epi16[4 * i + 3]);
    }
    for (int i = KERNEL_PROD_ALIGNED / 4 * 4; i < KERNEL_PROD_ALIGNED; ++i) {
      b_interleaved_epi32[i] = b_interleaved_epi16[i];
    }

    for (int i = 0; i < KERNEL_PROD_ALIGNED; ++i) {
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(
              &pmat_[((k1 / 32) * KERNEL_PROD_ALIGNED + i) * 32]),
          b_interleaved_epi32[i]);
    }
  }
}

template <int KERNEL_PROD>
PackedDepthWiseConvMatrix<KERNEL_PROD>::~PackedDepthWiseConvMatrix() {
  free(pmat_);
}

template class PackedDepthWiseConvMatrix<3 * 3>;
template class PackedDepthWiseConvMatrix<3 * 3 * 3>;

// c = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
// A is in uint8_t
// B is in int8_t and pre-interleaved
// C is in int32_t and 4 registers have results in the following layout:
// c0_v:   c[0:4], c[16:20]
// c1_v:   c[4:8], c[20:24]
// c2_v:  c[8:12], c[24:28]
// c3_v: c[12:16], c[28:32]
template <bool SUM_A = false>
static inline __attribute__((always_inline)) void madd_epi16x4_packed(
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
static inline __attribute__((always_inline)) void madd_epi16x3_packed(
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
static inline __attribute__((always_inline)) void madd_epi16x2_packed(
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
static inline __attribute__((always_inline)) void madd_epi16_packed(
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
static inline __attribute__((always_inline)) void inner_prod_packed_(
    const __m256i* a_v,
    const __m256i* Bp,
    int32_t* C,
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
          a_v[k],
          a_v[k + 1],
          Bp + k,
          &c[0],
          &c[1],
          &c[2],
          &c[3],
          a_sum_temp);
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

template <bool SUM_A = false, bool REMAINDER = false>
static inline __attribute__((always_inline)) void inner_prod_3x3_packed_(
    const __m256i* a_v,
    const __m256i* Bp,
    int32_t* C,
    int remainder,
    __m256i* a_sum = nullptr) {
  return inner_prod_packed_<9, SUM_A, REMAINDER>(a_v, Bp, C, remainder, a_sum);
}

// Almost same as ReQuantizeOutput in OutputProcessing-inh.h but different
// row_offsets for each row because of depth-wise convolution
template <bool FUSE_RELU, bool HAS_BIAS, bool PER_CHANNEL_QUANTIZATION>
static inline __attribute__((always_inline)) void requantize_(
    int32_t A_zero_point,
    const float* C_multiplier,
    int32_t C_zero_point,
    const int32_t* C_int32,
    uint8_t* C_uint8,
    int n,
    const int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias) {
  __m256 multiplier_v = _mm256_setzero_ps();
  if (!PER_CHANNEL_QUANTIZATION) {
    multiplier_v = _mm256_set1_ps(*C_multiplier);
  }

  __m256i min_v = _mm256_set1_epi8(static_cast<uint8_t>(0));
  __m256i max_v = _mm256_set1_epi8(static_cast<uint8_t>(255));

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

    __m256i col_off_v = _mm256_mullo_epi32(
        A_zero_point_v,
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_offsets + j)));
    __m256i row_offset_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_offsets + j));
    x_v = _mm256_sub_epi32(_mm256_sub_epi32(x_v, col_off_v), row_offset_v);

    col_off_v = _mm256_mullo_epi32(
        A_zero_point_v,
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(col_offsets + j + VLEN)));
    row_offset_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(row_offsets + j + VLEN));
    y_v = _mm256_sub_epi32(_mm256_sub_epi32(y_v, col_off_v), row_offset_v);

    col_off_v = _mm256_mullo_epi32(
        A_zero_point_v,
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(col_offsets + j + 2 * VLEN)));
    row_offset_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(row_offsets + j + 2 * VLEN));
    z_v = _mm256_sub_epi32(_mm256_sub_epi32(z_v, col_off_v), row_offset_v);

    col_off_v = _mm256_mullo_epi32(
        A_zero_point_v,
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(col_offsets + j + 3 * VLEN)));
    row_offset_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(row_offsets + j + 3 * VLEN));
    w_v = _mm256_sub_epi32(_mm256_sub_epi32(w_v, col_off_v), row_offset_v);

    if (HAS_BIAS) { // static if
      x_v = _mm256_add_epi32(
          x_v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + j)));
      y_v = _mm256_add_epi32(
          y_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(bias + j + VLEN)));
      z_v = _mm256_add_epi32(
          z_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(bias + j + 2 * VLEN)));
      w_v = _mm256_add_epi32(
          w_v,
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(bias + j + 3 * VLEN)));
    }

    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j);
    }
    __m256 x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + VLEN);
    }
    __m256 y_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(y_v), multiplier_v);
    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 2 * VLEN);
    }
    __m256 z_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(z_v), multiplier_v);
    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j + 3 * VLEN);
    }
    __m256 w_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(w_v), multiplier_v);

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

    __m256i col_off_v = _mm256_mullo_epi32(
        A_zero_point_v,
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_offsets + j)));
    __m256i row_offset_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_offsets + j));
    x_v = _mm256_sub_epi32(_mm256_sub_epi32(x_v, col_off_v), row_offset_v);

    if (HAS_BIAS) { // static if
      x_v = _mm256_add_epi32(
          x_v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + j)));
    }

    if (PER_CHANNEL_QUANTIZATION) {
      multiplier_v = _mm256_loadu_ps(C_multiplier + j);
    }
    __m256 x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
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
    int32_t raw = C_int32[j] - A_zero_point * col_offsets[j] - row_offsets[j];
    if (HAS_BIAS) { // static if
      raw += bias[j];
    }

    float ab = raw * C_multiplier[PER_CHANNEL_QUANTIZATION ? j : 0];
    long rounded = lrintf(ab) + C_zero_point;

    C_uint8[j] = std::max(
        FUSE_RELU ? static_cast<long>(C_zero_point) : 0l,
        std::min(255l, rounded));
  }
}

template <bool FUSE_RELU, bool HAS_BIAS>
static inline __attribute__((always_inline)) void requantize_(
    int32_t A_zero_point,
    float C_multiplier,
    int32_t C_zero_point,
    const int32_t* C_int32,
    uint8_t* C_uint8,
    int n,
    const int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias) {
  requantize_<FUSE_RELU, HAS_BIAS, false /* PER_CHANNEL_QUANTIZATION */>(
      A_zero_point,
      &C_multiplier,
      C_zero_point,
      C_int32,
      C_uint8,
      n,
      row_offsets,
      col_offsets,
      bias);
}

template <bool FUSE_RELU, bool HAS_BIAS>
static inline __attribute__((always_inline)) void requantize_per_channel_(
    int32_t A_zero_point,
    const float* C_multiplier,
    int32_t C_zero_point,
    const int32_t* C_int32,
    uint8_t* C_uint8,
    int n,
    const int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias) {
  requantize_<FUSE_RELU, HAS_BIAS, true /* PER_CHANNEL_QUANTIZATION */>(
      A_zero_point,
      C_multiplier,
      C_zero_point,
      C_int32,
      C_uint8,
      n,
      row_offsets,
      col_offsets,
      bias);
}

template <bool REMAINDER>
static inline __attribute__((always_inline)) __m256i load_a(
    const uint8_t* A,
    __m256i mask_v) {
  if (REMAINDER) {
    return _mm256_maskload_epi32(reinterpret_cast<const int*>(A), mask_v);
  } else {
    return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(A));
  }
}

template <
    bool SUM_A,
    bool REMAINDER = false,
    bool PER_CHANNEL_QUANTIZATION = false>
static inline __attribute__((always_inline)) void inner_prod_3x3_packed_(
    int H,
    int W,
    int K,
    int h_in,
    int w_in,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* Bp,
    const int32_t* B_zero_point,
    int32_t* C,
    int remainder,
    int32_t* row_offsets) {
  __m256i A_zero_point_v = _mm256_set1_epi8(static_cast<uint8_t>(A_zero_point));
  __m256i mask_v = _mm256_setzero_si256();
  if (REMAINDER) {
    mask_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(masks[remainder / 4]));
  }

  // The code below can be written as a simple R*S loop but the compiler
  // doesn't unroll so we're manually unrolling it.
  // constexpr int R = 3, S = 3;
  // array<__m256i, R * S> a_v;
  // for (int r = 0; r < R; ++r) {
  //   for (int s = 0; s < S; ++s) {
  //     if (h_in + r >= 0 && h_in + r < H && w_in + s >= 0 && w_in + s < W) {
  //       if (REMAINDER) {
  //         a_v[r * S + s] =
  //             _mm256_maskload_epi32((const int *)(A + (r * W + s) * K),
  //             mask_v);
  //       } else {
  //         a_v[r * S + s] =
  //             _mm256_lddqu_si256((const __m256i *)(A + (r * W + s) * K));
  //       }
  //     } else {
  //       a_v[r * S + s] = A_zero_point_v;
  //     }
  //   }
  // }
  __m256i a_v[9] = {
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
  };

  if (h_in >= 0 && h_in < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[0] = load_a<REMAINDER>(A + (0 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[1] = load_a<REMAINDER>(A + (0 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[2] = load_a<REMAINDER>(A + (0 * W + 2) * K, mask_v);
    }
  }

  if (h_in + 1 >= 0 && h_in + 1 < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[3] = load_a<REMAINDER>(A + (1 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[4] = load_a<REMAINDER>(A + (1 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[5] = load_a<REMAINDER>(A + (1 * W + 2) * K, mask_v);
    }
  }

  if (h_in + 2 >= 0 && h_in + 2 < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[6] = load_a<REMAINDER>(A + (2 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[7] = load_a<REMAINDER>(A + (2 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[8] = load_a<REMAINDER>(A + (2 * W + 2) * K, mask_v);
    }
  }

  __m256i a_sum[4];
  inner_prod_3x3_packed_<SUM_A, REMAINDER>(
      a_v,
      reinterpret_cast<const __m256i*>(Bp),
      C,
      remainder,
      a_sum);
  if (SUM_A) {
    __m256i B_zero_point_v;
    for (int i = 0; i < (REMAINDER ? (remainder / 8) : 4); ++i) {
      if (PER_CHANNEL_QUANTIZATION) {
        B_zero_point_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B_zero_point + i * 8));
      } else {
        B_zero_point_v = _mm256_set1_epi32(B_zero_point[0]);
      }
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&row_offsets[i * 8]),
          _mm256_mullo_epi32(a_sum[i], B_zero_point_v));
    }
  }
}

template <
    bool SUM_A,
    bool REMAINDER = false,
    bool PER_CHANNEL_QUANTIZATION = false>
static inline __attribute__((always_inline)) void inner_prod_3x3x3_packed_(
    int T,
    int H,
    int W,
    int K,
    int t_in,
    int h_in,
    int w_in,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* Bp,
    const int32_t* B_zero_point,
    int32_t* C,
    int remainder,
    int32_t* row_offsets) {
  __m256i A_zero_point_v = _mm256_set1_epi8(static_cast<uint8_t>(A_zero_point));
  __m256i mask_v = _mm256_setzero_si256();
  if (REMAINDER) {
    mask_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(masks[remainder / 4]));
  }

  // The code below can be written as a simple R*S loop but the compiler
  // doesn't unroll so we're manually unrolling it.
  // constexpr int R = 3, S = 3;
  // array<__m256i, R * S> a_v;
  // for (int r = 0; r < R; ++r) {
  //   for (int s = 0; s < S; ++s) {
  //     if (h_in + r >= 0 && h_in + r < H && w_in + s >= 0 && w_in + s < W) {
  //       if (REMAINDER) {
  //         a_v[r * S + s] =
  //             _mm256_maskload_epi32((const int *)(A + (r * W + s) * K),
  //             mask_v);
  //       } else {
  //         a_v[r * S + s] =
  //             _mm256_lddqu_si256((const __m256i *)(A + (r * W + s) * K));
  //       }
  //     } else {
  //       a_v[r * S + s] = A_zero_point_v;
  //     }
  //   }
  // }
  __m256i a_v[8];
  a_v[0] = A_zero_point_v;
  a_v[1] = A_zero_point_v;
  a_v[2] = A_zero_point_v;
  a_v[3] = A_zero_point_v;
  a_v[4] = A_zero_point_v;
  a_v[5] = A_zero_point_v;
  a_v[6] = A_zero_point_v;
  a_v[7] = A_zero_point_v;

  if (t_in >= 0 && t_in < T) {
    if (h_in >= 0 && h_in < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[0] = load_a<REMAINDER>(A + ((0 * H + 0) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[1] = load_a<REMAINDER>(A + ((0 * H + 0) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[2] = load_a<REMAINDER>(A + ((0 * H + 0) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 1 >= 0 && h_in + 1 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[3] = load_a<REMAINDER>(A + ((0 * H + 1) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[4] = load_a<REMAINDER>(A + ((0 * H + 1) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[5] = load_a<REMAINDER>(A + ((0 * H + 1) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[6] = load_a<REMAINDER>(A + ((0 * H + 2) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[7] = load_a<REMAINDER>(A + ((0 * H + 2) * W + 1) * K, mask_v);
      }
    }
  }

  __m256i a_sum[4];
  inner_prod_packed_<8, SUM_A, REMAINDER>(
      a_v,
      reinterpret_cast<const __m256i*>(Bp),
      C,
      remainder,
      a_sum);

  a_v[0] = A_zero_point_v;
  a_v[1] = A_zero_point_v;
  a_v[2] = A_zero_point_v;
  a_v[3] = A_zero_point_v;
  a_v[4] = A_zero_point_v;
  a_v[5] = A_zero_point_v;
  a_v[6] = A_zero_point_v;
  a_v[7] = A_zero_point_v;

  if (t_in >= 0 && t_in < T) {
    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[0] = load_a<REMAINDER>(A + ((0 * H + 2) * W + 2) * K, mask_v);
      }
    }
  }

  if (t_in + 1 >= 0 && t_in + 1 < T) {
    if (h_in >= 0 && h_in < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[1] = load_a<REMAINDER>(A + ((1 * H + 0) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[2] = load_a<REMAINDER>(A + ((1 * H + 0) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[3] = load_a<REMAINDER>(A + ((1 * H + 0) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 1 >= 0 && h_in + 1 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[4] = load_a<REMAINDER>(A + ((1 * H + 1) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[5] = load_a<REMAINDER>(A + ((1 * H + 1) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[6] = load_a<REMAINDER>(A + ((1 * H + 1) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[7] = load_a<REMAINDER>(A + ((1 * H + 2) * W + 0) * K, mask_v);
      }
    }
  }

  __m256i a_sum_temp[4];
  inner_prod_packed_<8, SUM_A, REMAINDER, true /* acc */>(
      a_v,
      reinterpret_cast<const __m256i*>(Bp) + 8,
      C,
      remainder,
      a_sum_temp);
  if (SUM_A) {
    a_sum[0] = _mm256_add_epi32(a_sum[0], a_sum_temp[0]);
    a_sum[1] = _mm256_add_epi32(a_sum[1], a_sum_temp[1]);
    a_sum[2] = _mm256_add_epi32(a_sum[2], a_sum_temp[2]);
    a_sum[3] = _mm256_add_epi32(a_sum[3], a_sum_temp[3]);
  }

  a_v[0] = A_zero_point_v;
  a_v[1] = A_zero_point_v;
  a_v[2] = A_zero_point_v;
  a_v[3] = A_zero_point_v;
  a_v[4] = A_zero_point_v;
  a_v[5] = A_zero_point_v;
  a_v[6] = A_zero_point_v;
  a_v[7] = A_zero_point_v;

  if (t_in + 1 >= 0 && t_in + 1 < T) {
    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[0] = load_a<REMAINDER>(A + ((1 * H + 2) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[1] = load_a<REMAINDER>(A + ((1 * H + 2) * W + 2) * K, mask_v);
      }
    }
  }

  if (t_in + 2 >= 0 && t_in + 2 < T) {
    if (h_in >= 0 && h_in < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[2] = load_a<REMAINDER>(A + ((2 * H + 0) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[3] = load_a<REMAINDER>(A + ((2 * H + 0) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[4] = load_a<REMAINDER>(A + ((2 * H + 0) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 1 >= 0 && h_in + 1 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[5] = load_a<REMAINDER>(A + ((2 * H + 1) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[6] = load_a<REMAINDER>(A + ((2 * H + 1) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[7] = load_a<REMAINDER>(A + ((2 * H + 1) * W + 2) * K, mask_v);
      }
    }
  }

  inner_prod_packed_<8, SUM_A, REMAINDER, true /* acc */>(
      a_v,
      reinterpret_cast<const __m256i*>(Bp) + 16,
      C,
      remainder,
      a_sum_temp);
  if (SUM_A) {
    a_sum[0] = _mm256_add_epi32(a_sum[0], a_sum_temp[0]);
    a_sum[1] = _mm256_add_epi32(a_sum[1], a_sum_temp[1]);
    a_sum[2] = _mm256_add_epi32(a_sum[2], a_sum_temp[2]);
    a_sum[3] = _mm256_add_epi32(a_sum[3], a_sum_temp[3]);
  }

  a_v[0] = A_zero_point_v;
  a_v[1] = A_zero_point_v;
  a_v[2] = A_zero_point_v;

  if (t_in + 2 >= 0 && t_in + 2 < T) {
    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[0] = load_a<REMAINDER>(A + ((2 * H + 2) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[1] = load_a<REMAINDER>(A + ((2 * H + 2) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[2] = load_a<REMAINDER>(A + ((2 * H + 2) * W + 2) * K, mask_v);
      }
    }
  }

  inner_prod_packed_<3, SUM_A, REMAINDER, true /* acc */>(
      a_v,
      reinterpret_cast<const __m256i*>(Bp) + 24,
      C,
      remainder,
      a_sum_temp);

  if (SUM_A) {
    a_sum[0] = _mm256_add_epi32(a_sum[0], a_sum_temp[0]);
    a_sum[1] = _mm256_add_epi32(a_sum[1], a_sum_temp[1]);
    a_sum[2] = _mm256_add_epi32(a_sum[2], a_sum_temp[2]);
    a_sum[3] = _mm256_add_epi32(a_sum[3], a_sum_temp[3]);

    __m256i B_zero_point_v;
    for (int i = 0; i < (REMAINDER ? (remainder / 8) : 4); ++i) {
      if (PER_CHANNEL_QUANTIZATION) {
        B_zero_point_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B_zero_point + i * 8));
      } else {
        B_zero_point_v = _mm256_set1_epi32(B_zero_point[0]);
      }
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&row_offsets[i * 8]),
          _mm256_mullo_epi32(a_sum[i], B_zero_point_v));
    }
  }
}

template <bool SUM_A, bool FUSE_RELU>
static inline __attribute__((always_inline)) void depthwise_3x3_kernel_(
    int H,
    int W,
    int K,
    int h,
    int w,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const int8_t* Bp,
    float C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias) {
  constexpr int S = 3;
  constexpr int PAD_T = 1, PAD_L = 1, PAD_R = 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  int k;
  for (k = 0; k < K / 32 * 32; k += 32) {
    inner_prod_3x3_packed_<SUM_A>(
        H,
        W,
        K,
        h_in,
        w_in,
        A + (h_in * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 10,
        &B_zero_point,
        C_int32 + k,
        0,
        &row_offsets[k]);
  }
  int remainder = K - k;
  if (remainder) {
    inner_prod_3x3_packed_<SUM_A, true>(
        H,
        W,
        K,
        h_in,
        w_in,
        A + (h_in * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 10,
        &B_zero_point,
        C_int32 + k,
        remainder,
        &row_offsets[k]);
  }
  if (SUM_A) {
    requantize_<FUSE_RELU, true>(
        A_zero_point,
        C_multiplier,
        C_zero_point,
        C_int32,
        C_uint8 + (h * W_OUT + w) * K,
        K,
        row_offsets,
        col_offsets,
        bias);
  }
}

template <bool SUM_A, bool FUSE_RELU>
static inline __attribute__((always_inline)) void depthwise_3x3x3_kernel_(
    int T,
    int H,
    int W,
    int K,
    int t,
    int h,
    int w,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const int8_t* Bp,
    float C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias) {
  constexpr int R = 3, S = 3;
  constexpr int PAD_P = 1, PAD_T = 1, PAD_B = 1, PAD_L = 1, PAD_R = 1;
  int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int t_in = -PAD_P + t * stride_t;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  int k;
  for (k = 0; k < K / 32 * 32; k += 32) {
    inner_prod_3x3x3_packed_<SUM_A>(
        T,
        H,
        W,
        K,
        t_in,
        h_in,
        w_in,
        A + ((t_in * H + h_in) * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 28,
        &B_zero_point,
        C_int32 + k,
        0,
        &row_offsets[k]);
  }
  int remainder = K - k;
  if (remainder) {
    inner_prod_3x3x3_packed_<SUM_A, true>(
        T,
        H,
        W,
        K,
        t_in,
        h_in,
        w_in,
        A + ((t_in * H + h_in) * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 28,
        &B_zero_point,
        C_int32 + k,
        remainder,
        &row_offsets[k]);
  }
  if (SUM_A) {
    requantize_<FUSE_RELU, true>(
        A_zero_point,
        C_multiplier,
        C_zero_point,
        C_int32,
        C_uint8 + ((t * H_OUT + h) * W_OUT + w) * K,
        K,
        row_offsets,
        col_offsets,
        bias);
  }
}

template <bool SUM_A>
static inline __attribute__((always_inline)) void
depthwise_3x3_per_channel_quantization_kernel_(
    int H,
    int W,
    int K,
    int h,
    int w,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const int8_t* Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias) {
  constexpr int S = 3;
  constexpr int PAD_T = 1, PAD_L = 1, PAD_R = 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  int k;
  for (k = 0; k < K / 32 * 32; k += 32) {
    inner_prod_3x3_packed_<SUM_A, false /*remainder*/, true /*per-channel*/>(
        H,
        W,
        K,
        h_in,
        w_in,
        A + (h_in * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 10,
        B_zero_point + k,
        C_int32 + k,
        0,
        &row_offsets[k]);
  }
  int remainder = K - k;
  if (remainder) {
    inner_prod_3x3_packed_<SUM_A, true /*remainder*/, true /*per-channel*/>(
        H,
        W,
        K,
        h_in,
        w_in,
        A + (h_in * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 10,
        B_zero_point + k,
        C_int32 + k,
        remainder,
        &row_offsets[k]);
  }
  if (SUM_A) {
    requantize_per_channel_<false, true>(
        A_zero_point,
        C_multiplier,
        C_zero_point,
        C_int32,
        C_uint8 + (h * W_OUT + w) * K,
        K,
        row_offsets,
        col_offsets,
        bias);
  }
}

static pair<int, int> closest_factors_(int n) {
  int a = (int)std::sqrt(n);
  while (n % a != 0) {
    a--;
  }
  return {a, n / a}; // a <= n / a
}

// TODO: short-circuit when B_zero_point is 0 or A_zero_point is 0
// This implemntation should be general enough to handle not just 3x3 but other
// filter shapes by parameterizing with R and S but restricting it to just 3x3
// for now.
template <bool FUSE_RESCALE = true, bool FUSE_RELU = false>
static inline __attribute__((always_inline)) void depthwise_3x3_pad_1_(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const Packed3x3ConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    const int32_t* col_offsets,
    const int32_t* bias,
    int thread_id,
    int num_threads) {
  assert(K % 8 == 0);
  constexpr int R = 3, S = 3;
  constexpr int PAD_T = 1, PAD_B = 1, PAD_L = 1, PAD_R = 1;
  int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  const int8_t* Bp = B.PackedMat();

  int32_t row_offsets[(K + 31) / 32 * 32] __attribute__((aligned(64)));
  int32_t* C_temp;

  int n_begin, n_end;
  int h_begin, h_end, w_begin, w_end;
  if (N >= num_threads) {
    int n_per_thread = (N + num_threads - 1) / num_threads;
    n_begin = std::min(thread_id * n_per_thread, N);
    n_end = std::min(n_begin + n_per_thread, N);
    h_begin = 0;
    h_end = H_OUT;
    w_begin = 0;
    w_end = W_OUT;
  } else {
    int nthreads_per_n = num_threads / N;
    n_begin = std::min(thread_id / nthreads_per_n, N);
    n_end = std::min(n_begin + 1, N);

    int tid_of_n_begin = std::min(n_begin * nthreads_per_n, num_threads);
    int tid_of_n_end = std::min(tid_of_n_begin + nthreads_per_n, num_threads);
    int nthreads_of_n = tid_of_n_end - tid_of_n_begin;
    int tid_within_n = thread_id - tid_of_n_begin;
    assert(tid_within_n >= 0);
    assert(tid_within_n < nthreads_of_n);

    // n is processed by num_threads_h * num_threads_w 2D grid of threads
    int num_threads_h, num_threads_w;
    // num_threads_w <= num_threads_h
    tie(num_threads_w, num_threads_h) = closest_factors_(nthreads_of_n);
    int tid_h = tid_within_n / num_threads_w;
    int tid_w = tid_within_n % num_threads_w;

    int h_per_thread = (H_OUT + num_threads_h - 1) / num_threads_h;
    h_begin = std::min(tid_h * h_per_thread, H_OUT);
    h_end = std::min(h_begin + h_per_thread, H_OUT);

    int w_per_thread = (W_OUT + num_threads_w - 1) / num_threads_w;
    w_begin = std::min(tid_w * w_per_thread, W_OUT);
    w_end = std::min(w_begin + w_per_thread, W_OUT);
  }

  for (int n = n_begin; n < n_end; ++n) {
    const uint8_t* A_base = A + n * H * W * K;
    uint8_t* C_uint8_base = C_uint8 + n * H_OUT * W_OUT * K;

    int h = 0;
    int w = 0;

    if (h_begin == 0) {
      if (w_begin == 0) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      for (w = std::max(1, w_begin); w < std::min(W_OUT - 1, w_end); ++w) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      if (w_end == W_OUT) {
        w = W_OUT - 1;
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }
    }

    for (h = std::max(1, h_begin); h < std::min(H - 1, h_end); ++h) {
      if (w_begin == 0) {
        w = 0;
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      for (w = std::max(1, w_begin); w < std::min(W_OUT - 1, w_end); ++w) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      if (w_end == W_OUT) {
        w = W_OUT - 1;
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }
    }

    if (h_end == H_OUT) {
      h = H_OUT - 1;
      w = 0;
      if (w_begin == 0) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      for (w = std::max(1, w_begin); w < std::min(W_OUT - 1, w_end); ++w) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      if (w_end == W_OUT) {
        w = W_OUT - 1;
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }
    }
  } // for each n
};

template <bool FUSE_RESCALE = true, bool FUSE_RELU = false>
static inline __attribute__((always_inline)) void depthwise_3x3x3_pad_1_(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const Packed3x3x3ConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    const int32_t* col_offsets,
    const int32_t* bias,
    int thread_id,
    int num_threads) {
  assert(K % 8 == 0);
  constexpr int K_T = 3, K_H = 3, K_W = 3;
  constexpr int PAD_P = 1, PAD_N = 1, PAD_T = 1, PAD_B = 1, PAD_L = 1,
                PAD_R = 1;
  int T_OUT = (T + PAD_P + PAD_N - K_T) / stride_t + 1;
  int H_OUT = (H + PAD_T + PAD_B - K_H) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - K_W) / stride_w + 1;
  const int8_t* Bp = B.PackedMat();

  int32_t row_offsets[(K + 31) / 32 * 32] __attribute__((aligned(64)));
  int32_t* C_temp;

  int n_begin, n_end;
  int t_begin, t_end, h_begin, h_end;
  if (N >= num_threads) {
    int n_per_thread = (N + num_threads - 1) / num_threads;
    n_begin = std::min(thread_id * n_per_thread, N);
    n_end = std::min(n_begin + n_per_thread, N);
    t_begin = 0;
    t_end = T_OUT;
    h_begin = 0;
    h_end = H_OUT;
  } else {
    int nthreads_per_n = num_threads / N;
    n_begin = std::min(thread_id / nthreads_per_n, N);
    n_end = std::min(n_begin + 1, N);

    int tid_of_n_begin = std::min(n_begin * nthreads_per_n, num_threads);
    int tid_of_n_end = std::min(tid_of_n_begin + nthreads_per_n, num_threads);
    int nthreads_of_n = tid_of_n_end - tid_of_n_begin;
    int tid_within_n = thread_id - tid_of_n_begin;
    assert(tid_within_n >= 0);
    assert(tid_within_n < nthreads_of_n);

    // n is processed by num_threads_t * num_threads_h 2D grid of threads
    int num_threads_t, num_threads_h;
    // num_threads_w <= num_threads_h
    tie(num_threads_t, num_threads_h) = closest_factors_(nthreads_of_n);
    int tid_t = tid_within_n / num_threads_h;
    int tid_h = tid_within_n % num_threads_h;

    int t_per_thread = (T_OUT + num_threads_t - 1) / num_threads_t;
    t_begin = std::min(tid_t * t_per_thread, T_OUT);
    t_end = std::min(t_begin + t_per_thread, T_OUT);

    int h_per_thread = (H_OUT + num_threads_h - 1) / num_threads_h;
    h_begin = std::min(tid_h * h_per_thread, H_OUT);
    h_end = std::min(h_begin + h_per_thread, H_OUT);
  }

  for (int n = n_begin; n < n_end; ++n) {
    const uint8_t* A_base = A + n * T * H * W * K;
    uint8_t* C_uint8_base = C_uint8 + n * T_OUT * H_OUT * W_OUT * K;

    for (int t = t_begin; t < t_end; ++t) {
      for (int h = h_begin; h < h_end; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          C_temp = FUSE_RESCALE
              ? C_int32
              : C_int32 + (((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K;
          depthwise_3x3x3_kernel_<FUSE_RESCALE, FUSE_RELU>(
              T,
              H,
              W,
              K,
              t,
              h,
              w,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_temp,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias);
        } // w
      } // h
    } // t
  } // for each n
};

template <bool FUSE_RESCALE = true>
static inline __attribute__((always_inline)) void
depthwise_3x3_per_channel_quantization_pad_1_(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const Packed3x3ConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    const int32_t* col_offsets,
    const int32_t* bias,
    int thread_id,
    int num_threads) {
  assert(K % 8 == 0);
  constexpr int R = 3, S = 3;
  constexpr int PAD_T = 1, PAD_B = 1, PAD_L = 1, PAD_R = 1;
  int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  const int8_t* Bp = B.PackedMat();

  int32_t row_offsets[(K + 31) / 32 * 32] __attribute__((aligned(64)));
  int32_t* C_temp;

  int n_begin, n_end;
  int h_begin, h_end, w_begin, w_end;
  if (N >= num_threads) {
    int n_per_thread = (N + num_threads - 1) / num_threads;
    n_begin = std::min(thread_id * n_per_thread, N);
    n_end = std::min(n_begin + n_per_thread, N);
    h_begin = 0;
    h_end = H_OUT;
    w_begin = 0;
    w_end = W_OUT;
  } else {
    int nthreads_per_n = num_threads / N;
    n_begin = std::min(thread_id / nthreads_per_n, N);
    n_end = std::min(n_begin + 1, N);

    int tid_of_n_begin = std::min(n_begin * nthreads_per_n, num_threads);
    int tid_of_n_end = std::min(tid_of_n_begin + nthreads_per_n, num_threads);
    int nthreads_of_n = tid_of_n_end - tid_of_n_begin;
    int tid_within_n = thread_id - tid_of_n_begin;
    assert(tid_within_n >= 0);
    assert(tid_within_n < nthreads_of_n);

    // n is processed by num_threads_h * num_threads_w 2D grid of threads
    int num_threads_h, num_threads_w;
    // num_threads_w <= num_threads_h
    tie(num_threads_w, num_threads_h) = closest_factors_(nthreads_of_n);
    int tid_h = tid_within_n / num_threads_w;
    int tid_w = tid_within_n % num_threads_w;

    int h_per_thread = (H_OUT + num_threads_h - 1) / num_threads_h;
    h_begin = std::min(tid_h * h_per_thread, H_OUT);
    h_end = std::min(h_begin + h_per_thread, H_OUT);

    int w_per_thread = (W_OUT + num_threads_w - 1) / num_threads_w;
    w_begin = std::min(tid_w * w_per_thread, W_OUT);
    w_end = std::min(w_begin + w_per_thread, W_OUT);
  }

  for (int n = n_begin; n < n_end; ++n) {
    const uint8_t* A_base = A + n * H * W * K;
    uint8_t* C_uint8_base = C_uint8 + n * H_OUT * W_OUT * K;

    int h = 0;
    int w = 0;

    if (h_begin == 0) {
      if (w_begin == 0) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_per_channel_quantization_kernel_<FUSE_RESCALE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      for (w = std::max(1, w_begin); w < std::min(W_OUT - 1, w_end); ++w) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_per_channel_quantization_kernel_<FUSE_RESCALE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      if (w_end == W_OUT) {
        w = W_OUT - 1;
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_per_channel_quantization_kernel_<FUSE_RESCALE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }
    }

    for (h = std::max(1, h_begin); h < std::min(H - 1, h_end); ++h) {
      if (w_begin == 0) {
        w = 0;
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_per_channel_quantization_kernel_<FUSE_RESCALE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      for (w = std::max(1, w_begin); w < std::min(W_OUT - 1, w_end); ++w) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_per_channel_quantization_kernel_<FUSE_RESCALE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      if (w_end == W_OUT) {
        w = W_OUT - 1;
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_per_channel_quantization_kernel_<FUSE_RESCALE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }
    }

    if (h_end == H_OUT) {
      h = H_OUT - 1;
      w = 0;
      if (w_begin == 0) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_per_channel_quantization_kernel_<FUSE_RESCALE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      for (w = std::max(1, w_begin); w < std::min(W_OUT - 1, w_end); ++w) {
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_per_channel_quantization_kernel_<FUSE_RESCALE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }

      if (w_end == W_OUT) {
        w = W_OUT - 1;
        C_temp = FUSE_RESCALE ? C_int32
                              : C_int32 + ((n * H_OUT + h) * W_OUT + w) * K;
        depthwise_3x3_per_channel_quantization_kernel_<FUSE_RESCALE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_temp,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias);
      }
    }
  } // for each n
};

// assumption: W > 3 and H > 3
void depthwise_3x3_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const Packed3x3ConvMatrix& B,
    int32_t* C,
    int thread_id,
    int num_threads) {
  if (7 == H && 7 == W && 1 == stride_h && 1 == stride_w) {
    depthwise_3x3_pad_1_<false>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        0,
        B,
        0.0f,
        0,
        C,
        nullptr,
        nullptr,
        nullptr,
        thread_id,
        num_threads);
  } else if (14 == H && 14 == W && 2 == stride_h && 2 == stride_w) {
    depthwise_3x3_pad_1_<false>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        0,
        B,
        0.0f,
        0,
        C,
        nullptr,
        nullptr,
        nullptr,
        thread_id,
        num_threads);
  } else if (1 == stride_h && 1 == stride_w) {
    depthwise_3x3_pad_1_<false>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        0,
        B,
        0.0f,
        0,
        C,
        nullptr,
        nullptr,
        nullptr,
        thread_id,
        num_threads);
  } else if (2 == stride_h && 2 == stride_w) {
    depthwise_3x3_pad_1_<false>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        0,
        B,
        0.0f,
        0,
        C,
        nullptr,
        nullptr,
        nullptr,
        thread_id,
        num_threads);
  } else {
    depthwise_3x3_pad_1_<false>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        0,
        B,
        0.0f,
        0,
        C,
        nullptr,
        nullptr,
        nullptr,
        thread_id,
        num_threads);
  }
}

void depthwise_3x3_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const Packed3x3ConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    int thread_id,
    int num_threads,
    bool fuse_relu) {
  int32_t C_int32_temp[(K + 31) / 32 * 32];
  if (fuse_relu) {
    if (7 == H && 7 == W && 1 == stride_h && 1 == stride_w) {
      depthwise_3x3_pad_1_<true /* FUSE_RESCALE */, true /* FUSE_RELU */>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    } else if (14 == H && 14 == W && 2 == stride_h && 2 == stride_w) {
      depthwise_3x3_pad_1_<true /* FUSE_RESCALE */, true /* FUSE_RELU */>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    } else if (1 == stride_h && 1 == stride_w) {
      depthwise_3x3_pad_1_<true /* FUSE_RESCALE */, true /* FUSE_RELU */>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    } else if (2 == stride_h && 2 == stride_w) {
      depthwise_3x3_pad_1_<true /* FUSE_RESCALE */, true /* FUSE_RELU */>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    } else {
      depthwise_3x3_pad_1_<true /* FUSE_RESCALE */, true /* FUSE_RELU */>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    }
  } else {
    if (7 == H && 7 == W && 1 == stride_h && 1 == stride_w) {
      depthwise_3x3_pad_1_(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    } else if (14 == H && 14 == W && 2 == stride_h && 2 == stride_w) {
      depthwise_3x3_pad_1_(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    } else if (1 == stride_h && 1 == stride_w) {
      depthwise_3x3_pad_1_(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    } else if (2 == stride_h && 2 == stride_w) {
      depthwise_3x3_pad_1_(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    } else {
      depthwise_3x3_pad_1_(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          thread_id,
          num_threads);
    }
  }
}

void depthwise_3x3x3_pad_1(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const Packed3x3x3ConvMatrix& B,
    int32_t* C,
    int thread_id,
    int num_threads) {
  depthwise_3x3x3_pad_1_<false /* FUSE_RESCALE */>(
      N,
      T,
      H,
      W,
      K,
      stride_t,
      stride_h,
      stride_w,
      A_zero_point,
      A,
      0,
      B,
      0.0f,
      0,
      C,
      nullptr,
      nullptr,
      nullptr,
      thread_id,
      num_threads);
}

static void depthwise_3x3x3_pad_1_(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const Packed3x3x3ConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    int thread_id,
    int num_threads) {
  int32_t C_int32_temp[(K + 31) / 32 * 32];
  depthwise_3x3x3_pad_1_<true /* FUSE_RESCALE */, false /* FUSE_RELU */>(
      N,
      T,
      H,
      W,
      K,
      stride_t,
      stride_h,
      stride_w,
      A_zero_point,
      A,
      B_zero_point,
      B,
      C_multiplier,
      C_zero_point,
      C_int32_temp,
      C,
      col_offsets,
      bias,
      thread_id,
      num_threads);
}

static void depthwise_3x3x3_pad_1_relu_fused_(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const Packed3x3x3ConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    int thread_id,
    int num_threads) {
  int32_t C_int32_temp[(K + 31) / 32 * 32];
  depthwise_3x3x3_pad_1_<true /* FUSE_RESCALE */, true /* FUSE_RELU */>(
      N,
      T,
      H,
      W,
      K,
      stride_t,
      stride_h,
      stride_w,
      A_zero_point,
      A,
      B_zero_point,
      B,
      C_multiplier,
      C_zero_point,
      C_int32_temp,
      C,
      col_offsets,
      bias,
      thread_id,
      num_threads);
}

void depthwise_3x3x3_pad_1(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const Packed3x3x3ConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    int thread_id,
    int num_threads) {
  // If we inline the following two functions, I see stack overflow.
  if (fuse_relu) {
    depthwise_3x3x3_pad_1_relu_fused_(
        N,
        T,
        H,
        W,
        K,
        stride_t,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        thread_id,
        num_threads);
  } else {
    depthwise_3x3x3_pad_1_(
        N,
        T,
        H,
        W,
        K,
        stride_t,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        thread_id,
        num_threads);
  }
}

void depthwise_3x3_per_channel_quantization_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const Packed3x3ConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    int thread_id,
    int num_threads) {
  int32_t C_int32_temp[(K + 31) / 32 * 32];
  if (7 == H && 7 == W && 1 == stride_h && 1 == stride_w) {
    depthwise_3x3_per_channel_quantization_pad_1_(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        Bp,
        C_multiplier,
        C_zero_point,
        C_int32_temp,
        C,
        col_offsets,
        bias,
        thread_id,
        num_threads);
  } else if (14 == H && 14 == W && 2 == stride_h && 2 == stride_w) {
    depthwise_3x3_per_channel_quantization_pad_1_(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        Bp,
        C_multiplier,
        C_zero_point,
        C_int32_temp,
        C,
        col_offsets,
        bias,
        thread_id,
        num_threads);
  } else if (1 == stride_h && 1 == stride_w) {
    depthwise_3x3_per_channel_quantization_pad_1_(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        Bp,
        C_multiplier,
        C_zero_point,
        C_int32_temp,
        C,
        col_offsets,
        bias,
        thread_id,
        num_threads);
  } else if (2 == stride_h && 2 == stride_w) {
    depthwise_3x3_per_channel_quantization_pad_1_(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        Bp,
        C_multiplier,
        C_zero_point,
        C_int32_temp,
        C,
        col_offsets,
        bias,
        thread_id,
        num_threads);
  } else {
    depthwise_3x3_per_channel_quantization_pad_1_(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        Bp,
        C_multiplier,
        C_zero_point,
        C_int32_temp,
        C,
        col_offsets,
        bias,
        thread_id,
        num_threads);
  }
}

} // namespace fbgemm
