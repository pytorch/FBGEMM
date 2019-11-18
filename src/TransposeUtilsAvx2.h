/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <immintrin.h>
#include "./MaskAvx2.h"

namespace fbgemm {

namespace internal {

#ifdef __AVX2__
// NOTE: Make sure every function defined in here has static linkage because
// this header file is included by UtilsAvx512.cc compiled with -mavx512f option

// 4 * 4 = 16 instructions
static inline void
transpose_kernel_4x4_sse(const float* src, int ld_src, float* dst, int ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3
  // b : b0 b1 b2 b3
  // c : c0 c1 c2 c3
  // d : d0 d1 d2 d3
  __m128 a = _mm_loadu_ps(&src[0 * ld_src]);
  __m128 b = _mm_loadu_ps(&src[1 * ld_src]);
  __m128 c = _mm_loadu_ps(&src[2 * ld_src]);
  __m128 d = _mm_loadu_ps(&src[3 * ld_src]);

  // transpose the 4x4 matrix formed by 32-bit elements: Macro from SSE
  // a : a0 b0 c0 d0
  // b : a1 b1 c1 d1
  // c : a2 b2 c2 d2
  // d : a3 b3 c3 d3
  _MM_TRANSPOSE4_PS(a, b, c, d);

  // store from registers to dst
  _mm_storeu_ps(&dst[0 * ld_dst], a);
  _mm_storeu_ps(&dst[1 * ld_dst], b);
  _mm_storeu_ps(&dst[2 * ld_dst], c);
  _mm_storeu_ps(&dst[3 * ld_dst], d);
}

// kernel for transpose mxn where m, n <= 4
// M + (M + 1) / 2 * 2 + 2 * N instructions
template <int M>
static void transpose_kernel_mxn_sse(
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  // clang-format off
  alignas(64) static const int masks[5][4] = {
    {  0,  0,  0,  0, },
    { -1,  0,  0,  0, },
    { -1, -1,  0,  0, },
    { -1, -1, -1,  0, },
    { -1, -1, -1, -1, },
  };
  // clang-format on

  // load from src to registers
  __m128i mask_v = _mm_load_si128(reinterpret_cast<const __m128i*>(masks[N]));
  __m128 input[4];
  int i;
  for (i = 0; i < M; ++i) {
    input[i] = _mm_maskload_ps(&src[i * ld_src], mask_v);
  }
  for (; i < 4; ++i) {
    // Not really needed but to avoid uninitialized variable warning.
    // Shouldn't be much overhead because xor can be executed in parallel with
    // other instructions.
    input[i] = _mm_setzero_ps();
  }

  __m128 temp[4];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = _mm_unpacklo_ps(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = _mm_unpackhi_ps(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 4; ++i) {
    temp[i] = _mm_setzero_ps();
  }

  mask_v = _mm_load_si128(reinterpret_cast<const __m128i*>(masks[M]));
  for (i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      input[i] = _mm_movelh_ps(temp[i / 2], temp[2 + i / 2]);
    } else {
      input[i] = _mm_movehl_ps(temp[2 + i / 2], temp[i / 2]);
    }
    _mm_maskstore_ps(&dst[i * ld_dst], mask_v, input[i]);
  }
}

// 8 * 5 = 40 instructions
static inline void transpose_kernel_8x8_avx2(
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3 a4 a5 a6 a7
  // b : b0 b1 b2 b3 b4 b5 b6 b7
  // c : c0 c1 c2 c3 c4 c5 c6 c7
  // d : d0 d1 d2 d3 d4 d5 d6 d7
  // e : e0 e1 e2 e3 e4 e5 e6 e7
  // f : f0 f1 f2 f3 f4 f5 f6 f7
  // g : g0 g1 g2 g3 g4 g5 g6 g7
  // h : h0 h1 h2 h3 h4 h5 h6 h7
  __m256 a = _mm256_loadu_ps(&src[0 * ld_src]);
  __m256 b = _mm256_loadu_ps(&src[1 * ld_src]);
  __m256 c = _mm256_loadu_ps(&src[2 * ld_src]);
  __m256 d = _mm256_loadu_ps(&src[3 * ld_src]);
  __m256 e = _mm256_loadu_ps(&src[4 * ld_src]);
  __m256 f = _mm256_loadu_ps(&src[5 * ld_src]);
  __m256 g = _mm256_loadu_ps(&src[6 * ld_src]);
  __m256 h = _mm256_loadu_ps(&src[7 * ld_src]);

  __m256 ab0145, ab2367, cd0145, cd2367, ef0145, ef2367, gh0145, gh2367;
  __m256 abcd04, abcd15, efgh04, efgh15, abcd26, abcd37, efgh26, efgh37;
  // unpacking and interleaving 32-bit elements
  // ab0145 : a0 b0 a1 b1 a4 b4 a5 b5
  // ab2367 : a2 b2 a3 b3 a6 b6 a7 b7
  // cd0145 : c0 d0 c1 d1 c4 d4 c5 d5
  // cd2367 : c2 d2 c3 d3 c6 d6 c7 d7
  // ef0145 : e0 f0 e1 f1 e4 f4 e5 f5
  // ef2367 : e2 f2 e3 f3 e6 f6 e7 f7
  // gh0145 : g0 h0 g1 h1 g4 h4 g5 h5
  // gh2367 : g2 h2 g3 h3 g6 h6 g7 h7
  ab0145 = _mm256_unpacklo_ps(a, b);
  ab2367 = _mm256_unpackhi_ps(a, b);
  cd0145 = _mm256_unpacklo_ps(c, d);
  cd2367 = _mm256_unpackhi_ps(c, d);
  ef0145 = _mm256_unpacklo_ps(e, f);
  ef2367 = _mm256_unpackhi_ps(e, f);
  gh0145 = _mm256_unpacklo_ps(g, h);
  gh2367 = _mm256_unpackhi_ps(g, h);

  // shuffling the 32-bit elements
  // abcd04 : a0 b0 c0 d0 a4 b4 c4 d4
  // abcd15 : a1 b1 c1 d1 a5 b5 c5 d5
  // efgh04 : e0 f0 g0 h0 e4 f4 g4 h4
  // efgh15 : e1 f1 g1 h1 e5 b5 c5 d5
  // abcd26 : a2 b2 c2 d2 a6 b6 c6 d6
  // abcd37 : a3 b3 c3 d3 a7 b7 c7 d7
  // efgh26 : e2 f2 g2 h2 e6 f6 g6 h6
  // efgh37 : e3 f3 g3 h3 e7 f7 g7 h7
  abcd04 = _mm256_shuffle_ps(ab0145, cd0145, 0x44);
  abcd15 = _mm256_shuffle_ps(ab0145, cd0145, 0xee);
  efgh04 = _mm256_shuffle_ps(ef0145, gh0145, 0x44);
  efgh15 = _mm256_shuffle_ps(ef0145, gh0145, 0xee);
  abcd26 = _mm256_shuffle_ps(ab2367, cd2367, 0x44);
  abcd37 = _mm256_shuffle_ps(ab2367, cd2367, 0xee);
  efgh26 = _mm256_shuffle_ps(ef2367, gh2367, 0x44);
  efgh37 = _mm256_shuffle_ps(ef2367, gh2367, 0xee);

  // shuffling 128-bit elements
  // a : a0 b0 c0 d0 e0 f0 g0 h0
  // b : a1 b1 c1 d1 e1 f1 g1 h1
  // c : a2 b2 c2 d2 e2 f2 g2 h2
  // d : a3 b3 c3 d3 e3 f3 g3 h3
  // e : a4 b4 c4 d4 e4 f4 g4 h4
  // f : a5 b5 c5 d5 e5 f5 g5 h5
  // g : a6 b6 c6 d6 e6 f6 g6 h6
  // h : a7 b7 c7 d7 e7 f7 g7 h7
  a = _mm256_permute2f128_ps(efgh04, abcd04, 0x02);
  b = _mm256_permute2f128_ps(efgh15, abcd15, 0x02);
  c = _mm256_permute2f128_ps(efgh26, abcd26, 0x02);
  d = _mm256_permute2f128_ps(efgh37, abcd37, 0x02);
  e = _mm256_permute2f128_ps(efgh04, abcd04, 0x13);
  f = _mm256_permute2f128_ps(efgh15, abcd15, 0x13);
  g = _mm256_permute2f128_ps(efgh26, abcd26, 0x13);
  h = _mm256_permute2f128_ps(efgh37, abcd37, 0x13);

  // store from registers to dst
  _mm256_storeu_ps(&dst[0 * ld_dst], a);
  _mm256_storeu_ps(&dst[1 * ld_dst], b);
  _mm256_storeu_ps(&dst[2 * ld_dst], c);
  _mm256_storeu_ps(&dst[3 * ld_dst], d);
  _mm256_storeu_ps(&dst[4 * ld_dst], e);
  _mm256_storeu_ps(&dst[5 * ld_dst], f);
  _mm256_storeu_ps(&dst[6 * ld_dst], g);
  _mm256_storeu_ps(&dst[7 * ld_dst], h);
}

// kernel for transposing mxn where m, n <= 8
// M + (M + 1) / 2 * 2 + (M + 3) / 4 * 4 + 2 * N instructions
template <int M>
static void transpose_kernel_mxn_avx2(
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  // load from src to registers
  __m256i mask_v = _mm256_load_si256(
      reinterpret_cast<const __m256i*>(internal::avx2_ps_or_epi32_masks[N]));
  __m256 input[8];
  int i;
  for (i = 0; i < M; ++i) {
    input[i] = _mm256_maskload_ps(&src[i * ld_src], mask_v);
  }
  for (; i < 8; ++i) {
    // Not really needed but to avoid uninitialized variable warning.
    // Shouldn't be much overhead because xor can be executed in parallel with
    // other instructions.
    input[i] = _mm256_setzero_ps();
  }

  // unpacking and interleaving 32-bit elements
  __m256 temp[8];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = _mm256_unpacklo_ps(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = _mm256_unpackhi_ps(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 8; ++i) {
    temp[i] = _mm256_setzero_ps();
  }

  // shuffling the 32-bit elements
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = _mm256_shuffle_ps(temp[4 * i], temp[4 * i + 2], 0x44);
    input[4 * i + 1] = _mm256_shuffle_ps(temp[4 * i], temp[4 * i + 2], 0xee);
    input[4 * i + 2] =
        _mm256_shuffle_ps(temp[4 * i + 1], temp[4 * i + 3], 0x44);
    input[4 * i + 3] =
        _mm256_shuffle_ps(temp[4 * i + 1], temp[4 * i + 3], 0xee);
  }

  // shuffling 128-bit elements
  // store from registers to dst
  mask_v = _mm256_load_si256(
      reinterpret_cast<const __m256i*>(internal::avx2_ps_or_epi32_masks[M]));
  for (i = 0; i < N; ++i) {
    if (i < 4) {
      temp[i] = _mm256_permute2f128_ps(input[4 + i], input[i], 0x02);
    } else {
      temp[i] = _mm256_permute2f128_ps(input[i], input[i - 4], 0x13);
    }
    _mm256_maskstore_ps(&dst[i * ld_dst], mask_v, temp[i]);
  }
}
#endif // __AVX2__

} // namespace internal

} // namespace fbgemm
