/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>
#endif
#include "./TransposeUtils.h"
#include "./TransposeUtilsAvx2.h"
namespace fbgemm {

namespace {

// 16 * 6 = 96 instructions
inline void transpose_kernel_16x16_avx512(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  // load from src to registers
  // a: a0  a1  a2  a3  a4  a5  a6  a7  a8  a9  a10 a11 a12 a13 a14 a15
  // b: b0  b1  b2  b3  b4  b5  b6  b7  b8  b9  b10 b11 b12 b13 b14 b15
  // c: c0  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10 c11 c12 c13 c14 c15
  // d: d0  d1  d2  d3  d4  d5  d6  d7  d8  d9  d10 d11 d12 d13 d14 d15
  // e: e0  e1  e2  e3  e4  e5  e6  e7  e8  e9  e10 e11 e12 e13 e14 e15
  // f: f0  f1  f2  f3  f4  f5  f6  f7  f8  f9  f10 f11 f12 f13 f14 f15
  // g: g0  g1  g2  g3  g4  g5  g6  g7  g8  g9  g10 g11 g12 g13 g14 g15
  // h: h0  h1  h2  h3  h4  h5  h6  h7  h8  h9  h10 h11 h12 h13 h14 h15
  // i: i0  i1  i2  i3  i4  i5  i6  i7  i8  i9  i10 i11 i12 i13 i14 i15
  // j: j0  j1  j2  j3  j4  j5  j6  j7  j8  j9  j10 j11 j12 j13 j14 j15
  // k: k0  k1  k2  k3  k4  k5  k6  k7  k8  k9  k10 k11 k12 k13 k14 k15
  // l: l0  l1  l2  l3  l4  l5  l6  l7  l8  l9  l10 l11 l12 l13 l14 l15
  // m: m0  m1  m2  m3  m4  m5  m6  m7  m8  m9  m10 m11 m12 m13 m14 m15
  // n: n0  n1  n2  n3  n4  n5  n6  n7  n8  n9  n10 n11 n12 n13 n14 n15
  // o: o0  o1  o2  o3  o4  o5  o6  o7  o8  o9  o10 o11 o12 o13 o14 o15
  // p: p0  p1  p2  p3  p4  p5  p6  p7  p8  p9  p10 p11 p12 p13 p14 p15
  __m512 a = _mm512_loadu_ps(&src[0 * ld_src]);
  __m512 b = _mm512_loadu_ps(&src[1 * ld_src]);
  __m512 c = _mm512_loadu_ps(&src[2 * ld_src]);
  __m512 d = _mm512_loadu_ps(&src[3 * ld_src]);
  __m512 e = _mm512_loadu_ps(&src[4 * ld_src]);
  __m512 f = _mm512_loadu_ps(&src[5 * ld_src]);
  __m512 g = _mm512_loadu_ps(&src[6 * ld_src]);
  __m512 h = _mm512_loadu_ps(&src[7 * ld_src]);
  __m512 i = _mm512_loadu_ps(&src[8 * ld_src]);
  __m512 j = _mm512_loadu_ps(&src[9 * ld_src]);
  __m512 k = _mm512_loadu_ps(&src[10 * ld_src]);
  __m512 l = _mm512_loadu_ps(&src[11 * ld_src]);
  __m512 m = _mm512_loadu_ps(&src[12 * ld_src]);
  __m512 n = _mm512_loadu_ps(&src[13 * ld_src]);
  __m512 o = _mm512_loadu_ps(&src[14 * ld_src]);
  __m512 p = _mm512_loadu_ps(&src[15 * ld_src]);

  __m512 ta, tb, tc, td, te, tf, tg, th, ti, tj, tk, tl, tm, tn, to, tq;
  // unpacking and interleaving 32-bit elements
  // a0  b0  a1  b1  a4  b4  a5  b5  a8  b8  a9  b9  a12  b12 a13 b13
  // a2  b2  a3  b3  a6  b6  a7  b7  a10 b10 a11 b11 a14  b14 a15 b15
  // c0  d0  c1  d1 ...
  // c2  d2  c3  d3 ...
  // e0  f0  e1  f1 ...
  // e2  f2  e3  f3 ...
  // g0  h0  g1  h1 ...
  // g2  h2  g3  h3 ...
  // i0  ...
  // i2  ...
  // k0  ...
  // k2  ...
  // m0  ...
  // m2  ...
  // o0  ...
  // o1  ...
  ta = _mm512_unpacklo_ps(a, b);
  tb = _mm512_unpackhi_ps(a, b);
  tc = _mm512_unpacklo_ps(c, d);
  td = _mm512_unpackhi_ps(c, d);
  te = _mm512_unpacklo_ps(e, f);
  tf = _mm512_unpackhi_ps(e, f);
  tg = _mm512_unpacklo_ps(g, h);
  th = _mm512_unpackhi_ps(g, h);
  ti = _mm512_unpacklo_ps(i, j);
  tj = _mm512_unpackhi_ps(i, j);
  tk = _mm512_unpacklo_ps(k, l);
  tl = _mm512_unpackhi_ps(k, l);
  tm = _mm512_unpacklo_ps(m, n);
  tn = _mm512_unpackhi_ps(m, n);
  to = _mm512_unpacklo_ps(o, p);
  tq = _mm512_unpackhi_ps(o, p);

  // unpacking and interleaving 64-bit elements
  //  a0  b0  c0  d0  a4  b4  c4  d4  a8  b8  c8  d8  a12 b12 c12 d12
  //  a1  b1  c1  d1 ...
  //  a2  b2  c2  d2 ...
  //  a3  b3  c3  d3 ...
  //  e0  f0  g0  h0  e4  f4  g4  h4  e8  f8  g8  h8  e12 f12 g12 h12
  //  e1  f1  g1  h1 ...
  //  e2  f2  g2  h2 ...
  //  e3  f3  g3  h3 ...
  //  i0  j0  k0  l0 ...
  //  i1  j1  k1  l1 ...
  //  i2  j2  k2  l2 ...
  //  i3  j3  k3  l3 ...
  //  m0  n0  o0  p0 ...
  //  m1  n1  o1  p1 ...
  //  m2  n2  o2  p2 ...
  //  m3  n3  o3  p3 ...
  a = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(ta), _mm512_castps_pd(tc)));
  b = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(ta), _mm512_castps_pd(tc)));
  c = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tb), _mm512_castps_pd(td)));
  d = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tb), _mm512_castps_pd(td)));
  e = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(te), _mm512_castps_pd(tg)));
  f = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(te), _mm512_castps_pd(tg)));
  g = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tf), _mm512_castps_pd(th)));
  h = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tf), _mm512_castps_pd(th)));
  i = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(ti), _mm512_castps_pd(tk)));
  j = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(ti), _mm512_castps_pd(tk)));
  k = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tj), _mm512_castps_pd(tl)));
  l = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tj), _mm512_castps_pd(tl)));
  m = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tm), _mm512_castps_pd(to)));
  n = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tm), _mm512_castps_pd(to)));
  o = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tn), _mm512_castps_pd(tq)));
  p = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tn), _mm512_castps_pd(tq)));

  //  shuffle 128-bits (composed of 4 32-bit elements)
  //  a0  b0  c0  d0  a8  b8  c8  d8  e0  f0  g0  h0  e8  f8  g8  h8
  //  a1  b1  c1  d1 ...
  //  a2  b2  c2  d2 ...
  //  a3  b3  c3  d3 ...
  //  a4  b4  c4  d4 ...
  //  a5  b5  c5  d5 ...
  //  a6  b6  c6  d6 ...
  //  a7  b7  c7  d7 ...
  //  i0  j0  k0  l0  i8  j8  k8  l8  m0  n0  o0  p0  m8  n8  o8  p8
  //  i1  j1  k1  l1 ...
  //  i2  j2  k2  l2 ...
  //  i3  j3  k3  l3 ...
  //  i4  j4  k4  l4 ...
  //  i5  j5  k5  l5 ...
  //  i6  j6  k6  l6 ...
  //  i7  j7  k7  l7 ...
  ta = _mm512_shuffle_f32x4(a, e, 0x88);
  tb = _mm512_shuffle_f32x4(b, f, 0x88);
  tc = _mm512_shuffle_f32x4(c, g, 0x88);
  td = _mm512_shuffle_f32x4(d, h, 0x88);
  te = _mm512_shuffle_f32x4(a, e, 0xdd);
  tf = _mm512_shuffle_f32x4(b, f, 0xdd);
  tg = _mm512_shuffle_f32x4(c, g, 0xdd);
  th = _mm512_shuffle_f32x4(d, h, 0xdd);
  ti = _mm512_shuffle_f32x4(i, m, 0x88);
  tj = _mm512_shuffle_f32x4(j, n, 0x88);
  tk = _mm512_shuffle_f32x4(k, o, 0x88);
  tl = _mm512_shuffle_f32x4(l, p, 0x88);
  tm = _mm512_shuffle_f32x4(i, m, 0xdd);
  tn = _mm512_shuffle_f32x4(j, n, 0xdd);
  to = _mm512_shuffle_f32x4(k, o, 0xdd);
  tq = _mm512_shuffle_f32x4(l, p, 0xdd);

  //  shuffle 128-bits (composed of 4 32-bit elements)
  //  a0  b0  c0  d0  ...  o0
  //  a1  b1  c1  d1  ...  o1
  //  a2  b2  c2  d2  ...  o2
  //  a3  b3  c3  d3  ...  o3
  //  a4  ...
  //  a5  ...
  //  a6  ...
  //  a7  ...
  //  a8  ...
  //  a9  ...
  //  a10 ...
  //  a11 ...
  //  a12 ...
  //  a13 ...
  //  a14 ...
  //  a15 b15 c15 d15 ...  o15
  a = _mm512_shuffle_f32x4(ta, ti, 0x88);
  b = _mm512_shuffle_f32x4(tb, tj, 0x88);
  c = _mm512_shuffle_f32x4(tc, tk, 0x88);
  d = _mm512_shuffle_f32x4(td, tl, 0x88);
  e = _mm512_shuffle_f32x4(te, tm, 0x88);
  f = _mm512_shuffle_f32x4(tf, tn, 0x88);
  g = _mm512_shuffle_f32x4(tg, to, 0x88);
  h = _mm512_shuffle_f32x4(th, tq, 0x88);
  i = _mm512_shuffle_f32x4(ta, ti, 0xdd);
  j = _mm512_shuffle_f32x4(tb, tj, 0xdd);
  k = _mm512_shuffle_f32x4(tc, tk, 0xdd);
  l = _mm512_shuffle_f32x4(td, tl, 0xdd);
  m = _mm512_shuffle_f32x4(te, tm, 0xdd);
  n = _mm512_shuffle_f32x4(tf, tn, 0xdd);
  o = _mm512_shuffle_f32x4(tg, to, 0xdd);
  p = _mm512_shuffle_f32x4(th, tq, 0xdd);

  // store from registers to dst
  _mm512_storeu_ps(&dst[0 * ld_dst], a);
  _mm512_storeu_ps(&dst[1 * ld_dst], b);
  _mm512_storeu_ps(&dst[2 * ld_dst], c);
  _mm512_storeu_ps(&dst[3 * ld_dst], d);
  _mm512_storeu_ps(&dst[4 * ld_dst], e);
  _mm512_storeu_ps(&dst[5 * ld_dst], f);
  _mm512_storeu_ps(&dst[6 * ld_dst], g);
  _mm512_storeu_ps(&dst[7 * ld_dst], h);
  _mm512_storeu_ps(&dst[8 * ld_dst], i);
  _mm512_storeu_ps(&dst[9 * ld_dst], j);
  _mm512_storeu_ps(&dst[10 * ld_dst], k);
  _mm512_storeu_ps(&dst[11 * ld_dst], l);
  _mm512_storeu_ps(&dst[12 * ld_dst], m);
  _mm512_storeu_ps(&dst[13 * ld_dst], n);
  _mm512_storeu_ps(&dst[14 * ld_dst], o);
  _mm512_storeu_ps(&dst[15 * ld_dst], p);
}

// kernel for transposing mxn where m, n <= 16
// M + (M + 1) / 2 * 2 + (M + 3) / 4 * 4 + (M + 7) / 8 * 8 + 2 * N instructions
template <int M>
void transpose_kernel_mxn_avx512(
    int N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  // load from src to registers
  __mmask16 src_mask = (1 << N) - 1;
  __m512 input[16];
  int i;
  for (i = 0; i < M; ++i) {
    input[i] = _mm512_maskz_loadu_ps(src_mask, &src[i * ld_src]);
  }
  for (; i < 16; ++i) {
    // Not really needed but to avoid uninitialized variable warning.
    // Shouldn't be much overhead because xor can be executed in parallel with
    // other instructions.
    input[i] = _mm512_setzero_ps();
  }

  // unpacking and interleaving 32-bit elements
  __m512 temp[16];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = _mm512_unpacklo_ps(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = _mm512_unpackhi_ps(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 16; ++i) {
    temp[i] = _mm512_setzero_ps();
  }

  // unpacking and interleaving 64-bit elements
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = _mm512_castpd_ps(_mm512_unpacklo_pd(
        _mm512_castps_pd(temp[4 * i]), _mm512_castps_pd(temp[4 * i + 2])));
    input[4 * i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(
        _mm512_castps_pd(temp[4 * i]), _mm512_castps_pd(temp[4 * i + 2])));
    input[4 * i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(
        _mm512_castps_pd(temp[4 * i + 1]), _mm512_castps_pd(temp[4 * i + 3])));
    input[4 * i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(
        _mm512_castps_pd(temp[4 * i + 1]), _mm512_castps_pd(temp[4 * i + 3])));
  }

  //  shuffle 128-bits (composed of 4 32-bit elements)
  for (i = 0; i < (M + 7) / 8; ++i) {
    temp[8 * i] = _mm512_shuffle_f32x4(input[8 * i], input[8 * i + 4], 0x88);
    temp[8 * i + 1] =
        _mm512_shuffle_f32x4(input[8 * i + 1], input[8 * i + 5], 0x88);
    temp[8 * i + 2] =
        _mm512_shuffle_f32x4(input[8 * i + 2], input[8 * i + 6], 0x88);
    temp[8 * i + 3] =
        _mm512_shuffle_f32x4(input[8 * i + 3], input[8 * i + 7], 0x88);
    temp[8 * i + 4] =
        _mm512_shuffle_f32x4(input[8 * i], input[8 * i + 4], 0xdd);
    temp[8 * i + 5] =
        _mm512_shuffle_f32x4(input[8 * i + 1], input[8 * i + 5], 0xdd);
    temp[8 * i + 6] =
        _mm512_shuffle_f32x4(input[8 * i + 2], input[8 * i + 6], 0xdd);
    temp[8 * i + 7] =
        _mm512_shuffle_f32x4(input[8 * i + 3], input[8 * i + 7], 0xdd);
  }

  // store from registers to dst
  __mmask16 dst_mask = (1 << M) - 1;
  for (i = 0; i < N; ++i) {
    if (i < 8) {
      input[i] = _mm512_shuffle_f32x4(temp[i], temp[8 + i], 0x88);
    } else {
      input[i] = _mm512_shuffle_f32x4(temp[i - 8], temp[i], 0xdd);
    }
    _mm512_mask_storeu_ps(&dst[i * ld_dst], dst_mask, input[i]);
  }
}

} // namespace

namespace internal {

template <typename T>
void transpose_avx512_contiguous_thin(
    const int64_t M,
    const int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);

template <typename T>
void transpose_avx512_contiguous_wide(
    const int64_t M,
    const int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);

// Permute elements in 128 bit lane
// e.g., if a 128-bit lane has the following elements:
// 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
//
// After this function call, it becomes
// 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
// The same happens with other 3 lanes.
static inline __m512i permute_row(__m512i row) {
  // clang-format off
  __m256i shuffle_256v0 = _mm256_set_epi8(
      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  // clang-format on
  __m512i shuffle_512v = _mm512_castsi256_si512(shuffle_256v0);
  row = _mm512_shuffle_epi8(
      row, _mm512_inserti64x4(shuffle_512v, shuffle_256v0, 1));
  return row;
}

static inline void core_transpose_16x32_block_i8(__m512i r[], __m512i u[]) {
  // Result after this operation; Read in conjunction with comments in
  // transpose_16x32_block
  // 00_00 00_01 01_00 01_01 00_04 00_05 01_04 01_05 04_00 04_01 05_00 05_01
  // 04_04 04_05 05_04 05_05
  u[0] = _mm512_unpacklo_epi64(r[0], r[1]);
  // 00_02 00_03 01_02 01_03 00_06 00_07 01_06 01_07 04_02 04_03 05_02 05_03
  // 04_06 04_07 05_06 05_07
  u[1] = _mm512_unpackhi_epi64(r[0], r[1]);
  // 02_00 02_01 03_00 03_01 02_04 02_05 03_04 03_05 06_00 06_01 07_00 07_01
  // 06_04 06_05 07_04 07_05
  u[2] = _mm512_unpacklo_epi64(r[2], r[3]);
  // 02_02 02_03 03_02 03_03 02_06 02_07 03_06 03_07 06_02 06_03 07_02 07_03
  // 06_06 06_07 07_06 07_07
  u[3] = _mm512_unpackhi_epi64(r[2], r[3]);
  // 08_00 08_01 09_00 09_01 08_04 08_05 09_04 09_05 12_00 12_01 13_00 13_01
  // 12_04 12_05 13_04 13_05
  u[4] = _mm512_unpacklo_epi64(r[4], r[5]);
  u[5] = _mm512_unpackhi_epi64(r[4], r[5]);
  u[6] = _mm512_unpacklo_epi64(r[6], r[7]);
  u[7] = _mm512_unpackhi_epi64(r[6], r[7]);

  // This instruction doesn't exist for epi32 so casting to ps
  // 00_00 01_00 02_00 03_00 00_04 01_04 02_04 03_04 04_00 05_00 06_00 07_00
  // 04_04 05_04 06_04 07_04
  r[0] = _mm512_castps_si512(_mm512_shuffle_ps(
      _mm512_castsi512_ps(u[0]), _mm512_castsi512_ps(u[2]), 0x88));
  // 00_01 01_01 02_01 03_01 00_05 01_05 02_05 03_05 04_01 05_01 06_01 07_01
  // 04_05 05_05 06_05 07_05
  r[1] = _mm512_castps_si512(_mm512_shuffle_ps(
      _mm512_castsi512_ps(u[0]), _mm512_castsi512_ps(u[2]), 0xDD));
  r[2] = _mm512_castps_si512(_mm512_shuffle_ps(
      _mm512_castsi512_ps(u[1]), _mm512_castsi512_ps(u[3]), 0x88));
  r[3] = _mm512_castps_si512(_mm512_shuffle_ps(
      _mm512_castsi512_ps(u[1]), _mm512_castsi512_ps(u[3]), 0xDD));
  // 08_00 09_00 10_00 11_00 08_04 09_04 10_04 11_04 12_00 13_00 14_00 15_00
  // 12_04 13_04 14_04 15_04
  r[4] = _mm512_castps_si512(_mm512_shuffle_ps(
      _mm512_castsi512_ps(u[4]), _mm512_castsi512_ps(u[6]), 0x88));
  r[5] = _mm512_castps_si512(_mm512_shuffle_ps(
      _mm512_castsi512_ps(u[4]), _mm512_castsi512_ps(u[6]), 0xDD));
  r[6] = _mm512_castps_si512(_mm512_shuffle_ps(
      _mm512_castsi512_ps(u[5]), _mm512_castsi512_ps(u[7]), 0x88));
  r[7] = _mm512_castps_si512(_mm512_shuffle_ps(
      _mm512_castsi512_ps(u[5]), _mm512_castsi512_ps(u[7]), 0xDD));

  // permute among 128-bit lanes
  r[0] = permute_row(r[0]);
  r[1] = permute_row(r[1]);
  r[2] = permute_row(r[2]);
  r[3] = permute_row(r[3]);
  r[4] = permute_row(r[4]);
  r[5] = permute_row(r[5]);
  r[6] = permute_row(r[6]);
  r[7] = permute_row(r[7]);

  __m512i const1 = _mm512_set_epi32(
      27, 19, 11, 3, 26, 18, 10, 2, 25, 17, 9, 1, 24, 16, 8, 0);
  __m512i const2 = _mm512_set_epi32(
      31, 23, 15, 7, 30, 22, 14, 6, 29, 21, 13, 5, 28, 20, 12, 4);

  // merge 128-bit values from two regs
  u[0] = _mm512_permutex2var_epi32(r[0], const1, r[4]);
  u[1] = _mm512_permutex2var_epi32(r[0], const2, r[4]);
  u[2] = _mm512_permutex2var_epi32(r[1], const1, r[5]);
  u[3] = _mm512_permutex2var_epi32(r[1], const2, r[5]);
  u[4] = _mm512_permutex2var_epi32(r[2], const1, r[6]);
  u[5] = _mm512_permutex2var_epi32(r[2], const2, r[6]);
  u[6] = _mm512_permutex2var_epi32(r[3], const1, r[7]);
  u[7] = _mm512_permutex2var_epi32(r[3], const2, r[7]);
}

static inline void core_transpose_16x16_block(__m512i r[], __m512i u[]) {
  // a0a1 b0b1 a2a3 b2b3 a8a9 b8b9 a10a11 b10b11   e0e1 f0f1 e2e3 f2f3 e8e9 f8f9
  // e10e11 f10f11
  u[0] = _mm512_unpacklo_epi32(r[0], r[1]);
  // a4a5 b4b5 a6a7 b6b7 a12a13 b12b13 a14a15 b14b15   e4e5 f4f5 e6e7 f6f7
  // e12e13 f12f13 e14e15 f14f15
  u[1] = _mm512_unpackhi_epi32(r[0], r[1]);
  // c0c1 d0d1 c2c3 d2d3 c8c9 d8d9 c10c11 d10d11   g0g1 h0h1 g2g3 h2h3 g8g9 h8h9
  // g10g11 h10h11
  u[2] = _mm512_unpacklo_epi32(r[2], r[3]);
  // c4c5 d4b5 c6c7 d6b7 c12c13 d12d13 c14c15 d14d15   g4g5 h4h5 g6g7 h6h7
  // g12g13 h12h13 g14g15 h14h15
  u[3] = _mm512_unpackhi_epi32(r[2], r[3]);
  // i j  m n
  u[4] = _mm512_unpacklo_epi32(r[4], r[5]);
  u[5] = _mm512_unpackhi_epi32(r[4], r[5]);
  // k l  o p
  u[6] = _mm512_unpacklo_epi32(r[6], r[7]);
  u[7] = _mm512_unpackhi_epi32(r[6], r[7]);

  // a0a1 b0b1 c0c1 d0d1 a8a9 b8b9 c8c9 d8d9  e0e1 f0f1 g0g1 h0h1 e8e9 f8f9 g8g9
  // h8h9
  r[0] = _mm512_unpacklo_epi64(u[0], u[2]);
  // a2a3 b2b3 c2c3 d2d3 a10a11 b10b11 c10c11 d10d11  e2e3 f2f3 g2g3 h2h3 e10e11
  // f10f11 g10g11 h10h11
  r[1] = _mm512_unpackhi_epi64(u[0], u[2]);
  // a4a5 b4b5 c4c5 d4b5 a12a13 b12b13 c12c13 d12d13
  r[2] = _mm512_unpacklo_epi64(u[1], u[3]);
  // a6a7 b6b7 c6c7 d6b7 a14a15 b14b15 c14c15 d14d15
  r[3] = _mm512_unpackhi_epi64(u[1], u[3]);
  // i j k l m n o p
  r[4] = _mm512_unpacklo_epi64(u[4], u[6]);
  r[5] = _mm512_unpackhi_epi64(u[4], u[6]);
  r[6] = _mm512_unpacklo_epi64(u[5], u[7]);
  r[7] = _mm512_unpackhi_epi64(u[5], u[7]);

  __m512i const1 = _mm512_set_epi32(
      0x00370035,
      0x00330031,
      0x00270025,
      0x00230021,
      0x00170015,
      0x00130011,
      0x00070005,
      0x00030001,
      0x00360034,
      0x00320030,
      0x00260024,
      0x00220020,
      0x00160014,
      0x00120010,
      0x00060004,
      0x00020000);
  __m512i const2 = _mm512_set_epi32(
      0x003f003d,
      0x003b0039,
      0x002f002d,
      0x002b0029,
      0x001f001d,
      0x001b0019,
      0x000f000d,
      0x000b0009,
      0x003e003c,
      0x003a0038,
      0x002e002c,
      0x002a0028,
      0x001e001c,
      0x001a0018,
      0x000e000c,
      0x000a0008);

  // merge values from two regs
  u[0] = _mm512_permutex2var_epi16(r[0], const1, r[4]); // 0-- 1--
  u[4] = _mm512_permutex2var_epi16(r[0], const2, r[4]); // 8-- 9--
  u[2] = _mm512_permutex2var_epi16(r[2], const1, r[6]); // 4-- 5--
  u[6] = _mm512_permutex2var_epi16(r[2], const2, r[6]); // 12-- 13--
  u[1] = _mm512_permutex2var_epi16(r[1], const1, r[5]); // 2-- 3--
  u[5] = _mm512_permutex2var_epi16(r[1], const2, r[5]); // 10-- 11--
  u[3] = _mm512_permutex2var_epi16(r[3], const1, r[7]); // 6-- 7--
  u[7] = _mm512_permutex2var_epi16(r[3], const2, r[7]); // 14-- 15--
}

static inline void load_with_remainders_i16(
    const uint16_t* src,
    int64_t ld_src,
    __m512i r[],
    int mrem,
    int nrem) {
  __m512i t[16];
  if (nrem < 16) {
    __mmask32 mask_nrem_v = (((long long)1) << nrem) - 1;
    for (int i = 0; i < mrem; ++i) {
      // mask load
      t[i] = _mm512_maskz_loadu_epi16(mask_nrem_v, src + i * ld_src);
    }
  } else {
    for (int i = 0; i < mrem; ++i) {
      // normal load
      t[i] = _mm512_castsi256_si512(_mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(src + i * ld_src)));
    }
  }
  r[0] = _mm512_inserti64x4(t[0], _mm512_castsi512_si256(t[4]), 0x01);
  r[1] = _mm512_inserti64x4(t[1], _mm512_castsi512_si256(t[5]), 0x01);
  r[2] = _mm512_inserti64x4(t[2], _mm512_castsi512_si256(t[6]), 0x01);
  r[3] = _mm512_inserti64x4(t[3], _mm512_castsi512_si256(t[7]), 0x01);
  r[4] = _mm512_inserti64x4(t[8], _mm512_castsi512_si256(t[12]), 0x01);
  r[5] = _mm512_inserti64x4(t[9], _mm512_castsi512_si256(t[13]), 0x01);
  r[6] = _mm512_inserti64x4(t[10], _mm512_castsi512_si256(t[14]), 0x01);
  r[7] = _mm512_inserti64x4(t[11], _mm512_castsi512_si256(t[15]), 0x01);
}

static inline void load_with_remainders_i8(
    const uint8_t* src,
    int64_t ld_src,
    __m512i r[],
    int mrem,
    int nrem) {
  __m512i t[16];
  if (nrem < 32) {
    __mmask64 mask_nrem_v = (((long long)1) << nrem) - 1;
    for (int i = 0; i < mrem; ++i) {
      // mask load
      t[i] = _mm512_maskz_loadu_epi8(mask_nrem_v, src + i * ld_src);
    }
  } else {
    for (int i = 0; i < mrem; ++i) {
      // normal load
      t[i] = _mm512_castsi256_si512(_mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(src + i * ld_src)));
    }
  }
  r[0] = _mm512_inserti64x4(t[0], _mm512_castsi512_si256(t[4]), 0x01);
  r[1] = _mm512_inserti64x4(t[1], _mm512_castsi512_si256(t[5]), 0x01);
  r[2] = _mm512_inserti64x4(t[2], _mm512_castsi512_si256(t[6]), 0x01);
  r[3] = _mm512_inserti64x4(t[3], _mm512_castsi512_si256(t[7]), 0x01);
  r[4] = _mm512_inserti64x4(t[8], _mm512_castsi512_si256(t[12]), 0x01);
  r[5] = _mm512_inserti64x4(t[9], _mm512_castsi512_si256(t[13]), 0x01);
  r[6] = _mm512_inserti64x4(t[10], _mm512_castsi512_si256(t[14]), 0x01);
  r[7] = _mm512_inserti64x4(t[11], _mm512_castsi512_si256(t[15]), 0x01);
}

static inline void store_with_remainders_i16(
    uint16_t* dst,
    int64_t ld_dst,
    __m512i u[],
    int mrem,
    int nrem) {
  if (mrem < 16) {
    __mmask32 mask_mrem_v = (((long long)1) << mrem) - 1;
    int i = 0;

    for (; i < nrem / 2 * 2; i += 2) {
      // mask store
      int reg_idx = i / 2;
      _mm512_mask_storeu_epi16(
          dst + (i + 0) * ld_dst,
          mask_mrem_v,
          _mm512_castsi256_si512(_mm512_extracti32x8_epi32(u[reg_idx], 0x0)));
      _mm512_mask_storeu_epi16(
          dst + (i + 1) * ld_dst,
          mask_mrem_v,
          _mm512_castsi256_si512(_mm512_extracti32x8_epi32(u[reg_idx], 0x1)));
    }
    if (i < nrem) {
      int reg_idx = i / 2;
      _mm512_mask_storeu_epi16(
          dst + (i + 0) * ld_dst,
          mask_mrem_v,
          _mm512_castsi256_si512(_mm512_extracti32x8_epi32(u[reg_idx], 0x0)));
    }
  } else {
    int i = 0;
    for (; i < nrem / 2 * 2; i += 2) {
      // normal store
      int reg_idx = i / 2;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(dst + (i + 0) * ld_dst),
          _mm512_extracti32x8_epi32(u[reg_idx], 0x0));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(dst + (i + 1) * ld_dst),
          _mm512_extracti32x8_epi32(u[reg_idx], 0x1));
    }
    if (i < nrem) {
      int reg_idx = i / 2;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(dst + (i + 0) * ld_dst),
          _mm512_extracti32x8_epi32(u[reg_idx], 0x0));
    }
  }
}

static inline void store_with_remainders_i8(
    uint8_t* dst,
    int64_t ld_dst,
    __m512i u[],
    int mrem,
    int nrem) {
  if (mrem < 16) {
    __mmask64 mask_mrem_v = (((long long)1) << mrem) - 1;
    int i = 0;
    for (; i < nrem / 4 * 4; i += 4) {
      // mask store
      // we need 0, 4, 8, 16 => 0, 2, 4, 6
      // and 16, 20, 24, 28 => 1, 3, 5, 7
      // See stores for non-rem case
      int reg_idx = i / 16 + 2 * ((i % 16) / 4);
      _mm512_mask_storeu_epi8(
          dst + (i + 0) * ld_dst,
          mask_mrem_v,
          _mm512_castsi128_si512(_mm512_extracti32x4_epi32(u[reg_idx], 0x0)));
      _mm512_mask_storeu_epi8(
          dst + (i + 1) * ld_dst,
          mask_mrem_v,
          _mm512_castsi128_si512(_mm512_extracti32x4_epi32(u[reg_idx], 0x1)));
      _mm512_mask_storeu_epi8(
          dst + (i + 2) * ld_dst,
          mask_mrem_v,
          _mm512_castsi128_si512(_mm512_extracti32x4_epi32(u[reg_idx], 0x2)));
      _mm512_mask_storeu_epi8(
          dst + (i + 3) * ld_dst,
          mask_mrem_v,
          _mm512_castsi128_si512(_mm512_extracti32x4_epi32(u[reg_idx], 0x3)));
    }
    int rem = nrem - i;
    int reg_rem_idx = i / 16 + 2 * ((i % 16) / 4);
    switch (rem) {
      case 1:
        _mm512_mask_storeu_epi8(
            dst + (i + 0) * ld_dst,
            mask_mrem_v,
            _mm512_castsi128_si512(
                _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x0)));
        break;
      case 2:
        _mm512_mask_storeu_epi8(
            dst + (i + 0) * ld_dst,
            mask_mrem_v,
            _mm512_castsi128_si512(
                _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x0)));
        _mm512_mask_storeu_epi8(
            dst + (i + 1) * ld_dst,
            mask_mrem_v,
            _mm512_castsi128_si512(
                _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x1)));
        break;
      case 3:
        _mm512_mask_storeu_epi8(
            dst + (i + 0) * ld_dst,
            mask_mrem_v,
            _mm512_castsi128_si512(
                _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x0)));
        _mm512_mask_storeu_epi8(
            dst + (i + 1) * ld_dst,
            mask_mrem_v,
            _mm512_castsi128_si512(
                _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x1)));
        _mm512_mask_storeu_epi8(
            dst + (i + 2) * ld_dst,
            mask_mrem_v,
            _mm512_castsi128_si512(
                _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x2)));
        break;
      default:
        break;
    }

  } else {
    int i = 0;
    for (; i < nrem / 4 * 4; i += 4) {
      // normal store
      int reg_idx = i / 16 + 2 * ((i % 16) / 4);
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(dst + (i + 0) * ld_dst),
          _mm512_extracti32x4_epi32(u[reg_idx], 0x0));
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(dst + (i + 1) * ld_dst),
          _mm512_extracti32x4_epi32(u[reg_idx], 0x1));
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(dst + (i + 2) * ld_dst),
          _mm512_extracti32x4_epi32(u[reg_idx], 0x2));
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(dst + (i + 3) * ld_dst),
          _mm512_extracti32x4_epi32(u[reg_idx], 0x3));
    }
    int rem = nrem - i;
    int reg_rem_idx = i / 16 + 2 * ((i % 16) / 4);
    switch (rem) {
      case 1:
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst + (i + 0) * ld_dst),
            _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x0));
        break;
      case 2:
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst + (i + 0) * ld_dst),
            _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x0));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst + (i + 1) * ld_dst),
            _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x1));
        break;
      case 3:
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst + (i + 0) * ld_dst),
            _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x0));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst + (i + 1) * ld_dst),
            _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x1));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst + (i + 2) * ld_dst),
            _mm512_extracti32x4_epi32(u[reg_rem_idx], 0x2));
        break;
      default:
        break;
    }
  }
}

static inline void transpose_contiguous_4x16_block(
    const float* src,
    float* dst,
    int64_t ld_src,
    int nrem = 16) {
  __m512i r[4];
  // load
  if (nrem < 16) {
    __mmask16 mask_mrem_v = (((long long)1) << nrem) - 1;
    r[0] = _mm512_maskz_loadu_epi32(mask_mrem_v, src);
    r[1] = _mm512_maskz_loadu_epi32(mask_mrem_v, src + ld_src);
    r[2] = _mm512_maskz_loadu_epi32(mask_mrem_v, src + 2 * ld_src);
    r[3] = _mm512_maskz_loadu_epi32(mask_mrem_v, src + 3 * ld_src);

  } else {
    r[0] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src));
    r[1] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + ld_src));
    r[2] =
        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + 2 * ld_src));
    r[3] =
        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + 3 * ld_src));
  }
  // transpose
  // a0b0 a1b1 a4b4 a5b5 a8b8 a9b9 a12b12 a13b13
  // a2b2 a3b3 a6b6 a7b7 a10b10 a11b11 a14b14 a15b15
  // c0d0 c1d1 c4d4 c5d5 c8d8 c9d9 c12d12 c13d13
  // c2d2 c3d3 c6d6 c7d7 c10b10 c11d11 c14d14 c15d15
  __m512i t0 = _mm512_unpacklo_epi32(r[0], r[1]);
  __m512i t1 = _mm512_unpackhi_epi32(r[0], r[1]);
  __m512i t2 = _mm512_unpacklo_epi32(r[2], r[3]);
  __m512i t3 = _mm512_unpackhi_epi32(r[2], r[3]);

  r[0] = _mm512_unpacklo_epi64(t0, t2);
  r[1] = _mm512_unpackhi_epi64(t0, t2);
  r[2] = _mm512_unpacklo_epi64(t1, t3);
  r[3] = _mm512_unpackhi_epi64(t1, t3);

  t0 = _mm512_shuffle_i32x4(r[0], r[1], 0x44);
  t1 = _mm512_shuffle_i32x4(r[0], r[1], 0xee);
  t2 = _mm512_shuffle_i32x4(r[2], r[3], 0x44);
  t3 = _mm512_shuffle_i32x4(r[2], r[3], 0xee);

  r[0] = _mm512_shuffle_i32x4(t0, t2, 0x88);
  r[1] = _mm512_shuffle_i32x4(t0, t2, 0xdd);
  r[2] = _mm512_shuffle_i32x4(t1, t3, 0x88);
  r[3] = _mm512_shuffle_i32x4(t1, t3, 0xdd);
  // store
  int i = 0;
  for (; (i + 1) * 16 <= nrem * 4; i++) {
    // normal store
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i * 16), r[i]);
  }
  int erem = nrem * 4 - i * 16;
  if (erem > 0) {
    // mask store
    __mmask16 mask_rem_v = (((long long)1) << erem) - 1;
    _mm512_mask_storeu_epi32(dst + i * 16, mask_rem_v, r[i]);
  }
}

static inline void transpose_contiguous_4x32_block(
    const uint16_t* src,
    uint16_t* dst,
    int64_t ld_src,
    int nrem = 32) {
  __m512i r[4], d[4];
  // load
  if (nrem < 32) {
    __mmask32 mask_mrem_v = (((long long)1) << nrem) - 1;
    r[0] = _mm512_maskz_loadu_epi16(mask_mrem_v, src);
    r[1] = _mm512_maskz_loadu_epi16(mask_mrem_v, src + ld_src);
    r[2] = _mm512_maskz_loadu_epi16(mask_mrem_v, src + 2 * ld_src);
    r[3] = _mm512_maskz_loadu_epi16(mask_mrem_v, src + 3 * ld_src);
  } else {
    r[0] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src));
    r[1] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + ld_src));
    r[2] =
        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + 2 * ld_src));
    r[3] =
        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + 3 * ld_src));
  }
  // transpose
  d[0] = _mm512_unpacklo_epi16(r[0], r[1]);
  d[1] = _mm512_unpackhi_epi16(r[0], r[1]);
  d[2] = _mm512_unpacklo_epi16(r[2], r[3]);
  d[3] = _mm512_unpackhi_epi16(r[2], r[3]);

  r[0] = _mm512_unpacklo_epi32(d[0], d[2]);
  r[1] = _mm512_unpackhi_epi32(d[0], d[2]);
  r[2] = _mm512_unpacklo_epi32(d[1], d[3]);
  r[3] = _mm512_unpackhi_epi32(d[1], d[3]);

  d[0] = _mm512_shuffle_i32x4(r[0], r[1], 0x44);
  d[1] = _mm512_shuffle_i32x4(r[0], r[1], 0xee);
  d[2] = _mm512_shuffle_i32x4(r[2], r[3], 0x44);
  d[3] = _mm512_shuffle_i32x4(r[2], r[3], 0xee);

  r[0] = _mm512_shuffle_i32x4(d[0], d[2], 0x88);
  r[1] = _mm512_shuffle_i32x4(d[0], d[2], 0xdd);
  r[2] = _mm512_shuffle_i32x4(d[1], d[3], 0x88);
  r[3] = _mm512_shuffle_i32x4(d[1], d[3], 0xdd);
  // store
  int i = 0;
  for (; (i + 1) * 32 <= nrem * 4; i++) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i * 32), r[i]);
  }
  int erem = nrem * 4 - i * 32;
  if (erem > 0) {
    // mask store
    __mmask32 mask_rem_v = (((long long)1) << erem) - 1;
    _mm512_mask_storeu_epi16(dst + i * 32, mask_rem_v, r[i]);
  }
}

static inline void transpose_contiguous_16x4_block(
    const float* src,
    float* dst,
    int64_t ld_dst,
    int mrem = 16) {
  __m512i r[4], d[4];
  int i = 0;
  for (; (i + 1) * 16 <= mrem * 4; i++) {
    // normal load
    r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i * 16));
  }
  if (i * 16 < mrem * 4) {
    __mmask16 mask_mrem_v = (((long long)1) << (mrem * 4 - i * 16)) - 1;
    r[i] = _mm512_maskz_loadu_epi32(mask_mrem_v, src + i * 16);
  }

  // transpose
  __m512i index1 = _mm512_set_epi32(
      0x0f,
      0x0b,
      0x07,
      0x03,
      0x0e,
      0x0a,
      0x06,
      0x02,
      0x0d,
      0x09,
      0x05,
      0x01,
      0x0c,
      0x08,
      0x04,
      0x00);
  d[0] = _mm512_permutexvar_epi32(index1, r[0]);
  d[1] = _mm512_permutexvar_epi32(index1, r[1]);
  d[2] = _mm512_permutexvar_epi32(index1, r[2]);
  d[3] = _mm512_permutexvar_epi32(index1, r[3]);

  r[0] = _mm512_shuffle_i32x4(d[0], d[1], 0x44);
  r[1] = _mm512_shuffle_i32x4(d[0], d[1], 0xee);
  r[2] = _mm512_shuffle_i32x4(d[2], d[3], 0x44);
  r[3] = _mm512_shuffle_i32x4(d[2], d[3], 0xee);

  d[0] = _mm512_shuffle_i32x4(r[0], r[2], 0x88);
  d[1] = _mm512_shuffle_i32x4(r[0], r[2], 0xdd);
  d[2] = _mm512_shuffle_i32x4(r[1], r[3], 0x88);
  d[3] = _mm512_shuffle_i32x4(r[1], r[3], 0xdd);

  if (mrem < 16) {
    // mask store
    __mmask16 mask_rem_v = (((long long)1) << mrem) - 1;
    _mm512_mask_storeu_epi32(dst + 0 * ld_dst, mask_rem_v, d[0]);
    _mm512_mask_storeu_epi32(dst + 1 * ld_dst, mask_rem_v, d[1]);
    _mm512_mask_storeu_epi32(dst + 2 * ld_dst, mask_rem_v, d[2]);
    _mm512_mask_storeu_epi32(dst + 3 * ld_dst, mask_rem_v, d[3]);
  } else {
    // normal load
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 0 * ld_dst), d[0]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 1 * ld_dst), d[1]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 2 * ld_dst), d[2]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 3 * ld_dst), d[3]);
  }
}

static inline void transpose_contiguous_16x2_block(
    const float* src,
    float* dst,
    int64_t ld_dst,
    int mrem = 16) {
  __m512i r[2], d[2];
  int i = 0;
  for (; (i + 1) * 16 <= mrem * 2; i++) {
    // normal load
    r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i * 16));
  }
  if (i * 16 < mrem * 2) {
    __mmask16 mask_mrem_v = (((long long)1) << (mrem * 2 - i * 16)) - 1;
    r[i] = _mm512_maskz_loadu_epi32(mask_mrem_v, src + i * 16);
  }
  // transpose
  __m512i index1 = _mm512_set_epi32(
      0x1e,
      0x1c,
      0x1a,
      0x18,
      0x16,
      0x14,
      0x12,
      0x10,
      0x0e,
      0x0c,
      0x0a,
      0x08,
      0x06,
      0x04,
      0x02,
      0x00);
  __m512i index2 = _mm512_set_epi32(
      0x1f,
      0x1d,
      0x1b,
      0x19,
      0x17,
      0x15,
      0x13,
      0x11,
      0x0f,
      0x0d,
      0x0b,
      0x09,
      0x07,
      0x05,
      0x03,
      0x01);

  // a0--p0
  // a1--p1
  d[0] = _mm512_permutex2var_epi32(r[0], index1, r[1]);
  d[1] = _mm512_permutex2var_epi32(r[0], index2, r[1]);

  // store
  if (mrem < 16) {
    __mmask16 mask_rem_v = (((long long)1) << mrem) - 1;
    // mask store
    _mm512_mask_storeu_epi32(dst, mask_rem_v, d[0]);
    _mm512_mask_storeu_epi32(dst + ld_dst, mask_rem_v, d[1]);
  } else {
    // normal store
    _mm512_storeu_si512(dst, d[0]);
    _mm512_storeu_si512(dst + ld_dst, d[1]);
  }
}

static inline void transpose_contiguous_64x4_block(
    const uint8_t* src,
    uint8_t* dst,
    int64_t ld_dst,
    int mrem = 64) {
  __m512i r[4], d[4];
  // normal load
  int i = 0;
  for (; (i + 1) * 64 <= mrem * 4; i++) {
    r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i * 64));
  }
  int erem = mrem * 4 - i * 64;
  if (erem > 0) {
    __mmask64 mask_mrem_v = (((long long)1) << erem) - 1;
    r[i] = _mm512_maskz_loadu_epi8(mask_mrem_v, src + i * 64);
  }

  // transpose
  __m512i index = _mm512_set_epi32(
      0x0f0b0703,
      0x0e0a0602,
      0x0d090501,
      0x0c080400,
      0x0f0b0703,
      0x0e0a0602,
      0x0d090501,
      0x0c080400,
      0x0f0b0703,
      0x0e0a0602,
      0x0d090501,
      0x0c080400,
      0x0f0b0703,
      0x0e0a0602,
      0x0d090501,
      0x0c080400);

  d[0] = _mm512_shuffle_epi8(r[0], index);
  d[1] = _mm512_shuffle_epi8(r[1], index);
  d[2] = _mm512_shuffle_epi8(r[2], index);
  d[3] = _mm512_shuffle_epi8(r[3], index);

  __m512i index2 =
      _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  r[0] = _mm512_permutexvar_epi32(index2, d[0]);
  r[1] = _mm512_permutexvar_epi32(index2, d[1]);
  r[2] = _mm512_permutexvar_epi32(index2, d[2]);
  r[3] = _mm512_permutexvar_epi32(index2, d[3]);

  __m512i t0 = _mm512_shuffle_i32x4(r[0], r[1], 0x44);
  __m512i t1 = _mm512_shuffle_i32x4(r[0], r[1], 0xee);
  __m512i t2 = _mm512_shuffle_i32x4(r[2], r[3], 0x44);
  __m512i t3 = _mm512_shuffle_i32x4(r[2], r[3], 0xee);

  d[0] = _mm512_shuffle_i32x4(t0, t2, 0x88);
  d[1] = _mm512_shuffle_i32x4(t0, t2, 0xdd);
  d[2] = _mm512_shuffle_i32x4(t1, t3, 0x88);
  d[3] = _mm512_shuffle_i32x4(t1, t3, 0xdd);

  // store
  if (mrem < 64) {
    __mmask64 mask_rem_v = (((long long)1) << mrem) - 1;
    // mask store
    _mm512_mask_storeu_epi8(dst, mask_rem_v, d[0]);
    _mm512_mask_storeu_epi8(dst + ld_dst, mask_rem_v, d[1]);
    _mm512_mask_storeu_epi8(dst + 2 * ld_dst, mask_rem_v, d[2]);
    _mm512_mask_storeu_epi8(dst + 3 * ld_dst, mask_rem_v, d[3]);
  } else {
    // normal store
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 0 * ld_dst), d[0]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 1 * ld_dst), d[1]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 2 * ld_dst), d[2]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 3 * ld_dst), d[3]);
  }
}

static inline void transpose_contiguous_32x4_block(
    const uint16_t* src,
    uint16_t* dst,
    int64_t ld_dst,
    int mrem = 32) {
  __m512i r[4], d[4];
  int i = 0;
  for (; (i + 1) * 32 <= mrem * 4; i++) {
    // normal load
    r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i * 32));
  }
  if (i * 32 < mrem * 4) {
    __mmask32 mask_mrem_v = (((long long)1) << (mrem * 4 - i * 32)) - 1;
    r[i] = _mm512_maskz_loadu_epi16(mask_mrem_v, src + i * 32);
  }
  // transpose
  __m512i index = _mm512_set_epi32(
      0x001f001b,
      0x00170013,
      0x000f000b,
      0x00070003,
      0x001e001a,
      0x00160012,
      0x000e000a,
      0x00060002,
      0x001d0019,
      0x00150011,
      0x000d0009,
      0x00050001,
      0x001c0018,
      0x00140010,
      0x000c0008,
      0x00040000);

  d[0] = _mm512_permutexvar_epi16(index, r[0]);
  d[1] = _mm512_permutexvar_epi16(index, r[1]);
  d[2] = _mm512_permutexvar_epi16(index, r[2]);
  d[3] = _mm512_permutexvar_epi16(index, r[3]);

  r[0] = _mm512_shuffle_i32x4(d[0], d[1], 0x44);
  r[1] = _mm512_shuffle_i32x4(d[0], d[1], 0xee);
  r[2] = _mm512_shuffle_i32x4(d[2], d[3], 0x44);
  r[3] = _mm512_shuffle_i32x4(d[2], d[3], 0xee);

  d[0] = _mm512_shuffle_i32x4(r[0], r[2], 0x88);
  d[1] = _mm512_shuffle_i32x4(r[0], r[2], 0xdd);
  d[2] = _mm512_shuffle_i32x4(r[1], r[3], 0x88);
  d[3] = _mm512_shuffle_i32x4(r[1], r[3], 0xdd);

  if (mrem < 32) {
    // mask store
    __mmask32 mask_rem_v = (((long long)1) << mrem) - 1;
    _mm512_mask_storeu_epi16(dst + 0 * ld_dst, mask_rem_v, d[0]);
    _mm512_mask_storeu_epi16(dst + ld_dst, mask_rem_v, d[1]);
    _mm512_mask_storeu_epi16(dst + 2 * ld_dst, mask_rem_v, d[2]);
    _mm512_mask_storeu_epi16(dst + 3 * ld_dst, mask_rem_v, d[3]);
  } else {
    // normal load
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 0 * ld_dst), d[0]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 1 * ld_dst), d[1]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 2 * ld_dst), d[2]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 3 * ld_dst), d[3]);
  }
}

static inline void transpose_contiguous_2x16_block(
    const float* src,
    float* dst,
    int64_t ld_src,
    int nrem = 16) {
  __m512i r0, r1;
  // load
  if (nrem < 16) {
    __mmask16 mask_mrem_v = (((long long)1) << nrem) - 1;
    r0 = _mm512_maskz_loadu_epi32(mask_mrem_v, src);
    r1 = _mm512_maskz_loadu_epi32(mask_mrem_v, src + ld_src);
  } else {
    r0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src));
    r1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + ld_src));
  }
  // transpose
  __m512i index1 = _mm512_set_epi32(
      0x0017,
      0x0007,
      0x0016,
      0x0006,
      0x0015,
      0x0005,
      0x0014,
      0x0004,
      0x0013,
      0x0003,
      0x0012,
      0x0002,
      0x0011,
      0x0001,
      0x0010,
      0x0000);
  __m512i index2 = _mm512_set_epi32(
      0x001f,
      0x000f,
      0x001e,
      0x000e,
      0x001d,
      0x000d,
      0x001c,
      0x000c,
      0x001b,
      0x000b,
      0x001a,
      0x000a,
      0x0019,
      0x0009,
      0x0018,
      0x0008);
  // a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 a6 b6 a7 b7
  // a8 b8 a9 b9 a10 b10 a11 b11 a12 b12 a13 b13 a14 b14 a15 b15
  __m512i u0 = _mm512_permutex2var_epi32(r0, index1, r1);
  __m512i u1 = _mm512_permutex2var_epi32(r0, index2, r1);
  // store
  if (nrem < 16) {
    // mask store
    if (nrem < 8) {
      __mmask16 mask_rem_v = (((long long)1) << (nrem * 2)) - 1;
      _mm512_mask_storeu_epi32(dst, mask_rem_v, u0);
    } else {
      __mmask16 mask_rem_v = (((long long)1) << ((nrem - 8) * 2)) - 1;
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), u0);
      _mm512_mask_storeu_epi32(dst + 16, mask_rem_v, u1);
    }
  } else {
    // normal store
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), u0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 16), u1);
  }
}

static inline void transpose_contiguous_64x2_block(
    const uint8_t* src,
    uint8_t* dst,
    int64_t ld_dst,
    int mrem = 64) {
  __m512i r[2], d[2];
  // normal load
  int i = 0;
  for (; (i + 1) * 64 <= mrem * 2; i++) {
    r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i * 64));
  }
  int erem = mrem * 2 - i * 64;
  if (erem > 0) {
    __mmask64 mask_mrem_v = (((long long)1) << erem) - 1;
    r[i] = _mm512_maskz_loadu_epi8(mask_mrem_v, src + i * 64);
  }

  // transpose
  __m512i index1 = _mm512_set_epi32(
      0x0f0d0b09,
      0x07050301,
      0x0e0c0a08,
      0x06040200,
      0x0f0d0b09,
      0x07050301,
      0x0e0c0a08,
      0x06040200,
      0x0f0d0b09,
      0x07050301,
      0x0e0c0a08,
      0x06040200,
      0x0f0d0b09,
      0x07050301,
      0x0e0c0a08,
      0x06040200);
  r[0] = _mm512_shuffle_epi8(r[0], index1);
  r[1] = _mm512_shuffle_epi8(r[1], index1);

  __m512i index2 = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  __m512i index3 = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

  d[0] = _mm512_permutex2var_epi64(r[0], index2, r[1]);
  d[1] = _mm512_permutex2var_epi64(r[0], index3, r[1]);

  // store
  if (mrem < 64) {
    __mmask64 mask_rem_v = (((long long)1) << mrem) - 1;
    // mask store
    _mm512_mask_storeu_epi8(dst, mask_rem_v, d[0]);
    _mm512_mask_storeu_epi8(dst + ld_dst, mask_rem_v, d[1]);
  } else {
    // normal store
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d[0]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + ld_dst), d[1]);
  }
}

static inline void transpose_contiguous_4x64_block(
    const uint8_t* src,
    uint8_t* dst,
    int64_t ld_src,
    int nrem = 64) {
  __m512i r[4], d[4];
  // load
  if (nrem < 64) {
    __mmask64 mask_mrem_v = (((long long)1) << nrem) - 1;
    r[0] = _mm512_maskz_loadu_epi8(mask_mrem_v, src);
    r[1] = _mm512_maskz_loadu_epi8(mask_mrem_v, src + ld_src);
    r[2] = _mm512_maskz_loadu_epi8(mask_mrem_v, src + 2 * ld_src);
    r[3] = _mm512_maskz_loadu_epi8(mask_mrem_v, src + 3 * ld_src);
  } else {
    r[0] = _mm512_loadu_si512(reinterpret_cast<const __m256i*>(src));
    r[1] = _mm512_loadu_si512(reinterpret_cast<const __m256i*>(src + ld_src));
    r[2] =
        _mm512_loadu_si512(reinterpret_cast<const __m256i*>(src + 2 * ld_src));
    r[3] =
        _mm512_loadu_si512(reinterpret_cast<const __m256i*>(src + 3 * ld_src));
  }
  // transpose
  d[0] = _mm512_unpacklo_epi32(r[0], r[1]);
  d[1] = _mm512_unpackhi_epi32(r[0], r[1]);
  d[2] = _mm512_unpacklo_epi32(r[2], r[3]);
  d[3] = _mm512_unpackhi_epi32(r[2], r[3]);

  r[0] = _mm512_unpacklo_epi64(d[0], d[2]);
  r[1] = _mm512_unpackhi_epi64(d[0], d[2]);
  r[2] = _mm512_unpacklo_epi64(d[1], d[3]);
  r[3] = _mm512_unpackhi_epi64(d[1], d[3]);

  d[0] = _mm512_shuffle_i32x4(r[0], r[1], 0x44);
  d[1] = _mm512_shuffle_i32x4(r[0], r[1], 0xee);
  d[2] = _mm512_shuffle_i32x4(r[2], r[3], 0x44);
  d[3] = _mm512_shuffle_i32x4(r[2], r[3], 0xee);

  r[0] = _mm512_shuffle_i32x4(d[0], d[2], 0x88);
  r[1] = _mm512_shuffle_i32x4(d[0], d[2], 0xdd);
  r[2] = _mm512_shuffle_i32x4(d[1], d[3], 0x88);
  r[3] = _mm512_shuffle_i32x4(d[1], d[3], 0xdd);

  __m512i index = _mm512_set_epi32(
      0x0f0b0703,
      0x0e0a0602,
      0x0d090501,
      0x0c080400,
      0x0f0b0703,
      0x0e0a0602,
      0x0d090501,
      0x0c080400,
      0x0f0b0703,
      0x0e0a0602,
      0x0d090501,
      0x0c080400,
      0x0f0b0703,
      0x0e0a0602,
      0x0d090501,
      0x0c080400);

  d[0] = _mm512_shuffle_epi8(r[0], index);
  d[1] = _mm512_shuffle_epi8(r[1], index);
  d[2] = _mm512_shuffle_epi8(r[2], index);
  d[3] = _mm512_shuffle_epi8(r[3], index);

  // store
  int i = 0;
  for (; (i + 1) * 64 <= nrem * 4; i++) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i * 64), d[i]);
  }
  int erem = nrem * 4 - i * 64;
  if (erem > 0) {
    __mmask64 mask_rem_v = (((long long)1) << erem) - 1;
    _mm512_mask_storeu_epi8(dst + i * 64, mask_rem_v, d[i]);
  }
}

static inline void transpose_contiguous_2x64_block(
    const uint8_t* src,
    uint8_t* dst,
    int64_t ld_src,
    int nrem = 64) {
  __m512i r[2];
  __m512i d[2];
  // load
  if (nrem < 64) {
    __mmask64 mask_mrem_v = (((long long)1) << nrem) - 1;
    r[0] = _mm512_maskz_loadu_epi8(mask_mrem_v, src);
    r[1] = _mm512_maskz_loadu_epi8(mask_mrem_v, src + ld_src);
  } else {
    r[0] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src));
    r[1] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + ld_src));
  }
  // transpose
  // _mm512_mask_blend_epi8(0xaaaaaaaaaaaaaaaa, r0, r1);
  d[0] = _mm512_unpacklo_epi16(r[0], r[1]);
  d[1] = _mm512_unpackhi_epi16(r[0], r[1]);
  __m512i index1 = _mm512_set_epi32(
      0x0f0d0e0c,
      0x0b090a08,
      0x07050604,
      0x03010200,
      0x0f0d0e0c,
      0x0b090a08,
      0x07050604,
      0x03010200,
      0x0f0d0e0c,
      0x0b090a08,
      0x07050604,
      0x03010200,
      0x0f0d0e0c,
      0x0b090a08,
      0x07050604,
      0x03010200);
  r[0] = _mm512_shuffle_epi8(d[0], index1);
  r[1] = _mm512_shuffle_epi8(d[1], index1);
  __m512i index2 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
  __m512i index3 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
  // a0b0 a1b1 ... a31b31
  // a32b32 ... a63b63
  d[0] = _mm512_permutex2var_epi64(r[0], index2, r[1]);
  d[1] = _mm512_permutex2var_epi64(r[0], index3, r[1]);

  int i = 0;
  for (; (i + 1) * 64 <= nrem * 2; i++) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i * 64), d[i]);
  }
  int erem = nrem * 2 - i * 64;
  if (erem > 0) {
    __mmask64 mask_rem_v = (((long long)1) << erem) - 1;
    _mm512_mask_storeu_epi8(dst + i * 64, mask_rem_v, d[i]);
  }
}

static inline void transpose_contiguous_2x32_block(
    const uint16_t* src,
    uint16_t* dst,
    int64_t ld_src,
    int nrem = 32) {
  __m512i r0, r1;
  __m512i d0, d1;
  // load
  if (nrem < 32) {
    __mmask32 mask_mrem_v = (((long long)1) << nrem) - 1;
    r0 = _mm512_maskz_loadu_epi16(mask_mrem_v, src);
    r1 = _mm512_maskz_loadu_epi16(mask_mrem_v, src + ld_src);
  } else {
    r0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src));
    r1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + ld_src));
  }
  // transpose
  d0 = _mm512_unpacklo_epi16(r0, r1);
  d1 = _mm512_unpackhi_epi16(r0, r1);
  r0 = _mm512_shuffle_i32x4(d0, d1, 0x88);
  r1 = _mm512_shuffle_i32x4(d0, d1, 0xdd);
  d0 = _mm512_shuffle_i32x4(r0, r1, 0x88);
  d1 = _mm512_shuffle_i32x4(r0, r1, 0xdd);

  // store
  if (nrem < 16) {
    __mmask32 mask_rem_v = (((long long)1) << (nrem * 2)) - 1;
    _mm512_mask_storeu_epi16(dst, mask_rem_v, d0);
  } else if (nrem == 16) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d0);
  } else if (nrem < 32) {
    __mmask32 mask_rem_v = (((long long)1) << (nrem * 2 - 32)) - 1;
    _mm512_mask_storeu_epi16(dst, mask_rem_v, d0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d0);
    _mm512_mask_storeu_epi16(
        reinterpret_cast<__m512i*>(dst + 32), mask_rem_v, d1);
  } else {
    // normal store
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 32), d1);
  }
}

static inline void transpose_contiguous_32x2_block(
    const uint16_t* src,
    uint16_t* dst,
    int64_t ld_dst,
    int mrem = 32) {
  __m512i r[2], d[2];
  // load
  int i = 0;
  for (; (i + 1) * 32 <= mrem * 2; i++) {
    r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i * 32));
  }
  int erem = mrem * 2 - i * 32;
  if (erem > 0) {
    __mmask32 mask_mrem_v = (((long long)1) << erem) - 1;
    r[i] = _mm512_maskz_loadu_epi16(mask_mrem_v, src + i * 32);
  }
  // transpose
  __m512i index = _mm512_set_epi32(
      0x001f001d,
      0x001b0019,
      0x00170015,
      0x00130011,
      0x000f000d,
      0x000b0009,
      0x00070005,
      0x00030001,
      0x001e001c,
      0x001a0018,
      0x00160014,
      0x00120010,
      0x000e000c,
      0x000a0008,
      0x00060004,
      0x00020000);
  d[0] = _mm512_permutexvar_epi16(index, r[0]);
  d[1] = _mm512_permutexvar_epi16(index, r[1]);
  r[0] = _mm512_shuffle_i32x4(d[0], d[1], 0x44);
  r[1] = _mm512_shuffle_i32x4(d[0], d[1], 0xee);

  // store
  if (mrem < 32) {
    __mmask32 mask_rem_v = (((long long)1) << mrem) - 1;
    // mask store
    _mm512_mask_storeu_epi16(dst, mask_rem_v, r[0]);
    _mm512_mask_storeu_epi16(dst + ld_dst, mask_rem_v, r[1]);
  } else {
    // normal store
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), r[0]);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + ld_dst), r[1]);
  }
}

template <bool MREM = false, bool NREM = false>
void transpose_16x16_block(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst,
    int mrem = 16,
    int nrem = 16) {
  __m512i r[8];
  if (MREM || NREM) {
    load_with_remainders_i16(src, ld_src, r, mrem, nrem);
  } else {
    __m256i t00 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 0 * ld_src));
    __m256i t01 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 1 * ld_src));
    __m256i t02 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * ld_src));
    __m256i t03 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 3 * ld_src));
    __m256i t04 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 4 * ld_src));
    __m256i t05 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 5 * ld_src));
    __m256i t06 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 6 * ld_src));
    __m256i t07 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 7 * ld_src));
    __m256i t08 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 8 * ld_src));
    __m256i t09 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 9 * ld_src));
    __m256i t10 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 10 * ld_src));
    __m256i t11 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 11 * ld_src));
    __m256i t12 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 12 * ld_src));
    __m256i t13 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 13 * ld_src));
    __m256i t14 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 14 * ld_src));
    __m256i t15 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 15 * ld_src));

    // a0a1 a2a3 a4a5 a6a7 a8a9 a10a11 a12a13 a14a15
    // e0e1 e2e3 e4e5 e6e7 e8e9 e10e11 e12e13 e14e15
    r[0] = _mm512_inserti64x4(_mm512_castsi256_si512(t00), t04, 0x01);
    // b0-b15
    // f0-f15
    r[1] = _mm512_inserti64x4(_mm512_castsi256_si512(t01), t05, 0x01);
    // c0-c15
    // g0-g15
    r[2] = _mm512_inserti64x4(_mm512_castsi256_si512(t02), t06, 0x01);
    // d0-d15
    // g0-h15
    r[3] = _mm512_inserti64x4(_mm512_castsi256_si512(t03), t07, 0x01);
    // i0-i15
    // m0-m15
    r[4] = _mm512_inserti64x4(_mm512_castsi256_si512(t08), t12, 0x01);
    // j0-j15
    // n0-n15
    r[5] = _mm512_inserti64x4(_mm512_castsi256_si512(t09), t13, 0x01);
    // k0-k15
    // o0-o15
    r[6] = _mm512_inserti64x4(_mm512_castsi256_si512(t10), t14, 0x01);
    // l0-l15
    // p0-p15
    r[7] = _mm512_inserti64x4(_mm512_castsi256_si512(t11), t15, 0x01);
  }
  __m512i u[8];
  core_transpose_16x16_block(r, u);
  if (MREM || NREM) {
    store_with_remainders_i16(dst, ld_dst, u, mrem, nrem);
  } else {
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 0 * ld_dst),
        _mm512_extracti32x8_epi32(u[0], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 1 * ld_dst),
        _mm512_extracti32x8_epi32(u[0], 0x01));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 2 * ld_dst),
        _mm512_extracti32x8_epi32(u[1], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 3 * ld_dst),
        _mm512_extracti32x8_epi32(u[1], 0x01));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 4 * ld_dst),
        _mm512_extracti32x8_epi32(u[2], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 5 * ld_dst),
        _mm512_extracti32x8_epi32(u[2], 0x01));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 6 * ld_dst),
        _mm512_extracti32x8_epi32(u[3], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 7 * ld_dst),
        _mm512_extracti32x8_epi32(u[3], 0x01));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 8 * ld_dst),
        _mm512_extracti32x8_epi32(u[4], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 9 * ld_dst),
        _mm512_extracti32x8_epi32(u[4], 0x01));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 10 * ld_dst),
        _mm512_extracti32x8_epi32(u[5], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 11 * ld_dst),
        _mm512_extracti32x8_epi32(u[5], 0x01));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 12 * ld_dst),
        _mm512_extracti32x8_epi32(u[6], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 13 * ld_dst),
        _mm512_extracti32x8_epi32(u[6], 0x01));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 14 * ld_dst),
        _mm512_extracti32x8_epi32(u[7], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + 15 * ld_dst),
        _mm512_extracti32x8_epi32(u[7], 0x01));
  }
}

template <bool MREM = false, bool NREM = false>
void transpose_16x32_block(
    const uint8_t* src,
    int64_t ld_src,
    uint8_t* dst,
    int64_t ld_dst,
    int mrem = 16,
    int nrem = 32) {
  // Treat the numbers in a row as 4-Byte integers.
  // Thus 03_04 is is 4-byte element in 03 row and 04 column
  //
  // 00_00 00_01 00_02 00_03 00_04 00_05 00_06 00_07
  // 01_00 01_01 01_02 01_03 01_04 01_05 01_06 01_07
  // 02_00 02_01 02_02 02_03 02_04 02_05 02_06 02_07
  // 03_00 03_01 03_02 03_03 03_04 03_05 03_06 03_07
  // 04_00 04_01 04_02 04_03 04_04 04_05 04_06 04_07
  // 05_00 05_01 05_02 05_03 05_04 05_05 05_06 05_07
  // 06_00 06_01 06_02 06_03 06_04 06_05 06_06 06_07
  // 07_00 07_01 07_02 07_03 07_04 07_05 07_06 07_07
  // 08_00 08_01 08_02 08_03 08_04 08_05 08_06 08_07
  // 09_00 09_01 09_02 09_03 09_04 09_05 09_06 09_07
  // 10_00 10_01 10_02 10_03 10_04 10_05 10_06 10_07
  // 11_00 11_01 11_02 11_03 11_04 11_05 11_06 11_07
  // 12_00 12_01 12_02 12_03 12_04 12_05 12_06 12_07
  // 13_00 13_01 13_02 13_03 13_04 13_05 13_06 13_07
  // 14_00 14_01 14_02 14_03 14_04 14_05 14_06 14_07
  // 15_00 15_01 15_02 15_03 15_04 15_05 15_06 15_07

  __m512i r[8];
  if (MREM || NREM) {
    load_with_remainders_i8(src, ld_src, r, mrem, nrem);
  } else {
    __m256i t00 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 0 * ld_src));
    __m256i t04 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 4 * ld_src));
    // 00_00 00_01 00_02 00_03 00_04 00_05 00_06 00_07 04_00 04_01 04_02 04_03
    // 04_04 04_05 04_06 04_07
    r[0] = _mm512_inserti64x4(_mm512_castsi256_si512(t00), t04, 0x01);

    __m256i t01 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 1 * ld_src));
    __m256i t05 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 5 * ld_src));
    // 01_00 01_01 01_02 01_03 01_04 01_05 01_06 01_07 05_00 05_01 05_02 05_03
    // 05_04 05_05 05_06 05_07
    r[1] = _mm512_inserti64x4(_mm512_castsi256_si512(t01), t05, 0x01);

    __m256i t02 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * ld_src));
    __m256i t06 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 6 * ld_src));
    // 02_00 02_01 02_02 02_03 02_04 02_05 02_06 02_07 06_00 06_01 06_02 06_03
    // 06_04 06_05 06_06 06_07
    r[2] = _mm512_inserti64x4(_mm512_castsi256_si512(t02), t06, 0x01);

    __m256i t03 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 3 * ld_src));
    __m256i t07 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 7 * ld_src));
    // 03_00 03_01 03_02 03_03 03_04 03_05 03_06 03_07 07_00 07_01 07_02 07_03
    // 07_04 07_05 07_06 07_07
    r[3] = _mm512_inserti64x4(_mm512_castsi256_si512(t03), t07, 0x01);

    __m256i t08 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 8 * ld_src));
    __m256i t12 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 12 * ld_src));
    // 08_00 08_01 08_02 08_03 08_04 08_05 08_06 08_07 12_00 12_01 12_02 12_03
    // 12_04 12_05 12_06 12_07
    r[4] = _mm512_inserti64x4(_mm512_castsi256_si512(t08), t12, 0x01);

    __m256i t09 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 9 * ld_src));
    __m256i t13 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 13 * ld_src));
    // 09_00 09_01 09_02 09_03 09_04 09_05 09_06 09_07 13_00 13_01 13_02 13_03
    // 13_04 13_05 13_06 13_07
    r[5] = _mm512_inserti64x4(_mm512_castsi256_si512(t09), t13, 0x01);

    __m256i t10 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 10 * ld_src));
    __m256i t14 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 14 * ld_src));
    // 10_00 10_01 10_02 10_03 10_04 10_05 10_06 10_07 14_00 14_01 14_02 14_03
    // 14_04 14_05 14_06 14_07
    r[6] = _mm512_inserti64x4(_mm512_castsi256_si512(t10), t14, 0x01);

    __m256i t11 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 11 * ld_src));
    __m256i t15 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 15 * ld_src));
    // 11_00 11_01 11_02 11_03 11_04 11_05 11_06 11_07 15_00 15_01 15_02 15_03
    // 15_04 15_05 15_06 15_07
    r[7] = _mm512_inserti64x4(_mm512_castsi256_si512(t11), t15, 0x01);
  }

  __m512i u[8];
  core_transpose_16x32_block_i8(r, u);

  if (MREM || NREM) {
    store_with_remainders_i8(dst, ld_dst, u, mrem, nrem);
  } else {
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 0 * ld_dst),
        _mm512_extracti32x4_epi32(u[0], 0x0));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 1 * ld_dst),
        _mm512_extracti32x4_epi32(u[0], 0x1));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 2 * ld_dst),
        _mm512_extracti32x4_epi32(u[0], 0x2));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 3 * ld_dst),
        _mm512_extracti32x4_epi32(u[0], 0x3));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 16 * ld_dst),
        _mm512_extracti32x4_epi32(u[1], 0x0));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 17 * ld_dst),
        _mm512_extracti32x4_epi32(u[1], 0x1));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 18 * ld_dst),
        _mm512_extracti32x4_epi32(u[1], 0x2));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 19 * ld_dst),
        _mm512_extracti32x4_epi32(u[1], 0x3));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 4 * ld_dst),
        _mm512_extracti32x4_epi32(u[2], 0x0));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 5 * ld_dst),
        _mm512_extracti32x4_epi32(u[2], 0x1));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 6 * ld_dst),
        _mm512_extracti32x4_epi32(u[2], 0x2));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 7 * ld_dst),
        _mm512_extracti32x4_epi32(u[2], 0x3));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 20 * ld_dst),
        _mm512_extracti32x4_epi32(u[3], 0x0));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 21 * ld_dst),
        _mm512_extracti32x4_epi32(u[3], 0x1));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 22 * ld_dst),
        _mm512_extracti32x4_epi32(u[3], 0x2));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 23 * ld_dst),
        _mm512_extracti32x4_epi32(u[3], 0x3));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 8 * ld_dst),
        _mm512_extracti32x4_epi32(u[4], 0x0));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 9 * ld_dst),
        _mm512_extracti32x4_epi32(u[4], 0x1));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 10 * ld_dst),
        _mm512_extracti32x4_epi32(u[4], 0x2));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 11 * ld_dst),
        _mm512_extracti32x4_epi32(u[4], 0x3));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 24 * ld_dst),
        _mm512_extracti32x4_epi32(u[5], 0x0));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 25 * ld_dst),
        _mm512_extracti32x4_epi32(u[5], 0x1));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 26 * ld_dst),
        _mm512_extracti32x4_epi32(u[5], 0x2));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 27 * ld_dst),
        _mm512_extracti32x4_epi32(u[5], 0x3));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 12 * ld_dst),
        _mm512_extracti32x4_epi32(u[6], 0x0));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 13 * ld_dst),
        _mm512_extracti32x4_epi32(u[6], 0x1));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 14 * ld_dst),
        _mm512_extracti32x4_epi32(u[6], 0x2));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 15 * ld_dst),
        _mm512_extracti32x4_epi32(u[6], 0x3));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 28 * ld_dst),
        _mm512_extracti32x4_epi32(u[7], 0x0));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 29 * ld_dst),
        _mm512_extracti32x4_epi32(u[7], 0x1));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 30 * ld_dst),
        _mm512_extracti32x4_epi32(u[7], 0x2));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + 31 * ld_dst),
        _mm512_extracti32x4_epi32(u[7], 0x3));
  }
}

template <>
void transpose_avx512_contiguous_thin(
    int64_t M,
    int64_t N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  if (N == 2) {
    int64_t i = 0;
    for (; i < M / 16 * 16; i += 16) {
      transpose_contiguous_16x2_block(src + i * ld_src, dst + i, ld_dst);
    }
    int mrem = M - i;
    if (mrem > 0) {
      transpose_contiguous_16x2_block(src + i * ld_src, dst + i, ld_dst, mrem);
    }
  } else if (N == 4) {
    int64_t i = 0;
    for (; i < M / 16 * 16; i += 16) {
      transpose_contiguous_16x4_block(src + i * ld_src, dst + i, ld_dst);
    }
    int mrem = M - i;
    if (mrem > 0) {
      transpose_contiguous_16x4_block(src + i * ld_src, dst + i, ld_dst, mrem);
    }
  }
}

template <>
void transpose_avx512_contiguous_thin(
    int64_t M,
    int64_t N,
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  if (N == 2) {
    int64_t i = 0;
    for (; i < M / 32 * 32; i += 32) {
      transpose_contiguous_32x2_block(src + i * ld_src, dst + i, ld_dst);
    }
    int mrem = M - i;
    if (mrem > 0) {
      transpose_contiguous_32x2_block(src + i * ld_src, dst + i, ld_dst, mrem);
    }
  } else if (N == 4) {
    int64_t i = 0;
    for (; i < M / 32 * 32; i += 32) {
      transpose_contiguous_32x4_block(src + i * ld_src, dst + i, ld_dst);
    }
    int mrem = M - i;
    if (mrem > 0) {
      transpose_contiguous_32x4_block(src + i * ld_src, dst + i, ld_dst, mrem);
    }
  }
}

template <>
void transpose_avx512_contiguous_thin(
    int64_t M,
    int64_t N,
    const uint8_t* src,
    int64_t ld_src,
    uint8_t* dst,
    int64_t ld_dst) {
  if (N == 2) {
    int64_t i = 0;
    for (; i < M / 64 * 64; i += 64) {
      transpose_contiguous_64x2_block(src + i * ld_src, dst + i, ld_dst);
    }
    int mrem = M - i;
    if (mrem > 0) {
      transpose_contiguous_64x2_block(src + i * ld_src, dst + i, ld_dst, mrem);
    }
  } else if (N == 4) {
    int64_t i = 0;
    for (; i < M / 64 * 64; i += 64) {
      transpose_contiguous_64x4_block(src + i * ld_src, dst + i, ld_dst);
    }
    int mrem = M - i;
    if (mrem > 0) {
      transpose_contiguous_64x4_block(src + i * ld_src, dst + i, ld_dst, mrem);
    }
  }
}

template <>
void transpose_avx512_contiguous_wide(
    int64_t M,
    int64_t N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  if (M == 2) {
    int64_t i = 0;
    for (; i < N / 16 * 16; i += 16) {
      transpose_contiguous_2x16_block(src + i, dst + i * ld_dst, ld_src);
    }
    int nrem = N - i;
    if (nrem > 0) {
      transpose_contiguous_2x16_block(src + i, dst + i * ld_dst, ld_src, nrem);
    }
  } else if (M == 4) {
    int64_t i = 0;
    for (; i < N / 16 * 16; i += 16) {
      transpose_contiguous_4x16_block(src + i, dst + i * ld_dst, ld_src);
    }
    int nrem = N - i;
    if (nrem > 0) {
      transpose_contiguous_4x16_block(src + i, dst + i * ld_dst, ld_src, nrem);
    }
  }
}

template <>
void transpose_avx512_contiguous_wide(
    int64_t M,
    int64_t N,
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  if (M == 2) {
    int64_t i = 0;
    for (; i < N / 32 * 32; i += 32) {
      transpose_contiguous_2x32_block(src + i, dst + i * ld_dst, ld_src);
    }
    int nrem = N - i;
    if (nrem > 0) {
      transpose_contiguous_2x32_block(src + i, dst + i * ld_dst, ld_src, nrem);
    }
  } else if (M == 4) {
    int64_t i = 0;
    for (; i < N / 32 * 32; i += 32) {
      transpose_contiguous_4x32_block(src + i, dst + i * ld_dst, ld_src);
    }
    int nrem = N - i;
    if (nrem > 0) {
      transpose_contiguous_4x32_block(src + i, dst + i * ld_dst, ld_src, nrem);
    }
  }
}

template <>
void transpose_avx512_contiguous_wide(
    int64_t M,
    int64_t N,
    const uint8_t* src,
    int64_t ld_src,
    uint8_t* dst,
    int64_t ld_dst) {
  if (M == 2) {
    int64_t i = 0;
    for (; i < N / 64 * 64; i += 64) {
      transpose_contiguous_2x64_block(src + i, dst + i * ld_dst, ld_src);
    }
    int nrem = N - i;
    if (nrem > 0) {
      transpose_contiguous_2x64_block(src + i, dst + i * ld_dst, ld_src, nrem);
    }
  } else if (M == 4) {
    int64_t i = 0;
    for (; i < N / 64 * 64; i += 64) {
      transpose_contiguous_4x64_block(src + i, dst + i * ld_dst, ld_src);
    }
    int nrem = N - i;
    if (nrem > 0) {
      transpose_contiguous_4x64_block(src + i, dst + i * ld_dst, ld_src, nrem);
    }
  }
}

template <>
void transpose_avx512(
    int64_t M,
    int64_t N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  if (M == ld_dst && (M == 2 || M == 4)) {
    transpose_avx512_contiguous_wide(M, N, src, ld_src, dst, ld_dst);
  } else if (N == ld_src && (N == 2 || N == 4)) {
    transpose_avx512_contiguous_thin(M, N, src, ld_src, dst, ld_dst);
  } else {
    int64_t ib = 0, jb = 0;
    if (N % 16 > 0 && N % 16 < 4) {
      // If the remainder has n < 4 columns, we use the SSE kernel for the
      // remainder because it requires 4 * (2 * 4 + 2 * N) = 32 + 8N
      // instructions instead of 4 * 16 + 2 * N = 64 + 2N instructions needed in
      // the masked AVX512 kernel.
      for (ib = 0; ib + 16 <= M; ib += 16) {
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_16x16_avx512(
              &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
        }
        for (int64_t i = ib; i < ib + 16; i += 4) {
          transpose_kernel_mxn_sse<4>(
              N - jb,
              &src[i * ld_src + jb],
              ld_src,
              &dst[i + jb * ld_dst],
              ld_dst);
        }
      }
    } else if (N % 16 == 4) {
      // If the remainder has 4 columns, we use the SSE kernel for the remainder
      // because it requires 4 * 16 = 64 instructions instead of 4 * 16 + 2 * 4
      // = 72 instructions needed in the masked AVX512 kernel.
      for (ib = 0; ib + 16 <= M; ib += 16) {
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_16x16_avx512(
              &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
        }
        for (int64_t i = ib; i < ib + 16; i += 4) {
          transpose_kernel_4x4_sse(
              &src[i * ld_src + jb], ld_src, &dst[i + jb * ld_dst], ld_dst);
        }
      }
    } else if (N % 16 == 8) {
      // If the remainder has 8 columns, we use the AVX kenrel for the remainder
      // because it requires 2 * 40 = 80 instructions instead of 4 * 16 + 2 * 8
      // = 80 instructions + looping overhead in the masked AVX512 kernel.
      for (ib = 0; ib + 16 <= M; ib += 16) {
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_16x16_avx512(
              &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
        }
        for (int64_t i = ib; i < ib + 16; i += 8) {
          transpose_kernel_8x8_avx2(
              &src[i * ld_src + jb], ld_src, &dst[i + jb * ld_dst], ld_dst);
        }
      }
    } else {
      for (ib = 0; ib + 16 <= M; ib += 16) {
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_16x16_avx512(
              &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx512<16>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
      }
    }

    // Specialization for small M - ib cases so that the compiler can inline
    // transpose_kernel_mxn_avx512 and unroll the loops whose iteration count
    // depends on by M - ib .
    // Specialization for m helps more than for n in transpose_kernel_mxn_avx512
    // because we have more loops in that function whose iteration count depends
    // on m.
    switch (M - ib) {
      case 1:
        for (int64_t j = 0; j < N; ++j) {
          dst[ib + j * ld_dst] = src[ib * ld_src + j];
        }
        break;
      case 2:
        for (jb = 0; jb + 4 <= N; jb += 4) {
          transpose_kernel_mxn_sse<2>(
              4,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_sse<2>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 3:
        for (jb = 0; jb + 4 <= N; jb += 4) {
          transpose_kernel_mxn_sse<3>(
              4,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_sse<3>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 4:
        for (jb = 0; jb + 4 <= N; jb += 4) {
          transpose_kernel_4x4_sse(
              &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_sse<4>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 5:
        for (jb = 0; jb + 8 <= N; jb += 8) {
          transpose_kernel_mxn_avx2<5>(
              8,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx2<5>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 6:
        for (jb = 0; jb + 8 <= N; jb += 8) {
          transpose_kernel_mxn_avx2<6>(
              8,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx2<6>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 7:
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_mxn_avx512<7>(
              16,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx512<7>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 8:
        for (jb = 0; jb + 8 <= N; jb += 8) {
          transpose_kernel_8x8_avx2(
              &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx2<8>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 9:
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_mxn_avx512<9>(
              16,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx512<9>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 10:
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_mxn_avx512<10>(
              16,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx512<10>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 11:
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_mxn_avx512<11>(
              16,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx512<11>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 12:
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_mxn_avx512<12>(
              16,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx512<12>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 13:
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_mxn_avx512<13>(
              16,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx512<13>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 14:
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_mxn_avx512<14>(
              16,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx512<14>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
      case 15:
        for (jb = 0; jb + 16 <= N; jb += 16) {
          transpose_kernel_mxn_avx512<15>(
              16,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        if (jb < N) {
          transpose_kernel_mxn_avx512<15>(
              N - jb,
              &src[ib * ld_src + jb],
              ld_src,
              &dst[ib + jb * ld_dst],
              ld_dst);
        }
        break;
    }
  }
}

template <>
void transpose_avx512(
    int64_t M,
    int64_t N,
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  if (M == ld_dst && (M == 2 || M == 4)) {
    transpose_avx512_contiguous_wide(M, N, src, ld_src, dst, ld_dst);
  } else if (N == ld_src && (N == 2 || N == 4)) {
    transpose_avx512_contiguous_thin(M, N, src, ld_src, dst, ld_dst);
  } else {
    int64_t i = 0;
    for (; i < M / 16 * 16; i += 16) {
      int64_t j = 0;
      for (; j < N / 16 * 16; j += 16) {
        transpose_16x16_block<false, false>(
            src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst);
      }
      // handle j rem
      int nrem = N - j;
      if (nrem > 0) {
        transpose_16x16_block<false, true>(
            src + i * ld_src + j,
            ld_src,
            dst + j * ld_dst + i,
            ld_dst,
            16,
            nrem);
      }
    }
    // handle i rem
    int mrem = M - i;
    if (mrem > 0) {
      int j = 0;
      for (; j < N / 16 * 16; j += 16) {
        transpose_16x16_block<true, false>(
            src + i * ld_src + j,
            ld_src,
            dst + j * ld_dst + i,
            ld_dst,
            mrem,
            16);
      }
      // handle j rem
      int nrem = N - j;
      transpose_16x16_block<true, true>(
          src + i * ld_src + j,
          ld_src,
          dst + j * ld_dst + i,
          ld_dst,
          mrem,
          nrem);
    }
  }
}

template <>
void transpose_avx512(
    int64_t M,
    int64_t N,
    const uint8_t* src,
    int64_t ld_src,
    uint8_t* dst,
    int64_t ld_dst) {
  if (M == ld_dst && (M == 2 || M == 4)) {
    transpose_avx512_contiguous_wide(M, N, src, ld_src, dst, ld_dst);
  } else if (N == ld_src && (N == 2 || N == 4)) {
    transpose_avx512_contiguous_thin(M, N, src, ld_src, dst, ld_dst);
  } else {
    int64_t i = 0;
    for (; i < M / 16 * 16; i += 16) {
      int64_t j = 0;
      for (; j < N / 32 * 32; j += 32) {
        transpose_16x32_block<false, false>(
            src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst);
      }
      // handle j rem
      int nrem = N - j;
      if (nrem > 0) {
        transpose_16x32_block<false, true>(
            src + i * ld_src + j,
            ld_src,
            dst + j * ld_dst + i,
            ld_dst,
            16,
            nrem);
      }
    }

    // handle i rem
    int mrem = M - i;
    if (mrem > 0) {
      int64_t j = 0;
      for (; j < N / 32 * 32; j += 32) {
        transpose_16x32_block<true, false>(
            src + i * ld_src + j,
            ld_src,
            dst + j * ld_dst + i,
            ld_dst,
            mrem,
            32);
      }
      // handle j rem
      int nrem = N - j;
      transpose_16x32_block<true, true>(
          src + i * ld_src + j,
          ld_src,
          dst + j * ld_dst + i,
          ld_dst,
          mrem,
          nrem);
    }
  }
}

} // namespace internal

} // namespace fbgemm
