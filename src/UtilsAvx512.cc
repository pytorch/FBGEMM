/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>
#include "./TransposeUtils.h"
#include "./TransposeUtilsAvx2.h"

namespace fbgemm {

namespace {

// 16 * 6 = 96 instructions
inline void transpose_kernel_16x16_avx512(
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
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
    int ld_src,
    float* dst,
    int ld_dst) {
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

void transpose_avx512(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  int ib = 0, jb = 0;
  if (N % 16 > 0 && N % 16 < 4) {
    // If the remainder has n < 4 columns, we use the SSE kernel for the
    // remainder because it requires 4 * (2 * 4 + 2 * N) = 32 + 8N instructions
    // instead of 4 * 16 + 2 * N = 64 + 2N instructions needed in the masked
    // AVX512 kernel.
    for (ib = 0; ib + 16 <= M; ib += 16) {
      for (jb = 0; jb + 16 <= N; jb += 16) {
        transpose_kernel_16x16_avx512(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      for (int i = ib; i < ib + 16; i += 4) {
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
    // because it requires 4 * 16 = 64 instructions instead of 4 * 16 + 2 * 4 =
    // 72 instructions needed in the masked AVX512 kernel.
    for (ib = 0; ib + 16 <= M; ib += 16) {
      for (jb = 0; jb + 16 <= N; jb += 16) {
        transpose_kernel_16x16_avx512(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      for (int i = ib; i < ib + 16; i += 4) {
        transpose_kernel_4x4_sse(
            &src[i * ld_src + jb], ld_src, &dst[i + jb * ld_dst], ld_dst);
      }
    }
  } else if (N % 16 == 8) {
    // If the remainder has 8 columns, we use the AVX kenrel for the remainder
    // because it requires 2 * 40 = 80 instructions instead of 4 * 16 + 2 * 8 =
    // 80 instructions + looping overhead in the masked AVX512 kernel.
    for (ib = 0; ib + 16 <= M; ib += 16) {
      for (jb = 0; jb + 16 <= N; jb += 16) {
        transpose_kernel_16x16_avx512(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      for (int i = ib; i < ib + 16; i += 8) {
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
      for (int j = 0; j < N; ++j) {
        dst[ib + j * ld_dst] = src[ib * ld_src + j];
      }
      break;
    case 2:
      for (jb = 0; jb + 4 <= N; jb += 4) {
        transpose_kernel_mxn_sse<2>(
            4, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            4, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            16, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            16, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            16, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            16, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            16, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            16, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            16, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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
            16, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
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

} // namespace internal

} // namespace fbgemm
