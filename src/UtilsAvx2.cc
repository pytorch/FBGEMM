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

namespace internal {

template <>
void transpose_avx2(
    int64_t M,
    int64_t N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  int64_t ib = 0, jb = 0;
  if (N % 8 > 0 && N % 8 < 4) {
    // If the remainder has n < 4 columns, we use the SSE kernel for the
    // remainder because it requires 2 * (2 * 4 + 2 * N) = 16 + 4N instructions
    // instead of 3 * 8 + 2 * N = 24 + 2N instructions in the masked AVX2
    // kernel.
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_8x8_avx2(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      for (int64_t i = ib; i < ib + 8; i += 4) {
        transpose_kernel_mxn_sse<4>(
            N - jb,
            &src[i * ld_src + jb],
            ld_src,
            &dst[i + jb * ld_dst],
            ld_dst);
      }
    }
  } else if (N % 8 == 4) {
    // If the remainder has 4 columns, we use the SSE kernel for the remainder
    // because it requires 2 * 16 = 32 instructions instead of 3 * 8 + 2 * 4 =
    // 32 instructions + looping overhead needed in the masked AVX2 kernel.
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_8x8_avx2(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      for (int64_t i = ib; i < ib + 8; i += 4) {
        transpose_kernel_4x4_sse(
            &src[i * ld_src + jb], ld_src, &dst[i + jb * ld_dst], ld_dst);
      }
    }
  } else {
    for (ib = 0; ib + 8 <= M; ib += 8) {
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
    }
  }

  // Specialization for small M - ib cases so that the compiler can inline
  // transpose_kernel_mxn_avx2 and unroll the loops whose iteration count
  // depends on by M - ib .
  // Specialization for m helps more than for n in transpose_kernel_mxn_avx2
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
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_mxn_avx2<7>(
            8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_avx2<7>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
  }
}

template <>
void transpose_avx2(
    int64_t M,
    int64_t N,
    const uint8_t* src,
    int64_t ld_src,
    uint8_t* dst,
    int64_t ld_dst) {
  int64_t ib = 0, jb = 0;
  if (M >= 8) {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_8x32_avx2(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }

      if (jb < N) {
        transpose_kernel_mxn_avx2_uint8<8>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
    }
  }

  // Specialization for small M - ib cases
  switch (M - ib) {
    case 1:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_avx2_uint8<1>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }

      if (jb < N)
        transpose_kernel_mxn_avx2_uint8<1>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);

      break;
    case 2:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_avx2_uint8<2>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_avx2_uint8<2>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 3:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_avx2_uint8<3>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_avx2_uint8<3>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 4:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_avx2_uint8<4>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_avx2_uint8<4>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 5:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_avx2_uint8<5>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_avx2_uint8<5>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 6:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_avx2_uint8<6>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_avx2_uint8<6>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 7:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_avx2_uint8<7>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_avx2_uint8<7>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
  }
}

template <>
void transpose_avx2(
    int64_t M,
    int64_t N,
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  int64_t i = 0;
  for (; i < M / 8 * 8; i += 8) {
    int64_t j = 0;
    for (; j < N / 16 * 16; j += 16) {
      transpose_kernel_8x16_avx2<false, false>(
          src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst);
    }
    // handle j rem
    unsigned nrem = N - j;
    if (nrem > 0) {
      transpose_kernel_8x16_avx2<false, true>(
          src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst, 8, nrem);
    }
  }

  // handle i rem
  unsigned mrem = M - i;
  if (mrem > 0) {
    int64_t j = 0;
    for (; j < N / 16 * 16; j += 16) {
      transpose_kernel_8x16_avx2<true, false>(
          src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst, mrem, 16);
    }
    // handle j rem
    unsigned nrem = N - j;
    transpose_kernel_8x16_avx2<true, true>(
        src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst, mrem, nrem);
  }
}

} // namespace internal

} // namespace fbgemm
