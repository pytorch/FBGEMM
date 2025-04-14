/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliate
 * <open-source-office@arm.com> SPDX-License-Identifier: BSD-3-Clause
 */

#ifdef __aarch64__

#include "./TransposeUtils.h"
#include "./TransposeUtilsNeon.h"

namespace fbgemm {

namespace internal {

static inline void transpose_kernel_mx1(
    const float* src,
    int64_t ld_src,
    float* dst,
    const int64_t M) {
  for (int64_t i = 0; i < M; ++i) {
    dst[i] = src[i * ld_src];
  }
}

template <>
void transpose_neon(
    int64_t M,
    int64_t N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  int64_t jb = 0;
  while (jb + 7 < M) {
    int64_t ib = 0;
    while (ib + 7 < N) {
      transpose_kernel_8x8_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 8;
    }
    while (ib + 3 < N) {
      transpose_kernel_8x4_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 4;
    }
    while (ib + 1 < N) {
      transpose_kernel_8x2_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 2;
    }
    if (ib < N) {
      transpose_kernel_mx1(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], 8);
    }
    jb += 8;
  }
  while (jb + 3 < M) {
    int64_t ib = 0;
    while (ib + 7 < N) {
      transpose_kernel_4x8_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 8;
    }
    while (ib + 3 < N) {
      transpose_kernel_4x4_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 4;
    }
    while (ib + 1 < N) {
      transpose_kernel_4x2_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 2;
    }
    if (ib < N) {
      transpose_kernel_mx1(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], 4);
    }
    jb += 4;
  }
  while (jb + 1 < M) {
    int64_t ib = 0;
    while (ib + 7 < N) {
      transpose_kernel_2x8_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 8;
    }
    while (ib + 3 < N) {
      transpose_kernel_2x4_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 4;
    }
    while (ib + 1 < N) {
      transpose_kernel_2x2_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 2;
    }
    if (ib < N) {
      transpose_kernel_mx1(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], 2);
    }
    jb += 2;
  }
  if (jb < M) {
    for (int64_t ib = 0; ib < N; ++ib) {
      dst[jb + ib * ld_dst] = src[ib + jb * ld_src];
    }
  }
}

template <>
void transpose_neon(int64_t M, int64_t N, const __fp16 *src,
                    int64_t ld_src, __fp16 *dst, int64_t ld_dst) {
  int64_t ib = 0, jb = 0;
  if (N % 8 > 0 && N % 8 < 4) {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_8x8_neon(&src[ib * ld_src + jb], ld_src,
                                  &dst[ib + jb * ld_dst], ld_dst);
      }
      for (int64_t i = ib; i < ib + 8; i += 4) {
        transpose_kernel_mxn_neon_64<4>(N - jb, &src[i * ld_src + jb], ld_src,
                                         &dst[i + jb * ld_dst], ld_dst);
      }
    }
  } else if (N % 8 == 4) {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_8x8_neon(&src[ib * ld_src + jb], ld_src,
                                  &dst[ib + jb * ld_dst], ld_dst);
      }
      for (int64_t i = ib; i < ib + 8; i += 4) {
        transpose_kernel_4x4_neon(&src[i * ld_src + jb], ld_src,
                                  &dst[i + jb * ld_dst], ld_dst);
      }
    }
  } else {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_8x8_neon(&src[ib * ld_src + jb], ld_src,
                                  &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_128<8>(N - jb, &src[ib * ld_src + jb], ld_src,
                                         &dst[ib + jb * ld_dst], ld_dst);
      }
    }
  }
  switch (M - ib) {
  case 1:
    for (int64_t j = 0; j < N; ++j) {
      dst[ib + j * ld_dst] = src[ib * ld_src + j];
    }
    break;
  case 2:
    for (jb = 0; jb + 4 <= N; jb += 4) {
      transpose_kernel_mxn_neon_64<2>(4, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    if (jb < N) {
      transpose_kernel_mxn_neon_64<2>(N - jb, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    break;
  case 3:
    for (jb = 0; jb + 4 <= N; jb += 4) {
      transpose_kernel_mxn_neon_64<3>(4, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    if (jb < N) {
      transpose_kernel_mxn_neon_64<3>(N - jb, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    break;
  case 4:
    for (jb = 0; jb + 4 <= N; jb += 4) {
      transpose_kernel_4x4_neon(&src[ib * ld_src + jb], ld_src,
                                &dst[ib + jb * ld_dst], ld_dst);
    }
    if (jb < N) {
      transpose_kernel_mxn_neon_64<4>(N - jb, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    break;
  case 5:
    for (jb = 0; jb + 8 <= N; jb += 8) {
      transpose_kernel_mxn_neon_128<5>(8, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    if (jb < N) {
      transpose_kernel_mxn_neon_128<5>(N - jb, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    break;
  case 6:
    for (jb = 0; jb + 8 <= N; jb += 8) {
      transpose_kernel_mxn_neon_128<6>(8, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    if (jb < N) {
      transpose_kernel_mxn_neon_128<6>(N - jb, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    break;
  case 7:
    for (jb = 0; jb + 8 <= N; jb += 8) {
      transpose_kernel_mxn_neon_128<7>(8, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    if (jb < N) {
      transpose_kernel_mxn_neon_128<7>(N - jb, &src[ib * ld_src + jb], ld_src,
                                       &dst[ib + jb * ld_dst], ld_dst);
    }
    break;
  }
}

} // namespace internal

} // namespace fbgemm

#endif // #ifdef __aarch64__
