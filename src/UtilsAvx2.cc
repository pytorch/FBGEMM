/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <immintrin.h>
#include <iostream>
#include "./TransposeUtils.h"
#include "./TransposeUtilsAvx2.h"

namespace fbgemm {

namespace internal {

template <>
void transpose_avx2(
    unsigned M,
    unsigned N,
    const float* src,
    unsigned ld_src,
    float* dst,
    unsigned ld_dst) {
  unsigned ib = 0, jb = 0;
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
      for (unsigned i = ib; i < ib + 8; i += 4) {
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
      for (unsigned i = ib; i < ib + 8; i += 4) {
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
      for (unsigned j = 0; j < N; ++j) {
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
/*
template <>
void transpose_avx2(
    unsigned M,
    unsigned N,
    const uint8_t* src,
    unsigned ld_src,
    uint8_t* dst,
    unsigned ld_dst) {
  for (unsigned j = 0; j < N; j++) {
    for (unsigned i = 0; i < M; i++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  } // for each output row
}
*/

void print_matrix(uint8_t* matrix, int rows, int cols, unsigned ld_src) {
  int i, j, k;
  // int8_t* v = (int8_t*)matrix;
  uint8_t* v = matrix;
  i = 0;
  /* Print the matrix , i = 0, 1, 2, 3,..., 255                          */
  /* rows and cols only controls the positions of the new lines printf("\n") */
  for (k = 0; k < rows; k++) {
    for (j = 0; j < cols; j++) {
      std::cout << static_cast<int32_t>(v[k * ld_src + j]) << " ";
      i = i + 1;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <>
void transpose_avx2(
    unsigned M,
    unsigned N,
    const uint8_t* src,
    unsigned ld_src,
    uint8_t* dst,
    unsigned ld_dst) {
  std::cout << "inside transpose avx2 " << M << " " << N << " " << ld_src << " "
            << ld_dst << std::endl;
  std::cout << "before transpose " << std::endl;
  print_matrix((uint8_t*)src, M, N, ld_src);
  unsigned ib = 0, jb = 0;
  if (M >= 8) {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_8x32_avx2(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      // transpose_kernel_8x32_avx2((src), ld_src, (dst), ld_dst);
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
  // transpose_kernel_mxn_avx2_uint8<7>(N, src, ld_src, dst, ld_dst);
  print_matrix(dst, N, M, ld_dst);
}
} // namespace internal

} // namespace fbgemm
