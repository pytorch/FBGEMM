/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "./TransposeUtils.h"
#include "fbgemm/Utils.h"
#include <cstring>

namespace fbgemm {

void transpose_ref(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  } // for each output row
}

void transpose_simd(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  if ((M == 1 && ld_dst == 1) || (N == 1 && ld_src == 1)) {
    if (dst != src) {
      memcpy(dst, src, M * N * sizeof(float));
    }
    return;
  }
  static const auto iset = fbgemmInstructionSet();
  // Run time CPU detection
  if (isZmm(iset)) {
    internal::transpose_avx512(M, N, src, ld_src, dst, ld_dst);
  } else if (isYmm(iset)) {
    internal::transpose_avx2(M, N, src, ld_src, dst, ld_dst);
  } else {
    transpose_ref(M, N, src, ld_src, dst, ld_dst);
  }
}

} // namespace fbgemm
