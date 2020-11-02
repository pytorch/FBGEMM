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

template <typename T>
void transpose_ref(int M, int N, const T* src, int ld_src, T* dst, int ld_dst) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  } // for each output row
}

template <typename T>
void transpose_simd(
    int M,
    int N,
    const T* src,
    int ld_src,
    T* dst,
    int ld_dst) {
  if ((M == 1 && ld_dst == 1) || (N == 1 && ld_src == 1)) {
    if (dst != src) {
      // sizeof must be first operand force dims promotion to OS-bitness type
      memcpy(dst, src, sizeof(T) * M * N);
    }
    return;
  }
  static const auto iset = fbgemmInstructionSet();
  // Run time CPU detection
  if (isZmm(iset)) {
    internal::transpose_avx512<T>(M, N, src, ld_src, dst, ld_dst);
  } else if (isYmm(iset)) {
    internal::transpose_avx2<T>(M, N, src, ld_src, dst, ld_dst);
  } else {
    transpose_ref<T>(M, N, src, ld_src, dst, ld_dst);
  }
}

template void transpose_ref<float>(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);

template void transpose_ref<uint8_t>(
    int M,
    int N,
    const uint8_t* src,
    int ld_src,
    uint8_t* dst,
    int ld_dst);

template FBGEMM_API void transpose_simd<float>(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);

template FBGEMM_API void transpose_simd<uint8_t>(
    int M,
    int N,
    const uint8_t* src,
    int ld_src,
    uint8_t* dst,
    int ld_dst);

} // namespace fbgemm
