/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "./TransposeUtils.h"
#include <cstring>
#include "fbgemm/Utils.h"

namespace fbgemm {

template <typename T>
void transpose_ref(
    int64_t M,
    int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst) {
  for (int64_t j = 0; j < N; j++) {
    for (int64_t i = 0; i < M; i++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  } // for each output row
}

template <typename T>
void transpose_simd(
    int64_t M,
    int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst) {
  if (M == 0 || N == 0) {
    return;
  }
  if ((M == 1 && ld_dst == 1) || (N == 1 && ld_src == 1)) {
    if (dst != src) {
      // sizeof must be first operand force dims promotion to OS-bitness type
      memcpy(dst, src, sizeof(T) * M * N);
    }
    return;
  }

#ifdef __aarch64__
  if constexpr (std::is_same<T, float>::value) {
    internal::transpose_neon<T>(M, N, src, ld_src, dst, ld_dst);
  } else {
    transpose_ref<T>(M, N, src, ld_src, dst, ld_dst);
  }
#else
  static const auto iset = fbgemmInstructionSet();
  // Run time CPU detection
  if (isZmm(iset)) {
    internal::transpose_avx512<T>(M, N, src, ld_src, dst, ld_dst);
  } else if (isYmm(iset)) {
    internal::transpose_avx2<T>(M, N, src, ld_src, dst, ld_dst);
  } else {
    transpose_ref<T>(M, N, src, ld_src, dst, ld_dst);
  }

#endif
}

template void transpose_ref<float>(
    int64_t M,
    int64_t N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst);

template void transpose_ref<uint16_t>(
    int64_t M,
    int64_t N,
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst);

template void transpose_ref<uint8_t>(
    int64_t M,
    int64_t N,
    const uint8_t* src,
    int64_t ld_src,
    uint8_t* dst,
    int64_t ld_dst);

template FBGEMM_API void transpose_simd<float>(
    int64_t M,
    int64_t N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst);

template FBGEMM_API void transpose_simd<uint8_t>(
    int64_t M,
    int64_t N,
    const uint8_t* src,
    int64_t ld_src,
    uint8_t* dst,
    int64_t ld_dst);

template FBGEMM_API void transpose_simd<uint16_t>(
    int64_t M,
    int64_t N,
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst);
} // namespace fbgemm
