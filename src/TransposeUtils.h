/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "fbgemm/FbgemmBuild.h"

namespace fbgemm {

/**
 * @brief Reference implementation of matrix transposition: B = A^T.
 * @param M The height of the matrix.
 * @param N The width of the matrix.
 * @param src The memory buffer of the source matrix A.
 * @param ld_src The leading dimension of the source matrix A.
 * @param dst The memory buffer of the destination matrix B.
 * @param ld_dst The leading dimension of the destination matrix B.
 */
FBGEMM_API void transpose_ref(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);

namespace internal {

/**
 * @brief Transpose a matrix using Intel AVX2.
 *
 * This is called if the code is running on a CPU with Intel AVX2 support.
 */
void transpose_avx2(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);

/**
 * @brief Transpose a matrix using Intel AVX512.
 *
 * This is called if the code is running on a CPU with Intel AVX512 support.
 */
void transpose_avx512(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);

} // namespace internal

} // namespace fbgemm
