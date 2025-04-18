/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "fbgemm/FbgemmBuild.h"

#include <cstdint>

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
template <typename T>
FBGEMM_API void transpose_ref(
    int64_t M,
    int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);

namespace internal {

/**
 * @brief Transpose a matrix using Intel AVX2.
 *
 * This is called if the code is running on a CPU with Intel AVX2 support.
 */
template <typename T>
void transpose_avx2(
    int64_t M,
    int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);

/**
 * @brief Transpose a matrix using Intel AVX512.
 *
 * This is called if the code is running on a CPU with Intel AVX512 support.
 */
template <typename T>
void transpose_avx512(
    int64_t M,
    int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);

#ifdef __aarch64__
/**
 * @brief Transpose a matrix using SVE.
 *
 * This is called if the code is running on a CPU with SVE support.
 */
template <typename T>
void transpose_sve(
    int64_t M,
    int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);

/**
 * @brief Transpose a matrix using NEON.
 *
 * This is called if the code is running on a CPU with NEON support.
 */
template <typename T>
void transpose_neon(
    int64_t M,
    int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);
#endif // __aarch64__

} // namespace internal

} // namespace fbgemm
