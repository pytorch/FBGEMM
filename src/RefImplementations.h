/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm>
#include <cstdint>

#include "fbgemm/ConvUtils.h"
#include "fbgemm/FbgemmI8Spmdm.h"

namespace fbgemm {

/**
 * @brief Reference implementation of requantization step.
 * int32 multiplier
 * @params bias can be nullptr
 */
FBGEMM_API void requantize_u8acc32_ref(
    int M,
    int N,
    int ld,
    const std::int32_t* inp,
    std::uint8_t* out,
    std::int32_t C_multiplier,
    std::int32_t C_right_shift,
    std::int32_t C_zero_point,
    std::int32_t A_zero_point,
    std::int32_t B_zero_point,
    const std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    bool fuse_relu = false);

/**
 * @brief Reference implementation of requantization step.
 * float multiplier
 * @params bias can be nullptr
 * @params ncols_per_quant_group the number of columns share the same
 *         quantization parameter.
 *         ncols_per_quant_group == N : per-tensor quantization
 *         ncols_per_quant_group == N / groups : per-group quantization
 *         ncols_per_quant_group == 1 : per-channel quantization
 */
FBGEMM_API void requantize_u8acc32_ref(
    int M,
    int N,
    int ld,
    const std::int32_t* inp,
    std::uint8_t* out,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t A_zero_point,
    const std::int32_t* B_zero_point,
    const std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    int ncols_per_quant_group,
    bool fuse_relu = false);

/**
 * @brief Reference implementation of matrix multiply with uint8 for A,
 * int8 for B, and 32-bit accumulation.
 */
FBGEMM_API void matmul_u8i8acc32_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const std::uint8_t* Aint8,
    const std::int8_t* Bint8,
    std::int32_t* Cint32);

/**
 * @brief Reference implementation of matrix multiply with uint 8 for A,
 * int8 for B, and 16-bit accumulation.
 */
void FBGEMM_API matmul_u8i8acc16_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int brow,
    const std::uint8_t* Aint8,
    const std::int8_t* Bint8,
    std::int32_t* Cint32);

/**
 * @brief Reference implementation of matrix multiply with fp32 (single
 * precision) floating point number.
 */
void FBGEMM_API matmul_fp_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const float* Afp32,
    const float* Bfp32,
    float* Cfp32);

/**
 * @brief Reference implementation of cblas_sgemm in MKL/BLAS.
 */
void FBGEMM_API cblas_sgemm_ref(
    const matrix_op_t transa,
    const matrix_op_t transb,
    const int m,
    const int n,
    const int k,
    float alpha,
    const float* Afp32,
    int lda,
    const float* Bfp32,
    int ldb,
    float beta,
    float* Cfp32,
    int ldc
    );

/**
 * @brief Reference implementation to compute row_offsets (sums of rows of A).
 */
FBGEMM_API void row_offsets_u8acc32_ref(
    int M,
    int K,
    int ld,
    const std::uint8_t* Aint8,
    std::int32_t* row_offsets);

/**
 * @brief Reference implementation to compute adjusted col_offsets (sum of
 * columns of B and adjusted with B_zero_point)
 *
 * @params ncols_per_quant_group see ncols_per_quant_group in
 *         requantize_u8acc32_ref
 */
FBGEMM_API void col_offsets_with_zero_pt_s8acc32_ref(
    int K,
    int N,
    int ld,
    const std::int8_t* Bint8,
    const std::int32_t* B_zero_point,
    std::int32_t* col_offsets,
    int ncols_per_quant_group);

/**
 * @brief Reference implementation of SPMDM (sparse matrix times dense matrix).
 *
 * @param groups when > 1, for gth group, we multiply
 *               A[:,g*(A.ncols/groups):(g+1)*(A.ncols/groups)] sub-matrix with
 *               B[:,g*(B.ncols/groups):(g+1)*(B.ncols/groups)] sub-matrix .
 */
FBGEMM_API void spmdm_ref(
    int M,
    const std::uint8_t* A,
    int lda,
    CompressedSparseColumn& B,
    bool accumulation,
    std::int32_t* C,
    int ldc,
    int groups = 1);

/*
 * @brief Trim a 32-bit integer to a 16-bit integer.
 */
int32_t clip_16bit(int32_t x);

/*
 * @brief Reference implementation of convolution operation.
 * The activations A are assumed to be in NHiWiC format.
 * The filters B are assumed to be in RSCK format.
 * The output C is assumed to be in NHoWoC format.
 */
template <int SPATIAL_DIM = 2>
FBGEMM_API void conv_ref(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    const std::int8_t* B,
    std::int32_t* C);

/*
 * @brief Transforms weights from  G K/G (R S C/G) to G (R S C/G) K/G format.
 */
template <int SPATIAL_DIM = 2>
FBGEMM_API void transposeConvWeights(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest);

/*
 * @brief Reference implementation of im2col operation.
 *
 * For 2D:
 * The input A is assumed to be in NHiWiC format.
 * The output A is assumed to be in NHoWoRSC format.
 *
 * For 3D:
 * The input A is assumed to be in NTiHiWiC format.
 * The output A is assumed to be in NToHoWoK0K1K2C format.
 */
template <int SPATIAL_DIM = 2>
FBGEMM_API void im2col_ref(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    std::uint8_t* Ao);

} // namespace fbgemm
