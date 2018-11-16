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
void requantize_u8acc32_ref(
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
 */
void requantize_u8acc32_ref(
    int M,
    int N,
    int ld,
    const std::int32_t* inp,
    std::uint8_t* out,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t A_zero_point,
    std::int32_t B_zero_point,
    const std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    bool fuse_relu = false);

/**
 * @brief Reference implementation of matrix multiply with uint8 for A,
 * int8 for B, and 32-bit accumulation.
 */
void matmul_u8i8acc32_ref(
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
void matmul_u8i8acc16_ref(
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
void matmul_fp_ref(
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
 * @brief Reference implementation to compute row_offsets (sums of rows of A).
 */
void row_offsets_u8acc32_ref(
    int M,
    int K,
    int ld,
    const std::uint8_t* Aint8,
    std::int32_t* row_offsets);

/**
 * @brief Reference implementation to compute adjusted col_offsets (sum of
 * columns of B and adjusted with B_zero_point)
 */
void col_offsets_with_zero_pt_s8acc32_ref(
    int K,
    int N,
    int ld,
    const std::int8_t* Bint8,
    std::int32_t B_zero_point,
    std::int32_t* col_offsets);

/**
 * @brief Reference implementation of SPMDM (sparse matrix times dense matrix).
 *
 * @param groups when > 1, for gth group, we multiply
 *               A[:,g*(A.ncols/groups):(g+1)*(A.ncols/groups)] sub-matrix with
 *               B[:,g*(B.ncols/groups):(g+1)*(B.ncols/groups)] sub-matrix .
 */
void spmdm_ref(
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
void conv_ref(
    const conv_param_t<>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    const std::int8_t* B,
    std::int32_t* C);

void conv3d_ref(
    const conv_param_t<3>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    const std::int8_t* B,
    std::int32_t* C);

/*
 * @brief Reference implementation of im2col operation.
 * The input A is assumed to be in NHiWiC format.
 * The output A is assumed to be in NHoWoRSC format.
 */
void im2col_ref(
    const conv_param_t<>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    std::uint8_t* Ao);

/*
 * @brief Reference implementation of im2col 3D operation.
 * The input A is assumed to be in NTiHiWiC format.
 * The output A is assumed to be in NToHoWoK0K1K2C format.
 */
void im2col3d_ref(
    const conv_param_t<3>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    std::uint8_t* Ao);

/*
 * @brief Reference implementation of depthwise convolution with a 3x3 filter
 * and padding size 1.
 */
void depthwise_3x3_pad_1_ref(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int8_t* B,
    std::int32_t* C);

/*
 * @brief Reference implementation of depthwise convolution with a 3x3 filter
 * and padding size 1, followed by requantization. (the same scaling factors and
 * zero points for each channel).
 */
void depthwise_3x3_pad_1_ref(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    std::int32_t B_zero_point,
    const std::int8_t* B,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const std::int32_t* bias);

/*
 * @brief Reference implementation of depthwise convolution with a 3x3 filter
 * and padding size 1, followed by requantization. (different scaling factors
 * and zero points for each channel).
 */
void depthwise_3x3_per_channel_quantization_pad_1_ref(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const std::int8_t* B,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const std::int32_t* bias);

/*
 * @brief Reference implementation of 3D depthwise convolution with a 3x3x3
 * filter and padding size 1.
 */
void depthwise_3x3x3_pad_1_ref(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int8_t* B,
    std::int32_t* C);

/*
 * @brief Reference implementation of 3D depthwise convolution with a 3x3x3
 * filter and padding size 1, followed by requantization.
 */
void depthwise_3x3x3_pad_1_ref(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    std::int32_t B_zero_point,
    const std::int8_t* B,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const std::int32_t* bias);

} // namespace fbgemm
