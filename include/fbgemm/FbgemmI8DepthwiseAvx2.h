/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>
#include "fbgemm/FbgemmBuild.h"

namespace fbgemm {

// KERNEL_PROD is the product of all kernels.
// For example, KERNEL_PROD = 9 for 3x3, and 27 for 3x3x3.
template <int KERNEL_PROD>
class FBGEMM_API PackedDepthWiseConvMatrix {
 public:
  // smat in GRS layout
  PackedDepthWiseConvMatrix(int K, const std::int8_t* smat);
  virtual ~PackedDepthWiseConvMatrix();

  const std::int8_t* PackedMat() const {
    return pmat_;
  }

  /**
   * @brief Unpacks pmat_ into unpack_data.
   * Used for recovering the weight matrix into the original format
   */
  void unpack(std::int8_t* unpacked_data);

  /**
   * @brief returns the index into pmat_ given the row and column for smat
   */
  int addr(int r, int c);

 private:
  int K_;
  std::int8_t* pmat_;
}; // Packed3x3ConvMatrix

using Packed3x3ConvMatrix = PackedDepthWiseConvMatrix<3 * 3>;
using Packed3x3x3ConvMatrix = PackedDepthWiseConvMatrix<3 * 3 * 3>;
using Packed1ConvMatrix = PackedDepthWiseConvMatrix<1>;
using Packed2ConvMatrix = PackedDepthWiseConvMatrix<2>;
using Packed3ConvMatrix = PackedDepthWiseConvMatrix<3>;
using Packed4ConvMatrix = PackedDepthWiseConvMatrix<4>;
using Packed5ConvMatrix = PackedDepthWiseConvMatrix<5>;
using Packed10ConvMatrix = PackedDepthWiseConvMatrix<10>;
using Packed11ConvMatrix = PackedDepthWiseConvMatrix<11>;

/**
 * Depth-wise 3x3 convolution with pad=1 and K a multiple of 8, fused with
 * requantization.
 *
 * @col_offsets nullptr if col_offsets are folded into bias
 */
FBGEMM_API void depthwise_3x3_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    std::int32_t B_zero_point,
    const Packed3x3ConvMatrix& Bp,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    bool fuse_relu = false,
    int thread_id = 0,
    int num_threads = 1);

/**
 * Depth-wise 3x3 convolution with pad=1 and K a multiple of 8, fused with
 * requantization, and using per-channel quantization.
 *
 * @col_offsets nullptr if col_offsets are folded into bias
 */
FBGEMM_API void depthwise_3x3_per_channel_quantization_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const Packed3x3ConvMatrix& Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    bool fuse_relu = false,
    int thread_id = 0,
    int num_threads = 1);

/**
 * @col_offsets nullptr if col_offsets are folded into bias
 */
FBGEMM_API void depthwise_3x3x3_pad_1(
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
    const Packed3x3x3ConvMatrix& Bp,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    bool fuse_relu = false,
    int thread_id = 0,
    int num_threads = 1);

/**
 * @col_offsets nullptr if col_offsets are folded into bias
 */
FBGEMM_API void depthwise_3x3x3_per_channel_quantization_pad_1(
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
    const std::int32_t* B_zero_point,
    const Packed3x3x3ConvMatrix& Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    bool fuse_relu = false,
    int thread_id = 0,
    int num_threads = 1);

} // namespace fbgemm
