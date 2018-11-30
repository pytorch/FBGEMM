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
  // smat in RSG layout
  PackedDepthWiseConvMatrix(int K, const std::int8_t* smat);
  virtual ~PackedDepthWiseConvMatrix();

  const std::int8_t* PackedMat() const {
    return pmat_;
  }

 private:
  int K_;
  std::int8_t* pmat_;
}; // Packed3x3ConvMatrix

using Packed3x3ConvMatrix = PackedDepthWiseConvMatrix<3 * 3>;
using Packed3x3x3ConvMatrix = PackedDepthWiseConvMatrix<3 * 3 * 3>;

/**
 * Depth-wise 3x3 convolution with pad=1 and stride=1 and K a multiple of 8
 * @params A The input image in NHWK layout
 * @params Bp The pre-packed filter
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
    const Packed3x3ConvMatrix& Bp,
    std::int32_t* C,
    int thread_id = 0,
    int num_threads = 1);

/**
 * Depth-wise 3x3 convolution with pad=1 and stride=1 and K a multiple of 8
 * This version is fused with requantization.
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
    int thread_id = 0,
    int num_threads = 1,
    bool fuse_relu = false);

/**
 * Depth-wise 3x3 convolution with pad=1 and stride=1 and K a multiple of 8
 * This version is fused with requantization and uses per-channel quantization.
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
    int thread_id = 0,
    int num_threads = 1);

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
    const Packed3x3x3ConvMatrix& Bp,
    std::int32_t* C,
    int thread_id = 0,
    int num_threads = 1);

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

} // namespace fbgemm
