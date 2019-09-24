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

class FBGEMM_API PackedDepthWiseConvMatrix {
 public:
  /**
   * @params K the number of channels (same as the number of groups because
   *           depth-wise convolution has one input/output channel per group)
   * @params kernel_prod the product of all kernels. For example, kernel_prod =
   *                     9 for 3x3 conv, and 27 for 3x3x3 conv.
   * @param smat the source unpacked weight in GRS layout
   */
  PackedDepthWiseConvMatrix(int K, int kernel_prod, const std::int8_t* smat);
  virtual ~PackedDepthWiseConvMatrix();

  const std::int8_t* PackedMat() const {
    return pmat_;
  }

  int GetKernelProduct() const {
    return kernel_prod_;
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
  const int K_; /**< the number of channels */
  const int kernel_prod_; /** the product of all kernel dims */
  std::int8_t* pmat_; /** packed weight */
}; // PackedDepthWiseConvMatrix

class FBGEMM_API Packed3x3ConvMatrix : public PackedDepthWiseConvMatrix {
 public:
  Packed3x3ConvMatrix(int K, const std::int8_t* smat)
      : PackedDepthWiseConvMatrix(K, 3 * 3, smat) {}
};

class FBGEMM_API Packed3x3x3ConvMatrix : public PackedDepthWiseConvMatrix {
 public:
  Packed3x3x3ConvMatrix(int K, const std::int8_t* smat)
      : PackedDepthWiseConvMatrix(K, 3 * 3 * 3, smat) {}
};

/** To be removed. Keeping it just to make sure we don't change C2 files and
 * fbgemm files in a single diff
 *
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
    const PackedDepthWiseConvMatrix& Bp,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    bool fuse_relu = false,
    int thread_id = 0,
    int num_threads = 1);

/**
 * Depth-wise 3x3 convolution with pad=1 and stride=1 and K a multiple of 8
 * This version is fused with requantization.
 *
 * @col_offsets nullptr if col_offsets are folded into bias
 * @act_times_w_scale Only used if BIAS_TYPE is float, i.e., bias is
 *                    unquantized.
 */
template <typename BIAS_TYPE = std::int32_t>
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
    const PackedDepthWiseConvMatrix& Bp,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    float act_times_w_scale = 1.0f,
    int thread_id = 0,
    int num_threads = 1);

/**
 * Depth-wise 3x3 convolution with pad=1 and K a multiple of 8, fused with
 * requantization, and using per-channel quantization.
 *
 * @col_offsets nullptr if col_offsets are folded into bias
 */
template <typename BIAS_TYPE = std::int32_t>
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
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    const float* act_times_w_scale = nullptr,
    int thread_id = 0,
    int num_threads = 1);

/** To be removed. Keeping it just to make sure we don't change C2 files and
 * fbgemm files in a single diff
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
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const std::int32_t* bias,
    bool fuse_relu = false,
    int thread_id = 0,
    int num_threads = 1);

/** To be removed. Keeping it just to make sure we don't change C2 files and
 * fbgemm files in a single diff
 *
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
    const PackedDepthWiseConvMatrix& Bp,
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
template <typename BIAS_TYPE = std::int32_t>
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
    const PackedDepthWiseConvMatrix& Bp,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    float act_times_w_scale = 1.0f,
    int thread_id = 0,
    int num_threads = 1);

/** To be removed. Keeping it just to make sure we don't change C2 files and
 * fbgemm files in a single diff
 *
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
    const PackedDepthWiseConvMatrix& Bp,
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
template <typename BIAS_TYPE = std::int32_t>
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
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    const float* act_times_w_scale = nullptr,
    int thread_id = 0,
    int num_threads = 1);

} // namespace fbgemm
