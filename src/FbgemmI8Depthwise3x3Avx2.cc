/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"

#include <stdexcept> // for logic_error
#include <string>

#include "./FbgemmI8Depthwise2DAvx2-inl.h"

using namespace std;

namespace fbgemm {

// Dispatch input shape and FUSE_RELU
// assumption: W > 3 and H > 3
template <typename BIAS_TYPE>
void depthwise_3x3_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (B.GetKernelProduct() != 3 * 3) {
    string msg =
        "[FBGEMM_CONV_ERROR] Packed weight is expected to have kernel_prod " +
        to_string(3 * 3) + " but has " + to_string(B.GetKernelProduct());
    throw logic_error(msg);
  }
  if (stride_h == 0 || stride_w == 0 || num_threads == 0) {
    assert(0 && "stride_h == 0 || stride_w == 0 || num_threads == 0");
    return;
  }
  if (N == 0) {
    // In C2, batch size 0 is allowed, so we should just early return.
    return;
  }
  if (fuse_relu) {
    if (7 == H && 7 == W && 1 == stride_h && 1 == stride_w) {
      depthwise_2d_<3, true /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (14 == H && 14 == W && 2 == stride_h && 2 == stride_w) {
      depthwise_2d_<3, true /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (1 == stride_h && 1 == stride_w) {
      depthwise_2d_<3, true /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (2 == stride_h && 2 == stride_w) {
      depthwise_2d_<3, true /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_2d_<3, true /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    }
  } else {
    if (7 == H && 7 == W && 1 == stride_h && 1 == stride_w) {
      depthwise_2d_<3, false /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (14 == H && 14 == W && 2 == stride_h && 2 == stride_w) {
      depthwise_2d_<3, false /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (1 == stride_h && 1 == stride_w) {
      depthwise_2d_<3, false /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (2 == stride_h && 2 == stride_w) {
      depthwise_2d_<3, false /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_2d_<3, false /* FUSE_RELU */, BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    }
  }
}

// Dispatch input shape and FUSE_RELU
template <typename BIAS_TYPE>
void depthwise_3x3_per_channel_quantization_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (Bp.GetKernelProduct() != 3 * 3) {
    string msg =
        "[FBGEMM_CONV_ERROR] Packed weight is expected to have kernel_prod " +
        to_string(3 * 3) + " but has " + to_string(Bp.GetKernelProduct());
    throw logic_error(msg);
  }
  if (stride_h == 0 || stride_w == 0 || num_threads == 0) {
    assert(0 && "stride_h == 0 || stride_w == 0 || num_threads == 0");
    return;
  }
  if (N == 0) {
    // In C2, batch size 0 is allowed, so we should just early return.
    return;
  }
  if (fuse_relu) {
    if (7 == H && 7 == W && 1 == stride_h && 1 == stride_w) {
      depthwise_2d_per_channel_quantization_<
          3,
          true /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (14 == H && 14 == W && 2 == stride_h && 2 == stride_w) {
      depthwise_2d_per_channel_quantization_<
          3,
          true /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (1 == stride_h && 1 == stride_w) {
      depthwise_2d_per_channel_quantization_<
          3,
          true /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (2 == stride_h && 2 == stride_w) {
      depthwise_2d_per_channel_quantization_<
          3,
          true /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_2d_per_channel_quantization_<
          3,
          true /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    }
  } else {
    if (7 == H && 7 == W && 1 == stride_h && 1 == stride_w) {
      depthwise_2d_per_channel_quantization_<
          3,
          false /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (14 == H && 14 == W && 2 == stride_h && 2 == stride_w) {
      depthwise_2d_per_channel_quantization_<
          3,
          false /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (1 == stride_h && 1 == stride_w) {
      depthwise_2d_per_channel_quantization_<
          3,
          false /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else if (2 == stride_h && 2 == stride_w) {
      depthwise_2d_per_channel_quantization_<
          3,
          false /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_2d_per_channel_quantization_<
          3,
          false /* FUSE_RELU */,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          Bp,
          C_multiplier,
          C_zero_point,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    }
  }
}

template void depthwise_3x3_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads);

template void depthwise_3x3_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads);

template void depthwise_3x3_per_channel_quantization_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template void depthwise_3x3_per_channel_quantization_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

} // namespace fbgemm
