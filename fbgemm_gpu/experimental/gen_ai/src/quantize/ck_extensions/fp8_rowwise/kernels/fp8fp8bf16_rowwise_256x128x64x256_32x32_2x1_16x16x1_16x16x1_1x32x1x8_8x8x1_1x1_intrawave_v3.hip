/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_common.h"

at::Tensor
fp8fp8bf16_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  return f8f8bf16_rowwise_wrapper<
      256,
      128,
      64,
      256,
      32,
      32,
      2,
      1,
      S<16, 16, 1>,
      S<16, 16, 1>,
      S<1, 32, 1, 8>,
      S<8, 8, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3>(XQ, WQ, x_scale, w_scale, Y, 1);
}
