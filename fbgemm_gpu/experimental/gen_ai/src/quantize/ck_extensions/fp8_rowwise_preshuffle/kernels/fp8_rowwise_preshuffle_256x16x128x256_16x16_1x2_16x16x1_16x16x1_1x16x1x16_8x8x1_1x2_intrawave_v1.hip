/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_preshuffle_common.h"

at::Tensor
fp8_rowwise_preshuffle_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  return f8f8bf16_rowwise_preshuffle_wrapper<
      256,
      16,
      128,
      256,
      16,
      16,
      1,
      2,
      S<16, 16, 1>,
      S<16, 16, 1>,
      S<1, 16, 1, 16>,
      S<8, 8, 1>,
      1,
      2,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v1>(XQ, WQ, x_scale, w_scale, Y, 1);
}
