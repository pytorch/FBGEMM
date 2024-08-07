/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_common.h"

at::Tensor
fp8_rowwise_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  // The smallest kernel we have available. Works well for memory bound shapes.
  using DeviceGemmInstance = DeviceGemmHelper<
      64,
      16,
      16,
      512,
      16,
      16,
      1,
      1,
      S<8, 8, 1>,
      S<8, 8, 1>,
      S<1, 16, 1, 4>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Interwave,
      ck::BlockGemmPipelineVersion::v2>;
  // Run kernel instance.
  return f8f8bf16_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}
