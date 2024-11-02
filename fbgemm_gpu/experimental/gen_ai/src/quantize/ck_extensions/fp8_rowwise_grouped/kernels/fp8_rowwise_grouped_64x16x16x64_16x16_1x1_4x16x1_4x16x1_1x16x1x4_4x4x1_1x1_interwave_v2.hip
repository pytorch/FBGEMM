/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_grouped_common.h"

std::vector<at::Tensor>
fp8_rowwise_grouped_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::vector<at::Tensor> Y) {
  // The smallest kernel we have available. Works well for memory bound shapes.
  using DeviceGemmInstance = DeviceGemmHelper<
      64,
      16,
      16,
      64,
      16,
      16,
      1,
      1,
      S<4, 16, 1>,
      S<4, 16, 1>,
      S<1, 16, 1, 4>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Interwave,
      ck::BlockGemmPipelineVersion::v2>;
  // Run kernel instance.
  return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}
