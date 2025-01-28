/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_batched_common.h"

at::Tensor
fp8_rowwise_batched_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  using DeviceGemmInstance = DeviceGemmHelper<
      256,
      64,
      64,
      512,
      32,
      32,
      1,
      1,
      S<32, 8, 1>,
      S<32, 8, 1>,
      S<1, 32, 1, 8>,
      S<8, 8, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3,
      ck::tensor_operation::device::GemmSpecialization::Default>;
  // Run kernel instance.
  return f8f8bf16_rowwise_batched_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}

