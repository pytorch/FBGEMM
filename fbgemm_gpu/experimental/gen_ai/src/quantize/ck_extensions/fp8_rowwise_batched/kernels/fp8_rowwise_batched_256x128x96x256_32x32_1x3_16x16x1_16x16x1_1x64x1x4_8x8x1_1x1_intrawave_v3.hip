/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_batched_common.h"

at::Tensor
fp8_rowwise_batched_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  using DeviceGemmInstance = DeviceGemmHelper<
      256,
      128,
      96,
      256,
      32,
      32,
      1,
      3,
      S<16, 16, 1>,
      S<16, 16, 1>,
      S<1, 64, 1, 4>,
      S<8, 8, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3,
      ck::tensor_operation::device::GemmSpecialization::Default>;
  // Run kernel instance.
  return f8f8bf16_rowwise_batched_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}

