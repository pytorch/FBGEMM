/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_common.h"

at::Tensor
fp8_rowwise_256x16x64x128_16x16_1x1_16x16x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v2_8(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  using DeviceGemmInstance = DeviceGemmHelper<
      256,
      16,
      64,
      128,
      16,
      16,
      1,
      1,
      S<16, 16, 1>,
      S<8, 32, 1>,
      S<1, 16, 1, 16>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v2,
      ck::tensor_operation::device::GemmSpecialization::Default,
      8,  // AReadVecLength
      16, // BReadVecLength
      8,  // ADstVecLength
      16, // BDstVecLength
      8,  // AK1
      16  // BK1
      >;
  // Run kernel instance.
  return f8f8bf16_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, 8);
}
