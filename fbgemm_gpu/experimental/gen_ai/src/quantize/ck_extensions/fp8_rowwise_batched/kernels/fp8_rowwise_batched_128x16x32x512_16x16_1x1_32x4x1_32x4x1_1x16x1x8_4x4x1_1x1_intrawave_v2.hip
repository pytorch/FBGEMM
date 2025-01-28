/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_batched_common.h"

at::Tensor
fp8_rowwise_batched_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  using DeviceGemmInstance = DeviceGemmHelper<
      128,
      16,
      32,
      512,
      16,
      16,
      1,
      1,
      S<32, 4, 1>,
      S<32, 4, 1>,
      S<1, 16, 1, 8>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v2,
      ck::tensor_operation::device::GemmSpecialization::Default>;
  // Run kernel instance.
  return f8f8bf16_rowwise_batched_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}

