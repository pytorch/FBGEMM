/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_batched_common.h"

at::Tensor
fp8_rowwise_batched_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  // A kernel that works well on small but not super tiny shapes.

  // Check if this input needs to be padded.
  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = WQ.size(2);
  bool pad = (M % 32 != 0) || (N % 64 != 0) || (K % 128 != 0);

  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
        128,
        32,
        64,
        128,
        32,
        32,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2>;
    // Run kernel instance.
    return f8f8bf16_rowwise_batched_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
        128,
        32,
        64,
        128,
        32,
        32,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return f8f8bf16_rowwise_batched_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
}
