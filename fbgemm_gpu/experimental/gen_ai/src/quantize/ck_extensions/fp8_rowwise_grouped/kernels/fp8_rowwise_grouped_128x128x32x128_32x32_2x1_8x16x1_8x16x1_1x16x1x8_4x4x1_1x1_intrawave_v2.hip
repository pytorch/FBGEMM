/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_grouped_common.h"

std::vector<at::Tensor>
fp8_rowwise_grouped_128x128x32x128_32x32_2x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::vector<at::Tensor> Y) {
  // Check if this input needs to be padded.
  bool pad = false;
  for (int i = 0; i < XQ.size(); i++) {
    int M = XQ[i].size(0);
    int N = WQ[i].size(0);
    int K = XQ[i].size(1);
    if (M % 128 != 0 || N % 32 != 0 || K % 128 != 0) {
      pad = true;
    }
  }

  // This kernel seems optimal in the most purely compute bound tasks.
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
        128,
        128,
        32,
        128,
        32,
        32,
        2,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2>;
    // Run kernel instance.
    return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
        128,
        128,
        32,
        128,
        32,
        32,
        2,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, Y);
  }
}
