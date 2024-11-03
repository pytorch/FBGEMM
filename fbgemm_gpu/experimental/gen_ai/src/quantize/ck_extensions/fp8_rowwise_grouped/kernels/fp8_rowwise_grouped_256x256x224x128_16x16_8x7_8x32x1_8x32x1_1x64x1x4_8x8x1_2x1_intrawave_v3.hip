/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_grouped_common.h"

std::vector<at::Tensor>
fp8_rowwise_grouped_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
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
    if (M % 256 != 0 || N % 224 != 0 || K % 128 != 0) {
      pad = true;
    }
  }

  // This kernel seems optimal in the most purely compute bound tasks.
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        256,
        224,
        128,
        16,
        16,
        8,
        7,
        S<8, 32, 1>,
        S<8, 32, 1>,
        S<1, 64, 1, 4>,
        S<8, 8, 1>,
        2,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3>;
    // Run kernel instance.
    return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        256,
        224,
        128,
        16,
        16,
        8,
        7,
        S<8, 32, 1>,
        S<8, 32, 1>,
        S<1, 64, 1, 4>,
        S<8, 8, 1>,
        2,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, Y);
  }
}
