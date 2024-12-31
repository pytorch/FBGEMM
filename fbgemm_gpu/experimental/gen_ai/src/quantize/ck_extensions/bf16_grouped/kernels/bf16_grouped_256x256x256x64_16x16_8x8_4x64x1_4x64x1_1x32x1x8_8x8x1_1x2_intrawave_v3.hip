/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

std::vector<at::Tensor>
bf16_grouped_256x256x256x64_16x16_8x8_4x64x1_4x64x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::TensorList A,
    at::TensorList B,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y) {
  // Check if this input needs to be padded.
  bool pad = false;
  for (int i = 0; i < A.size(); i++) {
    int K = A[i].size(1);
    if (K % 64 != 0) {
      pad = true;
    }
  }
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        256,
        256,
        64,
        16,
        16,
        8,
        8,
        S<4, 64, 1>,
        S<4, 64, 1>,
        S<1, 32, 1, 8>,
        S<8, 8, 1>,
        1,
        2,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return bf16_grouped_impl<DeviceGemmInstance>(
        A, B, kernel_args, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        256,
        256,
        64,
        16,
        16,
        8,
        8,
        S<4, 64, 1>,
        S<4, 64, 1>,
        S<1, 32, 1, 8>,
        S<8, 8, 1>,
        1,
        2,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return bf16_grouped_impl<DeviceGemmInstance>(
        A, B, kernel_args, Y);
  }
}
