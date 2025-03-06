/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

std::vector<at::Tensor>
bf16_grouped_256x128x160x64_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3(
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
        128,
        160,
        64,
        32,
        32,
        1,
        5,
        S<8, 32, 1>,
        S<8, 32, 1>,
        S<1, 64, 1, 4>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return bf16_grouped_impl<DeviceGemmInstance>(
        A, B, kernel_args, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        128,
        160,
        64,
        32,
        32,
        1,
        5,
        S<8, 32, 1>,
        S<8, 32, 1>,
        S<1, 64, 1, 4>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return bf16_grouped_impl<DeviceGemmInstance>(
        A, B, kernel_args, Y);
  }
}
