/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

std::vector<at::Tensor>
bf16_grouped_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::TensorList A,
    at::TensorList B,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y) {
  // Check if this input needs to be padded.
  bool pad = false;
  for (int i = 0; i < A.size(); i++) {
    int K = A[i].size(1);
    if (K % 128 != 0) {
      pad = true;
    }
  }
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        64,
        64,
        128,
        32,
        32,
        1,
        1,
        S<8, 32, 1>,
        S<8, 32, 1>,
        S<1, 32, 1, 8>,
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
        64,
        64,
        128,
        32,
        32,
        1,
        1,
        S<8, 32, 1>,
        S<8, 32, 1>,
        S<1, 32, 1, 8>,
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
