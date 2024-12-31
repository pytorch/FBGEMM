/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

std::vector<at::Tensor>
bf16_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
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
        64,
        16,
        16,
        128,
        16,
        16,
        1,
        1,
        S<8, 8, 1>,
        S<8, 8, 1>,
        S<1, 16, 1, 4>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Interwave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return bf16_grouped_impl<DeviceGemmInstance>(
        A, B, kernel_args, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
        64,
        16,
        16,
        128,
        16,
        16,
        1,
        1,
        S<8, 8, 1>,
        S<8, 8, 1>,
        S<1, 16, 1, 4>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Interwave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return bf16_grouped_impl<DeviceGemmInstance>(
        A, B, kernel_args, Y);
  }
}
