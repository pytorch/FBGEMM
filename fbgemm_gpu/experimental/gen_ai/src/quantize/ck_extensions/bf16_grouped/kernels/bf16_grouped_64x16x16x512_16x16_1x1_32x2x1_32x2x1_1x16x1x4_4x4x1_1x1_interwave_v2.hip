/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

std::vector<at::Tensor>
bf16_grouped_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::TensorList A,
    at::TensorList B,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y) {
  // The smallest kernel we have available. Works well for memory bound shapes.
  using DeviceGemmInstance = DeviceGemmHelper<
      64,
      16,
      16,
      512,
      16,
      16,
      1,
      1,
      S<32, 2, 1>,
      S<32, 2, 1>,
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
