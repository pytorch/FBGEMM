/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

std::vector<at::Tensor>
bf16_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::TensorList A,
    at::TensorList B,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y) {
  // A kernel that works well on small but not super tiny shapes.
  using DeviceGemmInstance = DeviceGemmHelper<
      128,
      128,
      16,
      128,
      16,
      16,
      4,
      1,
      S<8, 16, 1>,
      S<8, 16, 1>,
      S<1, 16, 1, 8>,
      S<2, 2, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Interwave,
      ck::BlockGemmPipelineVersion::v2>;
  // Run kernel instance.
  return bf16_grouped_impl<DeviceGemmInstance>(
      A, B, kernel_args, Y);
}
