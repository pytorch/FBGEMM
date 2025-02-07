/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_grouped_common.h"

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y) {
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
  return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(
      XQ, WQ, x_scale, w_scale, kernel_args, Y);
}

template std::vector<at::Tensor>
fp8_rowwise_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

template at::Tensor
fp8_rowwise_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);

template at::Tensor
fp8_rowwise_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);
