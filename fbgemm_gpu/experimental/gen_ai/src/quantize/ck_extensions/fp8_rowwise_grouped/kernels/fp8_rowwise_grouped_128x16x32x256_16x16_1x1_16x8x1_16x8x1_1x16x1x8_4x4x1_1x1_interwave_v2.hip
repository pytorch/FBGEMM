/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_grouped_common.h"
namespace fbgemm_gpu {

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y) {
  using DeviceGemmInstance = DeviceGemmHelper<
      128,
      16,
      32,
      256,
      16,
      16,
      1,
      1,
      S<16, 8, 1>,
      S<16, 8, 1>,
      S<1, 16, 1, 8>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Interwave,
      ck::BlockGemmPipelineVersion::v2>;
  return DeviceGemmInstance::f8f8bf16_rowwise_grouped_impl(
      XQ, WQ, x_scale, w_scale, kernel_args, Y);
}

template std::vector<at::Tensor>
fp8_rowwise_grouped_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

template at::Tensor
fp8_rowwise_grouped_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);

template at::Tensor
fp8_rowwise_grouped_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);
} // namespace fbgemm_gpu
