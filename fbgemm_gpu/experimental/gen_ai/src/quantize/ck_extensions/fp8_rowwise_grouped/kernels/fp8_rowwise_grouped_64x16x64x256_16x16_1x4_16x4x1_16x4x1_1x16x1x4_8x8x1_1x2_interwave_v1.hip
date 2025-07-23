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
fp8_rowwise_grouped_64x16x64x256_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_interwave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y) {
  using DeviceGemmInstance = DeviceGemmHelper<
      64,
      16,
      64,
      256,
      16,
      16,
      1,
      4,
      S<16, 4, 1>,
      S<16, 4, 1>,
      S<1, 16, 1, 4>,
      S<8, 8, 1>,
      1,
      2,
      ck::BlockGemmPipelineScheduler::Interwave,
      ck::BlockGemmPipelineVersion::v1>;
  return DeviceGemmInstance::f8f8bf16_rowwise_grouped_impl(
      XQ, WQ, x_scale, w_scale, kernel_args, Y);
}

template std::vector<at::Tensor>
fp8_rowwise_grouped_64x16x64x256_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

template at::Tensor
fp8_rowwise_grouped_64x16x64x256_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_interwave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);

template at::Tensor
fp8_rowwise_grouped_64x16x64x256_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);
} // namespace fbgemm_gpu
