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
fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y) {
  // Check if this input needs to be padded.
  bool pad = false;
  if constexpr (std::is_same_v<InputType, at::Tensor>) {
    int K = WQ.size(2);
    pad = K % 64 != 0;
  } else {
    for (int i = 0; i < XQ.size(); i++) {
      int K = XQ[i].size(1);
      pad = K % 64 != 0;
    }
  }
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        256,
        128,
        64,
        32,
        32,
        4,
        2,
        S<4, 64, 1>,
        S<4, 64, 1>,
        S<1, 32, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Interwave,
        ck::BlockGemmPipelineVersion::v1,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, kernel_args, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        256,
        128,
        64,
        32,
        32,
        4,
        2,
        S<4, 64, 1>,
        S<4, 64, 1>,
        S<1, 32, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Interwave,
        ck::BlockGemmPipelineVersion::v1,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, kernel_args, Y);
  }
}

template std::vector<at::Tensor>
fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

template at::Tensor
fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);

template at::Tensor
fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);
