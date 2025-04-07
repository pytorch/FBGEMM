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
fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y)
{
    // Check if this input needs to be padded.
#if 0
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = WQ.size(2);
    bool pad = (M % 16 != 0) || (N % 64 != 0) || (K % (256 * KBatch) != 0);
#else
    // disable padding for packed tensor
    bool pad = false;
#endif
    if (pad)
    {
        // pad
        using DeviceGemmInstance = DeviceGemmHelper<
            256,
            16,
            64,
            256,
            16,
            16,
            1,
            1,
            S<16, 16, 1>,
            S<16, 16, 1>,
            S<1, 16, 1, 16>,
            S<4, 4, 1>,
            1,
            1,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
        // Run kernel instance.
        return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, kernel_args, Y);

        // pad
    }
    else
    {
        // no pad
        using DeviceGemmInstance = DeviceGemmHelper<
            256,
            16,
            64,
            256,
            16,
            16,
            1,
            1,
            S<16, 16, 1>,
            S<16, 16, 1>,
            S<1, 16, 1, 16>,
            S<4, 4, 1>,
            1,
            1,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, kernel_args, Y);

        // no pad
    }
}


template std::vector<at::Tensor>
fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

template at::Tensor
fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);

template at::Tensor
fp8_rowwise_grouped_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);
