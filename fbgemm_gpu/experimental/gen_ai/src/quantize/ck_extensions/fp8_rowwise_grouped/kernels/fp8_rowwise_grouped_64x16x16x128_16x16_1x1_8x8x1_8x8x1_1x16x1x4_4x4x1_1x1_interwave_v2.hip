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
fp8_rowwise_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
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
    bool pad = (M % 16 != 0) || (N % 16 != 0) || (K % (128 * KBatch) != 0);
#else
    // disable padding for packed tensor
    bool pad = false;
#endif
    if (pad)
    {
        // pad
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
            ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
        // Run kernel instance.
        return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, kernel_args, Y);

        // pad
    }
    else
    {
        // no pad
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
        return f8f8bf16_rowwise_grouped_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, kernel_args, Y);

        // no pad
    }
}


template std::vector<at::Tensor>
fp8_rowwise_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

template at::Tensor
fp8_rowwise_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);

template at::Tensor
fp8_rowwise_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    at::Tensor Y);
