/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

std::vector<at::Tensor>
bf16_grouped_256x128x256x32_32x32_2x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    at::TensorList A,
    at::TensorList B,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y)
{
    // Check if this input needs to be padded.
#if 0
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = WQ.size(2);
    bool pad = (M % 128 != 0) || (N % 256 != 0) || (K % (32 * KBatch) != 0);
#else
    // disable padding for packed tensor
    bool pad = false;
#endif
    if (pad)
    {
        // pad
        using DeviceGemmInstance = DeviceGemmHelper<
            256,
            128,
            256,
            32,
            32,
            32,
            2,
            4,
            S<4, 64, 1>,
            S<4, 64, 1>,
            S<1, 32, 1, 8>,
            S<8, 8, 1>,
            1,
            1,
            ck::BlockGemmPipelineScheduler::Interwave,
            ck::BlockGemmPipelineVersion::v1,
            ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
        // Run kernel instance.
        return bf16_grouped_impl<DeviceGemmInstance>(A, B, kernel_args, Y);

        // pad
    }
    else
    {
        // no pad
        using DeviceGemmInstance = DeviceGemmHelper<
            256,
            128,
            256,
            32,
            32,
            32,
            2,
            4,
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
        return bf16_grouped_impl<DeviceGemmInstance>(A, B, kernel_args, Y);

        // no pad
    }
}


