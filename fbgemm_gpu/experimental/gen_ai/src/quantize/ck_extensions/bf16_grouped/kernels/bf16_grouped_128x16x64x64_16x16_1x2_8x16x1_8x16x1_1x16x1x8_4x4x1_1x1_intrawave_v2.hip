/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

std::vector<at::Tensor>
bf16_grouped_128x16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
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
    bool pad = (M % 16 != 0) || (N % 64 != 0) || (K % (64 * KBatch) != 0);
#else
    // disable padding for packed tensor
    bool pad = false;
#endif
    if (pad)
    {
        // pad
        using DeviceGemmInstance = DeviceGemmHelper<
            128,
            16,
            64,
            64,
            16,
            16,
            1,
            2,
            S<8, 16, 1>,
            S<8, 16, 1>,
            S<1, 16, 1, 8>,
            S<4, 4, 1>,
            1,
            1,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v2,
            ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
        // Run kernel instance.
        return bf16_grouped_impl<DeviceGemmInstance>(A, B, kernel_args, Y);

        // pad
    }
    else
    {
        // no pad
        using DeviceGemmInstance = DeviceGemmHelper<
            128,
            16,
            64,
            64,
            16,
            16,
            1,
            2,
            S<8, 16, 1>,
            S<8, 16, 1>,
            S<1, 16, 1, 8>,
            S<4, 4, 1>,
            1,
            1,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v2,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return bf16_grouped_impl<DeviceGemmInstance>(A, B, kernel_args, Y);

        // no pad
    }
}


