/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x64x96x64_16x16_4x3_8x16x1_8x16x1_1x32x1x4_8x8x1_2x1_intrawave_v3(
    InputType A,
    InputType B,
    at::Tensor kernel_args,
    OutputType Y)
{
    // Check if this input needs to be padded.
#if 0
    bool pad = true;
	for (int i = 0; i < A.size(); i++) {
		int M = A[i].size(0);
		int K = A[i].size(1);
		int N = B[i].size(0);
        pad = pad & (M % 64 != 0) || (N % 96 != 0) || (K % (64 * KBatch) != 0);
	}
#else
    // disable padding for packed tensor
    bool pad = false;
#endif
    if (pad)
    {
        // pad
        using DeviceGemmInstance = DeviceGemmHelper<
            128,
            64,
            96,
            64,
            16,
            16,
            4,
            3,
            S<8, 16, 1>,
            S<8, 16, 1>,
            S<1, 32, 1, 4>,
            S<8, 8, 1>,
            2,
            1,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
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
            64,
            96,
            64,
            16,
            16,
            4,
            3,
            S<8, 16, 1>,
            S<8, 16, 1>,
            S<1, 32, 1, 4>,
            S<8, 8, 1>,
            2,
            1,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return bf16_grouped_impl<DeviceGemmInstance>(A, B, kernel_args, Y);

        // no pad
    }
}



template at::Tensor
bf16_grouped_128x64x96x64_16x16_4x3_8x16x1_8x16x1_1x32x1x4_8x8x1_2x1_intrawave_v3(
    at::TensorList X,
    at::TensorList W,
    at::Tensor kernel_args,
    at::Tensor Y);

template at::Tensor
bf16_grouped_128x64x96x64_16x16_4x3_8x16x1_8x16x1_1x32x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor X,
    at::Tensor W,
    at::Tensor kernel_args,
    at::Tensor Y);
