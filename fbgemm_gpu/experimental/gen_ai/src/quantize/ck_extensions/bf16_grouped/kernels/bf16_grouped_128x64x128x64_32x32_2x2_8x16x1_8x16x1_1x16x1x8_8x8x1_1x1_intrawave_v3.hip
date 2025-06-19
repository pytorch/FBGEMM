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
bf16_grouped_128x64x128x64_32x32_2x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v3(
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
        pad = pad & (M % 64 != 0) || (N % 128 != 0) || (K % (64 * KBatch) != 0);
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
            128,
            64,
            32,
            32,
            2,
            2,
            S<8, 16, 1>,
            S<8, 16, 1>,
            S<1, 16, 1, 8>,
            S<8, 8, 1>,
            1,
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
            128,
            64,
            32,
            32,
            2,
            2,
            S<8, 16, 1>,
            S<8, 16, 1>,
            S<1, 16, 1, 8>,
            S<8, 8, 1>,
            1,
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
bf16_grouped_128x64x128x64_32x32_2x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v3(
    at::TensorList X,
    at::TensorList W,
    at::Tensor kernel_args,
    at::Tensor Y);

template at::Tensor
bf16_grouped_128x64x128x64_32x32_2x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor X,
    at::Tensor W,
    at::Tensor kernel_args,
    at::Tensor Y);
