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
bf16_grouped_128x16x96x128_16x16_1x3_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v1(
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
        pad = pad & (M % 16 != 0) || (N % 96 != 0) || (K % (128 * KBatch) != 0);
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
            16,
            96,
            128,
            16,
            16,
            1,
            3,
            S<16, 8, 1>,
            S<16, 8, 1>,
            S<1, 16, 1, 8>,
            S<4, 4, 1>,
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
            128,
            16,
            96,
            128,
            16,
            16,
            1,
            3,
            S<16, 8, 1>,
            S<16, 8, 1>,
            S<1, 16, 1, 8>,
            S<4, 4, 1>,
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



template at::Tensor
bf16_grouped_128x16x96x128_16x16_1x3_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v1(
    at::TensorList X,
    at::TensorList W,
    at::Tensor kernel_args,
    at::Tensor Y);

template at::Tensor
bf16_grouped_128x16x96x128_16x16_1x3_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v1(
    at::Tensor X,
    at::Tensor W,
    at::Tensor kernel_args,
    at::Tensor Y);
