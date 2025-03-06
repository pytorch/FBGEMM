#pragma once
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>


template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x256x128_32x32_4x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x224x128_32x32_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x192x128_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x160x128_32x32_2x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x128x128_32x32_4x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x96x128_32x32_2x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x64x128_32x32_4x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x256x128_32x32_2x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x224x128_32x32_1x7_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x256x128_32x32_1x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x192x256_32x32_1x3_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x192x128_32x32_1x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x160x256_16x16_2x5_16x16x1_16x16x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x96x256_16x16_2x3_16x16x1_16x16x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x32x256x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x32x224x256_16x16_1x7_16x16x1_16x16x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x32x192x256_16x16_1x6_16x16x1_16x16x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x32x160x256_16x16_1x5_16x16x1_16x16x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x32x96x256_16x16_1x3_16x16x1_16x16x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

/*template <typename InputType, typename OutputType>*/
/*OutputType*/
/*fp8_rowwise_grouped_256x16x256x128_16x16_1x4_16x16x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(*/
/*InputType XQ,*/
/*InputType WQ,*/
/*InputType x_scale,*/
/*InputType w_scale,*/
/*at::Tensor kernel_args,*/
/*OutputType Y);*/

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x16x192x256_16x16_1x3_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v4(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x128x64_32x32_2x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x256x64_16x16_8x8_4x64x1_4x64x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x256x64_32x32_2x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x32x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x16x128_16x16_4x1_8x32x1_8x16x1_1x32x1x8_2x2x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x128x32x128_32x32_2x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x64x16x128_16x16_2x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x16x64x128_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x16x128x128_16x16_1x4_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x16x256x128_16x16_1x4_8x16x1_8x16x1_1x16x1x16_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x32x256x128_32x32_1x2_8x32x1_8x32x1_1x16x1x16_8x8x1_1x1_intrawave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v1(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x32x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x256x16x128_16x16_4x1_8x32x1_8x16x1_1x32x1x8_2x2x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x128x32x128_32x32_2x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x64x16x128_16x16_2x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x16x64x128_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x16x128x128_16x16_1x4_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x16x256x128_16x16_1x4_8x16x1_8x16x1_1x16x1x16_4x4x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
fp8_rowwise_grouped_256x32x256x128_32x32_1x2_8x32x1_8x32x1_1x16x1x16_8x8x1_1x1_interwave_v2(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y);

