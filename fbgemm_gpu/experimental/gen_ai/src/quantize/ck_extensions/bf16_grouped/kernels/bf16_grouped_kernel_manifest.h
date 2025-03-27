/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>

#include <ATen/ATen.h>

#define KERNEL_NAME_MAP_ENTRY(name) \
  { #name, name }

template <typename InputType, typename OutputType>
using GroupedKernel =
    std::function<OutputType(InputType, InputType, at::Tensor, OutputType)>;

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x32x128x128_16x16_1x4_16x16x1_16x16x1_1x32x1x8_8x8x1_1x2_intrawave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x256x256x64_32x32_4x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x256x192x64_32x32_4x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x128x96x64_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x16x128x128_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x32x96x128_16x16_2x3_16x8x1_16x8x1_1x32x1x4_8x8x1_2x1_intrawave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x64x192x128_16x16_4x3_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x128x256x64_32x32_4x2_8x32x1_8x32x1_1x16x1x16_8x8x1_1x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x16x128x128_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x96x128_16x16_1x3_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x32x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_64x16x16x128_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x32x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x32x64x128_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_intrawave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x32x224x64_16x16_1x7_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x96x128_16x16_1x3_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x256x128x32_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x32x64x128_32x32_1x1_16x8x1_16x8x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x96x128_16x16_1x3_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x32x96x64_16x16_1x3_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_interwave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x32x128_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_64x16x16x128_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_64x16x48x128_16x16_1x3_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x64x96x64_16x16_4x3_8x16x1_8x16x1_1x32x1x4_8x8x1_2x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x128x128x128_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x64x128x128_32x32_2x1_16x16x1_16x16x1_1x16x1x16_8x8x1_1x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x64x128_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x16x64x128_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_interwave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x64x128_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_intrawave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x96x64_16x16_1x3_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x32x16x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x96x128_16x16_1x3_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x16x64x128_16x16_1x2_16x8x1_16x8x1_1x16x1x8_8x8x1_1x2_interwave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x256x160x64_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x224x256x32_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_64x16x16x64_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x256x224x64_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_128x64x128x64_32x32_2x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x64x96x64_16x16_2x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x128x224x64_16x16_4x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_64x16x64x128_16x16_1x4_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_256x32x96x64_16x16_1x3_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_interwave_v1(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);

template <typename InputType, typename OutputType>
OutputType
bf16_grouped_64x16x32x128_16x16_1x2_16x4x1_16x4x1_1x16x1x4_8x8x1_1x2_intrawave_v2(
    InputType X,
    InputType W,
    at::Tensor kernel_args,
    OutputType Y);
