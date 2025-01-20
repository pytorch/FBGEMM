/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>
#include <unordered_map>

#include <ATen/ATen.h>

#define KERNEL_NAME_MAP_ENTRY(name) \
  { #name, name }

using GroupedKernel = std::function<std::vector<at::Tensor>(
    at::TensorList,
    at::TensorList,
    at::Tensor,
    std::vector<at::Tensor>)>;

std::vector<at::Tensor>
bf16_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_256x128x128x64_32x32_2x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_256x256x256x64_16x16_8x8_4x64x1_4x64x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_256x128x256x64_32x32_2x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x256_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x128x32x128_32x32_2x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x64x16x128_16x16_2x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x256_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x64x128_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x128x128_16x16_1x4_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x256_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v1(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x128x32x128_32x32_2x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x64x16x128_16x16_2x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x256_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x64x128_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x16x128x128_16x16_1x4_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

std::vector<at::Tensor>
bf16_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);

// Map function for string name to kernel implementation for manual
// specification.
static const std::unordered_map<std::string, GroupedKernel> kernel_name_map = {
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_256x128x128x64_32x32_2x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_256x256x256x64_16x16_8x8_4x64x1_4x64x1_1x32x1x8_8x8x1_1x2_intrawave_v3),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_256x128x256x64_32x32_2x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x256_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x128x32x128_32x32_2x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x64x16x128_16x16_2x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x256_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x64x128_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x128x128_16x16_1x4_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x256_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v1),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x128x32x128_32x32_2x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x64x16x128_16x16_2x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x256_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x64x128_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x16x128x128_16x16_1x4_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2),
    KERNEL_NAME_MAP_ENTRY(
        bf16_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2),
};
