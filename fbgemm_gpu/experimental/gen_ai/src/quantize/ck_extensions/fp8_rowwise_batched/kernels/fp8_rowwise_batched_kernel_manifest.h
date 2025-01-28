/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>

#include <ATen/ATen.h>

// Default Kernel for now.
at::Tensor
fp8_rowwise_batched_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V4 Variant
at::Tensor
fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v4(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V5 variant
at::Tensor
fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Super tiny kernel.
at::Tensor
fp8_rowwise_batched_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_1x1_interwave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V2 variant
at::Tensor
fp8_rowwise_batched_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Still tiny but better for larger B.
at::Tensor
fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Intrawave variant.
at::Tensor
fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V1 variant
at::Tensor
fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V1 intrawave variant
at::Tensor
fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Relatively small shapes with M between 32 and 64.
at::Tensor
fp8_rowwise_batched_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Seems to work well for small M but large other shapes.
at::Tensor
fp8_rowwise_batched_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V2 variant
at::Tensor
fp8_rowwise_batched_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Good default kernel for M=64.
at::Tensor
fp8_rowwise_batched_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Works well for some M=128 cases.
at::Tensor
fp8_rowwise_batched_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Very effective large shape kernel.
at::Tensor
fp8_rowwise_batched_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Similar variant.
at::Tensor
fp8_rowwise_batched_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Kernel for M=32 and large NK.
at::Tensor
fp8_rowwise_batched_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x128x96x256_32x32_1x3_16x16x1_16x16x1_1x64x1x4_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x128x128x256_32x32_2x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x256x128x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x128x256x128_32x32_2x4_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x128x192x128_32x32_2x3_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x32x128x256_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x128x160x128_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x64x128x256_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_128x16x32x256_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4x4x1_1x1_interwave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x64x192x256_32x32_1x3_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_batched_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);
