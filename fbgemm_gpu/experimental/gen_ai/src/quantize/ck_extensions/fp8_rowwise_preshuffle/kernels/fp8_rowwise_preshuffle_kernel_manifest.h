/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

at::Tensor
fp8_rowwise_preshuffle_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x128x512_16x16_1x2_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x128x512_16x16_1x2_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x256x512_16x16_1x4_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x256x512_16x16_1x4_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_128x32x16x512_16x16_1x1_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_128x32x16x512_16x16_1x1_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x64x16x512_16x16_1x1_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x64x16x512_16x16_1x1_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x64x16x512_16x16_1x1_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x64x16x512_16x16_1x1_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x256x256_16x16_1x4_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x256x256_16x16_1x4_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x512x256_16x16_1x8_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x16x512x256_16x16_1x8_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x256x256x128_16x16_16x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x256x192x128_16x16_16x3_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x256x160x128_16x16_8x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x256x128x128_16x16_16x2_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x256x96x128_16x16_8x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x256x64x128_16x16_16x1_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x224x256x128_16x16_14x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x224x224x128_16x16_7x7_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x224x192x128_16x16_14x3_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x224x160x128_16x16_7x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x224x128x128_16x16_14x2_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x224x96x128_16x16_7x3_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x224x64x128_16x16_14x1_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x192x256x128_16x16_12x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x192x224x128_16x16_6x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x192x192x128_16x16_12x3_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x192x160x128_16x16_6x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x192x128x128_16x16_12x2_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x192x96x128_16x16_6x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x192x64x128_16x16_12x1_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x160x256x128_16x16_10x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x160x224x128_16x16_5x7_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x160x192x128_16x16_10x3_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x160x160x128_16x16_5x5_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x160x128x128_16x16_10x2_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x160x96x128_16x16_5x3_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x160x64x128_16x16_10x1_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x96x128_16x16_4x3_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x64x128_16x16_8x1_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x128x256_16x16_8x2_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x96x256_16x16_4x3_16x16x1_16x16x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x64x256_16x16_8x1_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x256x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x224x128_16x16_4x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x192x128_16x16_8x3_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x160x128_16x16_4x5_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_preshuffle_256x128x128x128_16x16_8x2_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);
