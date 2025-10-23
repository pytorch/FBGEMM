/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace fbgemm_gpu {

at::Tensor f8f8bf16_conv_64x64x64_1x1x1(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation);

at::Tensor f8f8bf16_conv_128x128x128_1x1x1(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation);

at::Tensor f8f8bf16_conv_128x256x128_2x1x1(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation);

at::Tensor f8f8bf16_conv_128x256x128_2x2x1(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation);

at::Tensor f8f8bf16_conv_256x256x128_2x1x1(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation);

at::Tensor f8f8bf16_conv_256x256x128_4x1x1(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation);

at::Tensor f8f8bf16_conv_256x512x128_2x2x1(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation);

at::Tensor f8f8bf16_conv_512x256x128_4x1x1(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation);

using Kernel_f8f8bf16_conv = at::Tensor (*)(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation); // [dilation_d, dilation_h, dilation_w]

} // namespace fbgemm_gpu
