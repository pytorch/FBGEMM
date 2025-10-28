/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f8f8bf16_conv_common.cuh"

namespace fbgemm_gpu {

at::Tensor f8f8bf16_conv_64x64x64_1x1x1(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation) { // [dilation_d, dilation_h, dilation_w]

  return f8f8bf16_conv_impl<64, 64, 64, 1, 1, 1>(
      activation, filter, scale, padding, stride, dilation);
}

} // namespace fbgemm_gpu
