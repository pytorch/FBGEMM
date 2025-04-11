/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f8f8bf16_rowwise_common.cuh"

namespace fbgemm_gpu {

at::Tensor f8f8bf16_rowwise_128_256_128_4_4_1_f_t(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt) {
  // Dispatch this kernel to the correct underlying implementation.
  return f8f8bf16_rowwise_wrapper<128, 256, 128, 4, 4, 1, false, true>(
      XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
}

} // namespace fbgemm_gpu
