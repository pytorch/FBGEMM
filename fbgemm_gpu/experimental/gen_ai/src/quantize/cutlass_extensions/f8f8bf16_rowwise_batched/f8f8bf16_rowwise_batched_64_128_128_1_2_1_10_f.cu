/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f8f8bf16_rowwise_batched_common.cuh"

namespace fbgemm_gpu {

at::Tensor f8f8bf16_rowwise_batched_64_128_128_1_2_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt) {
  // Dispatch this kernel to the correct underlying implementation.
  return f8f8bf16_rowwise_batched_impl<64, 128, 128, 1, 2, 1, 10, false>(
      XQ, WQ, x_scale, w_scale, bias, output);
}

} // namespace fbgemm_gpu
