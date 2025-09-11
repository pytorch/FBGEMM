/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16bf16bf16_grouped_grad_common.cuh"

namespace fbgemm_gpu {

at::Tensor bf16bf16bf16_grouped_grad_256_16_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes) {
  return bf16bf16bf16_grouped_grad_impl<256, 16, 128, 1, 1, 1, false>(
      X, W, output, M_sizes);
}

} // namespace fbgemm_gpu
