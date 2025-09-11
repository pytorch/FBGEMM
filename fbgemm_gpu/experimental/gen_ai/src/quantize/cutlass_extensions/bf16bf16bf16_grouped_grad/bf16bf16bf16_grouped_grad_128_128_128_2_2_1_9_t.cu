/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16bf16bf16_grouped_grad_common.cuh"

namespace fbgemm_gpu {

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_2_2_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes) {
  return bf16bf16bf16_grouped_grad_impl<128, 128, 128, 2, 2, 1, true>(
      X, W, output, M_sizes);
}

} // namespace fbgemm_gpu
