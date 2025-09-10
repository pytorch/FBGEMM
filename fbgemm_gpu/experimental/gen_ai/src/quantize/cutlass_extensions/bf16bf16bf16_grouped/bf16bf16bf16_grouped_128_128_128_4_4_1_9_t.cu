/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16bf16bf16_grouped_common.cuh"

namespace fbgemm_gpu {

at::Tensor bf16bf16bf16_grouped_128_128_128_4_4_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes) {
  return bf16bf16bf16_grouped_impl<at::Tensor, 128, 128, 128, 4, 4, 1, true>(
      X, W, output, zero_start_index_M, M_sizes);
}

at::Tensor bf16bf16bf16_grouped_128_128_128_4_4_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes) {
  return bf16bf16bf16_grouped_impl<
      at::TensorList,
      128,
      128,
      128,
      4,
      4,
      1,
      true>(X, W, output, zero_start_index_M, M_sizes);
}

} // namespace fbgemm_gpu
