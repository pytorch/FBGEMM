/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f8f8bf16_rowwise_grouped_common.cuh"

namespace fbgemm_gpu {

at::Tensor f8f8bf16_rowwise_grouped_128_32_128_2_1_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes) {
  // Dispatch this kernel to the correct underlying implementation.
  return f8f8bf16_rowwise_grouped_sm100_impl<
      at::Tensor,
      128,
      32,
      128,
      2,
      1,
      1,
      10,
      false>(XQ, WQ, x_scale, w_scale, output, zero_start_index_M, M_sizes);
}

at::Tensor f8f8bf16_rowwise_grouped_128_32_128_2_1_1_10_f(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes) {
  // Dispatch this kernel to the correct underlying implementation.
  return f8f8bf16_rowwise_grouped_sm100_impl<
      at::TensorList,
      128,
      32,
      128,
      2,
      1,
      1,
      10,
      false>(XQ, WQ, x_scale, w_scale, output, zero_start_index_M, M_sizes);
}

} // namespace fbgemm_gpu
