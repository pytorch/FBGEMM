/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f4f4bf16_grouped_common.cuh"

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor f4f4bf16_grouped_256_256_256_2_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding) {
  return f4f4bf16_grouped_impl<
      at::Tensor,
      cutlass::mx_float4_t<cutlass::float_e2m1_t>,
      256,
      256,
      256,
      2,
      1,
      1>(
      XQ,
      WQ,
      x_scale,
      w_scale,
      output,
      G,
      zero_start_index_M,
      M_sizes,
      global_scale,
      starting_row_after_padding);
}

at::Tensor f4f4bf16_grouped_256_256_256_2_1_1_t(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding) {
  return f4f4bf16_grouped_impl<
      at::TensorList,
      cutlass::mx_float4_t<cutlass::float_e2m1_t>,
      256,
      256,
      256,
      2,
      1,
      1>(
      XQ,
      WQ,
      x_scale,
      w_scale,
      output,
      G,
      zero_start_index_M,
      M_sizes,
      global_scale,
      starting_row_after_padding);
}

#endif

} // namespace fbgemm_gpu
