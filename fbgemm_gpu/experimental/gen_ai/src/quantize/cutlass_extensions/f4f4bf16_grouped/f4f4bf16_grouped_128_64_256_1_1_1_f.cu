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

at::Tensor f4f4bf16_grouped_128_64_256_1_1_1_f(
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
      cutlass::nv_float4_t<cutlass::float_e2m1_t>,
      128,
      64,
      256,
      1,
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
