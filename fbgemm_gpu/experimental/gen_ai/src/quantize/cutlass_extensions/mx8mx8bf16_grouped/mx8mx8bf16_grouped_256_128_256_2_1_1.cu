/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "mx8mx8bf16_grouped_common.cuh"

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor mx8mx8bf16_grouped_256_128_256_2_1_1(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> starting_row_after_padding) {
  return mx8mx8bf16_grouped_impl<at::Tensor, 256, 128, 256, 2, 1, 1>(
      XQ,
      WQ,
      x_scale,
      w_scale,
      output,
      G,
      zero_start_index_M,
      M_sizes,
      starting_row_after_padding);
}

#endif

} // namespace fbgemm_gpu
