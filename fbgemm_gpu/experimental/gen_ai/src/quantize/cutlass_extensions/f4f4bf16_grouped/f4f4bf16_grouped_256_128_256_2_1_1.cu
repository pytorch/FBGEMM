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

at::Tensor f4f4bf16_grouped_256_64_256_2_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> offsets,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding) {
  if (global_scale) {
    return f4f4bf16_grouped_impl<NVFP4, 256, 64, 256, 2, 1, 1>(
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        offsets,
        M_sizes,
        global_scale,
        starting_row_after_padding);
  } else {
    return f4f4bf16_grouped_impl<MXFP4, 256, 64, 256, 2, 1, 1>(
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        offsets,
        M_sizes,
        global_scale,
        starting_row_after_padding);
  }
}

#endif

} // namespace fbgemm_gpu
