/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f8f8bf16_groupwise_grouped_common.cuh"

namespace fbgemm_gpu {

at::Tensor f8f8bf16_groupwise_grouped_128_128_128_1_2_1_9_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    at::Tensor M_sizes) {
  // Dispatch this kernel to the correct underlying implementation.
  return f8f8bf16_groupwise_grouped_impl<
      at::Tensor,
      128,
      128,
      128,
      1,
      2,
      1,
      9,
      false>(XQ, WQ, x_scale, w_scale, output, M_sizes);
}

} // namespace fbgemm_gpu
