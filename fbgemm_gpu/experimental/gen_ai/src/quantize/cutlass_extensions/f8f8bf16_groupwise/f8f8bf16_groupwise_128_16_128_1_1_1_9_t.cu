/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f8f8bf16_groupwise_common.cuh"

namespace fbgemm_gpu {

at::Tensor f8f8bf16_groupwise_128_16_128_1_1_1_9_t(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale) {
  // Dispatch this kernel to the correct underlying implementation.
  return f8f8bf16_groupwise_wrapper<128, 16, 128, 1, 1, 1, 9, true>(
      XQ, WQ, x_scale, w_scale);
}

} // namespace fbgemm_gpu
