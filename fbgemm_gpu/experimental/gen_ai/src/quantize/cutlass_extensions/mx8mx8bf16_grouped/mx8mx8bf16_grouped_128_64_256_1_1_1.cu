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

at::Tensor mx8mx8bf16_grouped_128_64_256_1_1_1(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets) {
  return mx8mx8bf16_grouped_impl<at::Tensor, 128, 64, 256, 1, 1, 1>(
      XQ, WQ, x_scale, w_scale, output, G, offsets);
}

#endif

} // namespace fbgemm_gpu
