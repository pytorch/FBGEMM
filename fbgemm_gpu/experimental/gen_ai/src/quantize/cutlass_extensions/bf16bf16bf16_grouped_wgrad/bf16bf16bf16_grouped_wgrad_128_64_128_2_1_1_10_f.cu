/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16bf16bf16_grouped_wgrad_common.cuh"

namespace fbgemm_gpu {

at::Tensor bf16bf16bf16_grouped_wgrad_128_64_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum) {
  if (output_accum) {
    return bf16bf16bf16_grouped_wgrad_sm100_impl<
        128,
        64,
        128,
        2,
        1,
        1,
        true,
        false>(X, W, M_sizes, output);
  } else {
    return bf16bf16bf16_grouped_wgrad_sm100_impl<
        128,
        64,
        128,
        2,
        1,
        1,
        false,
        false>(X, W, M_sizes, output);
  }
}

} // namespace fbgemm_gpu
