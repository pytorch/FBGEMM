/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f4f4bf16_common.cuh"

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor f4f4bf16_128_256_2_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale = std::nullopt) {
  // Dispatch this kernel to the correct underlying implementation.
  return _f4f4bf16<
      cutlass::nv_float4_t<cutlass::float_e2m1_t>,
      128,
      256,
      2,
      1,
      1>(XQ, WQ, x_scale, w_scale, global_scale);
}

#endif

} // namespace fbgemm_gpu
