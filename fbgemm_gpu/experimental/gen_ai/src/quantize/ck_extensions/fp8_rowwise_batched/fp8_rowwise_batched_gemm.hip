/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <torch/torch.h>

#include "kernels/fp8_rowwise_batched_kernel_manifest.h"

namespace fbgemm_gpu {

at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::Tensor> output = std::nullopt) {
  // Check that input datatypes are valid.
  TORCH_CHECK(
      XQ.dim() == 3 && WQ.dim() == 3,
      "Inputs must have 3 dimensions, with the first being batch.");
  TORCH_CHECK(
      (XQ.dtype() == at::kFloat8_e4m3fnuz) &&
          (WQ.dtype() == at::kFloat8_e4m3fnuz),
      "Inputs must be type float8_e4m3fnuz.");
  TORCH_CHECK(
      (x_scale.dtype() == at::kFloat) && (w_scale.dtype() == at::kFloat),
      "Scales must be float32.");
  TORCH_CHECK(use_fast_accum, "AMD does not support disabling use_fast_accum.");

  // Check inputs are in expected format.
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  // XQ: M x K
  // WQ: N x K
  // output: M x N
  int B = XQ.size(0);
  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = WQ.size(2);

  // Prepare output tensor if needed.
  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    // Make sure the provided output has the proper shape and dtype.
    TORCH_CHECK(Y.dim() == 3, "Output tensor must have three dimensions.");
    int Y_B = Y.size(0);
    int Y_M = Y.size(1);
    int Y_N = Y.size(2);
    TORCH_CHECK(Y_B == B && Y_M == M && Y_N == N);
    TORCH_CHECK(Y.dtype() == at::kBFloat16);
  } else {
    auto out_sizes = XQ.sizes().vec();
    out_sizes.back() = N;
    Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  return fp8_rowwise_batched_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
      XQ, WQ, x_scale, w_scale, Y);
}

} // namespace fbgemm_gpu
