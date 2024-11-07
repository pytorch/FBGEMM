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

using RowwiseBatchedKernel = std::function<
    at::Tensor(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)>;

RowwiseBatchedKernel
rowwise_batched_heuristic_dispatch(int B, int M, int N, int K) {
  // Use shape heuristics to guess what the best kernel might be for the given
  // We divide cases based on the size of M to simplify sub-shape reasoning.
  // Start by handling cases with small M.
  if (M < 16) {
    // workload. For super small shapes, use specialized thin kernel.
    if (B <= 16 && (N <= 2048 && K >= 2048)) {
      return fp8_rowwise_batched_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_1x1_interwave_v2;
    }
    // Kernel for small m but larger other dims.
    if (B < 32 && K < 8192) {
      return fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2;
    }
    if (B < 32 && N < 8192) {
      return fp8_rowwise_batched_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_1x1_interwave_v1;
    }
    // Catch other small m cases.
    return fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1;
  }
  if (M < 32) {
    if (B < 8 && K < 8192) {
      return fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2;
    }
    if (B < 8 && N < 8192) {
      return fp8_rowwise_batched_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1;
    }
    // handle large shapes
    if (B < 8 && K > 8192 && N > 8192) {
      return fp8_rowwise_batched_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1;
    }
    // Fallback kernel that works well for most other gemms.
    return fp8_rowwise_batched_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2;
  }
  if (M < 64) {
    // Some small batch size but large N and K require special handling.
    if (B < 8) {
      if (K >= 20480) {
        return fp8_rowwise_batched_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2;
      }
      if (N >= 8192 && K >= 8192) {
        return fp8_rowwise_batched_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2;
      }
    }
    return fp8_rowwise_batched_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2;
  }
  if (M < 128) {
    if (K >= 20480) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
    }
    return fp8_rowwise_batched_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  if (M < 256) {
    if (N <= 8192 && K <= 8192) {
      return fp8_rowwise_batched_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
    }
    if (N >= 8192 && K >= 8192) {
      return fp8_rowwise_batched_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
    }
    return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v4;
  }
  if (M < 512) {
    if (N <= 8192 && K <= 5120) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
    }
    if (N <= 8192 && K <= 8192) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v4;
    }
    if (N < 20480 && K >= 20480) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
    }
    return fp8_rowwise_batched_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3;
  }
  if (M < 1024) {
    if (B < 8 && N <= 8192 && K <= 5120) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
    }
    if (B < 8 && N <= 8192 && K <= 8192) {
      return fp8_rowwise_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
    }
    if (N >= 20480 && K < 20480) {
      return fp8_rowwise_batched_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
    }
    return fp8_rowwise_batched_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3;
  }
  // Otherwise use super large kernel.
  return fp8_rowwise_batched_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
}

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

  // XQ: B x M x K
  // WQ: B x N x K
  // output: B x M x N
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

  RowwiseBatchedKernel selected_kernel =
      rowwise_batched_heuristic_dispatch(B, M, N, K);
  return selected_kernel(XQ, WQ, x_scale, w_scale, Y);
}

} // namespace fbgemm_gpu
