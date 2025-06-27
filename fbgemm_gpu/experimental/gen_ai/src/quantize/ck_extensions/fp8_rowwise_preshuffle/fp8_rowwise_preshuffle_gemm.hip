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

#include "kernels/fp8_rowwise_preshuffle_kernel_manifest.h"

namespace fbgemm_gpu {

using RowwiseKernel = std::function<
    at::Tensor(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)>;

RowwiseKernel rowwise_preshuffle_heuristic_dispatch(int M, int N, int K) {
  // Apply shape heuristics to find a suitable kernel implementation.
  if (M <= 16) {
    if (K <= 1024) {
      return fp8_rowwise_preshuffle_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1;
    } else if (N <= 1024) {
      return fp8_rowwise_preshuffle_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2;
    } else {
      return fp8_rowwise_preshuffle_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1;
    }
  } else if (M <= 32) {
    if (K <= 1024) {
      return fp8_rowwise_preshuffle_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1;
    } else if (N <= 1024) {
      return fp8_rowwise_preshuffle_128x16x32x512_16x16_1x1_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1;
    } else {
      return fp8_rowwise_preshuffle_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v2;
    }
  } else if (M <= 64) {
    if (K <= 1024) {
      return fp8_rowwise_preshuffle_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1;
    } else if (N <= 1024) {
      return fp8_rowwise_preshuffle_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v2;
    } else if (N <= 5120) {
      return fp8_rowwise_preshuffle_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1;
    } else {
      return fp8_rowwise_preshuffle_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1;
    }
  } else if (M <= 128) {
    if (K <= 1024) {
      return fp8_rowwise_preshuffle_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1;
    } else if (N <= 1024) {
      return fp8_rowwise_preshuffle_256x32x64x512_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1;
    } else {
      return fp8_rowwise_preshuffle_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v1;
    }
  } else if (M <= 512) {
    if (K <= 1024) {
      return fp8_rowwise_preshuffle_256x16x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1;
    } else if (N <= 1024) {
      return fp8_rowwise_preshuffle_256x64x64x512_32x32_1x1_32x8x1_32x8x1_1x32x1x8_8x8x1_1x1_intrawave_v1;
    } else {
      return fp8_rowwise_preshuffle_256x128x128x256_16x16_8x2_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3;
    }
  } else if (M <= 1024) {
    if (N <= 5120) {
      return fp8_rowwise_preshuffle_256x128x128x256_16x16_8x2_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3;
    } else {
      return fp8_rowwise_preshuffle_256x128x256x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3;
    }
  } else if (M <= 2048) {
    if (N <= 4096 || K <= 1024) {
      return fp8_rowwise_preshuffle_256x128x256x128_16x16_8x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3;
    } else {
      return fp8_rowwise_preshuffle_256x160x256x128_16x16_10x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3;
    }
  } else if (M <= 4096) {
    if (N <= 4096 || K <= 1024) {
      return fp8_rowwise_preshuffle_256x160x256x128_16x16_10x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3;
    } else {
      return fp8_rowwise_preshuffle_256x224x256x128_16x16_14x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3;
    }
  } else {
    return fp8_rowwise_preshuffle_256x224x256x128_16x16_14x4_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3;
  }
}

template <at::ScalarType OUTPUT_DTYPE>
at::Tensor f8f8_rowwise_preshuffle_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::Tensor> output = std::nullopt) {
  // Check that input datatypes are valid.
  TORCH_CHECK(
      (XQ.dtype() == at::kFloat8_e4m3fnuz) &&
          (WQ.dtype() == at::kFloat8_e4m3fnuz),
      "Inputs must be type float8_e4m3fnuz.");
  TORCH_CHECK(
      (x_scale.dtype() == at::kFloat) && (w_scale.dtype() == at::kFloat),
      "Scales must be float32.");
  TORCH_CHECK(use_fast_accum, "AMD does not support disabling use_fast_accum.");
  TORCH_CHECK(!bias.has_value(), "AMD does not support fused bias.");

  // Check inputs are in expected format.
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  // XQ: M x K
  // WQ: N x K
  // output: M x N
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  // Compute target output sizes.
  auto out_sizes = XQ.sizes().vec();
  out_sizes.back() = N;
  // Handle case where an input dimension is zero.
  if (M == 0 || N == 0 || K == 0) {
    // Return a tensor of zeros to handle case where K is 0.
    return at::zeros(out_sizes, XQ.options().dtype(OUTPUT_DTYPE));
  }

  // Prepare output tensor if needed.
  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    // Make sure the provided output has the proper shape and dtype.
    int Y_M = size_to_dim_(Y.dim() - 1, Y.sizes());
    TORCH_CHECK(Y_M == M && Y.sizes().vec().back() == N);
    TORCH_CHECK(Y.dtype() == OUTPUT_DTYPE);
  } else {
    Y = at::empty(out_sizes, XQ.options().dtype(OUTPUT_DTYPE));
  }

  RowwiseKernel heuristic_kernel = rowwise_preshuffle_heuristic_dispatch(M, N, K);

  return heuristic_kernel(XQ, WQ, x_scale, w_scale, Y);
}

at::Tensor f8f8bf16_rowwise_preshuffle(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum) {
  // Invoke f8f8bf16 rowwise without preallocated output.
  return f8f8_rowwise_preshuffle_wrapper<at::kBFloat16>(
      XQ, WQ, x_scale, w_scale, bias, use_fast_accum);
}

at::Tensor f8f8f16_rowwise_preshuffle(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum) {
  // Invoke f8f8bf16 rowwise without preallocated output.
  return f8f8_rowwise_preshuffle_wrapper<at::kHalf>(
      XQ, WQ, x_scale, w_scale, bias, use_fast_accum);
}

void f8f8bf16_rowwise_preshuffle_out(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> bias,
    bool use_fast_accum) {
  // Invoke f8f8bf16 rowwise with preallocated output.
  f8f8_rowwise_preshuffle_wrapper<at::kBFloat16>(
      XQ, WQ, x_scale, w_scale, bias, use_fast_accum, output);
}

} // namespace fbgemm_gpu
