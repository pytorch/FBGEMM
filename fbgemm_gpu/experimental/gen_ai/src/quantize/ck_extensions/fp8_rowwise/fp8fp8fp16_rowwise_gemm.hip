/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

#include "kernels/fp8fp8fp16_rowwise_kernel_manifest.h"

namespace fbgemm_gpu {
namespace {

using RowwiseKernel = std::function<
    at::Tensor(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)>;

RowwiseKernel rowwise_heuristic_dispatch(int M, int N, int K) {
  // Apply shape heuristics to find a suitable kernel implementation.

  //Fallback for irregular data types: some instances require K to be a multiple
  //of K Tile.
  //To-Do: Need a systemic solution for various restrictions from different
  //instances.
  if(!((N % 8 == 0) && (K % 16 == 0))) {
    return fp8fp8f16_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1;
  }

  if (M < 64 && N < 2048 && K < 2048) {
    // Kernel that generally works well on small shapes.
    return fp8fp8f16_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1;
  } else if (M < 64 && K < 2048) {
    // Kernel that works well for small batch size and small K.
    return fp8fp8f16_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1;
  } else if (M < 64 && N < 2048) {
    // Kernel that works well for small batch size and small N.
    return fp8fp8f16_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v1;
  } else if (M < 64 && N > 2048 && K > 2048) {
    // Kernel that works well for small M but larger N and K.
    return fp8fp8f16_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1;
  } else if (M < 64) {
    // Fallback to generic small batch kernel if we cant find a good match.
    return fp8fp8f16_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1;
  } else if (
      ((M < 512 && K < 8192) || (N <= 2048 && K <= 8192) ||
       (K <= 2048 && N <= 8192)) &&
      K >= 1024) {
    // Kernel that is optimized for larger batch sizes but otherwise small
    // tensors.
    return fp8fp8f16_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
  } else if (K < 1024) {
    // Special case for small K.
    return fp8fp8f16_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1;
  } else if (M < 1024) {
    // Kernel for generic medium batch sizes.
    return fp8fp8f16_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  } else if (M >= 1024 && N >= 1024 && K >= 1024) {
    // Kernel for very large gemm
    return fp8fp8f16_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
  } else {
    // Fallback large kernel.
    return fp8fp8f16_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
  }
}

at::Tensor f8f8_rowwise_wrapper(
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
    return at::zeros(out_sizes, XQ.options().dtype(at::kHalf));
  }

  // Prepare output tensor if needed.
  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    // Make sure the provided output has the proper shape and dtype.
    int Y_M = size_to_dim_(Y.dim() - 1, Y.sizes());
    TORCH_CHECK(Y_M == M && Y.sizes().vec().back() == N);
    TORCH_CHECK(Y.dtype() == at::kHalf);
  } else {
    Y = at::empty(out_sizes, XQ.options().dtype(at::kHalf));
  }

  RowwiseKernel rowwise_impl = rowwise_heuristic_dispatch(M, N, K);
  return rowwise_impl(XQ, WQ, x_scale, w_scale, Y);
}
} // namespace

at::Tensor f8f8f16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum) {
  // Invoke f8f8f16 rowwise without preallocated output.
  return f8f8_rowwise_wrapper(
      XQ, WQ, x_scale, w_scale, bias, use_fast_accum);
}

void f8f8f16_rowwise_out(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> bias,
    bool use_fast_accum) {
  // Invoke f8f8f16 rowwise with preallocated output.
  f8f8_rowwise_wrapper(
      XQ, WQ, x_scale, w_scale, bias, use_fast_accum, output);
}

} // namespace fbgemm_gpu
