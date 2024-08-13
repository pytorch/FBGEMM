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
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#if defined(USE_ROCM)

#include "kernels/fp8_rowwise_kernel_manifest.h"

namespace fbgemm_gpu {

using RowwiseKernel = std::function<
    at::Tensor(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)>;

// Define a custom hash function for std::tuple<int, int, int>
struct IntTupleHash {
  size_t operator()(const std::tuple<int, int, int>& t) const {
    auto hash1 = std::hash<int>{}(std::get<0>(t));
    auto hash2 = std::hash<int>{}(std::get<1>(t));
    auto hash3 = std::hash<int>{}(std::get<2>(t));
    return hash1 ^ hash2 ^ hash3;
  }
};

// For certain high priority shapes, we directly map to the best kernel rather
// than use heuristics.
static const std::unordered_map<
    std::tuple<int, int, int>,
    RowwiseKernel,
    IntTupleHash>
    rowwise_lookup_dispatch = {
        // LLama 70B Decode shapes.
        // Support for decode across batch sizes for [1280, 8192]
        {{16, 1280, 8192},
         fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},
        {{32, 1280, 8192},
         fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2},
        {{64, 1280, 8192},
         fp8_rowwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2},
        {{128, 1280, 8192},
         fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},
        // Support for decode across batch sizes for [8192, 1024]
        {{16, 8192, 1024},
         fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2},
        {{32, 8192, 1024},
         fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2},
        {{64, 8192, 1024},
         fp8_rowwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2},
        {{128, 8192, 1024},
         fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        // Support for decode across batch sizes for [7168, 8192]
        {{16, 7168, 8192},
         fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},
        {{32, 7168, 8192},
         fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2},
        {{64, 7168, 8192},
         fp8_rowwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2},
        {{128, 7168, 8192},
         fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        // Support for decode across batch sizes for [8192, 3584]
        {{16, 8192, 3584},
         fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},
        {{32, 8192, 3584},
         fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2},
        {{64, 8192, 3584},
         fp8_rowwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2},
        {{128, 8192, 3584},
         fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        // Llama 405B Decode Shapes.
        // Support for decode across batch sizes for [13312, 6656].
        {{16, 13312, 6656},
         fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1},
        {{32, 13312, 6656},
         fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2},
        {{64, 13312, 6656},
         fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        {{128, 13312, 6656},
         fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        // Support for decode across batch sizes for [13312, 16384].
        {{16, 13312, 16384},
         fp8_rowwise_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2},
        {{32, 13312, 16384},
         fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2},
        {{64, 13312, 16384},
         fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        {{128, 13312, 16384},
         fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        // Support for decode across batch sizes for [16384, 6656].
        {{16, 16384, 6656},
         fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1},
        {{32, 16384, 6656},
         fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2},
        {{64, 16384, 6656},
         fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        {{128, 16384, 6656},
         fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        // Support for decode across batch sizes for [16384, 16384].
        {{16, 16384, 16384},
         fp8_rowwise_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2},
        {{32, 16384, 16384},
         fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2},
        {{64, 16384, 16384},
         fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        {{128, 16384, 16384},
         fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        // EMU 1.6 Shapes.
        {{1536, 3584, 3584},
         fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1},
        {{8192, 9728, 3584},
         fp8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4},
        {{8192, 3584, 9728},
         fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5},
        {{8192, 3584, 3584},
         fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1},
        // Pro Shapes.
        {{32768, 128, 8192},
         fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        {{32768, 8192, 1024},
         fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},
        {{32768, 8192, 3072},
         fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
        {{32768, 3072, 8192},
         fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3},
        {{32768, 1024, 8192},
         fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3}};

RowwiseKernel rowwise_heuristic_dispatch(int M, int N, int K) {
  // Apply shape heuristics to find a suitable kernel implementation.

  if (M < 64 && N < 2048 && K < 2048) {
    // Kernel that generally works well on small shapes.
    return fp8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2;
  } else if (M < 64 && K < 2048) {
    // Kernel that works well for small batch size and small K.
    return fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2;
  } else if (M < 64 && N < 2048) {
    // Kernel that works well for small batch size and small N.
    return fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2;
  } else if (M < 64 && N > 2048 && K > 2048) {
    // Kernel that works well for small M but larger N and K.
    return fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1;
  } else if (M < 64) {
    // Fallback to generic small batch kernel if we cant find a good match.
    return fp8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2;
  } else if ((M < 512 && K < 8192) || (N <= 2048 && K <= 8192) || (K <= 2048 && N <= 8192)) {
    // Kernel that is optimized for larger batch sizes but otherwise small
    // tensors.
    return fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5;
  } else if (M < 1024) {
    // Kernel for generic medium batch sizes.
    return fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  } else if (M > 4096 && N > 4096 && K > 2048) {
    // Kernel for very large gemm
    return fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
  } else {
    // Fallback large kernel.
    return fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
  }
}

RowwiseKernel rowwise_dispatch(int M, int N, int K) {
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.
  int padded_m = M;
  if (M <= 16) {
    padded_m = 16;
  } else if (M <= 32) {
    padded_m = 32;
  } else if (M <= 64) {
    padded_m = 64;
  } else if (M <= 128) {
    padded_m = 128;
  }
  // First check if this shape is available in the direct lookup.
  auto it = rowwise_lookup_dispatch.find({padded_m, N, K});
  // If we found an optimal kernel, use it.
  if (it != rowwise_lookup_dispatch.end()) {
    return it->second;
  }
  // Otherwise, use heuristics.
  return rowwise_heuristic_dispatch(M, N, K);
}

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::Tensor> output = c10::nullopt) {
  // Check that input datatypes are valid.
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
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);

  // Prepare output tensor if needed.
  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    // Make sure the provided output has the proper shape and dtype.
    int Y_M = size_to_dim_(Y.dim() - 1, Y.sizes());
    TORCH_CHECK(Y_M == M && Y.sizes().vec().back() == N);
    TORCH_CHECK(Y.dtype() == at::kBFloat16);
  } else {
    // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
    // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
    auto out_sizes = XQ.sizes().vec();
    out_sizes.back() = N;
    Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  RowwiseKernel rowwise_impl = rowwise_dispatch(M, N, K);
  return rowwise_impl(XQ, WQ, x_scale, w_scale, Y);
}

} // namespace fbgemm_gpu

#endif // defined(USE_ROCM)
