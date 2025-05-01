/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
// clang-format on

#include "f8f8bf16_rowwise/f8f8bf16_rowwise_manifest.cuh"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

// FP8 Rowwise Cutlass kernel dispatch.
at::Tensor dispatch_fp8_rowwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt) {
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = size_to_dim_(WQ.dim() - 1, WQ.sizes());
  int K = XQ.size(-1);

  // Use shape heuristics to dispatch to optimized kernel configuration.
  if (M <= 16) {
    return f8f8bf16_rowwise_64_16_128_1_1_1_f_f(
        XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
  } else if (M <= 32) {
    if (N <= 4096) {
      return f8f8bf16_rowwise_64_16_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else {
      return f8f8bf16_rowwise_64_32_128_2_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    }
  } else if (M <= 64) {
    if (N <= 2048) {
      return f8f8bf16_rowwise_64_16_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (N <= 4096) {
      return f8f8bf16_rowwise_64_32_128_2_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else {
      return f8f8bf16_rowwise_64_64_128_2_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    }
  } else if (M <= 128) {
    if (N <= 1024) {
      return f8f8bf16_rowwise_64_16_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (N <= 2048) {
      return f8f8bf16_rowwise_64_32_128_2_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (N <= 4096) {
      return f8f8bf16_rowwise_64_64_128_2_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else {
      return f8f8bf16_rowwise_64_128_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    }
  } else if (M <= 256) {
    if (N <= 1024) {
      return f8f8bf16_rowwise_64_32_128_2_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (N <= 2048) {
      return f8f8bf16_rowwise_64_64_128_2_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (N <= 4096) {
      return f8f8bf16_rowwise_64_128_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else {
      return f8f8bf16_rowwise_64_256_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    }
  } else if (M <= 512) {
    if (N <= 1024) {
      return f8f8bf16_rowwise_64_64_128_2_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (N <= 2048) {
      return f8f8bf16_rowwise_64_128_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (N <= 4096 || use_fast_accum == false) {
      return f8f8bf16_rowwise_64_256_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else {
      return f8f8bf16_rowwise_128_256_128_2_1_1_f_t(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    }
  } else if (M <= 1024) {
    if (N <= 1024) {
      return f8f8bf16_rowwise_64_128_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (N <= 2048 || use_fast_accum == false) {
      return f8f8bf16_rowwise_64_256_128_1_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else {
      return f8f8bf16_rowwise_128_256_128_2_1_1_f_t(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    }
  } else {
    if (M <= 2048 && N <= 1024) {
      return f8f8bf16_rowwise_64_256_128_2_1_1_f_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (K <= 4096 || use_fast_accum == false) {
      return f8f8bf16_rowwise_128_128_128_2_1_1_t_f(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else if (M > 8192 && N > 8192) {
      return f8f8bf16_rowwise_128_256_128_4_4_1_f_t(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else {
      return f8f8bf16_rowwise_128_256_128_2_1_1_f_t(
          XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    }
  }
}

void f8f8bf16_rowwise_out(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true) {
  // Invoke rowwise kernel with output argument.
  dispatch_fp8_rowwise_kernel(
      XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
}

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true) {
  // Invoke and return rowwise kernel without output argument.
  return dispatch_fp8_rowwise_kernel(
      XQ, WQ, x_scale, w_scale, use_fast_accum, bias);
}

#else

void f8f8bf16_rowwise_out(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
#endif

} // namespace fbgemm_gpu
