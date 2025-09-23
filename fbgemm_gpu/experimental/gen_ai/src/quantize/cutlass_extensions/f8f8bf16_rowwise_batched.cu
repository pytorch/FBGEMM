/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include "f8f8bf16_rowwise_batched/f8f8bf16_rowwise_batched_manifest.cuh"

#include "fbgemm_gpu/quantize/utils.h"
#include "fbgemm_gpu/quantize/utils_gpu.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

constexpr int kNumSMsForH100 = 132;
constexpr int kNumSMsForGB200 = 600;

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

// FP8 Rowwise batched Cutlass kernel dispatch.
at::Tensor dispatch_fp8_rowwise_batched_kernel(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt) {
  TORCH_CHECK(
      (XQ.dim() == 3 && WQ.dim() == 3),
      "FP8 rowwise batched GEMM only supports 3D inputs");
  int M, N;
  M = XQ.size(1);
  N = WQ.size(1);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(XQ.dtype() == at::kFloat8_e4m3fn, "XQ must be FP8 e4m3fn");
  TORCH_CHECK(WQ.dtype() == at::kFloat8_e4m3fn, "WQ must be FP8 e4m3fn");
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat && w_scale.dtype() == at::kFloat,
      "Scale tensors must be float32.");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias.value().dtype() == at::kFloat,
        "Bias type must be float32 if provided.");
  }

  const int arch = getDeviceArch();
  if (arch == 10) {
    if ((M * N <= 4096 * 4096) || (N % 256 > 0 && M % 256 == 0) ||
        (M % 256 > 0 && N % 256 > 0) || M >= 1024 && N >= 1024) {
      if ((ceildiv(M, 64 * 2) * ceildiv(N, 128 * 1)) <=
          kNumSMsForGB200 /
              cute::size(
                  cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>{})) {
        return f8f8bf16_rowwise_batched_64_128_128_2_1_1_10_f(
            XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return f8f8bf16_rowwise_batched_128_128_128_2_1_1_10_t(
            XQ, WQ, x_scale, w_scale, bias, output);
      }
    } else {
      if ((ceildiv(M, 64 * 2) * ceildiv(N, 128 * 1)) <=
          kNumSMsForGB200 /
              cute::size(
                  cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>{})) {
        return f8f8bf16_rowwise_batched_64_128_128_1_2_1_10_f(
            XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return f8f8bf16_rowwise_batched_128_128_128_1_2_1_10_t(
            XQ, WQ, x_scale, w_scale, bias, output);
      }
    }
  } else {
    if ((M * N <= 4096 * 4096) || (N % 256 > 0 && M % 256 == 0) ||
        (M % 256 > 0 && N % 256 > 0) || M >= 1024 && N >= 1024) {
      if ((ceildiv(M, 64 * 2) * ceildiv(N, 128 * 1)) <=
          kNumSMsForH100 /
              cute::size(
                  cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>{})) {
        return f8f8bf16_rowwise_batched_64_128_128_2_1_1_9_f(
            XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return f8f8bf16_rowwise_batched_128_128_128_2_1_1_9_t(
            XQ, WQ, x_scale, w_scale, bias, output);
      }
    } else {
      if ((ceildiv(M, 64 * 2) * ceildiv(N, 128 * 1)) <=
          kNumSMsForGB200 /
              cute::size(
                  cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>{})) {
        return f8f8bf16_rowwise_batched_64_128_128_1_2_1_9_f(
            XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return f8f8bf16_rowwise_batched_128_128_128_1_2_1_9_t(
            XQ, WQ, x_scale, w_scale, bias, output);
      }
    }
  }
}

at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt) {
  TORCH_CHECK(
      use_fast_accum, "f8f8bf16_rowwise_batched only supports fast_accum=True");
  return dispatch_fp8_rowwise_batched_kernel(
      XQ, WQ, x_scale, w_scale, bias, output);
}

#else

at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
