/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

// FP8 Rowwise batched Cutlass kernel dispatch.
template <typename InputDType, bool FastAccum, bool UseBias, typename BiasDType>
at::Tensor dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output);

at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt) {
  // Check datatypes.
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat && w_scale.dtype() == at::kFloat,
      "Scale tensors must be float32.");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias.value().dtype() == at::kFloat ||
            bias.value().dtype() == at::kBFloat16,
        "Bias type must be bfloat16 or float32 if provided.");
  }
  bool use_bias = bias.has_value();
  bool bf16_bias = use_bias && bias.value().dtype() == at::kBFloat16;

  // Templatize based on input dtype.
  bool use_e5m2 = XQ.dtype() == at::kFloat8_e5m2;

  if (use_bias) {
    if (bf16_bias) {
      if (use_fast_accum) {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e5m2_t,
              true,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e4m3_t,
              true,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      } else {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e5m2_t,
              false,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e4m3_t,
              false,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      }
    } else {
      if (use_fast_accum) {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e5m2_t,
              true,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e4m3_t,
              true,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      } else {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e5m2_t,
              false,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e4m3_t,
              false,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      }
    }
  } else {
    if (use_fast_accum) {
      if (use_e5m2) {
        return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
            cutlass::float_e5m2_t,
            true,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
            cutlass::float_e4m3_t,
            true,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      }
    } else {
      if (use_e5m2) {
        return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
            cutlass::float_e5m2_t,
            false,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
            cutlass::float_e4m3_t,
            false,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      }
    }
  }
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
