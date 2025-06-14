/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace fbgemm_gpu {

at::Tensor f8f8bf16_rowwise_64_16_128_1_1_1_9_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_64_32_128_2_1_1_9_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_64_64_128_2_1_1_9_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_64_128_128_1_1_1_9_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_64_256_128_1_1_1_9_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_128_256_128_2_1_1_9_f_t(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_64_256_128_2_1_1_9_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_128_128_128_2_1_1_9_t_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_128_256_128_4_4_1_9_f_t(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_128_32_128_1_1_1_10_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_128_64_128_1_1_1_10_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_128_128_128_2_2_1_10_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_128_256_128_1_2_1_10_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor f8f8bf16_rowwise_128_256_128_2_1_1_10_f_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);

using Kernel_f8f8bf16_rowwise = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    bool,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>);

inline const std::unordered_map<std::string, Kernel_f8f8bf16_rowwise>&
get_f8f8bf16_rowwise_kernels(int arch) {
  static const std::unordered_map<std::string, Kernel_f8f8bf16_rowwise>
      kernelsSM90 = {};
  static const std::unordered_map<std::string, Kernel_f8f8bf16_rowwise>
      kernelsSM100 = {};
  if (arch == 10) {
    return kernelsSM100;
  } else {
    return kernelsSM90;
  }
}

} // namespace fbgemm_gpu
