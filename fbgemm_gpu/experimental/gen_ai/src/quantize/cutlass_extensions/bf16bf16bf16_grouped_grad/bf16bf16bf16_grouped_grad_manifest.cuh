/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {

at::Tensor bf16bf16bf16_grouped_grad_128_16_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_16_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_32_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_32_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_64_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_64_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_64_128_2_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_64_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_2_2_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_4_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_256_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_256_128_1_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_256_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_256_128_2_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_256_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_256_128_4_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_16_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_16_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_16_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_32_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_32_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_32_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_64_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_64_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_64_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_128_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_128_128_1_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_128_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_128_128_2_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_128_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_128_128_4_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

// SM100
at::Tensor bf16bf16bf16_grouped_grad_128_32_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_64_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_128_128_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_32_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_64_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_128_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_grad_256_256_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes);

using Kernel_bf16bf16bf16_grouped_grad = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>);

const std::unordered_map<std::string, Kernel_bf16bf16bf16_grouped_grad>&
get_bf16bf16bf16_grouped_grad_kernels(int arch) {
  static const std::unordered_map<std::string, Kernel_bf16bf16bf16_grouped_grad>
      kernelsSM90 = {
          {"bf16bf16bf16_grouped_grad_128_16_128_1_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_16_128_1_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_16_128_4_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_16_128_4_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_32_128_1_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_32_128_1_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_32_128_4_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_32_128_4_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_64_128_1_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_64_128_1_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_64_128_2_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_64_128_2_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_64_128_2_2_1_9_f",
           bf16bf16bf16_grouped_grad_128_64_128_2_2_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_64_128_4_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_64_128_4_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_t",
           bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_t},
          {"bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_f",
           bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t",
           bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t},
          {"bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_t",
           bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_t},
          {"bf16bf16bf16_grouped_grad_128_128_128_2_2_1_9_t",
           bf16bf16bf16_grouped_grad_128_128_128_2_2_1_9_t},
          {"bf16bf16bf16_grouped_grad_128_128_128_4_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_128_128_4_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_128_128_4_1_1_9_t",
           bf16bf16bf16_grouped_grad_128_128_128_4_1_1_9_t},
          {"bf16bf16bf16_grouped_grad_128_256_128_1_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_256_128_1_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_256_128_1_1_1_9_t",
           bf16bf16bf16_grouped_grad_128_256_128_1_1_1_9_t},
          {"bf16bf16bf16_grouped_grad_128_256_128_2_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_256_128_2_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_256_128_2_1_1_9_t",
           bf16bf16bf16_grouped_grad_128_256_128_2_1_1_9_t},
          {"bf16bf16bf16_grouped_grad_128_256_128_4_1_1_9_f",
           bf16bf16bf16_grouped_grad_128_256_128_4_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_128_256_128_4_1_1_9_t",
           bf16bf16bf16_grouped_grad_128_256_128_4_1_1_9_t},
          {"bf16bf16bf16_grouped_grad_256_16_128_1_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_16_128_1_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_16_128_2_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_16_128_2_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_16_128_4_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_16_128_4_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_32_128_1_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_32_128_1_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_32_128_2_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_32_128_2_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_32_128_4_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_32_128_4_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_64_128_1_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_64_128_1_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_64_128_2_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_64_128_2_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_64_128_4_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_64_128_4_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_128_128_1_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_128_128_1_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_128_128_1_1_1_9_t",
           bf16bf16bf16_grouped_grad_256_128_128_1_1_1_9_t},
          {"bf16bf16bf16_grouped_grad_256_128_128_2_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_128_128_2_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_128_128_2_1_1_9_t",
           bf16bf16bf16_grouped_grad_256_128_128_2_1_1_9_t},
          {"bf16bf16bf16_grouped_grad_256_128_128_4_1_1_9_f",
           bf16bf16bf16_grouped_grad_256_128_128_4_1_1_9_f},
          {"bf16bf16bf16_grouped_grad_256_128_128_4_1_1_9_t",
           bf16bf16bf16_grouped_grad_256_128_128_4_1_1_9_t},
      };

  static const std::unordered_map<std::string, Kernel_bf16bf16bf16_grouped_grad>
      kernelsSM100 = {
          {"bf16bf16bf16_grouped_grad_256_32_128_2_1_1_10_f",
           bf16bf16bf16_grouped_grad_256_32_128_2_1_1_10_f},
          {"bf16bf16bf16_grouped_grad_256_64_128_2_1_1_10_f",
           bf16bf16bf16_grouped_grad_256_64_128_2_1_1_10_f},
          {"bf16bf16bf16_grouped_grad_256_128_128_2_1_1_10_f",
           bf16bf16bf16_grouped_grad_256_128_128_2_1_1_10_f},
          {"bf16bf16bf16_grouped_grad_256_256_128_2_1_1_10_f",
           bf16bf16bf16_grouped_grad_256_256_128_2_1_1_10_f},
          {"bf16bf16bf16_grouped_grad_128_32_128_2_1_1_10_f",
           bf16bf16bf16_grouped_grad_128_32_128_2_1_1_10_f},
          {"bf16bf16bf16_grouped_grad_128_64_128_2_1_1_10_f",
           bf16bf16bf16_grouped_grad_128_64_128_2_1_1_10_f},
          {"bf16bf16bf16_grouped_grad_128_128_128_2_1_1_10_f",
           bf16bf16bf16_grouped_grad_128_128_128_2_1_1_10_f},
      };
  if (arch == 10) {
    return kernelsSM100;
  } else {
    return kernelsSM90;
  }
}

} // namespace fbgemm_gpu
