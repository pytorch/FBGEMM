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

at::Tensor bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_32_128_1_4_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_32_128_2_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_32_128_2_4_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_64_128_1_4_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_64_128_2_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_64_128_4_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_2_2_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_2_4_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_4_2_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_256_32_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_256_64_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_256_128_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_256_256_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_32_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_64_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_4_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_32_128_1_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_256_64_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

at::Tensor bf16bf16bf16_grouped_wgrad_256_64_128_1_4_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum);

using Kernel_bf16bf16bf16_grouped_wgrad =
    at::Tensor (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool);

const std::unordered_map<std::string, Kernel_bf16bf16bf16_grouped_wgrad>&
get_bf16bf16bf16_grouped_wgrad_kernels(int arch) {
  static const std::
      unordered_map<std::string, Kernel_bf16bf16bf16_grouped_wgrad>
          kernelsSM90 = {
              {"bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_32_128_1_4_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_32_128_1_4_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_32_128_2_2_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_32_128_2_2_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_32_128_2_4_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_32_128_2_4_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_64_128_1_4_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_64_128_1_4_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_64_128_2_2_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_64_128_2_2_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_64_128_4_2_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_64_128_4_2_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t",
               bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t",
               bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t",
               bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_2_2_1_9_t",
               bf16bf16bf16_grouped_wgrad_128_128_128_2_2_1_9_t},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_2_4_1_9_t",
               bf16bf16bf16_grouped_wgrad_128_128_128_2_4_1_9_t},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_4_2_1_9_t",
               bf16bf16bf16_grouped_wgrad_128_128_128_4_2_1_9_t},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_t",
               bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_t},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_4_1_1_9_t",
               bf16bf16bf16_grouped_wgrad_128_128_128_4_1_1_9_t},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t",
               bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t},
              {"bf16bf16bf16_grouped_wgrad_128_32_128_1_2_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_32_128_1_2_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f",
               bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_256_64_128_1_1_1_9_f",
               bf16bf16bf16_grouped_wgrad_256_64_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f",
               bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f},
              {"bf16bf16bf16_grouped_wgrad_256_64_128_1_4_1_9_f",
               bf16bf16bf16_grouped_wgrad_256_64_128_1_4_1_9_f},
          };
  static const std::
      unordered_map<std::string, Kernel_bf16bf16bf16_grouped_wgrad>
          kernelsSM100 = {
              {"bf16bf16bf16_grouped_wgrad_256_32_128_2_1_1_10_f",
               bf16bf16bf16_grouped_wgrad_256_32_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_wgrad_256_64_128_2_1_1_10_f",
               bf16bf16bf16_grouped_wgrad_256_64_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_wgrad_256_128_128_2_1_1_10_f",
               bf16bf16bf16_grouped_wgrad_256_128_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_wgrad_256_256_128_2_1_1_10_f",
               bf16bf16bf16_grouped_wgrad_256_256_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_wgrad_128_32_128_2_1_1_10_f",
               bf16bf16bf16_grouped_wgrad_128_32_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_wgrad_128_64_128_2_1_1_10_f",
               bf16bf16bf16_grouped_wgrad_128_64_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_10_f",
               bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_10_f},
          };
  if (arch == 10) {
    return kernelsSM100;
  } else {
    return kernelsSM90;
  }
}

} // namespace fbgemm_gpu
