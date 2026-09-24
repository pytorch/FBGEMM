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

at::Tensor f8f8bf16_groupwise_64_128_128_1_1_1_9_t_f_4(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_64_128_128_1_1_1_9_t_f_8(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_64_128_128_1_1_1_9_t_f_0(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_16_128_1_1_1_9_f_t_0(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_16_128_1_1_1_9_f_t_8(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_16_128_1_2_1_9_f_t_0(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_16_128_4_1_1_9_f_t_0(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_16_128_4_1_1_9_f_t_8(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_32_128_1_1_1_9_f_t_0(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_32_128_1_1_1_9_f_t_8(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_32_128_4_1_1_9_f_t_8(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_64_128_1_1_1_9_f_t_8(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_64_128_4_1_1_9_t_t_4(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_128_128_1_1_1_9_f_f_0(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_128_128_1_1_1_9_f_f_4(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_128_128_1_1_1_9_f_f_8(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_128_128_1_2_1_9_f_f_0(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_128_128_1_2_1_9_f_f_8(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_128_128_2_1_1_9_f_f_0(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

using Kernel_f8f8bf16_groupwise =
    at::Tensor (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor);

const std::unordered_map<std::string, Kernel_f8f8bf16_groupwise>&
get_f8f8bf16_groupwise_kernels() {
  static const std::unordered_map<std::string, Kernel_f8f8bf16_groupwise>
      kernels = {
          {"f8f8bf16_groupwise_64_128_128_1_1_1_9_t_f_4",
           f8f8bf16_groupwise_64_128_128_1_1_1_9_t_f_4},
          {"f8f8bf16_groupwise_64_128_128_1_1_1_9_t_f_8",
           f8f8bf16_groupwise_64_128_128_1_1_1_9_t_f_8},
          {"f8f8bf16_groupwise_128_16_128_1_1_1_9_f_t_0",
           f8f8bf16_groupwise_128_16_128_1_1_1_9_f_t_0},
          {"f8f8bf16_groupwise_128_16_128_1_1_1_9_f_t_8",
           f8f8bf16_groupwise_128_16_128_1_1_1_9_f_t_8},
          {"f8f8bf16_groupwise_64_128_128_1_1_1_9_t_f_0",
           f8f8bf16_groupwise_64_128_128_1_1_1_9_t_f_0},
          {"f8f8bf16_groupwise_128_16_128_1_2_1_9_f_t_0",
           f8f8bf16_groupwise_128_16_128_1_2_1_9_f_t_0},
          {"f8f8bf16_groupwise_128_16_128_4_1_1_9_f_t_0",
           f8f8bf16_groupwise_128_16_128_4_1_1_9_f_t_0},
          {"f8f8bf16_groupwise_128_16_128_4_1_1_9_f_t_8",
           f8f8bf16_groupwise_128_16_128_4_1_1_9_f_t_8},
          {"f8f8bf16_groupwise_128_32_128_1_1_1_9_f_t_0",
           f8f8bf16_groupwise_128_32_128_1_1_1_9_f_t_0},
          {"f8f8bf16_groupwise_128_32_128_1_1_1_9_f_t_8",
           f8f8bf16_groupwise_128_32_128_1_1_1_9_f_t_8},
          {"f8f8bf16_groupwise_128_32_128_4_1_1_9_f_t_8",
           f8f8bf16_groupwise_128_32_128_4_1_1_9_f_t_8},
          {"f8f8bf16_groupwise_128_64_128_1_1_1_9_f_t_8",
           f8f8bf16_groupwise_128_64_128_1_1_1_9_f_t_8},
          {"f8f8bf16_groupwise_128_64_128_4_1_1_9_t_t_4",
           f8f8bf16_groupwise_128_64_128_4_1_1_9_t_t_4},
          {"f8f8bf16_groupwise_128_128_128_1_1_1_9_f_f_0",
           f8f8bf16_groupwise_128_128_128_1_1_1_9_f_f_0},
          {"f8f8bf16_groupwise_128_128_128_1_1_1_9_f_f_4",
           f8f8bf16_groupwise_128_128_128_1_1_1_9_f_f_4},
          {"f8f8bf16_groupwise_128_128_128_1_1_1_9_f_f_8",
           f8f8bf16_groupwise_128_128_128_1_1_1_9_f_f_8},
          {"f8f8bf16_groupwise_128_128_128_1_2_1_9_f_f_0",
           f8f8bf16_groupwise_128_128_128_1_2_1_9_f_f_0},
          {"f8f8bf16_groupwise_128_128_128_1_2_1_9_f_f_8",
           f8f8bf16_groupwise_128_128_128_1_2_1_9_f_f_8},
          {"f8f8bf16_groupwise_128_128_128_2_1_1_9_f_f_0",
           f8f8bf16_groupwise_128_128_128_2_1_1_9_f_f_0},
      };
  return kernels;
}

} // namespace fbgemm_gpu
