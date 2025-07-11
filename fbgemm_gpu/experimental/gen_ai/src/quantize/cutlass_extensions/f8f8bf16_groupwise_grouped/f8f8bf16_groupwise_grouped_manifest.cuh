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

at::Tensor f8f8bf16_groupwise_grouped_128_128_128_1_2_1_9_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    at::Tensor M_sizes);

template <typename InputType>
using Kernel_f8f8bf16_groupwise_grouped = at::Tensor (*)(
    InputType,
    InputType,
    InputType,
    InputType,
    at::Tensor,
    at::Tensor);

template <typename InputType>
const std::
    unordered_map<std::string, Kernel_f8f8bf16_groupwise_grouped<InputType>>&
    get_f8f8bf16_groupwise_grouped_kernels() {
  static const std::
      unordered_map<std::string, Kernel_f8f8bf16_groupwise_grouped<InputType>>
          kernels = {
              {"f8f8bf16_groupwise_grouped_128_128_128_1_2_1_9_f",
               f8f8bf16_groupwise_grouped_128_128_128_1_2_1_9_f},
          };
  return kernels;
}

} // namespace fbgemm_gpu
