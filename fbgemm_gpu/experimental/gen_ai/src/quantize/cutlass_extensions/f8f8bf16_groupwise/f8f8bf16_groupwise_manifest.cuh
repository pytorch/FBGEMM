/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace fbgemm_gpu {

at::Tensor f8f8bf16_groupwise_128_128_128_1_2_1_9_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_128_16_128_1_1_1_9_t(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

using Kernel_f8f8bf16_groupwise =
    at::Tensor (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor);

inline const std::unordered_map<std::string, Kernel_f8f8bf16_groupwise>&
get_f8f8bf16_groupwise_kernels(int arch) {
  static const std::unordered_map<std::string, Kernel_f8f8bf16_groupwise>
      kernelsSM90 = {};
  static const std::unordered_map<std::string, Kernel_f8f8bf16_groupwise>
      kernelsSM100 = {};
  if (arch == 10) {
    return kernelsSM100;
  } else {
    return kernelsSM90;
  }
}

} // namespace fbgemm_gpu
