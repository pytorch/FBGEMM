/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor mx8mx8bf16_grouped_128_64_256_1_1_1(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor mx8mx8bf16_grouped_128_128_256_1_1_1(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor mx8mx8bf16_grouped_256_64_256_2_1_1(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor mx8mx8bf16_grouped_256_128_256_2_1_1(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor mx8mx8bf16_grouped_256_256_256_2_1_1(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> starting_row_after_padding);

template <typename InputType>
using Kernel_mx8mx8bf16_grouped = at::Tensor (*)(
    InputType,
    InputType,
    InputType,
    InputType,
    at::Tensor,
    int64_t,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>);

template <typename InputType>
const std::unordered_map<std::string, Kernel_mx8mx8bf16_grouped<InputType>>&
get_mx8mx8bf16_grouped_kernels() {
  static const std::
      unordered_map<std::string, Kernel_mx8mx8bf16_grouped<InputType>>
          kernels = {
              {"mx8mx8bf16_grouped_128_64_256_1_1_1",
               mx8mx8bf16_grouped_128_64_256_1_1_1},
              {"mx8mx8bf16_grouped_128_128_256_1_1_1",
               mx8mx8bf16_grouped_128_128_256_1_1_1},
              {"mx8mx8bf16_grouped_256_64_256_2_1_1",
               mx8mx8bf16_grouped_256_64_256_2_1_1},
              {"mx8mx8bf16_grouped_256_128_256_2_1_1",
               mx8mx8bf16_grouped_256_128_256_2_1_1},
              {"mx8mx8bf16_grouped_256_256_256_2_1_1",
               mx8mx8bf16_grouped_256_256_256_2_1_1},
          };
  return kernels;
}

#endif
} // namespace fbgemm_gpu
