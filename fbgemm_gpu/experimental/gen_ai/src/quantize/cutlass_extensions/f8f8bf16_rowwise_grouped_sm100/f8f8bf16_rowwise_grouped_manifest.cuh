/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace fbgemm_gpu {

at::Tensor f8f8bf16_rowwise_grouped_128_32_128_2_1_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_128_32_128_2_1_1_10_f(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_128_64_128_2_1_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_128_64_128_2_1_1_10_f(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_128_128_128_2_1_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_128_128_128_2_1_1_10_f(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_128_256_128_2_1_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_128_256_128_2_1_1_10_f(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_256_32_128_2_1_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_256_32_128_2_1_1_10_f(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_256_64_128_2_1_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_256_64_128_2_1_1_10_f(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_256_128_128_2_1_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_256_128_128_2_1_1_10_f(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_256_256_128_2_1_1_10_f(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_256_256_128_2_1_1_10_f(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

template <typename InputType>
const std::
    unordered_map<std::string, Kernel_f8f8bf16_rowwise_grouped<InputType>>&
    get_f8f8bf16_rowwise_grouped_kernels_sm100() {
  static const std::
      unordered_map<std::string, Kernel_f8f8bf16_rowwise_grouped<InputType>>
          kernels = {
              {"f8f8bf16_rowwise_grouped_128_32_128_2_1_1_10_f",
               f8f8bf16_rowwise_grouped_128_32_128_2_1_1_10_f},
              {"f8f8bf16_rowwise_grouped_128_64_128_2_1_1_10_f",
               f8f8bf16_rowwise_grouped_128_64_128_2_1_1_10_f},
              {"f8f8bf16_rowwise_grouped_128_128_128_2_1_1_10_f",
               f8f8bf16_rowwise_grouped_128_128_128_2_1_1_10_f},
              {"f8f8bf16_rowwise_grouped_128_256_128_2_1_1_10_f",
               f8f8bf16_rowwise_grouped_128_256_128_2_1_1_10_f},
              {"f8f8bf16_rowwise_grouped_256_32_128_2_1_1_10_f",
               f8f8bf16_rowwise_grouped_256_32_128_2_1_1_10_f},
              {"f8f8bf16_rowwise_grouped_256_64_128_2_1_1_10_f",
               f8f8bf16_rowwise_grouped_256_64_128_2_1_1_10_f},
              {"f8f8bf16_rowwise_grouped_256_128_128_2_1_1_10_f",
               f8f8bf16_rowwise_grouped_256_128_128_2_1_1_10_f},
              {"f8f8bf16_rowwise_grouped_256_256_128_2_1_1_10_f",
               f8f8bf16_rowwise_grouped_256_256_128_2_1_1_10_f},
          };
  return kernels;
}

} // namespace fbgemm_gpu
