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

at::Tensor f4f4bf16_grouped_128_64_256_1_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_128_64_256_1_1_1_f(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_128_64_256_1_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_128_64_256_1_1_1_t(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_128_128_256_1_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_128_128_256_1_1_1_f(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_128_128_256_1_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_128_128_256_1_1_1_t(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_64_256_2_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_64_256_2_1_1_f(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_64_256_2_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_64_256_2_1_1_t(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_128_256_2_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_128_256_2_1_1_f(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_128_256_2_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_128_256_2_1_1_t(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_256_256_2_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_256_256_2_1_1_f(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_256_256_2_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

at::Tensor f4f4bf16_grouped_256_256_256_2_1_1_t(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor output,
    int64_t G,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::TensorList> global_scale,
    std::optional<at::Tensor> starting_row_after_padding);

template <typename InputType>
using Kernel_f4f4bf16_grouped = at::Tensor (*)(
    InputType,
    InputType,
    InputType,
    InputType,
    at::Tensor,
    int64_t,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<InputType>,
    std::optional<at::Tensor>);

template <typename InputType>
const std::unordered_map<std::string, Kernel_f4f4bf16_grouped<InputType>>&
get_f4f4bf16_grouped_kernels() {
  static const std::
      unordered_map<std::string, Kernel_f4f4bf16_grouped<InputType>>
          kernels = {
              {"f4f4bf16_grouped_128_64_256_1_1_1_f",
               f4f4bf16_grouped_128_64_256_1_1_1_f},
              {"f4f4bf16_grouped_128_64_256_1_1_1_t",
               f4f4bf16_grouped_128_64_256_1_1_1_t},
              {"f4f4bf16_grouped_128_128_256_1_1_1_f",
               f4f4bf16_grouped_128_128_256_1_1_1_f},
              {"f4f4bf16_grouped_128_128_256_1_1_1_t",
               f4f4bf16_grouped_128_128_256_1_1_1_t},
              {"f4f4bf16_grouped_256_64_256_2_1_1_f",
               f4f4bf16_grouped_256_64_256_2_1_1_f},
              {"f4f4bf16_grouped_256_64_256_2_1_1_t",
               f4f4bf16_grouped_256_64_256_2_1_1_t},
              {"f4f4bf16_grouped_256_128_256_2_1_1_f",
               f4f4bf16_grouped_256_128_256_2_1_1_f},
              {"f4f4bf16_grouped_256_128_256_2_1_1_t",
               f4f4bf16_grouped_256_128_256_2_1_1_t},
              {"f4f4bf16_grouped_256_256_256_2_1_1_f",
               f4f4bf16_grouped_256_256_256_2_1_1_f},
              {"f4f4bf16_grouped_256_256_256_2_1_1_t",
               f4f4bf16_grouped_256_256_256_2_1_1_t},
          };
  return kernels;
}

#endif
} // namespace fbgemm_gpu
