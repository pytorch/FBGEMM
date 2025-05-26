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

at::Tensor bf16bf16bf16_grouped_128_16_128_2_1_1_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_16_128_2_1_1_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_2_1_1_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_2_1_1_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_2_1_1_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_2_1_1_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_1_1_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_1_1_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_1_1_1_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_1_1_1_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_2_1_1_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_2_1_1_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_16_128_1_1_1_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_16_128_1_1_1_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_1_1_1_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_1_1_1_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_1_1_1_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_1_1_1_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_1_1_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_1_1_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_2_1_1_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_2_1_1_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_2_1_1_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_2_1_1_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

} // namespace fbgemm_gpu
