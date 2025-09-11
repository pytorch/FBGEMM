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

at::Tensor bf16bf16bf16_grouped_128_16_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_16_128_1_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_16_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_16_128_2_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_16_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_16_128_4_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_1_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_2_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_4_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_1_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_2_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_2_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_2_2_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_4_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_1_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_2_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_2_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_2_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_2_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_4_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_1_4_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_1_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_2_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_2_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_4_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_4_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_4_1_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_4_4_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_4_4_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_1_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_1_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_1_1_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_2_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_2_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_2_1_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_4_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_4_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_256_128_4_1_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_16_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_16_128_1_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_16_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_16_128_2_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_16_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_16_128_4_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_32_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_32_128_1_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_32_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_32_128_2_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_32_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_32_128_4_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_64_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_64_128_1_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_64_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_64_128_2_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_64_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_64_128_4_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_1_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_1_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_1_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_1_1_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_2_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_2_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_2_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_2_1_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_4_1_1_9_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_4_1_1_9_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_4_1_1_9_t(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_4_1_1_9_t(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

// SM100
at::Tensor bf16bf16bf16_grouped_128_32_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_32_128_2_1_1_10_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_64_128_2_1_1_10_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_128_128_128_2_1_1_10_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_32_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_32_128_2_1_1_10_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_64_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_64_128_2_1_1_10_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_128_128_2_1_1_10_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_256_128_2_1_1_10_f(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor bf16bf16bf16_grouped_256_256_128_2_1_1_10_f(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

template <typename InputType>
using Kernel_bf16bf16bf16_grouped = at::Tensor (*)(
    InputType,
    InputType,
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>);

template <typename InputType>
const std::unordered_map<std::string, Kernel_bf16bf16bf16_grouped<InputType>>&
get_bf16bf16bf16_grouped_kernels(int arch) {
  static const std::
      unordered_map<std::string, Kernel_bf16bf16bf16_grouped<InputType>>
          kernelsSM90 = {
              {"bf16bf16bf16_grouped_128_16_128_1_1_1_9_f",
               bf16bf16bf16_grouped_128_16_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_128_16_128_2_1_1_9_f",
               bf16bf16bf16_grouped_128_16_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_128_16_128_4_1_1_9_f",
               bf16bf16bf16_grouped_128_16_128_4_1_1_9_f},
              {"bf16bf16bf16_grouped_128_32_128_1_1_1_9_f",
               bf16bf16bf16_grouped_128_32_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_128_32_128_2_1_1_9_f",
               bf16bf16bf16_grouped_128_32_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_128_32_128_4_1_1_9_f",
               bf16bf16bf16_grouped_128_32_128_4_1_1_9_f},
              {"bf16bf16bf16_grouped_128_64_128_1_1_1_9_f",
               bf16bf16bf16_grouped_128_64_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_128_64_128_2_1_1_9_f",
               bf16bf16bf16_grouped_128_64_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_128_64_128_2_2_1_9_f",
               bf16bf16bf16_grouped_128_64_128_2_2_1_9_f},
              {"bf16bf16bf16_grouped_128_64_128_4_1_1_9_f",
               bf16bf16bf16_grouped_128_64_128_4_1_1_9_f},
              {"bf16bf16bf16_grouped_128_128_128_1_1_1_9_f",
               bf16bf16bf16_grouped_128_128_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_128_128_128_1_1_1_9_t",
               bf16bf16bf16_grouped_128_128_128_1_1_1_9_t},
              {"bf16bf16bf16_grouped_128_128_128_1_4_1_9_t",
               bf16bf16bf16_grouped_128_128_128_1_4_1_9_t},
              {"bf16bf16bf16_grouped_128_128_128_1_2_1_9_f",
               bf16bf16bf16_grouped_128_128_128_1_2_1_9_f},
              {"bf16bf16bf16_grouped_128_128_128_1_2_1_9_t",
               bf16bf16bf16_grouped_128_128_128_1_2_1_9_t},
              {"bf16bf16bf16_grouped_128_128_128_2_1_1_9_f",
               bf16bf16bf16_grouped_128_128_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_128_128_128_2_1_1_9_t",
               bf16bf16bf16_grouped_128_128_128_2_1_1_9_t},
              {"bf16bf16bf16_grouped_128_128_128_2_2_1_9_t",
               bf16bf16bf16_grouped_128_128_128_2_2_1_9_t},
              {"bf16bf16bf16_grouped_128_128_128_4_1_1_9_f",
               bf16bf16bf16_grouped_128_128_128_4_1_1_9_f},
              {"bf16bf16bf16_grouped_128_128_128_4_1_1_9_t",
               bf16bf16bf16_grouped_128_128_128_4_1_1_9_t},
              {"bf16bf16bf16_grouped_128_128_128_4_4_1_9_t",
               bf16bf16bf16_grouped_128_128_128_4_4_1_9_t},
              {"bf16bf16bf16_grouped_128_256_128_1_1_1_9_f",
               bf16bf16bf16_grouped_128_256_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_128_256_128_1_1_1_9_t",
               bf16bf16bf16_grouped_128_256_128_1_1_1_9_t},
              {"bf16bf16bf16_grouped_128_256_128_2_1_1_9_f",
               bf16bf16bf16_grouped_128_256_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_128_256_128_2_1_1_9_t",
               bf16bf16bf16_grouped_128_256_128_2_1_1_9_t},
              {"bf16bf16bf16_grouped_128_256_128_4_1_1_9_f",
               bf16bf16bf16_grouped_128_256_128_4_1_1_9_f},
              {"bf16bf16bf16_grouped_128_256_128_4_1_1_9_t",
               bf16bf16bf16_grouped_128_256_128_4_1_1_9_t},
              {"bf16bf16bf16_grouped_256_16_128_1_1_1_9_f",
               bf16bf16bf16_grouped_256_16_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_256_16_128_2_1_1_9_f",
               bf16bf16bf16_grouped_256_16_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_256_16_128_4_1_1_9_f",
               bf16bf16bf16_grouped_256_16_128_4_1_1_9_f},
              {"bf16bf16bf16_grouped_256_32_128_1_1_1_9_f",
               bf16bf16bf16_grouped_256_32_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_256_32_128_2_1_1_9_f",
               bf16bf16bf16_grouped_256_32_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_256_32_128_4_1_1_9_f",
               bf16bf16bf16_grouped_256_32_128_4_1_1_9_f},
              {"bf16bf16bf16_grouped_256_64_128_1_1_1_9_f",
               bf16bf16bf16_grouped_256_64_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_256_64_128_2_1_1_9_f",
               bf16bf16bf16_grouped_256_64_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_256_64_128_4_1_1_9_f",
               bf16bf16bf16_grouped_256_64_128_4_1_1_9_f},
              {"bf16bf16bf16_grouped_256_128_128_1_1_1_9_f",
               bf16bf16bf16_grouped_256_128_128_1_1_1_9_f},
              {"bf16bf16bf16_grouped_256_128_128_1_1_1_9_t",
               bf16bf16bf16_grouped_256_128_128_1_1_1_9_t},
              {"bf16bf16bf16_grouped_256_128_128_2_1_1_9_f",
               bf16bf16bf16_grouped_256_128_128_2_1_1_9_f},
              {"bf16bf16bf16_grouped_256_128_128_2_1_1_9_t",
               bf16bf16bf16_grouped_256_128_128_2_1_1_9_t},
              {"bf16bf16bf16_grouped_256_128_128_4_1_1_9_f",
               bf16bf16bf16_grouped_256_128_128_4_1_1_9_f},
              {"bf16bf16bf16_grouped_256_128_128_4_1_1_9_t",
               bf16bf16bf16_grouped_256_128_128_4_1_1_9_t},
          };

  static const std::
      unordered_map<std::string, Kernel_bf16bf16bf16_grouped<InputType>>
          kernelsSM100 = {
              {"bf16bf16bf16_grouped_256_32_128_2_1_1_10_f",
               bf16bf16bf16_grouped_256_32_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_256_64_128_2_1_1_10_f",
               bf16bf16bf16_grouped_256_64_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_256_128_128_2_1_1_10_f",
               bf16bf16bf16_grouped_256_128_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_256_256_128_2_1_1_10_f",
               bf16bf16bf16_grouped_256_256_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_128_32_128_2_1_1_10_f",
               bf16bf16bf16_grouped_128_32_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_128_64_128_2_1_1_10_f",
               bf16bf16bf16_grouped_128_64_128_2_1_1_10_f},
              {"bf16bf16bf16_grouped_128_128_128_2_1_1_10_f",
               bf16bf16bf16_grouped_128_128_128_2_1_1_10_f},
          };
  if (arch == 10) {
    return kernelsSM100;
  } else {
    return kernelsSM90;
  }
}

} // namespace fbgemm_gpu
