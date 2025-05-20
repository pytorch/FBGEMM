/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "bf16bf16bf16_grouped/bf16bf16bf16_grouped_manifest.cuh"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

// FP8 Tensorwise grouped cutlass kernel dispatch.
template <typename InputType>
at::Tensor dispatch_bf16_grouped_kernel(
    int total_M,
    InputType X, // BF16
    InputType W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt) {
  // Use heuristics to pick best kernel implementation.

  if (total_M <= 16) {
    return bf16bf16bf16_grouped_128_16_128_1_1_1_f(
        X, W, output, zero_start_index_M, M_sizes);
  } else if (total_M <= 32) {
    return bf16bf16bf16_grouped_128_32_128_1_1_1_f(
        X, W, output, zero_start_index_M, M_sizes);
  } else if (total_M <= 64) {
    return bf16bf16bf16_grouped_128_64_128_1_1_1_f(
        X, W, output, zero_start_index_M, M_sizes);
  } else if (total_M <= 128) {
    return bf16bf16bf16_grouped_128_128_128_1_1_1_f(
        X, W, output, zero_start_index_M, M_sizes);
  } else if (total_M <= 512) {
    return bf16bf16bf16_grouped_256_128_128_2_1_1_f(
        X, W, output, zero_start_index_M, M_sizes);
  } else {
    return bf16bf16bf16_grouped_128_256_128_2_1_1_f(
        X, W, output, zero_start_index_M, M_sizes);
  }
}

template <typename OutputType>
OutputType _bf16bf16bf16_grouped(at::TensorList X, at::TensorList W) {
  at::Tensor Y;
  int64_t total_M = 0;
  int64_t G = X.size();

  // Allocate output tensor.
  std::vector<int64_t> output_sizes;
  int64_t total_output_size = 0;
  for (int i = 0; i < G; ++i) {
    int64_t M = X[i].size(0);
    int64_t N = W[i].size(0);
    total_M += M;
    const int64_t output_size = M * N;
    total_output_size += output_size;
    output_sizes.push_back(output_size);
  }
  Y = at::empty(total_output_size, X[0].options().dtype(at::kBFloat16));

  // Run kernel.
  at::Tensor g_out =
      dispatch_bf16_grouped_kernel<at::TensorList>(total_M, X, W, Y);

  // Return appropriate output type.
  if constexpr (std::is_same_v<OutputType, at::Tensor>) {
    int64_t N = W[0].size(0);
    return g_out.view({total_M, N});
  } else {
    // Return grouped view of output.
    std::vector<at::Tensor> output_group = g_out.split(output_sizes);
    for (int i = 0; i < G; ++i) {
      output_group[i] = output_group[i].view({X[i].size(0), W[i].size(0)});
    }
    return output_group;
  }
}

std::vector<at::Tensor> bf16bf16bf16_grouped(
    at::TensorList X,
    at::TensorList W) {
  return _bf16bf16bf16_grouped<std::vector<at::Tensor>>(X, W);
}

at::Tensor bf16bf16bf16_grouped_cat(at::TensorList X, at::TensorList W) {
  return _bf16bf16bf16_grouped<at::Tensor>(X, W);
}

at::Tensor
bf16bf16bf16_grouped_stacked(at::Tensor X, at::Tensor W, at::Tensor M_sizes) {
  int64_t total_M = X.size(0);
  int64_t N = W.size(1);
  int64_t G = M_sizes.size(0);
  TORCH_CHECK(
      M_sizes.device() == X.device(),
      "M_sizes must be on same device as inputs.");
  TORCH_CHECK(
      W.dim() == 3 && W.size(0) == G, "Weights should be shape [G, N, K].")
  at::Tensor Y = at::empty(total_M * N, X.options().dtype(at::kBFloat16));
  // Early exit for empty inputs.
  if (total_M == 0) {
    return Y.view({total_M, N});
  }
  // Return continuous view of output.
  at::Tensor out = dispatch_bf16_grouped_kernel<at::Tensor>(
      total_M, X, W, Y, std::nullopt, M_sizes);
  return out.view({total_M, N});
}

at::Tensor bf16bf16bf16_grouped_dynamic(
    at::Tensor X,
    at::Tensor W,
    at::Tensor zero_start_index_M) {
  TORCH_CHECK(
      zero_start_index_M.device() == X.device(),
      "zero_start_index_M must be on same device as inputs.");
  int64_t G = X.size(0);
  int64_t M = X.size(1);
  int64_t N = W.size(1);
  int64_t total_output_size = G * M * N;
  at::Tensor Y;
  Y = at::zeros(total_output_size, X.options().dtype(at::kBFloat16));

  // Return continuous view of output.
  at::Tensor output = dispatch_bf16_grouped_kernel<at::Tensor>(
      G * M, X, W, Y, zero_start_index_M);
  // View as proper shape.
  return output.view({G, M, N});
}

#else

std::vector<at::Tensor> bf16bf16bf16_grouped(
    at::TensorList X,
    at::TensorList W) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor bf16bf16bf16_grouped_cat(at::TensorList X, at::TensorList W) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor bf16bf16bf16_grouped_dynamic(
    at::Tensor X,
    at::Tensor W,
    at::Tensor zero_start_index_M) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor bf16bf16bf16_grouped_stacked(at::Tensor, at::Tensor, at::Tensor) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
