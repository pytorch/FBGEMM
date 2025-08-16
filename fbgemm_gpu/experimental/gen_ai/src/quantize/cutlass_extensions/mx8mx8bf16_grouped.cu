/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include "mx8mx8bf16_grouped/mx8mx8bf16_grouped_manifest.cuh"
#endif

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

template <typename InputType>
Kernel_mx8mx8bf16_grouped<InputType>
get_kernel_via_heuristics(int total_M, int N, int K, int G) {
  // Llama4 shapes
  if (N == 5120 && K == 1024) {
    if (G <= 8) {
      if (total_M <= 256) {
        return mx8mx8bf16_grouped_256_64_256_2_1_1;
      } else if (total_M <= 512) {
        return mx8mx8bf16_grouped_128_64_256_1_1_1;
      } else if (total_M <= 1024) {
        return mx8mx8bf16_grouped_128_128_256_1_1_1;
      }
    } else if (G <= 16) {
      if (total_M <= 1024) {
        return mx8mx8bf16_grouped_128_64_256_1_1_1;
      } else if (total_M <= 2048) {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    } else {
      if (total_M <= 1024) {
        return mx8mx8bf16_grouped_256_64_256_2_1_1;
      } else if (total_M <= 4096) {
        return mx8mx8bf16_grouped_128_64_256_1_1_1;
      } else if (total_M <= 8192) {
        return mx8mx8bf16_grouped_256_64_256_2_1_1;
      }
    }
    return mx8mx8bf16_grouped_256_256_256_2_1_1;
  } else if (N == 2048 && K == 5120) {
    if (G <= 8) {
      if (total_M <= 256) {
        return mx8mx8bf16_grouped_256_64_256_2_1_1;
      } else if (total_M <= 512) {
        return mx8mx8bf16_grouped_128_64_256_1_1_1;
      } else if (total_M <= 1024) {
        return mx8mx8bf16_grouped_128_128_256_1_1_1;
      }
    } else if (G <= 16) {
      if (total_M <= 1024) {
        return mx8mx8bf16_grouped_256_64_256_2_1_1;
      } else if (total_M <= 2048) {
        return mx8mx8bf16_grouped_128_128_256_1_1_1;
      }
    } else {
      if (total_M <= 1024) {
        return mx8mx8bf16_grouped_256_64_256_2_1_1;
      } else if (total_M <= 16384) {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    }
    return mx8mx8bf16_grouped_256_256_256_2_1_1;
  }

  // Fallback to legacy heuristic
  if (total_M <= 1000) {
    return mx8mx8bf16_grouped_256_128_256_2_1_1;
  } else {
    return mx8mx8bf16_grouped_256_256_256_2_1_1;
  }
}

template <typename InputType>
at::Tensor dispatch_mx8_grouped_kernel(
    int total_M,
    int N,
    int K,
    int G,
    InputType XQ, // FP8
    InputType WQ, // FP8
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt) {
  TORCH_CHECK(
      zero_start_index_M.has_value() != M_sizes.has_value(),
      "One of zero_start_index_M or M_sizes must be provided.");
  TORCH_CHECK(M_sizes.has_value(), "M_sizes is assumed to be provided.");
  TORCH_CHECK(
      starting_row_after_padding.has_value(),
      "starting_row_after_padding is assumed to be provided.");
  at::Tensor starting_row_after_padding_actual =
      starting_row_after_padding.value_or(at::zeros({0}));
  TORCH_CHECK(starting_row_after_padding_actual.size(0) % (G + 1) == 0);

  // Select kernel to run via heuristics.
  auto kernel = [&]() {
    return get_kernel_via_heuristics<InputType>(total_M, N, K, G);
  }();
  // Invoke kernel
  return kernel(
      XQ,
      WQ,
      x_scale,
      w_scale,
      output,
      G,
      zero_start_index_M,
      M_sizes,
      starting_row_after_padding);
}

at::Tensor mx8mx8bf16_grouped_stacked(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt) {
  int64_t total_M = XQ.size(0);
  int64_t N = WQ.size(1);
  int64_t K = WQ.size(2);
  int64_t G = M_sizes.size(0);
  TORCH_CHECK(
      M_sizes.device() == XQ.device(),
      "M_sizes must be on same device as inputs.");
  TORCH_CHECK(
      WQ.dim() == 3 && WQ.size(0) == G, "Weights should be shape [G, N, K].")
  at::Tensor Y = at::empty({total_M, N}, XQ.options().dtype(at::kBFloat16));
  // Early exit for empty inputs.
  if (total_M == 0) {
    return Y;
  }
  // Return continuous view of output.
  return dispatch_mx8_grouped_kernel<at::Tensor>(
      total_M,
      N,
      K,
      G,
      XQ,
      WQ,
      x_scale,
      w_scale,
      Y,
      std::nullopt,
      M_sizes,
      starting_row_after_padding);
}

#else

at::Tensor mx8mx8bf16_grouped_stacked(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}
#endif

} // namespace fbgemm_gpu
