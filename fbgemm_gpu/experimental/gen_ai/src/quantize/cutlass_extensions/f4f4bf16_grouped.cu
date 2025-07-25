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
#include "f4f4bf16_grouped/f4f4bf16_grouped_manifest.cuh"
#endif

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

template <typename InputType>
Kernel_f4f4bf16_grouped<InputType>
get_kernel_via_heuristics(int total_M, int N, int K, int G, bool use_mx) {
  // MXFP4
  if (use_mx) {
    // Llama4 shapes
    if (N == 5120 && K == 1024) {
      if (G <= 8) {
        if (total_M <= 256) {
          return f4f4bf16_grouped_256_64_256_2_1_1_t;
        } else if (total_M <= 512) {
          return f4f4bf16_grouped_128_64_256_1_1_1_t;
        } else if (total_M <= 1024) {
          return f4f4bf16_grouped_128_128_256_1_1_1_t;
        }
      } else if (G <= 16) {
        if (total_M <= 1024) {
          return f4f4bf16_grouped_128_64_256_1_1_1_t;
        } else if (total_M <= 2048) {
          return f4f4bf16_grouped_256_128_256_2_1_1_t;
        }
      } else {
        if (total_M <= 1024) {
          return f4f4bf16_grouped_256_64_256_2_1_1_t;
        } else if (total_M <= 4096) {
          return f4f4bf16_grouped_128_64_256_1_1_1_t;
        } else if (total_M <= 8192) {
          return f4f4bf16_grouped_256_64_256_2_1_1_t;
        }
      }
      return f4f4bf16_grouped_256_256_256_2_1_1_t;
    } else if (N == 2048 && K == 5120) {
      if (G <= 8) {
        if (total_M <= 256) {
          return f4f4bf16_grouped_256_64_256_2_1_1_t;
        } else if (total_M <= 512) {
          return f4f4bf16_grouped_128_64_256_1_1_1_t;
        } else if (total_M <= 1024) {
          return f4f4bf16_grouped_128_128_256_1_1_1_t;
        }
      } else if (G <= 16) {
        if (total_M <= 1024) {
          return f4f4bf16_grouped_256_64_256_2_1_1_t;
        } else if (total_M <= 2048) {
          return f4f4bf16_grouped_128_128_256_1_1_1_t;
        }
      } else {
        if (total_M <= 1024) {
          return f4f4bf16_grouped_256_64_256_2_1_1_t;
        } else if (total_M <= 16384) {
          return f4f4bf16_grouped_256_128_256_2_1_1_t;
        }
      }
      return f4f4bf16_grouped_256_256_256_2_1_1_t;
    }

    // Fallback to legacy heuristic
    if (total_M <= 1000) {
      return f4f4bf16_grouped_256_128_256_2_1_1_t;
    } else {
      return f4f4bf16_grouped_256_256_256_2_1_1_t;
    }
  } // NVFP4
  else {
    // Llama4 shapes
    if (N == 5120 && K == 1024) {
      if (G <= 8) {
        if (total_M <= 256) {
          return f4f4bf16_grouped_256_64_256_2_1_1_f;
        } else if (total_M <= 512) {
          return f4f4bf16_grouped_128_64_256_1_1_1_f;
        } else if (total_M <= 1024) {
          return f4f4bf16_grouped_128_128_256_1_1_1_f;
        }
      } else if (G <= 16) {
        if (total_M <= 1024) {
          return f4f4bf16_grouped_128_64_256_1_1_1_f;
        } else if (total_M <= 2048) {
          return f4f4bf16_grouped_256_128_256_2_1_1_f;
        }
      } else {
        if (total_M <= 1024) {
          return f4f4bf16_grouped_256_64_256_2_1_1_f;
        } else if (total_M <= 4096) {
          return f4f4bf16_grouped_128_64_256_1_1_1_f;
        } else if (total_M <= 8192) {
          return f4f4bf16_grouped_256_64_256_2_1_1_f;
        }
      }
      return f4f4bf16_grouped_256_256_256_2_1_1_f;
    } else if (N == 2048 && K == 5120) {
      if (G <= 8) {
        if (total_M <= 256) {
          return f4f4bf16_grouped_256_64_256_2_1_1_f;
        } else if (total_M <= 512) {
          return f4f4bf16_grouped_128_64_256_1_1_1_f;
        } else if (total_M <= 1024) {
          return f4f4bf16_grouped_128_128_256_1_1_1_f;
        }
      } else if (G <= 16) {
        if (total_M <= 1024) {
          return f4f4bf16_grouped_256_64_256_2_1_1_f;
        } else if (total_M <= 2048) {
          return f4f4bf16_grouped_128_128_256_1_1_1_f;
        }
      } else {
        if (total_M <= 1024) {
          return f4f4bf16_grouped_256_64_256_2_1_1_f;
        } else if (total_M <= 16384) {
          return f4f4bf16_grouped_256_128_256_2_1_1_f;
        }
      }
      return f4f4bf16_grouped_256_256_256_2_1_1_f;
    }

    // Fallback to legacy heuristic
    if (total_M <= 1000) {
      return f4f4bf16_grouped_256_128_256_2_1_1_f;
    } else {
      return f4f4bf16_grouped_256_256_256_2_1_1_f;
    }
  }
}

template <typename InputType>
at::Tensor dispatch_fp4_grouped_kernel(
    int total_M,
    int N,
    int K,
    int G,
    InputType XQ, // FP4
    InputType WQ, // FP4
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt,
    std::optional<InputType> global_scale = std::nullopt,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt,
    bool use_mx = true) {
  if constexpr (std::is_same_v<InputType, at::TensorList>) {
    TORCH_CHECK(WQ.size() == G);
  } else {
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
  }

  // Select kernel to run via heuristics.
  auto kernel = [&]() {
    return get_kernel_via_heuristics<InputType>(total_M, N, K, G, use_mx);
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
      global_scale,
      starting_row_after_padding);
}

template <typename OutputType>
OutputType _f4f4bf16_grouped(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::TensorList> global_scale,
    bool use_mx) {
  at::Tensor Y;
  int64_t total_M = 0;
  int64_t max_N = 0;
  int64_t max_K = 0;
  int64_t G = XQ.size();

  // Allocate output tensor.
  std::vector<int64_t> output_sizes;
  int64_t total_output_size = 0;
  for (int i = 0; i < G; ++i) {
    int64_t M = XQ[i].size(0);
    int64_t N = WQ[i].size(0);
    int64_t K = WQ[i].size(1);
    total_M += M;
    if (N > max_N) {
      max_N = N;
    }
    if (K > max_K) {
      max_K = K;
    }
    const int64_t output_size = M * N;
    total_output_size += output_size;
    output_sizes.push_back(output_size);
  }
  Y = at::empty(total_output_size, XQ[0].options().dtype(at::kBFloat16));

  // Run kernel.
  at::Tensor g_out = dispatch_fp4_grouped_kernel<at::TensorList>(
      total_M,
      max_N,
      max_K * 2, // Since K is packed
      G,
      XQ,
      WQ,
      x_scale,
      w_scale,
      Y,
      std::nullopt,
      std::nullopt,
      global_scale,
      std::nullopt,
      use_mx);

  // Return appropriate output type.
  if constexpr (std::is_same_v<OutputType, at::Tensor>) {
    int64_t N = WQ[0].size(0);
    return g_out.view({total_M, N});
  } else {
    // Return grouped view of output.
    std::vector<at::Tensor> output_group = g_out.split(output_sizes);
    for (int i = 0; i < G; ++i) {
      output_group[i] = output_group[i].view({XQ[i].size(0), WQ[i].size(0)});
    }
    return output_group;
  }
}

std::vector<at::Tensor> f4f4bf16_grouped(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::TensorList> global_scale = std::nullopt,
    bool use_mx = true) {
  return _f4f4bf16_grouped<std::vector<at::Tensor>>(
      XQ, WQ, x_scale, w_scale, global_scale, use_mx);
}

at::Tensor f4f4bf16_grouped_stacked(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> global_scale = std::nullopt,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt,
    bool use_mx = true) {
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
  return dispatch_fp4_grouped_kernel<at::Tensor>(
      total_M,
      N,
      K * 2, // Since K is packed
      G,
      XQ,
      WQ,
      x_scale,
      w_scale,
      Y,
      std::nullopt,
      M_sizes,
      global_scale,
      starting_row_after_padding,
      use_mx);
}

#else

std::vector<at::Tensor> f4f4bf16_grouped(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::TensorList> global_scale = std::nullopt,
    bool use_mx = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}

at::Tensor f4f4bf16_grouped_stacked(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> global_scale = std::nullopt,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt,
    bool use_mx = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}
#endif

} // namespace fbgemm_gpu
