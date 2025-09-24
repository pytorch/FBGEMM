/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include "f4f4bf16_grouped/f4f4bf16_grouped_manifest.cuh"
#endif

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

Kernel_f4f4bf16_grouped
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

at::Tensor dispatch_fp4_grouped_kernel(
    int total_M,
    int N,
    int K,
    int G,
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> M_sizes = std::nullopt,
    std::optional<at::Tensor> global_scale = std::nullopt,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt,
    bool use_mx = true) {
  TORCH_CHECK(M_sizes.has_value(), "M_sizes is assumed to be provided.");
  TORCH_CHECK(
      starting_row_after_padding.has_value(),
      "starting_row_after_padding is assumed to be provided.");
  at::Tensor starting_row_after_padding_actual =
      starting_row_after_padding.value_or(at::zeros({0}));
  TORCH_CHECK(starting_row_after_padding_actual.size(0) % (G + 1) == 0);

  // Select kernel to run via heuristics.
  auto kernel = [&]() {
    return get_kernel_via_heuristics(total_M, N, K, G, use_mx);
  }();
  // Invoke kernel
  return kernel(
      XQ,
      WQ,
      x_scale,
      w_scale,
      output,
      M_sizes,
      global_scale,
      starting_row_after_padding);
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
  return dispatch_fp4_grouped_kernel(
      total_M,
      N,
      K * 2, // Since K is packed
      G,
      XQ,
      WQ,
      x_scale,
      w_scale,
      Y,
      M_sizes,
      global_scale,
      starting_row_after_padding,
      use_mx);
}

#else

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
