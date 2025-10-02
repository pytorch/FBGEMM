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
    std::optional<at::Tensor> offsets = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt,
    std::optional<at::Tensor> global_scale = std::nullopt,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt,
    bool use_mx = true) {
  TORCH_CHECK(
      offsets.has_value() ^ M_sizes.has_value(),
      "Exactly one of M_sizes or offsets must be present.");
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
      offsets,
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
  TORCH_CHECK(M_sizes.dtype() == at::kLong, "M_sizes must be int64.");
  TORCH_CHECK(
      x_scale.dim() == 2 || x_scale.dim() == 3,
      "x_scale must be either 2D or 3D tensor")
  TORCH_CHECK(
      w_scale.dim() == 2 || w_scale.dim() == 3,
      "w_scale must be either 2D or 3D tensor")
  TORCH_CHECK(
      WQ.dim() == 3 && WQ.size(0) == G, "Weights should be shape [G, N, K].")
  TORCH_CHECK(
      starting_row_after_padding.has_value(),
      "starting_row_after_padding is assumed to be provided when using f4f4bf16_grouped_stacked.");
  at::Tensor starting_row_after_padding_actual =
      starting_row_after_padding.value_or(at::zeros({0}));
  TORCH_CHECK(starting_row_after_padding_actual.size(0) % (G + 1) == 0);

  at::Tensor Y = at::empty({total_M, N}, XQ.options().dtype(at::kBFloat16));
  // Early exit for empty inputs.
  if (total_M == 0) {
    return Y;
  }
  // Return continuous view of output.
  return dispatch_fp4_grouped_kernel(
      total_M,
      N,
      K * 2, // 2 FP4 values are packed into uint8
      G,
      XQ,
      WQ,
      x_scale,
      w_scale,
      Y,
      std::nullopt, // offsets
      M_sizes,
      global_scale,
      starting_row_after_padding,
      use_mx);
}

at::Tensor f4f4bf16_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    std::optional<at::Tensor> output_maybe,
    std::optional<at::Tensor> global_scale = std::nullopt) {
  TORCH_CHECK(offsets.dtype() == at::kInt, "offsets must be int32.");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D tensor.");
  TORCH_CHECK(XQ.is_contiguous(), "XQ must be row major.");
  TORCH_CHECK(WQ.transpose(-2, -1).is_contiguous(), "WQ must be column major.");
  TORCH_CHECK(XQ.dtype() == at::kFloat4_e2m1fn_x2, "XQ must be FP4.")
  TORCH_CHECK(WQ.dtype() == at::kFloat4_e2m1fn_x2, "WQ must be FP4.")
  TORCH_CHECK(
      x_scale.dtype() == w_scale.dtype(),
      "x_scale and w_scale must be same type.")
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale must be contiguous.");
  TORCH_CHECK(w_scale.is_contiguous(), "w_scale must be contiguous.");

  const bool use_mx = [&]() {
    if (x_scale.dtype() == at::kFloat8_e4m3fn) {
      TORCH_CHECK(
          global_scale.has_value(), "global_scale must be provided for NVFP4.")
      TORCH_CHECK(
          global_scale->dtype() == at::kFloat, "global_scale must be FP32.")
      return false;
    } else if (x_scale.dtype() == at::kFloat8_e8m0fnu) {
      TORCH_CHECK(
          !global_scale.has_value(), "global_scale must be unset for MXFP4.")
      return true;
    } else {
      TORCH_CHECK(
          false, "Scales must be FP8 e8m0 for MXFP4 or FP8 e4m3 for NVFP4")
    }
  }();

  int64_t G = offsets.size(0);
  int64_t M = XQ.size(-2);
  int64_t N = WQ.size(-1);
  int64_t K = WQ.size(-2);

  at::Tensor out;

  if (XQ.dim() == 2 && WQ.dim() == 3) {
    out = output_maybe.has_value()
        ? output_maybe.value()
        : at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

    TORCH_CHECK(
        XQ.size(-1) == K && WQ.size(0) == G,
        "for 2d-3d grouped GEMM, XQ shape must be (total_M, K) and WQ shape must be (G, K, N).");
    TORCH_CHECK(
        out.dim() == 2 && out.size(0) == M && out.size(1) == N,
        "for 2d-3d grouped GEMM, output shape must be (total_M, N).");

  } else if (XQ.dim() == 2 && WQ.dim() == 2) {
    out = output_maybe.has_value()
        ? output_maybe.value()
        : at::empty({G, M, N}, XQ.options().dtype(at::kBFloat16));

    TORCH_CHECK(
        XQ.dim() == 2 && WQ.dim() == 2 && WQ.size(-2) == K,
        "for 2d-2d grouped GEMM, XQ shape must be (M, total_K) and WQ shape must be (total_K, N).");
    TORCH_CHECK(
        out.dim() == 3 && out.size(0) == G && out.size(1) == M &&
            out.size(2) == N,
        "for 2d-2d grouped GEMM, output shape must be (G, M, N).");

  } else {
    TORCH_CHECK(false, "Invalid input shapes. Must be one of 2D-2D, 2D-3D.");
  }

  // Early exit for empty inputs.
  if (out.numel() == 0) {
    return out;
  }

  return dispatch_fp4_grouped_kernel(
      M,
      N,
      K * 2, // 2 FP4 values are packed into float4_e2m1fn_x2
      G,
      XQ,
      // WQ is shape (K, N) or (G, K, N) in column major layout, to align with
      // torch._scaled_grouped_mm. We transpose here to match cutlass kernel
      // requirements.
      WQ.transpose(-2, -1),
      x_scale,
      w_scale,
      out,
      offsets,
      std::nullopt, // M_sizes
      global_scale,
      std::nullopt, // starting_row_after_padding
      use_mx);
}

#else

at::Tensor f4f4bf16_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    std::optional<at::Tensor> output,
    std::optional<at::Tensor> global_scale = std::nullopt) {
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
