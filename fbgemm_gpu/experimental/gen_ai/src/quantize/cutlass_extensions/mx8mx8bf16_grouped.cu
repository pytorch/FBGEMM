/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "fbgemm_gpu/quantize/tuning_cache.cuh"
#include "fbgemm_gpu/quantize/utils.h"

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include "mx8mx8bf16_grouped/mx8mx8bf16_grouped_manifest.cuh"
#endif

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

Kernel_mx8mx8bf16_grouped get_kernel_via_tuning(
    int M,
    int N,
    int K,
    int G,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    at::Tensor offsets) {
  static TuningCache cache("mx8mx8bf16_grouped");

  M = nextPowerOf2(M);
  N = nextPowerOf2(N);
  K = nextPowerOf2(K);
  const std::string shape_key =
      std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);

  const auto& kernels = get_mx8mx8bf16_grouped_kernels();
  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key, kernels, XQ, WQ, x_scale, w_scale, output, G, offsets);

  return kernel;
}

Kernel_mx8mx8bf16_grouped
get_kernel_via_heuristics(int M, int N, int K, int G) {
  if (M <= 128) {
    if (N <= 512) {
      return mx8mx8bf16_grouped_256_64_256_2_1_1;
    } else if (N <= 1024) {
      if (K <= 4096) {
        return mx8mx8bf16_grouped_256_64_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_128_64_256_1_1_1;
      }
    } else {
      return mx8mx8bf16_grouped_256_128_256_2_1_1;
    }
  } else if (M <= 512) {
    if (N <= 512) {
      return mx8mx8bf16_grouped_256_128_256_2_1_1;
    } else if (N <= 4096) {
      if (K <= 1024) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    } else if (N <= 8192) {
      return mx8mx8bf16_grouped_256_128_256_2_1_1;
    } else {
      if (K <= 512) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else if (K <= 4096) {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      } else if (K <= 8192) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    }
  } else if (M <= 1024) {
    if (N <= 2048) {
      if (K <= 1024) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    } else if (N <= 4096) {
      return mx8mx8bf16_grouped_256_128_256_2_1_1;
    } else if (N <= 8192) {
      if (K <= 512) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    } else {
      return mx8mx8bf16_grouped_256_128_256_2_1_1;
    }
  } else if (M <= 2048) {
    if (N <= 1024) {
      if (K <= 1024) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    } else if (N <= 2048) {
      return mx8mx8bf16_grouped_256_128_256_2_1_1;
    } else {
      if (K <= 512) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    }
  } else if (M <= 4096) {
    if (N <= 512) {
      if (K <= 512) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    } else if (N <= 1024) {
      return mx8mx8bf16_grouped_256_128_256_2_1_1;
    } else {
      if (K <= 512) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    }
  } else if (M <= 8192) {
    if (K <= 512) {
      return mx8mx8bf16_grouped_256_256_256_2_1_1;
    } else {
      return mx8mx8bf16_grouped_256_128_256_2_1_1;
    }
  } else {
    if (N <= 8192) {
      if (K <= 512) {
        return mx8mx8bf16_grouped_256_256_256_2_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    } else {
      if (K <= 512) {
        return mx8mx8bf16_grouped_128_64_256_1_1_1;
      } else {
        return mx8mx8bf16_grouped_256_128_256_2_1_1;
      }
    }
  }
}

at::Tensor dispatch_mx8_grouped_kernel(
    int M,
    int N,
    int K,
    int G,
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    at::Tensor offsets) {
  // Select kernel to run via heuristics.
  auto kernel = [&]() {
    if (std::getenv("FBGEMM_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(
          M, N, K, G, XQ, WQ, x_scale, w_scale, output, offsets);
    } else {
      return get_kernel_via_heuristics(M, N, K, G);
    }
  }();
  // Invoke kernel
  return kernel(XQ, WQ, x_scale, w_scale, output, G, offsets);
}

at::Tensor mx8mx8bf16_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    std::optional<at::Tensor> output) {
  TORCH_CHECK(offsets.dtype() == at::kInt, "offsets must be int32.");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D tensor.");
  TORCH_CHECK(XQ.is_contiguous(), "XQ must be row major.");
  TORCH_CHECK(WQ.transpose(-2, -1).is_contiguous(), "WQ must be column major.");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale must be contiguous.");
  TORCH_CHECK(w_scale.is_contiguous(), "w_scale must be contiguous.");

  int64_t G = offsets.size(0);
  int64_t M = XQ.size(0);
  int64_t N = WQ.size(-1);
  int64_t K = WQ.size(-2);

  at::Tensor output_actual;

  // 2d-3d case.
  if (XQ.dim() == 2 && WQ.dim() == 3) {
    // Alias for clarity that groups are along M dimension for 2d-3d case.
    int64_t total_M = M;

    // Allocate output tensor if necessary.
    output_actual = output.has_value()
        ? output.value()
        : at::empty({total_M, N}, XQ.options().dtype(at::kBFloat16));

    TORCH_CHECK(
        XQ.size(-1) == K && WQ.size(0) == G,
        "for 2d-3d grouped GEMM, XQ shape must be (total_M, K) and WQ shape must be (G, K, N).");

    TORCH_CHECK(
        output_actual.dim() == 2 && output_actual.size(0) == total_M &&
            output_actual.size(1) == N,
        "for 2d-3d grouped GEMM, output shape must be (total_M, N).");

    // Normalized jagged dim for heuristics
    M /= G;
    // 2d-2d case.
  } else if (XQ.dim() == 2 && WQ.dim() == 2) {
    // Alias for clarity that groups are along K dimension for 2d-2d case.
    int64_t total_K = K;

    // Allocate output tensor if necessary.
    output_actual = output.has_value()
        ? output.value()
        : at::empty({G, M, N}, XQ.options().dtype(at::kBFloat16));

    TORCH_CHECK(
        XQ.dim() == 2 && WQ.dim() == 2 && WQ.size(-2) == total_K,
        "for 2d-2d grouped GEMM, XQ shape must be (M, total_K) and WQ shape must be (total_K, N).");

    TORCH_CHECK(
        output_actual.dim() == 3 && output_actual.size(0) == G &&
            output_actual.size(1) == M && output_actual.size(2) == N,
        "for 2d-2d grouped GEMM, output shape must be (G, M, N).");
    // Normalized jagged dim for heuristics
    K /= G;
  } else {
    TORCH_CHECK(false, "Invalid input shapes. Must be one of 2D-2D, 2D-3D.");
  }

  // Early exit for empty inputs.
  if (M == 0) {
    return output_actual;
  }

  // Return continuous view of output.
  return dispatch_mx8_grouped_kernel(
      M, N, K, G, XQ, WQ, x_scale, w_scale, output_actual, offsets);
}

#else

at::Tensor mx8mx8bf16_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    std::optional<at::Tensor> output) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}

#endif

} // namespace fbgemm_gpu
