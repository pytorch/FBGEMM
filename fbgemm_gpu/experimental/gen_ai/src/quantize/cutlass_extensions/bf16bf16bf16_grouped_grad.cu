/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "bf16bf16bf16_grouped_grad/bf16bf16bf16_grouped_grad_manifest.cuh"
#include "fbgemm_gpu/quantize/tuning_cache.hpp"
#include "fbgemm_gpu/quantize/utils.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

namespace {
TuningCache& getTuningCache() {
  static TuningCache cache("bf16bf16bf16_grouped_grad");
  return cache;
}
} // namespace

Kernel_bf16bf16bf16_grouped_grad
get_kernel_via_heuristic(int arch, int G, int total_M, int N, int K) {
  // Use heuristics to pick best kernel implementation.
  if (arch == 10) {
    // Llama4 shapes
    if ((N == 5120 && K == 1024) || (N == 2048 && K == 5120)) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_grad_256_32_128_2_1_1_10_f;
      } else if (total_M <= 512) {
        return bf16bf16bf16_grouped_grad_256_64_128_2_1_1_10_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_grad_256_128_128_2_1_1_10_f;
      } else {
        return bf16bf16bf16_grouped_grad_256_256_128_2_1_1_10_f;
      }
    }

    // Fallback to legacy heuristic.
    if (total_M <= 64 || (total_M <= 256 and N <= 1024)) {
      if (K <= 4096) {
        return bf16bf16bf16_grouped_grad_256_32_128_2_1_1_10_f;
      } else {
        return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_10_f;
      }
    } else if (total_M <= 512) {
      if (N <= 1024) {
        return bf16bf16bf16_grouped_grad_128_64_128_2_1_1_10_f;
      } else if (N <= 8192) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_grad_256_32_128_2_1_1_10_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_10_f;
        } else {
          return bf16bf16bf16_grouped_grad_128_64_128_2_1_1_10_f;
        }
      }
    } else if (total_M <= 1024) {
      if (N <= 1024) {
        return bf16bf16bf16_grouped_grad_128_128_128_2_1_1_10_f;
      } else if (N <= 8192) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_grad_256_64_128_2_1_1_10_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_grad_128_64_128_2_1_1_10_f;
        } else {
          return bf16bf16bf16_grouped_grad_128_128_128_2_1_1_10_f;
        }
      }
    } else if (total_M <= 2048) {
      if (N <= 1024) {
        return bf16bf16bf16_grouped_grad_256_256_128_2_1_1_10_f;
      } else if (N <= 8192) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_grad_256_128_128_2_1_1_10_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_grad_128_128_128_2_1_1_10_f;
        }
      }
    }
    return bf16bf16bf16_grouped_grad_256_256_128_2_1_1_10_f;
  } else {
    // Llama4 128E
    if (G == 128) {
      if (N == 5120 && K == 1024) {
        if (total_M <= 128) {
          return bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f;
        } else if (total_M <= 256) {
          return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f;
        } else if (total_M <= 2048) {
          return bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f;
        } else if (total_M <= 4096) {
          return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f;
        } else if (total_M <= 8192) {
          return bf16bf16bf16_grouped_grad_128_64_128_1_1_1_9_f;
        } else if (total_M <= 16384) {
          return bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_grad_128_256_128_2_1_1_9_f;
        }
      }

      if (N == 2048 && K == 5120) {
        if (total_M <= 2048) {
          return bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_t;
        }
      }
    }

    // Llama4 64E
    if (G == 16) {
      if (N == 5120 && K == 1024) {
        if (total_M <= 32) {
          return bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f;
        } else if (total_M <= 64) {
          return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f;
        } else if (total_M <= 256) {
          return bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f;
        } else if (total_M <= 512) {
          return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f;
        } else if (total_M <= 1024) {
          return bf16bf16bf16_grouped_grad_128_64_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_grad_128_256_128_2_1_1_9_f;
        }
      }

      if (N == 2048 && K == 5120) {
        if (total_M <= 16) {
          return bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f;
        } else if (total_M <= 64) {
          return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f;
        } else if (total_M <= 256) {
          return bf16bf16bf16_grouped_grad_128_16_128_2_1_1_9_f;
        } else if (total_M <= 512) {
          return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f;
        } else if (total_M <= 1024) {
          return bf16bf16bf16_grouped_grad_128_64_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_t;
        }
      }
    }

    // Llama4.x pretraining
    if (N == 1280 && K == 5120) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_grad_128_64_128_2_2_1_9_f;
      } else if (total_M <= 4096) {
        return bf16bf16bf16_grouped_grad_128_128_128_2_2_1_9_t;
      } else {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t;
      }
    } else if (N == 2560 && K == 5120) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_grad_128_64_128_2_2_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_grad_128_64_128_2_2_1_9_f;
      } else {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t;
      }
    } else if (N == 1536 && K == 6144) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_f;
      } else if (total_M <= 4096) {
        return bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_t;
      } else {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t;
      }
    } else if (N == 3072 && K == 6144) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_grad_128_64_128_2_1_1_9_f;
      } else if (total_M <= 4096) {
        return bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_t;
      } else {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t;
      }
    } else if (N == 5120 && K == 2560) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_grad_128_128_128_2_1_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_grad_128_128_128_2_2_1_9_t;
      } else if (total_M <= 4096) {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t;
      } else {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t;
      }
    } else if (N == 5120 && K == 5120) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_t;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_grad_128_128_128_2_2_1_9_t;
      } else {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t;
      }
    } else if (N == 6144 && K == 3072) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_grad_128_32_128_2_1_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_f;
      } else if (total_M <= 4096) {
        return bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_t;
      } else {
        return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t;
      }
    } else if (N == 6144 && K == 6144) {
      return bf16bf16bf16_grouped_grad_128_128_128_1_2_1_9_t;
    }

    // Fallback to legacy heuristic for now.
    if (total_M <= 16) {
      return bf16bf16bf16_grouped_grad_128_16_128_1_1_1_9_f;
    } else if (total_M <= 32) {
      return bf16bf16bf16_grouped_grad_128_32_128_1_1_1_9_f;
    } else if (total_M <= 64) {
      return bf16bf16bf16_grouped_grad_128_64_128_1_1_1_9_f;
    } else if (total_M <= 128) {
      return bf16bf16bf16_grouped_grad_128_128_128_1_1_1_9_f;
    } else if (total_M <= 512) {
      return bf16bf16bf16_grouped_grad_256_128_128_2_1_1_9_f;
    } else {
      return bf16bf16bf16_grouped_grad_128_256_128_2_1_1_9_f;
    }
  }
}

Kernel_bf16bf16bf16_grouped_grad get_kernel_via_tuning(
    int arch,
    int G,
    int total_M,
    int N,
    int K,
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes = std::nullopt) {
  auto& cache = getTuningCache();

  // Reducing amount of auto tuning by rounding up total_m to next power of 2.
  total_M = nextPowerOf2(total_M);
  // Use (total_M, N, K, G) shape as the key.
  const std::string shape_key = std::to_string(total_M) + "_" +
      std::to_string(N) + "_" + std::to_string(K) + "_" + std::to_string(G);
  const auto& kernels = get_bf16bf16bf16_grouped_grad_kernels(arch);
  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key, kernels, X, W, output, M_sizes);

  return kernel;
}

// BF16 grouped cutlass kernel dispatch.
at::Tensor dispatch_bf16_grouped_kernel(
    int G,
    int total_M,
    int N,
    int K,
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> M_sizes = std::nullopt) {
  static int arch = -1;
  // Avoid expensive cudaGetDeviceProperties call.
  if (arch < 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major >= 10) {
      arch = 10;
      int runtimeVersion;
      C10_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
      TORCH_CHECK(
          runtimeVersion >= 12080,
          "FP8 grouped GEMM on sm100a or above requires cuda >= 12.8");
    } else {
      arch = 9;
    }
  }

  // Select kernel to run via heuristics or tuning.
  auto kernel = [&]() {
    if (std::getenv("FBGEMM_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(
          arch, G, total_M, N, K, X, W, output, M_sizes);
    } else {
      return get_kernel_via_heuristic(arch, G, total_M, N, K);
    }
  }();
  // Invoke kernel
  return kernel(X, W, output, M_sizes);
}

at::Tensor
bf16bf16bf16_grouped_grad(at::Tensor X, at::Tensor W, at::Tensor M_sizes) {
  int64_t total_M = X.size(0);
  int64_t N = W.size(1);
  int64_t K = W.size(2);
  int64_t G = M_sizes.size(0);
  TORCH_CHECK(
      M_sizes.device() == X.device(),
      "M_sizes must be on same device as inputs.");
  TORCH_CHECK(
      W.dim() == 3 && W.size(0) == G, "Weights should be shape [G, N, K].")

  TORCH_CHECK(X.stride(-1) == 1, "Activation memory layout must be row-major.");
  TORCH_CHECK(W.stride(-2) == 1, "Weight memory layout must be column-major.");

  at::Tensor Y = at::empty(total_M * N, X.options().dtype(at::kBFloat16));
  // Early exit for empty inputs.
  if (total_M == 0) {
    return Y.view({total_M, N});
  }
  // Return continuous view of output.
  at::Tensor out =
      dispatch_bf16_grouped_kernel(G, total_M, N, K, X, W, Y, M_sizes);
  return out.view({total_M, N});
}

#else

at::Tensor bf16bf16bf16_grouped_grad(at::Tensor, at::Tensor, at::Tensor) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
