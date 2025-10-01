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
#include "fbgemm_gpu/quantize/tuning_cache.cuh"
#include "fbgemm_gpu/quantize/utils.h"
#include "fbgemm_gpu/quantize/utils_gpu.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

namespace {
TuningCache& getTuningCache() {
  // This kernel has multiple APIs templated based on InputType, so we use this
  // to have a single cache instance across APIs.
  static TuningCache cache("bf16bf16bf16_grouped");
  return cache;
}
} // namespace

template <typename InputType>
Kernel_bf16bf16bf16_grouped<InputType>
get_kernel_via_heuristic(int arch, int G, int total_M, int N, int K) {
  // Use heuristics to pick best kernel implementation.
  if (arch == 10) {
    // Llama4 shapes
    if ((N == 5120 && K == 1024) || (N == 2048 && K == 5120)) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_256_32_128_2_1_1_10_f;
      } else if (total_M <= 512) {
        return bf16bf16bf16_grouped_256_64_128_2_1_1_10_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_256_128_128_2_1_1_10_f;
      } else {
        return bf16bf16bf16_grouped_256_256_128_2_1_1_10_f;
      }
    }

    // Fallback to legacy heuristic.
    if (total_M <= 64 || (total_M <= 256 and N <= 1024)) {
      if (K <= 4096) {
        return bf16bf16bf16_grouped_256_32_128_2_1_1_10_f;
      } else {
        return bf16bf16bf16_grouped_128_32_128_2_1_1_10_f;
      }
    } else if (total_M <= 512) {
      if (N <= 1024) {
        return bf16bf16bf16_grouped_128_64_128_2_1_1_10_f;
      } else if (N <= 8192) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_256_32_128_2_1_1_10_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_128_32_128_2_1_1_10_f;
        } else {
          return bf16bf16bf16_grouped_128_64_128_2_1_1_10_f;
        }
      }
    } else if (total_M <= 1024) {
      if (N <= 1024) {
        return bf16bf16bf16_grouped_128_128_128_2_1_1_10_f;
      } else if (N <= 8192) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_256_64_128_2_1_1_10_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_128_64_128_2_1_1_10_f;
        } else {
          return bf16bf16bf16_grouped_128_128_128_2_1_1_10_f;
        }
      }
    } else if (total_M <= 2048) {
      if (N <= 1024) {
        return bf16bf16bf16_grouped_256_256_128_2_1_1_10_f;
      } else if (N <= 8192) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_256_128_128_2_1_1_10_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_128_128_128_2_1_1_10_f;
        }
      }
    }
    return bf16bf16bf16_grouped_256_256_128_2_1_1_10_f;
  } else {
    // Llama4 128E
    if (G == 128) {
      if (N == 5120 && K == 1024) {
        if (total_M <= 128) {
          return bf16bf16bf16_grouped_128_16_128_2_1_1_9_f;
        } else if (total_M <= 256) {
          return bf16bf16bf16_grouped_128_32_128_2_1_1_9_f;
        } else if (total_M <= 2048) {
          return bf16bf16bf16_grouped_128_16_128_2_1_1_9_f;
        } else if (total_M <= 4096) {
          return bf16bf16bf16_grouped_128_32_128_2_1_1_9_f;
        } else if (total_M <= 8192) {
          return bf16bf16bf16_grouped_128_64_128_1_1_1_9_f;
        } else if (total_M <= 16384) {
          return bf16bf16bf16_grouped_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_128_256_128_2_1_1_9_f;
        }
      }

      if (N == 2048 && K == 5120) {
        if (total_M <= 2048) {
          return bf16bf16bf16_grouped_128_16_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_128_128_128_2_1_1_9_t;
        }
      }
    }

    // Llama4 64E
    if (G == 16) {
      if (N == 5120 && K == 1024) {
        if (total_M <= 32) {
          return bf16bf16bf16_grouped_128_16_128_2_1_1_9_f;
        } else if (total_M <= 64) {
          return bf16bf16bf16_grouped_128_32_128_2_1_1_9_f;
        } else if (total_M <= 256) {
          return bf16bf16bf16_grouped_128_16_128_2_1_1_9_f;
        } else if (total_M <= 512) {
          return bf16bf16bf16_grouped_128_32_128_2_1_1_9_f;
        } else if (total_M <= 1024) {
          return bf16bf16bf16_grouped_128_64_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_128_256_128_2_1_1_9_f;
        }
      }

      if (N == 2048 && K == 5120) {
        if (total_M <= 16) {
          return bf16bf16bf16_grouped_128_16_128_2_1_1_9_f;
        } else if (total_M <= 64) {
          return bf16bf16bf16_grouped_128_32_128_2_1_1_9_f;
        } else if (total_M <= 256) {
          return bf16bf16bf16_grouped_128_16_128_2_1_1_9_f;
        } else if (total_M <= 512) {
          return bf16bf16bf16_grouped_128_32_128_2_1_1_9_f;
        } else if (total_M <= 1024) {
          return bf16bf16bf16_grouped_128_64_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_128_128_128_2_1_1_9_t;
        }
      }
    }

    // Llama4.x pretraining
    if (N == 2560 && K == 5120) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_128_64_128_2_2_1_9_f;
      } else if (total_M <= 512) {
        return bf16bf16bf16_grouped_128_128_128_2_1_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_128_128_128_2_2_1_9_t;
      } else {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_t;
      }
    } else if (N == 5120 && K == 5120) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_128_128_128_2_1_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_128_128_128_2_2_1_9_t;
      } else if (total_M <= 4096) {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_t;
      } else {
        return bf16bf16bf16_grouped_128_128_128_4_4_1_9_t;
      }
    } else if (N == 3072 && K == 6144) {
      if (total_M <= 512) {
        return bf16bf16bf16_grouped_128_128_128_2_1_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_128_128_128_2_2_1_9_t;
      } else if (total_M <= 2048) {
        return bf16bf16bf16_grouped_128_128_128_2_1_1_9_t;
      } else {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_t;
      }
    } else if (N == 6144 && K == 6144) {
      if (total_M <= 512) {
        return bf16bf16bf16_grouped_128_128_128_4_1_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_t;
      } else {
        return bf16bf16bf16_grouped_128_128_128_4_4_1_9_t;
      }

    } else if (N == 5120 && K == 1280) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_128_128_128_4_1_1_9_f;
      } else {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_t;
      }
    } else if (N == 5120 && K == 2560) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_128_128_128_2_2_1_9_t;
      } else {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_t;
      }
    } else if (N == 6144 && K == 1536) {
      if (total_M <= 4096) {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_f;
      } else {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_t;
      }
    } else if (N == 6144 && K == 3072) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_f;
      } else if (total_M <= 4096) {
        return bf16bf16bf16_grouped_128_128_128_1_2_1_9_t;
      } else {
        return bf16bf16bf16_grouped_128_128_128_1_4_1_9_t;
      }
    }

    // Fallback to legacy heuristic for now.
    if (total_M <= 16) {
      return bf16bf16bf16_grouped_128_16_128_1_1_1_9_f;
    } else if (total_M <= 32) {
      return bf16bf16bf16_grouped_128_32_128_1_1_1_9_f;
    } else if (total_M <= 64) {
      return bf16bf16bf16_grouped_128_64_128_1_1_1_9_f;
    } else if (total_M <= 128) {
      return bf16bf16bf16_grouped_128_128_128_1_1_1_9_f;
    } else if (total_M <= 512) {
      return bf16bf16bf16_grouped_256_128_128_2_1_1_9_f;
    } else {
      return bf16bf16bf16_grouped_128_256_128_2_1_1_9_f;
    }
  }
}

template <typename InputType>
Kernel_bf16bf16bf16_grouped<InputType> get_kernel_via_tuning(
    int arch,
    int G,
    int total_M,
    int N,
    int K,
    InputType X, // BF16
    InputType W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt) {
  auto& cache = getTuningCache();

  // Reducing amount of auto tuning by rounding up total_m to next power of 2.
  total_M = nextPowerOf2(total_M);
  // Use (total_M, N, K, G) shape as the key.
  const std::string shape_key = std::to_string(total_M) + "_" +
      std::to_string(N) + "_" + std::to_string(K) + "_" + std::to_string(G);
  const auto& kernels = get_bf16bf16bf16_grouped_kernels<InputType>(arch);
  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key, kernels, X, W, output, zero_start_index_M, M_sizes);

  return kernel;
}

// BF16 grouped cutlass kernel dispatch.
template <typename InputType>
at::Tensor dispatch_bf16_grouped_kernel(
    int G,
    int total_M,
    int N,
    int K,
    InputType X, // BF16
    InputType W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt) {
  const int arch = getDeviceArch();

  // Select kernel to run via heuristics or tuning.
  auto kernel = [&]() {
    if (std::getenv("FBGEMM_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(
          arch, G, total_M, N, K, X, W, output, zero_start_index_M, M_sizes);
    } else {
      return get_kernel_via_heuristic<InputType>(arch, G, total_M, N, K);
    }
  }();
  // Invoke kernel
  return kernel(X, W, output, zero_start_index_M, M_sizes);
}

template <typename OutputType>
OutputType _bf16bf16bf16_grouped(at::TensorList X, at::TensorList W) {
  at::Tensor Y;
  int64_t total_M = 0;
  int64_t G = X.size();
  int64_t max_N = 0;
  int64_t max_K = 0;

  // Allocate output tensor.
  std::vector<int64_t> output_sizes;
  int64_t total_output_size = 0;
  for (int i = 0; i < G; ++i) {
    int64_t M = X[i].size(0);
    int64_t N = W[i].size(0);
    int64_t K = W[i].size(1);
    max_N = std::max(max_N, N);
    max_K = std::max(max_K, K);
    total_M += M;
    const int64_t output_size = M * N;
    total_output_size += output_size;
    output_sizes.push_back(output_size);
  }
  Y = at::empty(total_output_size, X[0].options().dtype(at::kBFloat16));

  // Run kernel.
  at::Tensor g_out = dispatch_bf16_grouped_kernel<at::TensorList>(
      G, total_M, max_N, max_K, X, W, Y);

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

at::Tensor bf16bf16bf16_grouped_stacked(
    at::Tensor X,
    at::Tensor W,
    at::Tensor M_sizes,
    std::optional<at::Tensor> out) {
  int64_t total_M = X.size(0);
  int64_t N = W.size(1);
  int64_t K = W.size(2);
  int64_t G = M_sizes.size(0);
  TORCH_CHECK(
      M_sizes.device() == X.device(),
      "M_sizes must be on same device as inputs.");
  TORCH_CHECK(
      W.dim() == 3 && W.size(0) == G, "Weights should be shape [G, N, K].")

  at::Tensor Y;
  if (out.has_value()) {
    Y = out.value();
  } else {
    Y = at::empty(total_M * N, X.options().dtype(at::kBFloat16));
  }

  // Early exit for empty inputs.
  if (total_M == 0) {
    return Y.view({total_M, N});
  }
  // Return continuous view of output.
  at::Tensor output = dispatch_bf16_grouped_kernel<at::Tensor>(
      G, total_M, N, K, X, W, Y, std::nullopt, M_sizes);
  return output.view({total_M, N});
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
  int64_t K = W.size(2);
  int64_t total_output_size = G * M * N;
  at::Tensor Y;
  Y = at::zeros(total_output_size, X.options().dtype(at::kBFloat16));

  // Return continuous view of output.
  at::Tensor output = dispatch_bf16_grouped_kernel<at::Tensor>(
      G, G * M, N, K, X, W, Y, zero_start_index_M);
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

at::Tensor bf16bf16bf16_grouped_stacked(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
