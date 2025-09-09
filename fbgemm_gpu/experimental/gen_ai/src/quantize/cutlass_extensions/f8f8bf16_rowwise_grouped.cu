/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "f8f8bf16_rowwise_grouped/f8f8bf16_rowwise_grouped_manifest.cuh"
#include "f8f8bf16_rowwise_grouped_sm100/f8f8bf16_rowwise_grouped_manifest.cuh"
#include "fbgemm_gpu/quantize/tuning_cache.hpp"
#include "fbgemm_gpu/quantize/utils.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

namespace {
TuningCache& getTuningCache() {
  // This kernel has multiple APIs templated based on InputType, so we use this
  // to have a single cache instance across APIs.
  static TuningCache cache("f8f8bf16_rowwise_grouped");
  return cache;
}
} // namespace

template <typename InputType>
Kernel_f8f8bf16_rowwise_grouped<InputType>
get_kernel_via_heuristics(int total_M, int max_N, int max_K, int G) {
  const int arch = getDeviceArch();

  // Use heuristics to pick the best kernel implementation.
  if (arch == 10) {
    // Llama4 shapes
    if ((max_N == 5120 && max_K == 1024) || (max_N == 2048 && max_K == 5120)) {
      if (total_M <= 256) {
        return f8f8bf16_rowwise_grouped_256_32_128_2_1_1_10_f;
      } else if (total_M <= 512) {
        return f8f8bf16_rowwise_grouped_256_64_128_2_1_1_10_f;
      } else if (total_M <= 1024) {
        return f8f8bf16_rowwise_grouped_256_128_128_2_1_1_10_f;
      } else {
        return f8f8bf16_rowwise_grouped_256_256_128_2_1_1_10_f;
      }
    }

    // Fallback to legacy heuristic.
    if (total_M <= 64 || (total_M <= 256 and max_N <= 1024)) {
      if (max_K <= 4096) {
        return f8f8bf16_rowwise_grouped_256_32_128_2_1_1_10_f;
      } else {
        return f8f8bf16_rowwise_grouped_128_32_128_2_1_1_10_f;
      }
    } else if (total_M <= 512) {
      if (max_N <= 1024) {
        return f8f8bf16_rowwise_grouped_128_64_128_2_1_1_10_f;
      } else if (max_N <= 8192) {
        if (max_K <= 2048) {
          return f8f8bf16_rowwise_grouped_256_32_128_2_1_1_10_f;
        } else if (max_K <= 4096) {
          return f8f8bf16_rowwise_grouped_128_32_128_2_1_1_10_f;
        } else {
          return f8f8bf16_rowwise_grouped_128_64_128_2_1_1_10_f;
        }
      }
    } else if (total_M <= 1024) {
      if (max_N <= 1024) {
        return f8f8bf16_rowwise_grouped_128_128_128_2_1_1_10_f;
      } else if (max_N <= 8192) {
        if (max_K <= 2048) {
          return f8f8bf16_rowwise_grouped_256_64_128_2_1_1_10_f;
        } else if (max_K <= 4096) {
          return f8f8bf16_rowwise_grouped_128_64_128_2_1_1_10_f;
        } else {
          return f8f8bf16_rowwise_grouped_128_128_128_2_1_1_10_f;
        }
      }
    } else if (total_M <= 2048) {
      if (max_N <= 1024) {
        return f8f8bf16_rowwise_grouped_256_256_128_2_1_1_10_f;
      } else if (max_N <= 8192) {
        if (max_K <= 2048) {
          return f8f8bf16_rowwise_grouped_256_128_128_2_1_1_10_f;
        } else if (max_K <= 4096) {
          return f8f8bf16_rowwise_grouped_128_128_128_2_1_1_10_f;
        }
      }
    }
    return f8f8bf16_rowwise_grouped_256_256_128_2_1_1_10_f;
  } else {
    // LLama4 16E
    if (max_N == 2048 && max_K == 5120 && G == 16) {
      if (total_M <= 256) {
        return f8f8bf16_rowwise_grouped_128_16_128_2_1_1_9_f;
      } else if (total_M <= 512) {
        return f8f8bf16_rowwise_grouped_128_32_128_2_1_1_9_f;
      } else if (total_M <= 1024) {
        return f8f8bf16_rowwise_grouped_128_64_128_2_1_1_9_f;
      } else if (total_M <= 2048) {
        return f8f8bf16_rowwise_grouped_128_128_128_2_1_1_9_f;
      } else {
        return f8f8bf16_rowwise_grouped_128_256_128_1_1_1_9_f;
      }
    }
    if (max_N == 5120 && max_K == 1024 && G == 16) {
      if (total_M <= 16) {
        return f8f8bf16_rowwise_grouped_128_16_128_2_1_1_9_f;
      } else if (total_M <= 256) {
        return f8f8bf16_rowwise_grouped_256_32_128_4_1_1_9_f;
      } else if (total_M <= 512) {
        return f8f8bf16_rowwise_grouped_256_32_128_1_1_1_9_f;
      } else if (total_M <= 1024) {
        return f8f8bf16_rowwise_grouped_256_64_128_1_1_1_9_f;
      } else if (total_M <= 1536) {
        return f8f8bf16_rowwise_grouped_256_128_128_4_1_1_9_f;
      } else if (total_M <= 2048) {
        return f8f8bf16_rowwise_grouped_256_64_128_1_1_1_9_f;
      } else if (total_M <= 4096) {
        return f8f8bf16_rowwise_grouped_128_256_128_2_1_1_9_f;
      } else {
        return f8f8bf16_rowwise_grouped_128_256_128_1_1_1_9_f;
      }
    }
    // LLama4 128E
    if (max_N == 5120 && max_K == 1024 && G == 128) {
      if (total_M <= 128) {
        return f8f8bf16_rowwise_grouped_256_16_128_2_1_1_9_f;
      } else if (total_M <= 256) {
        return f8f8bf16_rowwise_grouped_256_16_128_2_1_1_9_f;
      } else if (total_M <= 2048) {
        return f8f8bf16_rowwise_grouped_256_16_128_1_1_1_9_f;
      } else if (total_M <= 4096) {
        return f8f8bf16_rowwise_grouped_256_32_128_1_1_1_9_f;
      } else if (total_M <= 8192) {
        return f8f8bf16_rowwise_grouped_256_64_128_1_1_1_9_f;
      } else if (total_M <= 12288) {
        return f8f8bf16_rowwise_grouped_256_128_128_2_1_1_9_f;
      } else if (total_M <= 32768) {
        return f8f8bf16_rowwise_grouped_256_128_128_1_1_1_9_f;
      } else {
        return f8f8bf16_rowwise_grouped_128_256_128_2_1_1_9_f;
      }
    }
    if (max_N == 2048 && max_K == 5120 && G == 128) {
      if (total_M <= 128) {
        return f8f8bf16_rowwise_grouped_256_16_128_1_1_1_9_f;
      } else if (total_M <= 512) {
        return f8f8bf16_rowwise_grouped_256_16_128_4_1_1_9_f;
      } else if (total_M <= 1024) {
        return f8f8bf16_rowwise_grouped_256_32_128_4_1_1_9_f;
      } else if (total_M <= 2048) {
        return f8f8bf16_rowwise_grouped_256_16_128_2_1_1_9_f;
      } else if (total_M <= 4096) {
        return f8f8bf16_rowwise_grouped_256_32_128_4_1_1_9_f;
      } else if (total_M <= 8192) {
        return f8f8bf16_rowwise_grouped_256_64_128_4_1_1_9_f;
      } else if (total_M <= 16384) {
        return f8f8bf16_rowwise_grouped_128_128_128_1_1_1_9_f;
      } else {
        return f8f8bf16_rowwise_grouped_128_256_128_2_1_1_9_f;
      }
    }

    if (total_M <= 16) {
      return f8f8bf16_rowwise_grouped_128_16_128_1_1_1_9_f;
    } else if (total_M <= 32) {
      return f8f8bf16_rowwise_grouped_128_32_128_1_1_1_9_f;
    } else if (total_M <= 64) {
      return f8f8bf16_rowwise_grouped_128_64_128_1_1_1_9_f;
    } else if (total_M <= 128) {
      return f8f8bf16_rowwise_grouped_128_128_128_1_1_1_9_f;
    } else if (total_M <= 512) {
      return f8f8bf16_rowwise_grouped_256_128_128_2_1_1_9_f;
    } else {
      return f8f8bf16_rowwise_grouped_128_256_128_2_1_1_9_f;
    }
  }
}

template <typename InputType>
Kernel_f8f8bf16_rowwise_grouped<InputType> get_kernel_via_tuning(
    int total_M,
    int max_N,
    int max_K,
    int G,
    InputType XQ, // FP8
    InputType WQ, // FP8
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt) {
  auto& cache = getTuningCache();

  // Reducing amount of auto tuning by rounding up total_M to next power of 2.
  total_M = nextPowerOf2(total_M);
  // Use (total_M, max_N, max_K, G) shape as the key.
  const std::string shape_key = std::to_string(total_M) + "_" +
      std::to_string(max_N) + "_" + std::to_string(max_K) + "_" +
      std::to_string(G);

  const auto& kernels = []() {
    const int arch = getDeviceArch();
    if (arch == 9) {
      return get_f8f8bf16_rowwise_grouped_kernels<InputType>();
    } else {
      return get_f8f8bf16_rowwise_grouped_kernels_sm100<InputType>();
    }
  }();

  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key,
      kernels,
      XQ,
      WQ,
      x_scale,
      w_scale,
      output,
      zero_start_index_M,
      M_sizes);

  return kernel;
}

// FP8 rowwise grouped cutlass kernel dispatch.
template <typename InputType>
at::Tensor dispatch_fp8_grouped_kernel(
    int total_M,
    int max_N,
    int max_K,
    int G,
    InputType XQ, // FP8
    InputType WQ, // FP8
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt) {
  // Select kernel to run via heuristics or tuning.
  auto kernel = [&]() {
    if (std::getenv("FBGEMM_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(
          total_M,
          max_N,
          max_K,
          G,
          XQ,
          WQ,
          x_scale,
          w_scale,
          output,
          zero_start_index_M,
          M_sizes);
    } else {
      return get_kernel_via_heuristics<InputType>(total_M, max_N, max_K, G);
    }
  }();
  // Invoke kernel
  return kernel(XQ, WQ, x_scale, w_scale, output, zero_start_index_M, M_sizes);
}

template <typename OutputType>
OutputType _f8f8bf16_rowwise_grouped(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale) {
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
  at::Tensor g_out = dispatch_fp8_grouped_kernel<at::TensorList>(
      total_M, max_N, max_K, G, XQ, WQ, x_scale, w_scale, Y);

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

std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale) {
  return _f8f8bf16_rowwise_grouped<std::vector<at::Tensor>>(
      XQ, WQ, x_scale, w_scale);
}

at::Tensor f8f8bf16_rowwise_grouped_cat(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale) {
  return _f8f8bf16_rowwise_grouped<at::Tensor>(XQ, WQ, x_scale, w_scale);
}

at::Tensor f8f8bf16_rowwise_grouped_stacked(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes) {
  int64_t total_M = XQ.size(0);
  int64_t N = WQ.size(1);
  int64_t K = WQ.size(2);
  int64_t G = M_sizes.size(0);
  TORCH_CHECK(
      M_sizes.device() == XQ.device(),
      "M_sizes must be on same device as inputs.");
  TORCH_CHECK(
      WQ.dim() == 3 && WQ.size(0) == G, "Weights should be shape [G, N, K].")
  at::Tensor Y = at::empty(total_M * N, XQ.options().dtype(at::kBFloat16));
  // Early exit for empty inputs.
  if (total_M == 0) {
    return Y.view({total_M, N});
  }
  // Return continuous view of output.
  at::Tensor out = dispatch_fp8_grouped_kernel<at::Tensor>(
      total_M, N, K, G, XQ, WQ, x_scale, w_scale, Y, std::nullopt, M_sizes);
  return out.view({total_M, N});
}

at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor zero_start_index_M,
    bool zeroing_output_tensor = true) {
  TORCH_CHECK(
      zero_start_index_M.device() == XQ.device(),
      "zero_start_index_M must be on same device as inputs.");
  int64_t G = XQ.size(0);
  int64_t M = XQ.size(1);
  int64_t N = WQ.size(1);
  int64_t K = WQ.size(2);
  int64_t total_output_size = G * M * N;
  at::Tensor Y;
  if (zeroing_output_tensor) {
    Y = at::zeros(total_output_size, XQ.options().dtype(at::kBFloat16));
  } else {
    Y = at::empty(total_output_size, XQ.options().dtype(at::kBFloat16));
  }

  // Return continuous view of output.
  at::Tensor output = dispatch_fp8_grouped_kernel<at::Tensor>(
      G * M, N, K, G, XQ, WQ, x_scale, w_scale, Y, zero_start_index_M);
  // View as proper shape.
  return output.view({G, M, N});
}

#else

std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor f8f8bf16_rowwise_grouped_cat(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor f8f8bf16_rowwise_grouped_stacked(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor zero_start_index_M,
    bool zeroing_output_tensor = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
