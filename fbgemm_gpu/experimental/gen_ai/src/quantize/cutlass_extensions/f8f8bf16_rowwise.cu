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
// clang-format on

#include "f8f8bf16_rowwise/f8f8bf16_rowwise_manifest.cuh"
#include "fbgemm_gpu/quantize/tuning_cache.cuh"
#include "fbgemm_gpu/quantize/utils.h"
#include "fbgemm_gpu/quantize/utils_gpu.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

// FP8 Rowwise Cutlass kernel dispatch.
Kernel_f8f8bf16_rowwise
get_kernel_via_heuristic(int M, int N, int K, bool use_fast_accum) {
  // Use shape heuristics to dispatch to optimized kernel configuration.
  const int arch = getDeviceArch();

  if (arch == 10) {
    if (M <= 128) {
      if (N <= 1024) {
        return f8f8bf16_rowwise_128_32_128_1_1_1_10_f_f;
      } else {
        return f8f8bf16_rowwise_128_64_128_1_1_1_10_f_f;
      }
    } else if (M <= 1024) {
      if (N <= 1024) {
        return f8f8bf16_rowwise_128_256_128_2_1_1_10_f_f;
      } else {
        return f8f8bf16_rowwise_128_128_128_2_2_1_10_f_f;
      }
    } else if (M <= 2048) {
      return f8f8bf16_rowwise_128_256_128_2_1_1_10_f_f;
    } else {
      if (N <= 1024) {
        return f8f8bf16_rowwise_128_256_128_1_2_1_10_f_f;
      } else {
        return f8f8bf16_rowwise_128_256_128_2_1_1_10_f_f;
      }
    }
  } else {
    if (M <= 16) {
      return f8f8bf16_rowwise_64_16_128_1_1_1_9_f_f;
    } else if (M <= 32) {
      if (N <= 4096) {
        return f8f8bf16_rowwise_64_16_128_1_1_1_9_f_f;
      } else {
        return f8f8bf16_rowwise_64_32_128_2_1_1_9_f_f;
      }
    } else if (M <= 64) {
      if (N <= 2048) {
        return f8f8bf16_rowwise_64_16_128_1_1_1_9_f_f;
      } else if (N <= 4096) {
        return f8f8bf16_rowwise_64_32_128_2_1_1_9_f_f;
      } else {
        return f8f8bf16_rowwise_64_64_128_2_1_1_9_f_f;
      }
    } else if (M <= 128) {
      if (N <= 1024) {
        return f8f8bf16_rowwise_64_16_128_1_1_1_9_f_f;
      } else if (N <= 2048) {
        return f8f8bf16_rowwise_64_32_128_2_1_1_9_f_f;
      } else if (N <= 4096) {
        return f8f8bf16_rowwise_64_64_128_2_1_1_9_f_f;
      } else {
        return f8f8bf16_rowwise_64_128_128_1_1_1_9_f_f;
      }
    } else if (M <= 256) {
      if (N <= 1024) {
        return f8f8bf16_rowwise_64_32_128_2_1_1_9_f_f;
      } else if (N <= 2048) {
        return f8f8bf16_rowwise_64_64_128_2_1_1_9_f_f;
      } else if (N <= 4096) {
        return f8f8bf16_rowwise_64_128_128_1_1_1_9_f_f;
      } else {
        return f8f8bf16_rowwise_64_256_128_1_1_1_9_f_f;
      }
    } else if (M <= 512) {
      if (N <= 1024) {
        return f8f8bf16_rowwise_64_64_128_2_1_1_9_f_f;
      } else if (N <= 2048) {
        return f8f8bf16_rowwise_64_128_128_1_1_1_9_f_f;
      } else if (N <= 4096 || use_fast_accum == false) {
        return f8f8bf16_rowwise_64_256_128_1_1_1_9_f_f;
      } else {
        return f8f8bf16_rowwise_128_256_128_2_1_1_9_f_t;
      }
    } else if (M <= 1024) {
      if (N <= 1024) {
        return f8f8bf16_rowwise_64_128_128_1_1_1_9_f_f;
      } else if (N <= 2048 || use_fast_accum == false) {
        return f8f8bf16_rowwise_64_256_128_1_1_1_9_f_f;
      } else {
        return f8f8bf16_rowwise_128_256_128_2_1_1_9_f_t;
      }
    } else {
      if (M <= 2048 && N <= 1024) {
        return f8f8bf16_rowwise_64_256_128_2_1_1_9_f_f;
      } else if (K <= 4096 || use_fast_accum == false) {
        return f8f8bf16_rowwise_128_128_128_2_1_1_9_t_f;
      } else if (M > 8192 && N > 8192) {
        return f8f8bf16_rowwise_128_256_128_4_4_1_9_f_t;
      } else {
        return f8f8bf16_rowwise_128_256_128_2_1_1_9_f_t;
      }
    }
  }
}

Kernel_f8f8bf16_rowwise get_kernel_via_tuning(
    int M,
    int N,
    int K,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt) {
  // One cache per kernel type
  static TuningCache cache("f8f8bf16_rowwise");

  // Reducing amount of auto tuning by rounding up M to next power of 2.
  M = nextPowerOf2(M);
  // Use (M, N, K) shape as the key.
  const std::string shape_key =
      std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);
  const int arch = getDeviceArch();
  const auto& kernels = get_f8f8bf16_rowwise_kernels(arch);
  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key,
      kernels,
      XQ,
      WQ,
      x_scale,
      w_scale,
      use_fast_accum,
      bias,
      output);

  return kernel;
}

// FP8 Rowwise Cutlass kernel dispatch.
at::Tensor dispatch_fp8_rowwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt) {
  TORCH_CHECK(XQ.dtype() == at::kFloat8_e4m3fn);

  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = size_to_dim_(WQ.dim() - 1, WQ.sizes());
  int K = XQ.size(-1);

  // Select kernel to run via heuristics or tuning.
  auto kernel = [&]() {
    if (std::getenv("FBGEMM_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(
          M, N, K, XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
    } else {
      return get_kernel_via_heuristic(M, N, K, use_fast_accum);
    }
  }();
  // Invoke kernel
  return kernel(XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
}

void f8f8bf16_rowwise_out(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true) {
  // Invoke rowwise kernel with output argument.
  dispatch_fp8_rowwise_kernel(
      XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
}

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true) {
  // Invoke and return rowwise kernel without output argument.
  return dispatch_fp8_rowwise_kernel(
      XQ, WQ, x_scale, w_scale, use_fast_accum, bias);
}

#else

void f8f8bf16_rowwise_out(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
#endif

} // namespace fbgemm_gpu
