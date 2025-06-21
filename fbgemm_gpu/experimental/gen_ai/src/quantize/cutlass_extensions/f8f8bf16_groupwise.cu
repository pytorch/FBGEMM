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

#include "f8f8bf16_groupwise/f8f8bf16_groupwise_manifest.cuh"
#include "fbgemm_gpu/quantize/tuning_cache.hpp"
#include "fbgemm_gpu/quantize/utils.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

// FP8 Groupwise Cutlass kernel dispatch.
Kernel_f8f8bf16_groupwise
get_kernel_via_heuristic(int arch, int M, int N, int K) {
  // Use shape heuristics to dispatch to optimized kernel configuration.
  // Initial enablement includes only one schedule.
  if (M <= 16) {
    return f8f8bf16_groupwise_128_16_128_1_1_1_9_t;
  } else {
    return f8f8bf16_groupwise_128_128_128_1_2_1_9_f;
  }
}

Kernel_f8f8bf16_groupwise get_kernel_via_tuning(
    int arch,
    int M,
    int N,
    int K,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale) {
  // One cache per kernel type
  static TuningCache cache("f8f8bf16_groupwise");

  // Reducing amount of auto tuning by rounding up M to next power of 2.
  M = nextPowerOf2(M);
  // Use (M, N, K) shape as the key.
  const std::string shape_key =
      std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);
  const auto& kernels = get_f8f8bf16_groupwise_kernels(arch);
  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key, kernels, XQ, WQ, x_scale, w_scale);
  return kernel;
}

// FP8 Rowwise Cutlass kernel dispatch.
at::Tensor dispatch_fp8_groupwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale) {
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = size_to_dim_(WQ.dim() - 1, WQ.sizes());
  int K = XQ.size(-1);

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
          "FP8 GEMM on sm100a or above requires cuda >= 12.8");
    } else {
      arch = 9;
    }
  }

  // Select kernel to run via heuristics or tuning.
  auto kernel = [&]() {
    if (std::getenv("FBGEMM_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(arch, M, N, K, XQ, WQ, x_scale, w_scale);
    } else {
      return get_kernel_via_heuristic(arch, M, N, K);
    }
  }();
  // Invoke kernel
  return kernel(XQ, WQ, x_scale, w_scale);
}

at::Tensor f8f8bf16_groupwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale) {
  // Invoke and return rowwise kernel without output argument.
  return dispatch_fp8_groupwise_kernel(XQ, WQ, x_scale, w_scale);
}

#else

at::Tensor f8f8bf16_groupwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
#endif

} // namespace fbgemm_gpu
