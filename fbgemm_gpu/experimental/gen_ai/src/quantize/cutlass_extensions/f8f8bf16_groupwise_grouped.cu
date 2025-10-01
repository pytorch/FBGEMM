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

#include "f8f8bf16_groupwise_grouped/f8f8bf16_groupwise_grouped_manifest.cuh"
#include "fbgemm_gpu/quantize/tuning_cache.cuh"
#include "fbgemm_gpu/quantize/utils.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

namespace {
TuningCache& getTuningCache() {
  // This kernel has multiple APIs templated based on InputType, so we use this
  // to have a single cache instance across APIs.
  static TuningCache cache("f8f8bf16_groupwise_grouped");
  return cache;
}
} // namespace

template <typename InputType>
Kernel_f8f8bf16_groupwise_grouped<InputType>
get_kernel_via_heuristics(int total_M, int max_N, int max_K, int G) {
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

  // Use heuristics to pick the best kernel implementation.
  return f8f8bf16_groupwise_grouped_128_128_128_1_2_1_9_f;
}

template <typename InputType>
Kernel_f8f8bf16_groupwise_grouped<InputType> get_kernel_via_tuning(
    int total_M,
    int max_N,
    int max_K,
    int G,
    InputType XQ, // FP8
    InputType WQ, // FP8
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    at::Tensor M_sizes) {
  auto& cache = getTuningCache();

  // Reducing amount of auto tuning by rounding up total_M to next power of 2.
  total_M = nextPowerOf2(total_M);
  // Use (total_M, max_N, max_K, G) shape as the key.
  const std::string shape_key = std::to_string(total_M) + "_" +
      std::to_string(max_N) + "_" + std::to_string(max_K) + "_" +
      std::to_string(G);
  const auto& kernels = get_f8f8bf16_groupwise_grouped_kernels<InputType>();
  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key, kernels, XQ, WQ, x_scale, w_scale, output, M_sizes);

  return kernel;
}

// FP8 groupwise grouped cutlass kernel dispatch.
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
    at::Tensor M_sizes) {
  // Select kernel to run via heuristics or tuning.
  auto kernel = [&]() {
    if (std::getenv("FBGEMM_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(
          total_M, max_N, max_K, G, XQ, WQ, x_scale, w_scale, output, M_sizes);
    } else {
      return get_kernel_via_heuristics<InputType>(total_M, max_N, max_K, G);
    }
  }();
  // Invoke kernel
  return kernel(XQ, WQ, x_scale, w_scale, output, M_sizes);
}

at::Tensor f8f8bf16_groupwise_grouped(
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
      total_M, N, K, G, XQ, WQ, x_scale, w_scale, Y, M_sizes);
  return out.view({total_M, N});
}

#else

at::Tensor f8f8bf16_groupwise_grouped(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
