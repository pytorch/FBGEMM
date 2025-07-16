/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

namespace fbgemm_gpu::utils::cuda {

// Based on the empirical study, max grid size that is 64x larger than the
// number of SMs gives good performance across the board
constexpr int32_t MAX_THREAD_BLOCKS_FACTOR = 64;

inline auto get_max_thread_blocks(const c10::cuda::CUDAStream& stream) {
  const auto device = stream.device_index();
  return MAX_THREAD_BLOCKS_FACTOR *
      at::cuda::getDeviceProperties(device)->multiProcessorCount;
}

inline auto get_compute_versions() {
  static const auto versions = [] {
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);

    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);

    return std::make_tuple(runtime_version, driver_version);
  }();

  return versions;
}

template <typename func_t>
inline void set_max_dynamic_smem(
    func_t kernel,
    const int32_t smem_bytes,
    const int32_t device = at::cuda::current_device()) {
#ifndef USE_ROCM

  // Check
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // "Compute capability 7.x devices allow a single thread block to
  // address the full capacity of shared memory: 96 KB on Volta,
  // 64 KB on Turing. Kernels relying on shared memory allocations
  // over 48 KB per block are architecture-specific, as such they
  // must use dynamic shared memory (rather than statically sized
  // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

  TORCH_CHECK(smem_bytes > 0);

  int max_smem_bytes = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_smem_bytes,
#ifndef __HIP_PLATFORM_AMD__
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
#else
      hipDeviceAttributeMaxSharedMemoryPerBlock,
#endif
      device));

  TORCH_CHECK(
      smem_bytes <= max_smem_bytes,
      "Attempted to allocate ",
      smem_bytes / 1024,
      " KB of shared memory but only ",
      max_smem_bytes / 1024,
      " KB is available");

  C10_CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<void*>(kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      // V100: 64 KB; A100: 96 KB; H100: 144 KB
      smem_bytes));

#endif
}

} // namespace fbgemm_gpu::utils::cuda
