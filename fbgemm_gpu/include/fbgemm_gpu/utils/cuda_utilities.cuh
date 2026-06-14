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

#include <algorithm>
#include <limits>
#include <type_traits>

#include "fbgemm_gpu/utils/cuda_block_count.h"

namespace fbgemm_gpu::utils::cuda {

/// Empirical multiplier on `#SMs` that gives a good grid-size cap across
/// kernels: `max_blocks = MAX_THREAD_BLOCKS_FACTOR * #SMs`.
constexpr int32_t MAX_THREAD_BLOCKS_FACTOR = 64;

/// Selects how `cap_grid_dim_x{,_from_workload}` clamps the requested grid.
///
/// - `Always`: cap on both CUDA and ROCm at `MAX_THREAD_BLOCKS_FACTOR * #SMs`.
///   Mirrors the legacy platform-agnostic `get_max_thread_blocks` semantics
///   (D75543767, D65009966).
/// - `OverflowOnly` (default): cap on ROCm only when the unguarded launch
///   would exceed the HIP 2^32 thread-per-launch limit. No-op on CUDA.
///   Cheapest correct policy for kernels that already grid-stride.
/// - `Never`: do not cap.
enum class BlockCapPolicy { Always, OverflowOnly, Never };

/// Returns the grid-x dimension to launch with, optionally clamped per
/// `policy`. The caller is expected to have already computed
/// `blocks_uncapped` (e.g. via `cuda_calc_xblock_count(N, divisor)`) and to
/// pass the full per-block thread count for the threshold check.
///
/// Use this form when the block-count divisor differs from the full block
/// size (e.g. `dim3(a, b)` launches where `gridDim.x = ceil(N / b)` but the
/// threshold check needs `a * b`). For the common case where divisor equals
/// `threads_per_block`, prefer `cap_grid_dim_x_from_workload`.
///
/// The caller MUST guarantee the kernel grid-strides over its saturating
/// work dimension whenever the cap path is reachable.
///
/// References:
/// - HIP 2^32 thread-per-launch limit: https://github.com/ROCm/hip/issues/2253
/// - Pre-existing unconditional-cap pattern: D75543767, D65009966
///
/// @param blocks_uncapped     Pre-computed unclamped grid-x dimension.
/// @param threads_per_block   Full per-block thread count
///                            (`block.x * block.y * block.z`), used only by
///                            the `OverflowOnly` threshold check.
/// @param stream              Stream whose device supplies `#SMs` for the cap.
/// @param policy              Cap policy; see `BlockCapPolicy`. Default
///                            `OverflowOnly`.
/// @return Grid-x dimension to pass to the kernel launch.
inline uint32_t cap_grid_dim_x(
    uint32_t blocks_uncapped,
    [[maybe_unused]] int64_t threads_per_block,
    const c10::cuda::CUDAStream& stream,
    BlockCapPolicy policy = BlockCapPolicy::OverflowOnly) {
  if (policy == BlockCapPolicy::Never) {
    return blocks_uncapped;
  }

  const auto max_blocks = static_cast<uint32_t>(
      MAX_THREAD_BLOCKS_FACTOR *
      at::cuda::getDeviceProperties(stream.device_index())
          ->multiProcessorCount);

  if (policy == BlockCapPolicy::Always) {
    return std::min<uint32_t>(blocks_uncapped, max_blocks);
  }

  // policy == OverflowOnly
#ifdef USE_ROCM
  const auto threads_total = static_cast<uint64_t>(blocks_uncapped) *
      static_cast<uint64_t>(threads_per_block);
  if (threads_total > std::numeric_limits<uint32_t>::max()) {
    return std::min<uint32_t>(blocks_uncapped, max_blocks);
  }
#endif
  return blocks_uncapped;
}

/// Sugar over `cap_grid_dim_x` for the common case where the block-count
/// divisor equals `threads_per_block`. Computes `blocks_uncapped` via
/// `cuda_calc_xblock_count(num_items, threads_per_block)` (which preserves
/// its overflow-safe arithmetic, `threads <= 1024` guard, and CUDA grid-x
/// cap of 2^31 - 1) and forwards to `cap_grid_dim_x`.
///
/// @param num_items           Total work items (saturating dimension size).
/// @param threads_per_block   Per-block thread count, also used as the
///                            block-count divisor.
/// @param stream              See `cap_grid_dim_x`.
/// @param policy              See `cap_grid_dim_x`. Default `OverflowOnly`.
/// @return Grid-x dimension to pass to the kernel launch.
template <typename Integer1, typename Integer2>
inline uint32_t cap_grid_dim_x_from_workload(
    Integer1 num_items,
    Integer2 threads_per_block,
    const c10::cuda::CUDAStream& stream,
    BlockCapPolicy policy = BlockCapPolicy::OverflowOnly) {
  static_assert(std::is_integral_v<Integer1>);
  static_assert(std::is_integral_v<Integer2>);
  return cap_grid_dim_x(
      cuda_calc_xblock_count(num_items, threads_per_block),
      static_cast<int64_t>(threads_per_block),
      stream,
      policy);
}

inline auto get_compute_versions() {
  static const auto versions = [] {
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);

    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);

    return std::tuple{runtime_version, driver_version};
  }();

  return versions;
}

/// Opts a CUDA kernel into using more than 48 KB of dynamic shared memory
/// per block on compute capability 7.x+ devices, which requires an explicit
/// `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` call
/// (see
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x).
/// `TORCH_CHECK`s that `smem_bytes` is positive and within the device limit.
/// No-op on ROCm.
///
/// @param kernel       Kernel function pointer to configure.
/// @param smem_bytes   Requested dynamic shared-memory size in bytes
///                     (V100: up to 64 KB; A100: 96 KB; H100: 144 KB).
/// @param device       Target device; defaults to the current CUDA device.
template <typename func_t>
inline void set_max_dynamic_smem(
    func_t kernel [[maybe_unused]],
    const int32_t smem_bytes [[maybe_unused]],
    const int32_t device [[maybe_unused]] = at::cuda::current_device()) {
#ifndef USE_ROCM
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
      smem_bytes));
#endif
}

} // namespace fbgemm_gpu::utils::cuda
