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
/// kernels: `perf_block_cap = MAX_THREAD_BLOCKS_FACTOR * #SMs`.
constexpr int32_t MAX_THREAD_BLOCKS_FACTOR = 64;

/// The grid x-dimension is limited to 2^31 - 1 on both CUDA and HIP; a launch
/// with `gridDim.x` above this is rejected by the driver.
constexpr int64_t kMaxGridDimX = std::numeric_limits<int32_t>::max();

/// The grid y-dimension is limited to 65,535 on both CUDA and HIP.
constexpr int64_t kMaxGridDimY = std::numeric_limits<uint16_t>::max();

/// HIP requires the total threads per launch to be less than 2^32 (CUDA
/// silently wraps). This is the largest valid total thread count and is
/// distinct from the grid-x dimension limit above.
/// See: https://github.com/ROCm/hip/issues/2253
constexpr int64_t kMaxThreadsPerLaunch = std::numeric_limits<uint32_t>::max();

/// Selects how the grid-cap helpers clamp the requested dimension.
///
/// - `Always`: cap on both CUDA and ROCm at `MAX_THREAD_BLOCKS_FACTOR * #SMs`.
///   Mirrors the legacy platform-agnostic `get_max_thread_blocks` semantics
///   (D75543767, D65009966).
/// - `OverflowOnly` (default): cap on ROCm only when the unguarded launch
///   would reach the HIP 2^32 thread-per-launch limit. No-op on CUDA except
///   for the driver limit.
///   Cheapest correct policy for kernels that already grid-stride.
/// - `Never`: skip optional performance and overflow caps. Still clamp
///   the selected dimension to the driver range and reject a fixed grid plane
///   that cannot fit on HIP.
enum class BlockCapPolicy { Always, OverflowOnly, Never };

namespace detail {

/// Caps one grid dimension and leaves the other two dimensions unchanged.
///
/// The launch has `blocks_uncapped * other_grid_blocks * threads_per_block`
/// threads before capping. On ROCm, division-first arithmetic finds the
/// largest valid value of the selected dimension under the launch thread
/// limit. The function fails only when the fixed grid plane and one thread
/// block exceed that limit. Nonpositive launch multipliers bypass the overflow
/// cap so the kernel launcher retains responsibility for validating them.
///
/// @param blocks_uncapped Requested size of the selected grid dimension.
/// @param threads_per_block Product of `block.x`, `block.y`, and `block.z`.
/// @param other_grid_blocks Product of the two unchanged grid dimensions.
/// @param stream Stream whose device supplies the performance cap.
/// @param policy Optional cap policy.
/// @param max_grid_dim Driver limit for the selected grid dimension.
/// @param grid_dimension Name of the selected dimension for error messages.
/// @param fixed_grid_plane Name of the unchanged plane for error messages.
/// @return Valid size of the selected grid dimension.
inline uint32_t cap_grid_dimension(
    int64_t blocks_uncapped,
    [[maybe_unused]] int64_t threads_per_block,
    [[maybe_unused]] int64_t other_grid_blocks,
    const c10::cuda::CUDAStream& stream,
    BlockCapPolicy policy,
    int64_t max_grid_dim,
    [[maybe_unused]] const char* grid_dimension,
    [[maybe_unused]] const char* fixed_grid_plane) {
  // Apply the selected dimension's driver range after all policy caps.
  const auto to_grid_dim = [max_grid_dim](int64_t blocks) {
    return static_cast<uint32_t>(
        std::clamp<int64_t>(blocks, int64_t{1}, max_grid_dim));
  };

#ifdef USE_ROCM
  // Divide before multiplication because the uncapped product can overflow.
  const auto has_positive_launch_multipliers =
      threads_per_block > 0 && other_grid_blocks > 0;
  const auto safe_dimension_cap = has_positive_launch_multipliers
      ? (kMaxThreadsPerLaunch / threads_per_block) / other_grid_blocks
      : std::numeric_limits<int64_t>::max();
  TORCH_CHECK(
      !has_positive_launch_multipliers || safe_dimension_cap > 0,
      fixed_grid_plane,
      " and threads_per_block exceed the ROCm maximum per-launch thread count ",
      kMaxThreadsPerLaunch,
      "; no positive ",
      grid_dimension,
      " value can make the launch valid");
#endif

  if (policy == BlockCapPolicy::Never) {
    // The fixed-plane validity check above still applies on ROCm.
    return to_grid_dim(blocks_uncapped);
  }

  const int64_t performance_cap =
      static_cast<int64_t>(MAX_THREAD_BLOCKS_FACTOR) *
      at::cuda::getDeviceProperties(stream.device_index())->multiProcessorCount;

  if (policy == BlockCapPolicy::Always) {
    auto capped_blocks = blocks_uncapped;
    if (capped_blocks > performance_cap) {
      capped_blocks = performance_cap;
    }
#ifdef USE_ROCM
    if (capped_blocks > safe_dimension_cap) {
      capped_blocks = safe_dimension_cap;
    }
#endif
    return to_grid_dim(capped_blocks);
  }

#ifdef USE_ROCM
  if (blocks_uncapped > safe_dimension_cap) {
    return to_grid_dim(
        std::min(
            blocks_uncapped, std::min(performance_cap, safe_dimension_cap)));
  }
  return to_grid_dim(blocks_uncapped);
#else
  return to_grid_dim(blocks_uncapped);
#endif
}

} // namespace detail

/// Caps `grid.x` without changing `grid.y` or `grid.z`.
///
/// On ROCm, the result keeps
/// `grid.x * yz_blocks * threads_per_block <= kMaxThreadsPerLaunch` when the
/// selected policy permits capping. The kernel must grid-stride over X.
///
/// @param blocks_x_uncapped Requested `grid.x` size before capping.
/// @param threads_per_block Product of `block.x`, `block.y`, and `block.z`.
/// @param yz_blocks Product of the unchanged `grid.y` and `grid.z`.
/// @param stream Stream whose device supplies the performance cap.
/// @param policy Optional cap policy.
/// @return Valid `grid.x` size.
inline uint32_t cap_grid_dim_x_with_yz_blocks(
    int64_t blocks_x_uncapped,
    int64_t threads_per_block,
    int64_t yz_blocks,
    const c10::cuda::CUDAStream& stream,
    BlockCapPolicy policy = BlockCapPolicy::OverflowOnly) {
  return detail::cap_grid_dimension(
      blocks_x_uncapped,
      threads_per_block,
      yz_blocks,
      stream,
      policy,
      kMaxGridDimX,
      "grid.x",
      "grid.y * grid.z");
}

/// Caps `grid.y` without changing `grid.x` or `grid.z`.
///
/// On ROCm, the result keeps
/// `grid.y * xz_blocks * threads_per_block <= kMaxThreadsPerLaunch` when the
/// selected policy permits capping. The kernel must grid-stride over Y.
///
/// @param blocks_y_uncapped Requested `grid.y` size before capping.
/// @param threads_per_block Product of `block.x`, `block.y`, and `block.z`.
/// @param xz_blocks Product of the unchanged `grid.x` and `grid.z`.
/// @param stream Stream whose device supplies the performance cap.
/// @param policy Optional cap policy.
/// @return Valid `grid.y` size.
inline uint32_t cap_grid_dim_y_with_xz_blocks(
    int64_t blocks_y_uncapped,
    int64_t threads_per_block,
    int64_t xz_blocks,
    const c10::cuda::CUDAStream& stream,
    BlockCapPolicy policy = BlockCapPolicy::OverflowOnly) {
  return detail::cap_grid_dimension(
      blocks_y_uncapped,
      threads_per_block,
      xz_blocks,
      stream,
      policy,
      kMaxGridDimY,
      "grid.y",
      "grid.x * grid.z");
}

/// Caps `grid.x` for a one-dimensional grid.
///
/// This function forwards `yz_blocks = 1`. Use
/// `cap_grid_dim_x_with_yz_blocks` when `grid.y` or `grid.z` is greater than
/// one. The kernel must grid-stride over X.
///
/// @param blocks_x_uncapped Requested `grid.x` size before capping.
/// @param threads_per_block Product of `block.x`, `block.y`, and `block.z`.
/// @param stream Stream whose device supplies the performance cap.
/// @param policy Optional cap policy.
/// @return Valid `grid.x` size.
inline uint32_t cap_grid_dim_x(
    int64_t blocks_x_uncapped,
    int64_t threads_per_block,
    const c10::cuda::CUDAStream& stream,
    BlockCapPolicy policy = BlockCapPolicy::OverflowOnly) {
  return cap_grid_dim_x_with_yz_blocks(
      blocks_x_uncapped, threads_per_block, 1, stream, policy);
}

/// Derives and caps `grid.x` for a one-dimensional launch.
///
/// Computes `ceil(num_items / threads_per_block)`, then applies
/// `cap_grid_dim_x`. The kernel must grid-stride over the work items.
///
/// @param num_items Number of work items.
/// @param threads_per_block Product of `block.x`, `block.y`, and `block.z`.
/// @param stream Stream whose device supplies the performance cap.
/// @param policy Optional cap policy.
/// @return Valid `grid.x` size.
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
