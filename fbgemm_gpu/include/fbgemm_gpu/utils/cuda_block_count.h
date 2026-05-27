/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

/// Determine an appropriate CUDA block count along the x axis
///
/// When launching CUDA kernels the number of blocks B is often calculated
/// w.r.t. the number of threads T and items to be processed N as
/// B=(N+T-1)/T - which is integer division rounding up.
/// This function abstracts that calculation, performs it in an
/// overflow-safe manner, and limits the return value to the CUDA grid-x
/// dimension cap (2^31-1 for compute capability >= 3.5).
///
/// Accepts any pair of integral types. The `if constexpr` branches on
/// signedness emit the `>= 0` TORCH_CHECKs only for signed types, which
/// avoids "pointless comparison against zero" warnings on unsigned types
/// without needing per-signedness SFINAE overloads.
template <typename Integer1, typename Integer2>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  static_assert(
      std::is_integral_v<Integer1>,
      "cuda_calc_xblock_count: num_items must be an integral type");
  static_assert(
      std::is_integral_v<Integer2>,
      "cuda_calc_xblock_count: threads_per_block must be an integral type");

  // The number of threads can be as high as 2048 on some newer
  // architectures, but this is not portable.
  TORCH_CHECK(threads_per_block <= 1024, "Number of threads must be <=1024!");

  if constexpr (std::is_signed_v<Integer1>) {
    TORCH_CHECK(
        num_items >= 0,
        "When calculating block counts, the number of items must be positive!");
  }
  if constexpr (std::is_signed_v<Integer2>) {
    TORCH_CHECK(
        threads_per_block >= 0,
        "When calculating thread counts, the number of threads must be positive!");
  }

  // The CUDA specification at
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
  // states that for compute capability 3.5-* the grid dimension of a kernel
  // launch must be <=2^31-1.
  constexpr uint64_t max_blocks = 2147483647;
  const auto u_num_items = static_cast<uint64_t>(num_items);
  const auto u_threads = static_cast<uint64_t>(threads_per_block);
  // Overflow-safe variant of (a + b - 1) / b.
  const uint64_t blocks =
      u_num_items / u_threads + (u_num_items % u_threads != 0);
  return static_cast<uint32_t>(std::min(blocks, max_blocks));
}

/// Determine an appropriate CUDA block count for a y- or z-dim of the
/// launch grid.
///
/// The CUDA specification states that the grid dimension of a kernel
/// launch must generally be <=65535. (For compute capability 3.5-* the
/// grid's x-dimension may be <=2^31-1; that larger limit is enforced
/// by `cuda_calc_xblock_count` instead.) Because this function does not
/// know which dimension is being calculated, it uses the smaller limit.
///
/// See `cuda_calc_xblock_count` for the underlying arithmetic.
template <typename Integer1, typename Integer2>
constexpr uint32_t cuda_calc_block_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  constexpr uint32_t max_blocks = 65535;
  return std::min(
      cuda_calc_xblock_count(num_items, threads_per_block), max_blocks);
}
