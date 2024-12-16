/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <cstdint>

/// Determine an appropriate CUDA block count along the x axis
///
/// When launching CUDA kernels the number of blocks B is often calculated
/// w.r.t. the number of threads T and items to be processed N as
/// B=(N+T-1)/T - which is integer division rounding up.
/// This function abstracts that calculation, performs it in an
/// overflow-safe manner, and limits the return value appropriately.
///
/// This is a general function for all integral data types.
/// The goal of this set of functions is to ensure correct calculations
/// across a variety of data types without forcing the programmer to
/// cast to an appropriate type (which is dangerous because we don't
/// have conversion warnings enabled). The values of the variables
/// can then be checked for correctness at run-time.
/// Specialized functions below handle various combinations of signed
/// and unsigned inputs. This system prevents "pointless comparison
/// against zero" warnings from the compiler for unsigned types
/// (simpler ways of suppressing this warning didn't work) while
/// maintaining the various warnings.
///
/// Function is designed to facilitate run-time value checking.
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<std::is_integral<Integer1>::value, bool> = true,
    std::enable_if_t<std::is_integral<Integer2>::value, bool> = true>
constexpr uint32_t cuda_calc_xblock_count_base(
    Integer1 num_items,
    Integer2 threads_per_block) {
  // The number of threads can be as high as 2048 on some newer architectures,
  // but this is not portable.
  TORCH_CHECK(threads_per_block <= 1024, "Number of threads must be <=1024!");
  // The CUDA specification at
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
  // states that for compute capability 3.5-* the grid dimension of a kernel
  // launch must must be <=2^31-1.
  constexpr uint64_t max_blocks = 2147483647;
  const auto u_num_items = static_cast<uint64_t>(num_items);
  const auto u_threads = static_cast<uint64_t>(threads_per_block);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks =
      u_num_items / u_threads + (u_num_items % u_threads != 0);
  return static_cast<uint32_t>(std::min(blocks, max_blocks));
}

// See: cuda_calc_xblock_count_base
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<
        std::is_integral<Integer1>::value && std::is_signed<Integer2>::value,
        bool> = true,
    std::enable_if_t<
        std::is_integral<Integer2>::value && std::is_unsigned<Integer2>::value,
        bool> = true>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  TORCH_CHECK(
      num_items >= 0,
      "When calculating block counts, the number of items must be positive!");
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

// See: cuda_calc_xblock_count_base
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<
        std::is_integral<Integer1>::value && std::is_unsigned<Integer2>::value,
        bool> = true,
    std::enable_if_t<
        std::is_integral<Integer2>::value && std::is_signed<Integer2>::value,
        bool> = true>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  TORCH_CHECK(
      threads_per_block >= 0,
      "When calculating thread counts, the number of threads must be positive!");
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

// See: cuda_calc_xblock_count_base
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<
        std::is_integral<Integer1>::value && std::is_signed<Integer2>::value,
        bool> = true,
    std::enable_if_t<
        std::is_integral<Integer2>::value && std::is_signed<Integer2>::value,
        bool> = true>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  TORCH_CHECK(
      num_items >= 0,
      "When calculating block counts, the number of items must be positive!");
  TORCH_CHECK(
      threads_per_block >= 0,
      "When calculating thread counts, the number of threads must be positive!");
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

// See: cuda_calc_xblock_count_base
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<
        std::is_integral<Integer1>::value && std::is_unsigned<Integer2>::value,
        bool> = true,
    std::enable_if_t<
        std::is_integral<Integer2>::value && std::is_unsigned<Integer2>::value,
        bool> = true>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

/// Determine an appropriate CUDA block count.
///
/// See cuda_calc_xblock_count_base() for details.
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<std::is_integral<Integer1>::value, bool> = true,
    std::enable_if_t<std::is_integral<Integer2>::value, bool> = true>
constexpr uint32_t cuda_calc_block_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  // The CUDA specification at
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
  // states that the grid dimension of a kernel launch must generally
  // be <=65535. (For compute capability 3.5-* the grid's x-dimension must
  // be <=2^31-1.) Because this function does not know which dimension
  // is being calculated, we use the smaller limit.
  constexpr uint32_t max_blocks = 65535;
  return std::min(
      cuda_calc_xblock_count(num_items, threads_per_block), max_blocks);
}
