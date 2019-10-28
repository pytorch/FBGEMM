/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <type_traits>
#include "FbgemmBuild.h"
#include "UtilsAvx2.h"

namespace fbgemm {

/**
 * @brief Helper typedef for supported x86 ISA.
 */
enum ISA {
  unknown = -1,
  avx2 = 0,
  avx512 = 1,
  avx512_256 = 3,
  avx512_vnni = 4
};

/**
 * @brief Helper struct to type specialize for uint8 and int8 together.
 */
template <typename T>
struct is_8bit {
  static constexpr bool value =
      std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
};

/**
 * @brief Typed enum to specify matrix operations.
 */
enum class matrix_op_t { NoTranspose, Transpose };

/**
 * @brief Typed enum for supported instruction sets.
 */
enum class inst_set_t { anyarch, avx2, avx512, avx512_vnni };

/**
 * @brief Typed enum for optimized paths for convolutions
 */
enum class optimized_conv_t { depthwise, groupwise, pointwise, im2col };

/**
 * @brief Typed enum for implementation type.
 *
 * ref is reference and opt is optimized.
 */
enum class impl_type_t { ref, opt };

/**
 * @brief Typed enum to specify data layout.
 * KCX can be KCRS format or KCTRS format (e.g., for 3-D convolutions)
 * KXC can be KRSC format or KTRSC format (e.g., for 3-D convolutions)
 */
enum class layout_t { KCX, KXC };

/**
 * @brief Some commonly used variables for different instruction sets
 */
template <inst_set_t inst_set>
struct simd_info;

template <>
struct simd_info<inst_set_t::avx2> {
  static constexpr int WIDTH_BITS = 256;
  static constexpr int WIDTH_BYTES = 32;
  static constexpr int WIDTH_32BIT_ELEMS = 8;
};

template <>
struct simd_info<inst_set_t::avx512> {
  static constexpr int WIDTH_BITS = 512;
  static constexpr int WIDTH_BYTES = 64;
  static constexpr int WIDTH_32BIT_ELEMS = 16;
};

template <>
struct simd_info<inst_set_t::avx512_vnni> {
  static constexpr int WIDTH_BITS = 512;
  static constexpr int WIDTH_BYTES = 64;
  static constexpr int WIDTH_32BIT_ELEMS = 16;
};

/**
 * @brief A function to compare data in two buffers for closeness/equality.
 */
template <typename T>
FBGEMM_API int compare_buffers(
    const T* ref,
    const T* test,
    int m,
    int n,
    int ld,
    int max_mismatches_to_report,
    float atol = 1e-3);

/**
 * @brief Debugging helper.
 */
template <typename T>
void printMatrix(
    matrix_op_t trans,
    const T* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);

/**
 * @brief Transpose a matrix.
 *
 * @param M the number of rows of input matrix
 * @param N the number of columns of input matrix
 */
FBGEMM_API void transpose_simd(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);

/**
 * @brief Are we running on a Xeon-D cpu?
 */
FBGEMM_API bool fbgemmIsIntelXeonD();

/**
 * @brief Are we running on a AVX512 supported cpu?
 */
FBGEMM_API bool fbgemmHasAvx512Support();

/**
 * @brief Are we running on a AVX2 supported cpu?
 */
FBGEMM_API bool fbgemmHasAvx2Support();

/**
 * @brief Are we running on a AVX512_VNNI supported cpu?
 */
FBGEMM_API bool fbgemmHasAvx512VnniSupport();

/**
 * @brief Retrieve current CPU instruction set
 */
FBGEMM_API ISA fbgemmInstructionSet();

/**
 * @brief Helper struct to enable autotuning of FBGEMM packing and kernels.
 *
 * This structure is optional. If not used, the default values for these
 * parameters are picked up from PackingTraits-inl.h. Please see this
 * file for details on these parameters.
 */
struct FBGEMM_API BlockingFactors {
  int MR;
  int NR;
  int NR_MIN;
  int ROW_INTERLEAVE;
  int MCB;
  int KCB;
  int NCB;
};

template <int SIZE, typename T = std::int32_t>
FBGEMM_API std::string arrayToString(const std::array<T, SIZE>& inp) {
  std::string out = "[";
  for (int i = 0; i < SIZE; ++i) {
    out += std::to_string(inp[i]);
    out += (i != SIZE - 1) ? std::string(", ") : std::string("]");
  }
  return out;
}

template <typename accT = std::int32_t>
FBGEMM_API bool isValidBlockingFactor(BlockingFactors* param) {
  constexpr bool is_32bit = std::is_same<accT, int32_t>::value;
  constexpr bool is_16bit = std::is_same<accT, int16_t>::value;
  static const auto iset = fbgemmInstructionSet();

  if (is_32bit) {
    if (param->ROW_INTERLEAVE != 4)
      return false;

    if (iset == fbgemm::avx512 || iset == fbgemm::avx512_vnni) {
      if (param->NR_MIN != 16 || param->NR % param->NR_MIN)
        return false;
    } else if (iset == fbgemm::avx512_256 || iset == fbgemm::avx2) {
      if (param->NR_MIN != 8 || param->NR % param->NR_MIN)
        return false;
    }
  } else if (is_16bit) {
    if (param->ROW_INTERLEAVE != 2)
      return false;

    if (iset == fbgemm::avx512 || iset == fbgemm::avx512_vnni) {
      if (param->NR_MIN != 32 || param->NR % param->NR_MIN)
        return false;
    } else if (iset == fbgemm::avx512_256 || iset == fbgemm::avx2) {
      if (param->NR_MIN != 16 || param->NR % param->NR_MIN)
        return false;
    }
  }

  if (param->MCB % param->MR)
    return false;
  if (param->NCB % param->NR)
    return false;
  if (iset == fbgemm::avx512 || iset == fbgemm::avx512_vnni) {
    if (is_32bit) {
      // Zmm register usage for C
      if (param->MR * (param->NR / param->NR_MIN) > 28)
        return false;
    } else if (is_16bit) {
      // Zmm register usage for C + one row for loading B
      if ((param->MR * (param->NR / param->NR_MIN) +
           (param->NR / param->NR_MIN)) > 28)
        return false;
    }

  } else if (iset == fbgemm::avx512_256 || iset == fbgemm::avx2) {
    if (param->MR * (param->NR / param->NR_MIN) > 12)
      return false;
  }
  return true;
}

/**
 * @brief Partition work across given number of threads
 *
 * @param start Given thread_id should execute starting from the index
 *              start
 * @param stop Given thread_id should stop executing at the index stop
 *
 * i.e., the loop should be equivalent to for(int i = start; i < end; ++i)
 */
FBGEMM_API void fbgemmPartition1D(
    int thread_id,
    int num_threads,
    int total_work,
    int& start,
    int& end);

/**
 * @brief Partition work across given number of threads in blocks
 *        of size block_size. Each thread gets a multiple of block_size
 *        work or nothing, except the last one. The last one might
 *        receive the fringe case.
 *
 * @param start Given thread_id should execute starting from the index
 *              start
 * @param stop Given thread_id should stop executing at the index stop
 *
 * The loop can be equivalent to for(int i = start; i < end; i+=block_size)
 * except for the last thread. (i.e., thread_id = num_threads - 1)
 *
 * Example 1: block_size = 2, num_threads = 2
 *  total_work  start(th 0) end(th 0) start(th 1) end(th 1)
 *      4         0           2          2          4
 *      5         0           2          2          5
 *
 * Example 2: block_size = 2, num_threads = 3
 *  total_work  start(th 0) end(th 0) start(th 1) end(th 1)
 *      4         0           2          2          4
 *      5         0           2          2          4
 *
 *  total_work  start(th 2) end(th 2)
 *      4         4           4
 *      5         4           5
 *
 * Example 3: block_size = 2, num_threads = 4
 *  total_work  start(th 0) end(th 0) start(th 1) end(th 1)
 *      4         0           2          2          4
 *      5         0           2          2          4
 *
 *  total_work  start(th 2) end(th 2) start(th 3) end(th 3)
 *      4         4           4          4          4
 *      5         4           4          4          5
 */
FBGEMM_API void fbgemmPartition1DBlocked(
    int thread_id,
    int num_threads,
    int total_work,
    int block_size,
    int& start,
    int& end);
} // namespace fbgemm
