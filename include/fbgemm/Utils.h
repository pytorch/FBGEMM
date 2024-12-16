/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "./FbgemmBuild.h"
#include "./UtilsAvx2.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <type_traits>

#ifndef HAVE_SVE
#if defined(__aarch64__) && (__GNUC__ >= 8 || __clang_major__ >= 5) && \
    __ARM_FEATURE_SVE
#define HAVE_SVE 1
#else
#define HAVE_SVE 0
#endif
#endif

namespace fbgemm {

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
enum class inst_set_t {
  anyarch,
  avx2,
  avx512,
  avx512_ymm,
  avx512_vnni,
  avx512_vnni_ymm,
  sve
};

/**
 * @brief Typed enum for optimized paths for convolutions
 */
enum class optimized_conv_t {
  depthwise,
  groupwise,
  pointwise,
  fastpath1d,
  im2col,
  directconv
};

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
enum class FBGEMM_ENUM_CLASS_API layout_t { KCX, KXC };

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
    size_t max_mismatches_to_report,
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
template <typename T>
FBGEMM_API void transpose_simd(
    int64_t M,
    int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);

/**
 * @brief Explicitly set instruction set to be used
 */
FBGEMM_API void fbgemmForceIsa(inst_set_t);

/**
 * @brief Enable AVX512-256 path for Intel(r) Xeon(r) D servers
 */
FBGEMM_API void fbgemmEnableAvx512Ymm(bool);

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
 * @brief Are we running on a ARM Neon supported cpu?
 */
FBGEMM_API bool fbgemmHasArmNeonSupport();

/**
 * @brief Are we running on a ARM SVE supported cpu?
 */
FBGEMM_API bool fbgemmHasArmSveSupport();

/**
 * @brief Are we running on a ARM SVE2 supported cpu?
 */
FBGEMM_API bool fbgemmHasArmSve2Support();

/**
 * @brief Retrieve current CPU instruction set
 */
FBGEMM_API inst_set_t fbgemmInstructionSet();

/**
 * @brief Is ISA is wide vector ZMM
 */
FBGEMM_API bool isZmm(inst_set_t);

/**
 * @brief Is ISA is wide vector ZMM
 */
FBGEMM_API bool isYmm(inst_set_t);

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

/**
 * @brief A struct to represent the partition information for the threads on the
 * m and n dimensions.
 */
struct FBGEMM_API thread_type_t {
  int g_num_threads;
  int m_num_threads;
  int n_num_threads;
  int g_thread_id;
  int m_thread_id;
  int n_thread_id;

  std::string toString() const {
    std::string out = "";
    out += "g num threads: " + std::to_string(g_num_threads) + ", ";
    out += "m num threads: " + std::to_string(m_num_threads) + ", ";
    out += "n num threads: " + std::to_string(n_num_threads) + ", ";
    out += "g thread id: " + std::to_string(g_thread_id) + ", ";
    out += "m thread id: " + std::to_string(m_thread_id) + ", ";
    out += "n thread id: " + std::to_string(n_thread_id);
    return out;
  }
};

/**
 * @brief A heuristic algorithm to partition the threads across m and n
 * dimensions for parallelization, ensuring the ratio between the number of rows
 * allocated to each thread in the m dimension and the number of columns
 * allocated to each thread in the n dimension is approximately aspect_ratio.
 *
 * The less aspect_ratio is, the more favorable it is to parallelize the m
 * dimension over the n dimension.
 */
FBGEMM_API int fbgemmGet2DPartition(
    int m,
    int n,
    int nthreads,
    int n_align,
    double aspect_ratio);

/**
 * @brief A heuristic way to partition the threads across g, m and n dimensions
 * for parallelization.
 */
FBGEMM_API thread_type_t fbgemmGetThreadPartition(
    int g,
    int m,
    int n,
    int num_threads,
    int thread_id,
    int n_align = 64);

template <int SIZE, typename T = std::int32_t>
std::string arrayToString(const std::array<T, SIZE>& inp) {
  std::string out = "[";
  for (int i = 0; i < SIZE; ++i) {
    out += std::to_string(inp[i]);
    out += (i != SIZE - 1) ? std::string(", ") : std::string("]");
  }
  return out;
}

template <typename accT = std::int32_t>
bool isValidBlockingFactor(const BlockingFactors* const param) {
  constexpr bool is_32bit = std::is_same<accT, int32_t>::value;
  constexpr bool is_16bit = std::is_same<accT, int16_t>::value;
  static const auto iset = fbgemmInstructionSet();

  if (is_32bit) {
    if (param->ROW_INTERLEAVE != 4)
      return false;

    if (isZmm(iset)) {
      if (param->NR_MIN != 16 || param->NR % param->NR_MIN)
        return false;
    } else if (isYmm(iset)) {
      if (param->NR_MIN != 8 || param->NR % param->NR_MIN)
        return false;
    }
  } else if (is_16bit) {
    if (param->ROW_INTERLEAVE != 2)
      return false;

    if (isZmm(iset)) {
      if (param->NR_MIN != 32 || param->NR % param->NR_MIN)
        return false;
    } else if (isYmm(iset)) {
      if (param->NR_MIN != 16 || param->NR % param->NR_MIN)
        return false;
    }
  }

  if (param->MCB % param->MR)
    return false;
  if (param->NCB % param->NR)
    return false;
  if (isZmm(iset)) {
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

  } else if (isYmm(iset)) {
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
    std::int64_t total_work,
    std::int64_t& start,
    std::int64_t& end);

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
    std::int64_t total_work,
    int block_size,
    std::int64_t& start,
    std::int64_t& end);

/**
 * @brief A stable sorting algorithm. It sorts 8 bits at a time, hence in a
 * worst-case performing sizeof(K) / 8 passes. Providing meaningful max_value
 * may help reduce the number of passes performed by radix_sort. If
 * maybe_with_neg_vals is set to true, we are performing all possible passes,
 * up to a sign bit. If OpenMP is available in a build system, radix_sort works
 * in parallel.
 */
template <typename K, typename V>
FBGEMM_API std::pair<K*, V*> radix_sort_parallel(
    K* const inp_key_buf,
    V* const inp_value_buf,
    K* const tmp_key_buf,
    V* const tmp_value_buf,
    const int64_t elements_count,
    const int64_t max_value,
    const bool maybe_with_neg_vals = false);

/**
 * @brief Helper function that allows us to check whether radix_sort is
 * accelerated with OpenMP or not.
 */
FBGEMM_API bool is_radix_sort_accelerated_with_openmp();

/**
 * Choosing which kernel (autovec/asmjit/ref) to use for nbit-CPU-TBE
 * Available kernels:
 *   * ref: non-optimized, reference implementation that focuses on
 *      correctness, not performance
 *   * asmjit: hand-optimized kernel by having asmjit emit SIMD
 *      instructions during runtime. Only supports x86_64 CPUs with
 *      AVX2/AVX512 instruction sets
 *   * autovec: the kernel written in regular C++ code but in a
 *      way that makes compilers easier to generate vectorized SIMD
 *      instructions out of it. Supports both x86_64 and aarch64 CPUs.
 *      Currently only available on Linux.
 * How to set environment variables:
 *   * No environment variables: on x86_64 we will default to asmjit
 *      kernel, and on aarch64 and linux we will default to autovec.
 *      On non-linux aarch64 we will fall back to ref.
 *   * Set FBGEMM_NO_AUTOVEC: on aarch64 linux we will use ref. On other
 *      platforms this will have no effect.
 *   * Set FBGEMM_NO_ASMJIT: on x86_64 we will use ref. On other
 *      platforms this will have no effect.
 *   * Set FBGEMM_NO_ASMJIT AND FBGEMM_FORCE_AUTOVEC: on x86_64 we will
 *      use autovec if these two variables are set at the same time.
 *      No effect on other platforms.
 *   * FBGEMM_FORCE_AUTOVEC will override FBGEMM_NO_AUTOVEC if they
 *      are set at the same time.
 *   * These variables are considered set as long as they exist regardless
 *      of content. That means assigning values like "1", "true", "y", "0",
 *      "false" or "no" has the same effect. The easiest way of setting a
 *      variable is to prepend `<VARIABLE>=1` before the benchmarking command.
 */
FBGEMM_API bool is_autovec_disabled();
FBGEMM_API bool is_autovec_forced();
FBGEMM_API bool is_asmjit_disabled();

/**
 * @brief A function to check if the input parameter in the nbit CPU TBE kernel
 * is valid.
 */
template <typename OutType>
void nbit_embedding_sanity_check(
    // assertions are ignored in release mode, in which case these parameters
    // will be unused
    [[maybe_unused]] const int input_bit_rate,
    [[maybe_unused]] const int output_bit_rate,
    [[maybe_unused]] const bool no_bag) {
  assert(
      (input_bit_rate == 2 || input_bit_rate == 4) &&
      "input_bit_rate must be 2 or 4");
  if (std::is_same<OutType, uint8_t>::value) {
    assert(
        (no_bag && input_bit_rate == 4 && output_bit_rate == 4) &&
        "we currently only support int4 to int4 for sequential TBE");
  } else {
    assert(
        (output_bit_rate == 8 * sizeof(OutType)) &&
        "output_bit_rate should be equal to 8 * sizeof(OutType)");
  }
}

#define WARN_ONCE(...)              \
  do {                              \
    static bool _warned = false;    \
    if (!_warned) {                 \
      _warned = true;               \
      fprintf(stderr, __VA_ARGS__); \
    }                               \
  } while (0)

} // namespace fbgemm
