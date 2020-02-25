/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <array>
#include <string>
#include <type_traits>
#include "FbgemmBuild.h"
#include "UtilsAvx2.h"

#ifdef _MSC_VER
# define ALWAYS_INLINE // __forceinline
#else
# define ALWAYS_INLINE __attribute__((always_inline))
#endif

namespace fbgemm {

void * genericAlignedAlloc(size_t size, size_t alignment);
void genericFree(void * ptr);

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
void transpose_simd(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);

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

  if (is_32bit) {
    if (param->ROW_INTERLEAVE != 4)
      return false;

    if (fbgemmHasAvx512Support()) {
      if (param->NR_MIN != 16 || param->NR % param->NR_MIN)
        return false;
    } else if (fbgemmHasAvx2Support()) {
      if (param->NR_MIN != 8 || param->NR % param->NR_MIN)
        return false;
    }
  } else if (is_16bit) {
    if (param->ROW_INTERLEAVE != 2)
      return false;

    if (fbgemmHasAvx512Support()) {
      if (param->NR_MIN != 32 || param->NR % param->NR_MIN)
        return false;
    } else if (fbgemmHasAvx2Support()) {
      if (param->NR_MIN != 16 || param->NR % param->NR_MIN)
        return false;
    }
  }

  if (param->MCB % param->MR)
    return false;
  if (param->NCB % param->NR)
    return false;
  if (fbgemmHasAvx512Support()) {
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

  } else if (fbgemmHasAvx2Support()) {
    if (param->MR * (param->NR / param->NR_MIN) > 12)
      return false;
  }
  return true;
}
} // namespace fbgemm
