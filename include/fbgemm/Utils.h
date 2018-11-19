/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <string>
#include <type_traits>

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
enum class inst_set_t { anyarch, avx2, avx512 };

/**
 * @brief Typed enum for implementation type.
 *
 * ref is reference and opt is optimized.
 */
enum class impl_type_t { ref, opt };

/**
 * @brief A struct to represent a block of a matrix.
 */
struct block_type_t {
  int row_start;
  int row_size;
  int col_start;
  int col_size;

  std::string toString() const {
    std::string out = "";
    out += "row start:" + std::to_string(row_start) + ", ";
    out += "row size:" + std::to_string(row_size) + ", ";
    out += "col start:" + std::to_string(col_start) + ", ";
    out += "col size:" + std::to_string(col_size);
    return out;
  }
};

/**
 * @brief A function to compare data in two buffers for closeness/equality.
 */
template <typename T>
int compare_buffers(
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

namespace internal {

/**
 * @brief Transpose a matrix using Intel AVX2.
 *
 * This is called if the code is running on a CPU with Intel AVX2 support.
 */
void transpose_8x8(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);

/**
 * @brief Transpose a matrix using Intel AVX512.
 *
 * This is called if the code is running on a CPU with Intel AVX512 support.
 */
void transpose_16x16(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);

} // namespace internal

} // namespace fbgemm
