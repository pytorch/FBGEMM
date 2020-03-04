/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/Utils.h"
#include <cpuinfo.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "TransposeUtils.h"

namespace fbgemm {

void * genericAlignedAlloc(size_t size, size_t align) {
  void* aligned_mem = nullptr;
  int ret;
#ifdef _MSC_VER
  aligned_mem = _aligned_malloc(size, align);
  ret = 0;
#else
  ret = posix_memalign(&aligned_mem, align, size);
#endif
  // Throw std::bad_alloc in the case of memory allocation failure.
  if (ret || aligned_mem == nullptr) {
    throw std::bad_alloc();
  }
  return aligned_mem;
}

void genericFree(void * p) {
#ifdef _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

/**
 * @brief Compare the reference and test result matrix to check the correctness.
 * @param ref The buffer for the reference result matrix.
 * @param test The buffer for the test result matrix.
 * @param m The height of the reference and test result matrix.
 * @param n The width of the reference and test result matrix.
 * @param ld The leading dimension of the reference and test result matrix.
 * @param max_mismatches_to_report The maximum number of tolerable mismatches to
 * report.
 * @param atol The tolerable error.
 * @retval false If the number of mismatches for reference and test result
 * matrix exceeds max_mismatches_to_report.
 * @retval true If the number of mismatches for reference and test result matrix
 * is tolerable.
 */
template <typename T>
int compare_buffers(
    const T* ref,
    const T* test,
    int m,
    int n,
    int ld,
    int max_mismatches_to_report,
    float atol /*=1e-3*/) {
  size_t mismatches = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      T reference = ref[i * ld + j], actual = test[i * ld + j];
      if (std::abs(reference - actual) > atol) {
        std::cout << "\tmismatch at (" << i << ", " << j << ")" << std::endl;
        if (std::is_integral<T>::value) {
          std::cout << "\t  reference:" << static_cast<int64_t>(reference)
                    << " test:" << static_cast<int64_t>(actual) << std::endl;
        } else {
          std::cout << "\t  reference:" << reference << " test:" << actual
                    << std::endl;
        }

        mismatches++;
        if (mismatches > max_mismatches_to_report) {
          return 1;
        }
      }
    }
  }
  return 0;
}

/**
 * @brief Print the matrix.
 * @param op Transpose type of the matrix.
 * @param R The height of the matrix.
 * @param C The width of the matrix.
 * @param ld The leading dimension of the matrix.
 * @param name The prefix string before printing the matrix.
 */
template <typename T>
void printMatrix(
    matrix_op_t op,
    const T* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name) {
  // R: number of rows in op(inp)
  // C: number of cols in op(inp)
  // ld: leading dimension in inp
  std::cout << name << ":"
            << "[" << R << ", " << C << "]" << std::endl;
  bool tr = (op == matrix_op_t::Transpose);
  for (auto r = 0; r < R; ++r) {
    for (auto c = 0; c < C; ++c) {
      T res = tr ? inp[c * ld + r] : inp[r * ld + c];
      if (std::is_integral<T>::value) {
        std::cout << std::setw(5) << static_cast<int64_t>(res) << " ";
      } else {
        std::cout << std::setw(5) << res << " ";
      }
    }
    std::cout << std::endl;
  }
}

template int compare_buffers<float>(
    const float* ref,
    const float* test,
    int m,
    int n,
    int ld,
    int max_mismatches_to_report,
    float atol);

template int compare_buffers<int32_t>(
    const int32_t* ref,
    const int32_t* test,
    int m,
    int n,
    int ld,
    int max_mismatches_to_report,
    float atol);

template int compare_buffers<uint8_t>(
    const uint8_t* ref,
    const uint8_t* test,
    int m,
    int n,
    int ld,
    int max_mismatches_to_report,
    float atol);

template void printMatrix<float>(
    matrix_op_t op,
    const float* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);
template void printMatrix<int8_t>(
    matrix_op_t op,
    const int8_t* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);
template void printMatrix<uint8_t>(
    matrix_op_t op,
    const uint8_t* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);
template void printMatrix<int32_t>(
    matrix_op_t op,
    const int32_t* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);

void transpose_ref(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  } // for each output row
}

void transpose_simd(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst) {
  // Run time CPU detection
  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512Support()) {
      internal::transpose_16x16(M, N, src, ld_src, dst, ld_dst);
    } else if (fbgemmHasAvx2Support()) {
      internal::transpose_8x8(M, N, src, ld_src, dst, ld_dst);
    } else {
      transpose_ref(M, N, src, ld_src, dst, ld_dst);
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

bool fbgemmHasAvx512Support() {
  return (cpuinfo_initialize() &&
      cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() &&
      cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl());
}

bool fbgemmHasAvx2Support() {
  return (cpuinfo_initialize() && cpuinfo_has_x86_avx2());
}

bool fbgemmHasAvx512VnniSupport() {
  return (cpuinfo_has_x86_avx512vnni());
}
} // namespace fbgemm
