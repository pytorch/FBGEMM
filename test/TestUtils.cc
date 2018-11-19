/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "TestUtils.h"
#include <gtest/gtest.h>
#include "bench/AlignedVec.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T>
int compare_validate_buffers(
    const T* ref,
    const T* test,
    int m,
    int n,
    int ld,
    T atol) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (std::is_integral<T>::value) {
        EXPECT_EQ(ref[i * ld + j], test[i * ld + j])
            << "GEMM results differ at (" << i << ", " << j
            << ") reference: " << (int64_t)ref[i * ld + j]
            << ", FBGEMM: " << (int64_t)test[i * ld + j];
      } else {
        EXPECT_LE(std::abs(ref[i * ld + j] - test[i * ld + j]), atol)
            << "GEMM results differ at (" << i << ", " << j
            << ") reference: " << ref[i * ld + j]
            << ", FBGEMM: " << test[i * ld + j];
      }
    }
  }
  return 0;
}

template int compare_validate_buffers<float>(
    const float* ref,
    const float* test,
    int m,
    int n,
    int ld,
    float atol);

template int compare_validate_buffers<int32_t>(
    const int32_t* ref,
    const int32_t* test,
    int m,
    int n,
    int ld,
    int32_t atol);

template int compare_validate_buffers<uint8_t>(
    const uint8_t* ref,
    const uint8_t* test,
    int m,
    int n,
    int ld,
    uint8_t atol);

template <typename T>
bool check_all_zero_entries(const T* test, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (test[i * n + j] != 0)
        return true;
    }
  }
  return false;
}

template bool check_all_zero_entries<float>(const float* test, int m, int n);
template bool
check_all_zero_entries<int32_t>(const int32_t* test, int m, int n);
template bool
check_all_zero_entries<uint8_t>(const uint8_t* test, int m, int n);

template <typename T>
void transpose_matrix(
    int M,
    int N,
    const T* src,
    int ld_src,
    T* dst,
    int ld_dst) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      dst[i * ld_dst + j] = src[i + j * ld_src];
    }
  } // for each output row
}

template <typename T>
void transpose_matrix(T* ref, int n, int k) {
  std::vector<T> local(n * k);
  transpose_matrix(n, k, ref, k, local.data(), n);
  memcpy(ref, local.data(), n * k * sizeof(T));
}

template void transpose_matrix<float>(
    int M,
    int N,
    const float* src,
    int ld_src,
    float* dst,
    int ld_dst);
template void transpose_matrix<int32_t>(
    int M,
    int N,
    const int32_t* src,
    int ld_src,
    int32_t* dst,
    int ld_dst);
template void transpose_matrix<uint8_t>(
    int M,
    int N,
    const uint8_t* src,
    int ld_src,
    uint8_t* dst,
    int ld_dst);
template void transpose_matrix<int8_t>(
    int M,
    int N,
    const int8_t* src,
    int ld_src,
    int8_t* dst,
    int ld_dst);

template void transpose_matrix<float>(float* ref, int n, int k);
template void transpose_matrix<int32_t>(int32_t* ref, int n, int k);
template void transpose_matrix<uint8_t>(uint8_t* ref, int n, int k);
template void transpose_matrix<int8_t>(int8_t* ref, int n, int k);

} // namespace fbgemm
