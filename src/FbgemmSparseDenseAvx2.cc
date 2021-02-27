/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmSparse.h"

#include <immintrin.h>
#include "./MaskAvx2.h"

namespace fbgemm {
namespace internal {

void SparseDenseMMAvx2(
    int M,
    int N,
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const float* B,
    int ldb,
    float* C,
    int ldc,
    bool accum) {
  // Calcualtes accum ? C += A * B : C = A * B
  // size of values is equal to number of non-zeros (nnzs)
  // size of row_ptr is equal to M + 1
  // size of col_idx is equal to nnzs
  constexpr int VLEN = 8;
  for (int i = 0; i < M; ++i) {
    if (!accum) {
      int j = 0;
      __m256 c_v = _mm256_set1_ps(0.0f);
      for (; j < N / VLEN * VLEN; j += VLEN) {
        _mm256_storeu_ps(C + i * ldc + j, c_v);
      }
      // Handle remainder
      int rem = N - j;
      if (rem > 0) {
        __m256i mask_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            &avx2_ps_or_epi32_combined_mask[VLEN - rem]));
        _mm256_maskstore_ps(C + i * ldc + j, mask_v, c_v);
      }
    }
    int r = row_ptr[i];
    int r_end_aligned = row_ptr[i] + (row_ptr[i + 1] - row_ptr[i]) / 4 * 4;
    // unrolled by 4
    for (; r < r_end_aligned; r += 4) {
      int acbr_0 = col_idx[r + 0];
      int acbr_1 = col_idx[r + 1];
      int acbr_2 = col_idx[r + 2];
      int acbr_3 = col_idx[r + 3];
      float v_0 = values[r + 0];
      float v_1 = values[r + 1];
      float v_2 = values[r + 2];
      float v_3 = values[r + 3];
      __m256 a_v_0 = _mm256_set1_ps(v_0);
      __m256 a_v_1 = _mm256_set1_ps(v_1);
      __m256 a_v_2 = _mm256_set1_ps(v_2);
      __m256 a_v_3 = _mm256_set1_ps(v_3);
      int j = 0;
      for (; j < N / VLEN * VLEN; j += VLEN) {
        __m256 br_v_0 = _mm256_loadu_ps(B + acbr_0 * ldb + j);
        __m256 br_v_1 = _mm256_loadu_ps(B + acbr_1 * ldb + j);
        __m256 br_v_2 = _mm256_loadu_ps(B + acbr_2 * ldb + j);
        __m256 br_v_3 = _mm256_loadu_ps(B + acbr_3 * ldb + j);
        __m256 c_v = _mm256_loadu_ps(C + i * ldc + j);
        c_v = _mm256_fmadd_ps(a_v_0, br_v_0, c_v);
        c_v = _mm256_fmadd_ps(a_v_1, br_v_1, c_v);
        c_v = _mm256_fmadd_ps(a_v_2, br_v_2, c_v);
        c_v = _mm256_fmadd_ps(a_v_3, br_v_3, c_v);
        _mm256_storeu_ps(C + i * ldc + j, c_v);
      }
      // Handle remainder j loop
      int rem = N - j;
      if (rem > 0) {
        __m256i mask_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            &avx2_ps_or_epi32_combined_mask[VLEN - rem]));
        __m256 br_v_0 = _mm256_maskload_ps(B + acbr_0 * ldb + j, mask_v);
        __m256 br_v_1 = _mm256_maskload_ps(B + acbr_1 * ldb + j, mask_v);
        __m256 br_v_2 = _mm256_maskload_ps(B + acbr_2 * ldb + j, mask_v);
        __m256 br_v_3 = _mm256_maskload_ps(B + acbr_3 * ldb + j, mask_v);
        __m256 c_v = _mm256_maskload_ps(C + i * ldc + j, mask_v);
        c_v = _mm256_fmadd_ps(a_v_0, br_v_0, c_v);
        c_v = _mm256_fmadd_ps(a_v_1, br_v_1, c_v);
        c_v = _mm256_fmadd_ps(a_v_2, br_v_2, c_v);
        c_v = _mm256_fmadd_ps(a_v_3, br_v_3, c_v);
        _mm256_maskstore_ps(C + i * ldc + j, mask_v, c_v);
      }
    }
    // Handle remainder r loop
    for (; r < row_ptr[i + 1]; ++r) {
      int acbr = col_idx[r];
      float v = values[r];
      __m256 a_v = _mm256_set1_ps(v);
      int j = 0;
      for (; j < N / VLEN * VLEN; j += VLEN) {
        __m256 br_v = _mm256_loadu_ps(B + acbr * ldb + j);
        __m256 c_v = _mm256_loadu_ps(C + i * ldc + j);
        c_v = _mm256_fmadd_ps(a_v, br_v, c_v);
        _mm256_storeu_ps(C + i * ldc + j, c_v);
      }
      // Handle remainder j loop
      int rem = N - j;
      if (rem > 0) {
        __m256i mask_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            &avx2_ps_or_epi32_combined_mask[VLEN - rem]));
        __m256 br_v = _mm256_maskload_ps(B + acbr * ldb + j, mask_v);
        __m256 c_v = _mm256_maskload_ps(C + i * ldc + j, mask_v);
        c_v = _mm256_fmadd_ps(a_v, br_v, c_v);
        _mm256_maskstore_ps(C + i * ldc + j, mask_v, c_v);
      }
    }
  }
}

} // namespace internal
} // namespace fbgemm
