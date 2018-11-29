/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/FbgemmI8Spmdm.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>

#include <immintrin.h>

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double spmdm_initial_time = 0.0;
double spmdm_transpose_uint8_time = 0.0;
double spmdm_transpose_32xN_time = 0.0;
double spmdm_compute_time = 0.0;
double spmdm_transpose_Nx32_time = 0.0;
double spmdm_run_time = 0.0;
double sconv_run_time = 0.0;
#endif

using namespace std;

namespace fbgemm {

CompressedSparseColumn::CompressedSparseColumn(int num_of_rows, int num_of_cols)
    : num_rows_(num_of_rows),
      colptr_(num_of_cols + 1),
      hyper_sparse_(false),
      old_nnz_(-1) {}

double CompressedSparseColumn::Density() const {
  return (double)NumOfNonZeros() / (NumOfRows() * NumOfCols());
}

bool CompressedSparseColumn::IsHyperSparse() const {
  if (NumOfNonZeros() != old_nnz_) {
    old_nnz_ = NumOfNonZeros();
    // The number of non-zero per row is very small.
    hyper_sparse_ = (double)old_nnz_ / NumOfRows() < 0.08;
  }

  return hyper_sparse_;
}

static void transpose_8rows(
    int N,
    const uint8_t* src,
    int ld_src,
    uint8_t* dst,
    int ld_dst) {
  constexpr int M = 8;
  int j;
  // vectorized loop
  for (j = 0; j < N / 32 * 32; j += 32) {
    // a : a0 a1 ... a31
    // b : b0 b1 ... b31
    // c : c0 c1 ... c31
    // d : d0 d1 ... d31
    __m256i a = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 0 * ld_src));
    __m256i b = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 1 * ld_src));
    __m256i c = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 2 * ld_src));
    __m256i d = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 3 * ld_src));
    __m256i e = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 4 * ld_src));
    __m256i f = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 5 * ld_src));
    __m256i g = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 6 * ld_src));
    __m256i h = _mm256_lddqu_si256(
        reinterpret_cast<const __m256i*>(src + j + 7 * ld_src));

    // even-odd interleaving
    // ab_lo : a0 b0 a1 b1 ...  a7  b7 | a16 b16 ... a23 b23
    // ab_hi : a8 b8 a9 b9 ... a15 b15 | a24 b24 ... a31 b31
    // cd_lo : c0 d0 c1 d1 ...  c7  d7 | c16 d16 ... c23 d23
    // cd_hi : c8 d8 c9 d9 ... c15 d15 | c24 d24 ... c31 d31
    __m256i ab_lo = _mm256_unpacklo_epi8(a, b);
    __m256i ab_hi = _mm256_unpackhi_epi8(a, b);
    __m256i cd_lo = _mm256_unpacklo_epi8(c, d);
    __m256i cd_hi = _mm256_unpackhi_epi8(c, d);
    __m256i ef_lo = _mm256_unpacklo_epi8(e, f);
    __m256i ef_hi = _mm256_unpackhi_epi8(e, f);
    __m256i gh_lo = _mm256_unpacklo_epi8(g, h);
    __m256i gh_hi = _mm256_unpackhi_epi8(g, h);

    // 4-row interleaving but permuted at 128-bit granularity
    // abcd0 :  a0  b0  c0  d0 ...  a-d3 | a-d16 ... a-d19
    // abcd1 :  a4  b4  c4  d4 ...  a-d7 | a-d20 ... a-d23
    // abcd2 :  a8  b8  c8  d8 ... a-d11 | a-d24 ... a-d27
    // abcd3 : a12 b12 c12 d12 ... a-d15 | a-d28 ... a-d31
    __m256i abcd0 = _mm256_unpacklo_epi16(ab_lo, cd_lo);
    __m256i abcd1 = _mm256_unpackhi_epi16(ab_lo, cd_lo);
    __m256i abcd2 = _mm256_unpacklo_epi16(ab_hi, cd_hi);
    __m256i abcd3 = _mm256_unpackhi_epi16(ab_hi, cd_hi);
    __m256i efgh0 = _mm256_unpacklo_epi16(ef_lo, gh_lo);
    __m256i efgh1 = _mm256_unpackhi_epi16(ef_lo, gh_lo);
    __m256i efgh2 = _mm256_unpacklo_epi16(ef_hi, gh_hi);
    __m256i efgh3 = _mm256_unpackhi_epi16(ef_hi, gh_hi);

    // 8-row interleaving
    __m256i y0 = _mm256_unpacklo_epi32(abcd0, efgh0);
    __m256i y1 = _mm256_unpackhi_epi32(abcd0, efgh0);
    __m256i y2 = _mm256_unpacklo_epi32(abcd1, efgh1);
    __m256i y3 = _mm256_unpackhi_epi32(abcd1, efgh1);
    __m256i y4 = _mm256_unpacklo_epi32(abcd2, efgh2);
    __m256i y5 = _mm256_unpackhi_epi32(abcd2, efgh2);
    __m256i y6 = _mm256_unpacklo_epi32(abcd3, efgh3);
    __m256i y7 = _mm256_unpackhi_epi32(abcd3, efgh3);

    // Storing with 128-bit lanes are permuted so that everything is in order
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 0) * ld_dst),
        _mm256_castsi256_si128(y0));
    *reinterpret_cast<int64_t*>(dst + (j + 1) * ld_dst) =
        _mm256_extract_epi64(y0, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 2) * ld_dst),
        _mm256_castsi256_si128(y1));
    *reinterpret_cast<int64_t*>(dst + (j + 3) * ld_dst) =
        _mm256_extract_epi64(y1, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 4) * ld_dst),
        _mm256_castsi256_si128(y2));
    *reinterpret_cast<int64_t*>(dst + (j + 5) * ld_dst) =
        _mm256_extract_epi64(y2, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 6) * ld_dst),
        _mm256_castsi256_si128(y3));
    *reinterpret_cast<int64_t*>(dst + (j + 7) * ld_dst) =
        _mm256_extract_epi64(y3, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 8) * ld_dst),
        _mm256_castsi256_si128(y4));
    *reinterpret_cast<int64_t*>(dst + (j + 9) * ld_dst) =
        _mm256_extract_epi64(y4, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 10) * ld_dst),
        _mm256_castsi256_si128(y5));
    *reinterpret_cast<int64_t*>(dst + (j + 11) * ld_dst) =
        _mm256_extract_epi64(y5, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 12) * ld_dst),
        _mm256_castsi256_si128(y6));
    *reinterpret_cast<int64_t*>(dst + (j + 13) * ld_dst) =
        _mm256_extract_epi64(y6, 1);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + (j + 14) * ld_dst),
        _mm256_castsi256_si128(y7));
    *reinterpret_cast<int64_t*>(dst + (j + 15) * ld_dst) =
        _mm256_extract_epi64(y7, 1);
    *reinterpret_cast<int64_t*>(dst + (j + 16) * ld_dst) =
        _mm256_extract_epi64(y0, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 17) * ld_dst) =
        _mm256_extract_epi64(y0, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 18) * ld_dst) =
        _mm256_extract_epi64(y1, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 19) * ld_dst) =
        _mm256_extract_epi64(y1, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 20) * ld_dst) =
        _mm256_extract_epi64(y2, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 21) * ld_dst) =
        _mm256_extract_epi64(y2, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 22) * ld_dst) =
        _mm256_extract_epi64(y3, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 23) * ld_dst) =
        _mm256_extract_epi64(y3, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 24) * ld_dst) =
        _mm256_extract_epi64(y4, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 25) * ld_dst) =
        _mm256_extract_epi64(y4, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 26) * ld_dst) =
        _mm256_extract_epi64(y5, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 27) * ld_dst) =
        _mm256_extract_epi64(y5, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 28) * ld_dst) =
        _mm256_extract_epi64(y6, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 29) * ld_dst) =
        _mm256_extract_epi64(y6, 3);
    *reinterpret_cast<int64_t*>(dst + (j + 30) * ld_dst) =
        _mm256_extract_epi64(y7, 2);
    *reinterpret_cast<int64_t*>(dst + (j + 31) * ld_dst) =
        _mm256_extract_epi64(y7, 3);
  }

  // scalar loop for remainder
  for (; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      dst[j * ld_dst + i] = src[j + i * ld_src];
    }
  }
}

// TODO: fallback when AVX2 is not available
void CompressedSparseColumn::SpMDM(
    const block_type_t& block,
    const uint8_t* A,
    int lda,
    bool accumulation,
    int32_t* C,
    int ldc) const {
  int K = NumOfRows();
  int N = block.col_size;

  if (K == 0 || N == 0) {
    return;
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_very_start,
      t_start, t_end;
  double dt;
  t_start = std::chrono::high_resolution_clock::now();
  t_very_start = std::chrono::high_resolution_clock::now();
#endif

  alignas(64) uint8_t A_buffer[K * 32];
  alignas(64) int32_t C_buffer[N * 32];

  // If we compute C = C + A * B, where B is a sparse matrix in CSC format, for
  // each non-zero in B, we'd need to access the corresponding column in A.
  // This results in strided access, which we want to avoid.
  // Instead, we pre-transpose A and C, and compute C = (C^T + B^T * A^T)^T

  if (IsHyperSparse()) {
    // The cost of transpose is O(K*N) and we do O(NNZ*N) multiplications.
    // If NNZ/K is small, it's not worth doing transpose so we just use this
    // scalar loop.
    if (!accumulation) {
      for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
        for (int j = block.col_start; j < block.col_start + block.col_size;
             ++j) {
          C[(i - block.row_start) * ldc + j - block.col_start] = 0;
        }
      }
    }
    for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
      for (int k = colptr_[j]; k < colptr_[j + 1]; ++k) {
        int row = rowidx_[k];
        int w = values_[k];
        for (int i = block.row_start; i < block.row_start + block.row_size;
             ++i) {
          C[(i - block.row_start) * ldc + j - block.col_start] +=
              A[i * lda + row] * w;
        }
      }
    } // for each column of B
    return;
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
           .count();
  spmdm_initial_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif

  // Take 32 rows at a time
  int i_end = block.row_start + block.row_size;
  for (int i1 = block.row_start; i1 < i_end; i1 += 32) {
    // Transpose 32 x K submatrix of A
    if (i_end - i1 < 32) {
      alignas(64) uint8_t A_temp_buffer[K * 32];
      for (int i2 = 0; i2 < (i_end - i1) / 8 * 8; i2 += 8) {
        transpose_8rows(K, A + (i1 + i2) * lda, lda, A_buffer + i2, 32);
      }

      for (int i2 = (i_end - i1) / 8 * 8; i2 < i_end - i1; ++i2) {
        memcpy(
            A_temp_buffer + i2 * K, A + (i1 + i2) * lda, K * sizeof(uint8_t));
      }
      memset(
          A_temp_buffer + (i_end - i1) * K,
          0,
          (32 - (i_end - i1)) * K * sizeof(uint8_t));
      for (int i2 = (i_end - i1) / 8 * 8; i2 < 32; i2 += 8) {
        transpose_8rows(K, A_temp_buffer + i2 * K, K, A_buffer + i2, 32);
      }
    } else {
      for (int i2 = 0; i2 < 32; i2 += 8) {
        transpose_8rows(K, A + (i1 + i2) * lda, lda, A_buffer + i2, 32);
      }
    }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    spmdm_transpose_uint8_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif

    if (accumulation) {
      // Transpose 32 x N submatrix of C to fill N x 32 C_buffer
      transpose_simd(
          std::min(32, i_end - i1),
          N,
          reinterpret_cast<const float*>(C + (i1 - block.row_start) * ldc),
          ldc,
          reinterpret_cast<float*>(C_buffer),
          32);
    } else {
      memset(C_buffer, 0, N * 32 * sizeof(int32_t));
    }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    spmdm_transpose_32xN_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif

    for (int j = 0; j < block.col_size; ++j) {
      int j_start = j + block.col_start;
      int k = colptr_[j_start];
      int k_end_aligned =
          colptr_[j_start] + (colptr_[j_start + 1] - colptr_[j_start]) / 4 * 4;

      for (; k < k_end_aligned; k += 4) {
        __m256i w =
            _mm256_set1_epi32(*(reinterpret_cast<const int32_t*>(&values_[k])));
        array<__m256i, 4> a;
        a[0] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(&A_buffer[rowidx_[k + 0] * 32]));
        a[1] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(&A_buffer[rowidx_[k + 1] * 32]));
        a[2] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(&A_buffer[rowidx_[k + 2] * 32]));
        a[3] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(&A_buffer[rowidx_[k + 3] * 32]));

        __m256i a01_lo = _mm256_unpacklo_epi8(a[0], a[1]);
        __m256i a01_hi = _mm256_unpackhi_epi8(a[0], a[1]);
        __m256i a23_lo = _mm256_unpacklo_epi8(a[2], a[3]);
        __m256i a23_hi = _mm256_unpackhi_epi8(a[2], a[3]);

        a[0] = _mm256_unpacklo_epi16(a01_lo, a23_lo);
        a[1] = _mm256_unpackhi_epi16(a01_lo, a23_lo);
        a[2] = _mm256_unpacklo_epi16(a01_hi, a23_hi);
        a[3] = _mm256_unpackhi_epi16(a01_hi, a23_hi);

        array<__m256i, 4> ab;
        ab[0] = _mm256_maddubs_epi16(a[0], w);
        ab[1] = _mm256_maddubs_epi16(a[1], w);
        ab[2] = _mm256_maddubs_epi16(a[2], w);
        ab[3] = _mm256_maddubs_epi16(a[3], w);

        __m256i one = _mm256_set1_epi16(1);
        ab[0] = _mm256_madd_epi16(ab[0], one);
        ab[1] = _mm256_madd_epi16(ab[1], one);
        ab[2] = _mm256_madd_epi16(ab[2], one);
        ab[3] = _mm256_madd_epi16(ab[3], one);

        array<__m256i, 4> t;
        t[0] = _mm256_permute2f128_si256(ab[0], ab[1], 0x20);
        t[1] = _mm256_permute2f128_si256(ab[2], ab[3], 0x20);
        t[2] = _mm256_permute2f128_si256(ab[0], ab[1], 0x31);
        t[3] = _mm256_permute2f128_si256(ab[2], ab[3], 0x31);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 0 * 8]),
            _mm256_add_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(
                    &C_buffer[j * 32 + 0 * 8])),
                t[0]));
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 1 * 8]),
            _mm256_add_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(
                    &C_buffer[j * 32 + 1 * 8])),
                t[1]));
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 2 * 8]),
            _mm256_add_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(
                    &C_buffer[j * 32 + 2 * 8])),
                t[2]));
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 3 * 8]),
            _mm256_add_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(
                    &C_buffer[j * 32 + 3 * 8])),
                t[3]));
      }

      int remainder = colptr_[j_start + 1] - k;
      assert(remainder < 4);
      if (remainder > 0) {
        int32_t temp_w = 0;
        for (int r = 0; r < remainder; ++r) {
          (reinterpret_cast<int8_t*>(&temp_w))[r] = values_[k + r];
        }
        __m256i w = _mm256_set1_epi32(temp_w);
        array<__m256i, 4> a;
        a[0] = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(&A_buffer[rowidx_[k + 0] * 32]));
        a[1] = remainder > 1
            ? _mm256_load_si256(reinterpret_cast<const __m256i*>(
                  &A_buffer[rowidx_[k + 1] * 32]))
            : _mm256_setzero_si256();
        a[2] = remainder > 2
            ? _mm256_load_si256(reinterpret_cast<const __m256i*>(
                  &A_buffer[rowidx_[k + 2] * 32]))
            : _mm256_setzero_si256();
        a[3] = _mm256_setzero_si256();

        __m256i a01_lo = _mm256_unpacklo_epi8(a[0], a[1]);
        __m256i a01_hi = _mm256_unpackhi_epi8(a[0], a[1]);
        __m256i a23_lo = _mm256_unpacklo_epi8(a[2], a[3]);
        __m256i a23_hi = _mm256_unpackhi_epi8(a[2], a[3]);

        a[0] = _mm256_unpacklo_epi16(a01_lo, a23_lo);
        a[1] = _mm256_unpackhi_epi16(a01_lo, a23_lo);
        a[2] = _mm256_unpacklo_epi16(a01_hi, a23_hi);
        a[3] = _mm256_unpackhi_epi16(a01_hi, a23_hi);

        array<__m256i, 4> ab;
        ab[0] = _mm256_maddubs_epi16(a[0], w);
        ab[1] = _mm256_maddubs_epi16(a[1], w);
        ab[2] = _mm256_maddubs_epi16(a[2], w);
        ab[3] = _mm256_maddubs_epi16(a[3], w);

        __m256i one = _mm256_set1_epi16(1);
        ab[0] = _mm256_madd_epi16(ab[0], one);
        ab[1] = _mm256_madd_epi16(ab[1], one);
        ab[2] = _mm256_madd_epi16(ab[2], one);
        ab[3] = _mm256_madd_epi16(ab[3], one);

        array<__m256i, 4> t;
        t[0] = _mm256_permute2f128_si256(ab[0], ab[1], 0x20);
        t[1] = _mm256_permute2f128_si256(ab[2], ab[3], 0x20);
        t[2] = _mm256_permute2f128_si256(ab[0], ab[1], 0x31);
        t[3] = _mm256_permute2f128_si256(ab[2], ab[3], 0x31);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 0 * 8]),
            _mm256_add_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(
                    &C_buffer[j * 32 + 0 * 8])),
                t[0]));
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 1 * 8]),
            _mm256_add_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(
                    &C_buffer[j * 32 + 1 * 8])),
                t[1]));
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 2 * 8]),
            _mm256_add_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(
                    &C_buffer[j * 32 + 2 * 8])),
                t[2]));
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(&C_buffer[j * 32 + 3 * 8]),
            _mm256_add_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(
                    &C_buffer[j * 32 + 3 * 8])),
                t[3]));
      }
    } // for each column of B

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    spmdm_compute_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif

    // Transpose N x 32 C_buffer to fill 32 x N submatrix of C
    transpose_simd(
        N,
        std::min(32, i_end - i1),
        reinterpret_cast<const float*>(C_buffer),
        32,
        reinterpret_cast<float*>(C + (i1 - block.row_start) * ldc),
        ldc);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    spmdm_transpose_Nx32_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_very_start)
          .count();
  spmdm_run_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif
}

void CompressedSparseColumn::SparseConv(
    const conv_param_t<>& conv_p,
    const block_type_t& block,
    const uint8_t* A,
    int32_t A_zero_point,
    bool accumulation,
    int32_t* C,
    int ldc) const {
  int K = NumOfRows();
  int N = block.col_size;

  if (K == 0 || N == 0) {
    return;
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
  double dt;
  t_start = std::chrono::high_resolution_clock::now();
#endif

  // TODO: if not hyper sparse, transpose a block of A matrix as in SpMDM.
  if (!accumulation) {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      for (int j = block.col_start; j < block.col_start + block.col_size;
           ++j) {
        C[(i - block.row_start) * ldc + j - block.col_start] = 0;
      }
    }
  }
  for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
    for (int k = colptr_[j]; k < colptr_[j + 1]; ++k) {
      int v = values_[k];
      for (int i = block.row_start; i < block.row_start + block.row_size;
           ++i) {
        int ow = i % conv_p.OUT_DIM[1];
        int oh = i / conv_p.OUT_DIM[1] % conv_p.OUT_DIM[0];
        int n = i / conv_p.OUT_DIM[1] / conv_p.OUT_DIM[0];
        assert(n < conv_p.MB);
        int iw = -conv_p.pad[1] + ow * conv_p.stride[1] + kw_[k];
        int ih = -conv_p.pad[0] + oh * conv_p.stride[0] + kh_[k];

        if (ih >= 0 && ih < conv_p.IN_DIM[0] && iw >= 0 &&
            iw < conv_p.IN_DIM[1]) {
          C[(i - block.row_start) * ldc + j - block.col_start] +=
              A[((n * conv_p.IN_DIM[0] + ih) * conv_p.IN_DIM[1] + iw) *
                    conv_p.IC +
                ic_[k]] *
              v;
        } else {
          C[(i - block.row_start) * ldc + j - block.col_start] +=
              A_zero_point * v;
        }
      }
    }
  } // for each column of B

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
           .count();
  sconv_run_time += (dt);
#endif
}

} // namespace fbgemm
