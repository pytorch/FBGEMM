/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <cpuinfo.h>
#include <array>
#include <cmath>
#include <utility>

#include "./FbgemmFP16Common.h"
#include "./FbgemmFP16UKernelsAvx2.h"
#include "./FbgemmFP16UKernelsAvx512.h"
#include "./FbgemmFP16UKernelsAvx512_256.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmFP16.h"

using namespace std;

namespace fbgemm {

/// class that performs packing of matrix in
/// row-major or col-major format into
/// internal packed blocked-row major format

/// Todo: make it fast with AVX2 transpose
inline void PackA(int nrow, int ncol, const float* from, int ldim, float* to) {
  // for (int r = 0; r < nrow; ++r) {
  //   for (int c = 0; c < ncol; ++c) {
  //     to[r + c * nrow] = from[r * ldim + c];
  //   }
  // }
  transpose_simd(nrow, ncol, from, ldim, to, nrow);
}

// Each kernel does the following computation that multiplies
// mb x k A sub-matrix with k x b_block_cols*64 B sub-matrix
// for (int j = 0; j < b_block_cols * 64; j += 64) {
//   for (int kk = 0; kk < k; ++k) {
//     for (int i = 0; i < mb; ++i) {
//       c[i][j:j+64] += a[i][kk] * b[kk][j:j+64]
//     }
//   }
// }

namespace KernelInfo {
using knl_ptr = funcptr_t<float16>;
// optimized kernels to cover all cases
// 2 in ?x2 should be the same as kernel_ncol_blocks.
// Here with kernel_ncol_blocks = 2, we can provide up to 6x2 kernels, due to
// the restrictions of ymm register numbers (16).
constexpr std::array<knl_ptr, 15> kernel_avx2 = {
    nullptr,
    gemmkernel_1x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_2x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_3x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_4x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_5x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_6x2_Avx2_fp16_fA0fB0fC0};

constexpr std::array<knl_ptr, 15> kernel_avx512_256 = {
    nullptr,
    gemmkernel_1x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_2x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_3x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_4x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_5x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_6x2_Avx2_fp16_fA0fB0fC0,
    gemmkernel_7x2_Avx512_256_fp16_fA0fB0fC0,
    gemmkernel_8x2_Avx512_256_fp16_fA0fB0fC0,
    gemmkernel_9x2_Avx512_256_fp16_fA0fB0fC0,
    gemmkernel_10x2_Avx512_256_fp16_fA0fB0fC0,
    gemmkernel_11x2_Avx512_256_fp16_fA0fB0fC0,
    gemmkernel_12x2_Avx512_256_fp16_fA0fB0fC0,
    gemmkernel_13x2_Avx512_256_fp16_fA0fB0fC0,
    gemmkernel_14x2_Avx512_256_fp16_fA0fB0fC0};

constexpr std::array<knl_ptr, 15> kernel_avx512 = {
    nullptr,
    gemmkernel_1x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_2x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_3x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_4x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_5x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_6x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_7x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_8x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_9x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_10x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_11x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_12x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_13x2_Avx512_fp16_fA0fB0fC0,
    gemmkernel_14x2_Avx512_fp16_fA0fB0fC0};

// autotuned kernel splits for various cases m = 1:mb_max
// may need re-autotuning for new uarch
// clang-format off
  constexpr partition_array_t partition_avx2 = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } }, // 0
      {{ { 1, 1 }, { 0, 0 } } }, // 1
      {{ { 2, 1 }, { 0, 0 } } }, // 2
      {{ { 3, 1 }, { 0, 0 } } }, // 3
      {{ { 4, 1 }, { 0, 0 } } }, // 4
      {{ { 5, 1 }, { 0, 0 } } }, // 5
      {{ { 6, 1 }, { 0, 0 } } }, // 6
      {{ { 5, 1 }, { 2, 1 } } }, // 7
      {{ { 4, 2 }, { 0, 0 } } }, // 8
      {{ { 5, 1 }, { 4, 1 } } }, // 9
      {{ { 5, 2 }, { 0, 0 } } }, // 10
      {{ { 6, 1 }, { 5, 1 } } }, // 11
      {{ { 6, 2 }, { 0, 0 } } }, // 12
      {{ { 5, 2 }, { 3, 1 } } }, // 13
      {{ { 6, 2 }, { 2, 1 } } }, // 14
      {{ { 5, 3 }, { 0, 0 } } }, // 15
      {{ { 6, 2 }, { 4, 1 } } }, // 16
      {{ { 6, 2 }, { 5, 1 } } }, // 17
      {{ { 6, 3 }, { 0, 0 } } }, // 18
      {{ { 5, 3 }, { 4, 1 } } }, // 19
      {{ { 5, 4 }, { 0, 0 } } }, // 20
      {{ { 5, 3 }, { 6, 1 } } }, // 21
      {{ { 6, 3 }, { 4, 1 } } }, // 22
      {{ { 6, 3 }, { 5, 1 } } }, // 23
      {{ { 6, 4 }, { 0, 0 } } }, // 24
      {{ { 5, 5 }, { 0, 0 } } }, // 25
      {{ { 5, 4 }, { 6, 1 } } }, // 26
      {{ { 6, 4 }, { 3, 1 } } }, // 27
      {{ { 6, 4 }, { 4, 1 } } }, // 28
      {{ { 6, 4 }, { 5, 1 } } }, // 29
      {{ { 6, 5 }, { 0, 0 } } }, // 30
      {{ { 6, 5 }, { 1, 1 } } }, // 31
      {{ { 6, 5 }, { 2, 1 } } }, // 32
      {{ { 6, 5 }, { 3, 1 } } }, // 33
      {{ { 6, 5 }, { 4, 1 } } }, // 34
      {{ { 6, 5 }, { 5, 1 } } }, // 35
      {{ { 6, 6 }, { 0, 0 } } }, // 36
      {{ { 6, 6 }, { 1, 1 } } }, // 37
      {{ { 6, 6 }, { 2, 1 } } }, // 38
      {{ { 6, 6 }, { 3, 1 } } }, // 39
      {{ { 6, 6 }, { 4, 1 } } }, // 40
      {{ { 6, 6 }, { 5, 1 } } }, // 41
      {{ { 6, 7 }, { 0, 0 } } }, // 42
      {{ { 6, 7 }, { 1, 1 } } }, // 43
      {{ { 6, 7 }, { 2, 1 } } }, // 44
      {{ { 6, 7 }, { 3, 1 } } }, // 45
      {{ { 6, 7 }, { 4, 1 } } }, // 46
      {{ { 6, 7 }, { 5, 1 } } }, // 47
      {{ { 6, 8 }, { 0, 0 } } }, // 48
      {{ { 6, 8 }, { 1, 1 } } }, // 49
      {{ { 6, 8 }, { 2, 1 } } }, // 50
      {{ { 6, 8 }, { 3, 1 } } }, // 51
      {{ { 6, 8 }, { 4, 1 } } }, // 52
      {{ { 6, 8 }, { 5, 1 } } }, // 53
      {{ { 6, 9 }, { 0, 0 } } }, // 54
      {{ { 6, 9 }, { 1, 1 } } }, // 55
      {{ { 6, 9 }, { 2, 1 } } }, // 56
      {{ { 6, 9 }, { 3, 1 } } }, // 57
      {{ { 6, 9 }, { 4, 1 } } }, // 58
      {{ { 6, 9 }, { 5, 1 } } }, // 59
      {{ { 6, 10 }, { 0, 0 } } }, // 60
      {{ { 6, 10 }, { 1, 1 } } }, // 61
      {{ { 6, 10 }, { 2, 1 } } }, // 62
      {{ { 6, 10 }, { 3, 1 } } }, // 63
      {{ { 6, 10 }, { 4, 1 } } }, // 64
      {{ { 6, 10 }, { 5, 1 } } }, // 65
      {{ { 6, 11 }, { 0, 0 } } }, // 66
      {{ { 6, 11 }, { 1, 1 } } }, // 67
      {{ { 6, 11 }, { 2, 1 } } }, // 68
      {{ { 6, 11 }, { 3, 1 } } }, // 69
      {{ { 6, 11 }, { 4, 1 } } }, // 70
      {{ { 6, 11 }, { 5, 1 } } }, // 71
      {{ { 6, 12 }, { 0, 0 } } }, // 72
      {{ { 6, 12 }, { 1, 1 } } }, // 73
      {{ { 6, 12 }, { 2, 1 } } }, // 74
      {{ { 6, 12 }, { 3, 1 } } }, // 75
      {{ { 6, 12 }, { 4, 1 } } }, // 76
      {{ { 6, 12 }, { 5, 1 } } }, // 77
      {{ { 6, 13 }, { 0, 0 } } }, // 78
      {{ { 6, 13 }, { 1, 1 } } }, // 79
      {{ { 6, 13 }, { 2, 1 } } }, // 80
      {{ { 6, 13 }, { 3, 1 } } }, // 81
      {{ { 6, 13 }, { 4, 1 } } }, // 82
      {{ { 6, 13 }, { 5, 1 } } }, // 83
      {{ { 6, 14 }, { 0, 0 } } }, // 84
      {{ { 6, 14 }, { 1, 1 } } }, // 85
      {{ { 6, 14 }, { 2, 1 } } }, // 86
      {{ { 6, 14 }, { 3, 1 } } }, // 87
      {{ { 6, 14 }, { 4, 1 } } }, // 88
      {{ { 6, 14 }, { 5, 1 } } }, // 89
      {{ { 6, 15 }, { 0, 0 } } }, // 90
      {{ { 6, 15 }, { 1, 1 } } }, // 91
      {{ { 6, 15 }, { 2, 1 } } }, // 92
      {{ { 6, 15 }, { 3, 1 } } }, // 93
      {{ { 6, 15 }, { 4, 1 } } }, // 94
      {{ { 6, 15 }, { 5, 1 } } }, // 95
      {{ { 6, 16 }, { 0, 0 } } }, // 96
      {{ { 6, 16 }, { 1, 1 } } }, // 97
      {{ { 6, 16 }, { 2, 1 } } }, // 98
      {{ { 6, 16 }, { 3, 1 } } }, // 99
      {{ { 6, 16 }, { 4, 1 } } }, // 100
      {{ { 6, 16 }, { 5, 1 } } }, // 101
      {{ { 6, 17 }, { 0, 0 } } }, // 102
      {{ { 6, 17 }, { 1, 1 } } }, // 103
      {{ { 6, 17 }, { 2, 1 } } }, // 104
      {{ { 6, 17 }, { 3, 1 } } }, // 105
      {{ { 6, 17 }, { 4, 1 } } }, // 106
      {{ { 6, 17 }, { 5, 1 } } }, // 107
      {{ { 6, 18 }, { 0, 0 } } }, // 108
      {{ { 6, 18 }, { 1, 1 } } }, // 109
      {{ { 6, 18 }, { 2, 1 } } }, // 110
      {{ { 6, 18 }, { 3, 1 } } }, // 111
      {{ { 6, 18 }, { 4, 1 } } }, // 112
      {{ { 6, 18 }, { 5, 1 } } }, // 113
      {{ { 6, 19 }, { 0, 0 } } }, // 114
      {{ { 6, 19 }, { 1, 1 } } }, // 115
      {{ { 6, 19 }, { 2, 1 } } }, // 116
      {{ { 6, 19 }, { 3, 1 } } }, // 117
      {{ { 6, 19 }, { 4, 1 } } }, // 118
      {{ { 6, 19 }, { 5, 1 } } }, // 119
      {{ { 6, 20 }, { 0, 0 } } }, // 120
    }
  };
  constexpr partition_array_t partition_avx512 = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } }, // 0
      {{ { 1, 1 }, { 0, 0 } } }, // 1
      {{ { 2, 1 }, { 0, 0 } } }, // 2
      {{ { 3, 1 }, { 0, 0 } } }, // 3
      {{ { 4, 1 }, { 0, 0 } } }, // 4
      {{ { 5, 1 }, { 0, 0 } } }, // 5
      {{ { 6, 1 }, { 0, 0 } } }, // 6
      {{ { 7, 1 }, { 0, 0 } } }, // 7
      {{ { 8, 1 }, { 0, 0 } } }, // 8
      {{ { 9, 1 }, { 0, 0 } } }, // 9
      {{ { 10, 1 }, { 0, 0 } } }, // 10
      {{ { 11, 1 }, { 0, 0 } } }, // 11
      {{ { 12, 1 }, { 0, 0 } } }, // 12
      {{ { 13, 1 }, { 0, 0 } } }, // 13
      {{ { 14, 1 }, { 0, 0 } } }, // 14
      {{ { 8, 1 }, { 7, 1 } } }, // 15
      {{ { 8, 2 }, { 0, 0 } } }, // 16
      {{ { 9, 1 }, { 8, 1 } } }, // 17
      {{ { 9, 2 }, { 0, 0 } } }, // 18
      {{ { 10, 1 }, { 9, 1 } } }, // 19
      {{ { 10, 2 }, { 0, 0 } } }, // 20
      {{ { 11, 1 }, { 10, 1 } } }, // 21
      {{ { 11, 2 }, { 0, 0 } } }, // 22
      {{ { 12, 1 }, { 11, 1 } } }, // 23
      {{ { 12, 2 }, { 0, 0 } } }, // 24
      {{ { 13, 1 }, { 12, 1 } } }, // 25
      {{ { 13, 2 }, { 0, 0 } } }, // 26
      {{ { 14, 1 }, { 13, 1 } } }, // 27
      {{ { 14, 2 }, { 0, 0 } } }, // 28
      {{ { 10, 2 }, { 9, 1 } } }, // 29
      {{ { 10, 3 }, { 0, 0 } } }, // 30
      {{ { 11, 2 }, { 9, 1 } } }, // 31
      {{ { 11, 2 }, { 10, 1 } } }, // 32
      {{ { 11, 3 }, { 0, 0 } } }, // 33
      {{ { 12, 2 }, { 10, 1 } } }, // 34
      {{ { 12, 2 }, { 11, 1 } } }, // 35
      {{ { 12, 3 }, { 0, 0 } } }, // 36
      {{ { 13, 2 }, { 11, 1 } } }, // 37
      {{ { 13, 2 }, { 12, 1 } } }, // 38
      {{ { 13, 3 }, { 0, 0 } } }, // 39
      {{ { 14, 2 }, { 12, 1 } } }, // 40
      {{ { 14, 2 }, { 13, 1 } } }, // 41
      {{ { 14, 3 }, { 0, 0 } } }, // 42
      {{ { 11, 3 }, { 10, 1 } } }, // 43
      {{ { 11, 4 }, { 0, 0 } } }, // 44
      {{ { 12, 3 }, { 9, 1 } } }, // 45
      {{ { 12, 3 }, { 10, 1 } } }, // 46
      {{ { 12, 3 }, { 11, 1 } } }, // 47
      {{ { 12, 4 }, { 0, 0 } } }, // 48
      {{ { 13, 3 }, { 10, 1 } } }, // 49
      {{ { 13, 3 }, { 11, 1 } } }, // 50
      {{ { 13, 3 }, { 12, 1 } } }, // 51
      {{ { 13, 4 }, { 0, 0 } } }, // 52
      {{ { 14, 3 }, { 11, 1 } } }, // 53
      {{ { 14, 3 }, { 12, 1 } } }, // 54
      {{ { 14, 3 }, { 13, 1 } } }, // 55
      {{ { 14, 4 }, { 0, 0 } } }, // 56
      {{ { 12, 4 }, { 9, 1 } } }, // 57
      {{ { 12, 4 }, { 10, 1 } } }, // 58
      {{ { 12, 4 }, { 11, 1 } } }, // 59
      {{ { 12, 5 }, { 0, 0 } } }, // 60
      {{ { 13, 4 }, { 9, 1 } } }, // 61
      {{ { 13, 4 }, { 10, 1 } } }, // 62
      {{ { 13, 4 }, { 11, 1 } } }, // 63
      {{ { 13, 4 }, { 12, 1 } } }, // 64
      {{ { 13, 5 }, { 0, 0 } } }, // 65
      {{ { 14, 4 }, { 10, 1 } } }, // 66
      {{ { 14, 4 }, { 11, 1 } } }, // 67
      {{ { 14, 4 }, { 12, 1 } } }, // 68
      {{ { 14, 4 }, { 13, 1 } } }, // 69
      {{ { 14, 5 }, { 0, 0 } } }, // 70
      {{ { 12, 5 }, { 11, 1 } } }, // 71
      {{ { 12, 6 }, { 0, 0 } } }, // 72
      {{ { 13, 5 }, { 8, 1 } } }, // 73
      {{ { 13, 5 }, { 9, 1 } } }, // 74
      {{ { 13, 5 }, { 10, 1 } } }, // 75
      {{ { 13, 5 }, { 11, 1 } } }, // 76
      {{ { 13, 5 }, { 12, 1 } } }, // 77
      {{ { 13, 6 }, { 0, 0 } } }, // 78
      {{ { 14, 5 }, { 9, 1 } } }, // 79
      {{ { 14, 5 }, { 10, 1 } } }, // 80
      {{ { 14, 5 }, { 11, 1 } } }, // 81
      {{ { 14, 5 }, { 12, 1 } } }, // 82
      {{ { 14, 5 }, { 13, 1 } } }, // 83
      {{ { 14, 6 }, { 0, 0 } } }, // 84
      {{ { 13, 6 }, { 7, 1 } } }, // 85
      {{ { 13, 6 }, { 8, 1 } } }, // 86
      {{ { 13, 6 }, { 9, 1 } } }, // 87
      {{ { 13, 6 }, { 10, 1 } } }, // 88
      {{ { 13, 6 }, { 11, 1 } } }, // 89
      {{ { 13, 6 }, { 12, 1 } } }, // 90
      {{ { 13, 7 }, { 0, 0 } } }, // 91
      {{ { 14, 6 }, { 8, 1 } } }, // 92
      {{ { 14, 6 }, { 9, 1 } } }, // 93
      {{ { 14, 6 }, { 10, 1 } } }, // 94
      {{ { 14, 6 }, { 11, 1 } } }, // 95
      {{ { 14, 6 }, { 12, 1 } } }, // 96
      {{ { 14, 6 }, { 13, 1 } } }, // 97
      {{ { 14, 7 }, { 0, 0 } } }, // 98
      {{ { 13, 7 }, { 8, 1 } } }, // 99
      {{ { 13, 7 }, { 9, 1 } } }, // 100
      {{ { 13, 7 }, { 10, 1 } } }, // 101
      {{ { 13, 7 }, { 11, 1 } } }, // 102
      {{ { 13, 7 }, { 12, 1 } } }, // 103
      {{ { 13, 8 }, { 0, 0 } } }, // 104
      {{ { 14, 7 }, { 7, 1 } } }, // 105
      {{ { 14, 7 }, { 8, 1 } } }, // 106
      {{ { 14, 7 }, { 9, 1 } } }, // 107
      {{ { 14, 7 }, { 10, 1 } } }, // 108
      {{ { 14, 7 }, { 11, 1 } } }, // 109
      {{ { 14, 7 }, { 12, 1 } } }, // 110
      {{ { 14, 7 }, { 13, 1 } } }, // 111
      {{ { 14, 8 }, { 0, 0 } } }, // 112
      {{ { 13, 8 }, { 9, 1 } } }, // 113
      {{ { 13, 8 }, { 10, 1 } } }, // 114
      {{ { 13, 8 }, { 11, 1 } } }, // 115
      {{ { 13, 8 }, { 12, 1 } } }, // 116
      {{ { 13, 9 }, { 0, 0 } } }, // 117
      {{ { 14, 8 }, { 6, 1 } } }, // 118
      {{ { 14, 8 }, { 7, 1 } } }, // 119
      {{ { 14, 8 }, { 8, 1 } } }, // 120
    }
  };
// clang-format on
}; // namespace KernelInfo

// define this to debug fp16 kernel using a reference C implementation
// #define FBGEMM_FP16_FALLBACK_TO_REF_KERNEL
#if defined(FBGEMM_FP16_FALLBACK_TO_REF_KERNEL)
namespace {
void ref_kernel(
    int kernel_nrows,
    GemmParams<float16>* gp,
    const float* C_base,
    int m_total,
    int n_total,
    bool use_avx512) {
  int vlen = use_avx512 ? simd_info<inst_set_t::avx512>::WIDTH_32BIT_ELEMS
                        : simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
  int kernel_ncol_blocks = 2;
  int block_col_size = vlen * kernel_ncol_blocks;
  for (int jb = 0; jb < gp->b_block_cols; ++jb) {
    for (int k = 0; k < gp->k; ++k) {
      for (int i = 0; i < kernel_nrows; ++i) {
        float a = gp->A[i + k * kernel_nrows];
        for (int j = 0; j < block_col_size; ++j) {
          float* C_ptr =
              gp->C + i * (gp->ldc / sizeof(float)) + jb * block_col_size + j;
          assert(C_ptr < C_base + m_total * n_total);
          float b =
              cpu_half2float(gp->B[(jb * gp->k + k) * block_col_size + j]);
          if (k == 0) {
            if (gp->beta) {
              *C_ptr = std::fma(a, b, (gp->beta) * (*C_ptr));
            } else {
              *C_ptr = a * b;
            }
          } else {
            *C_ptr = std::fma(a, b, *C_ptr);
          }
        }
      }
    }
  }
}
} // anonymous namespace
#endif // FBGEMM_FP16_FALLBACK_TO_REF_KERNEL

// autotuned kernel splits for various cases m = 1:mb_max
void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C,
    int thread_id,
    int num_threads) {
  // ground truth
  assert(cpuinfo_initialize());
  assert(cpuinfo_has_x86_fma3());
  assert(cpuinfo_has_x86_f16c());
  assert(transa == matrix_op_t::NoTranspose);

  // constants
  const int n = Bp.numCols(), k = Bp.numRows(), ldc = n;
  const int mb_max = 120;

  static inst_set_t isa = fbgemmInstructionSet();
  bool use_avx512 = isZmm(isa);

  // private scratchpad storage
  static thread_local unique_ptr<std::array<float, 256 * 1024>> scratchpad(
      new std::array<float, 256 * 1024>());

  GemmParams<float16> gp;

  const auto& kernels = use_avx512
      ? KernelInfo::kernel_avx512
      : isa == inst_set_t::avx512_ymm ? KernelInfo::kernel_avx512_256
                                      : KernelInfo::kernel_avx2;
  const auto& partition = use_avx512 || isa == inst_set_t::avx512_ymm
      ? KernelInfo::partition_avx512
      : KernelInfo::partition_avx2;

  int i_begin, i_end;
  // fbgemmPartition1D(thread_id, num_threads, m, i_begin, i_end);
  i_begin = 0;
  i_end = m;
  for (auto m0 = i_begin; m0 < i_end; m0 += mb_max) {
    int mb = std::min(mb_max, i_end - m0);
    assert(mb < partition.size());
    for (auto k_ind = 0; k_ind < k; k_ind += Bp.blockRowSize()) {
      // set up proper accumulation to avoid "Nan" problem
      float beta_;
      if (k_ind == 0) {
        // accumulate of beta != 0.0
        // do not!!! accumulate otherwise
        beta_ = beta;
      } else {
        // always accumulate with beta_ = 1.0f
        beta_ = 1.0f;
      }

      const int kb = std::min(Bp.blockRowSize(), Bp.numRows() - k_ind);

      auto m1 = m0;
      for (auto c = 0; c < 2; c++) {
        auto kernel_nrows = partition[mb][c][0];
        auto nkernel_nrows = partition[mb][c][1];

        auto m_start = m1, m_end = m1 + kernel_nrows * nkernel_nrows;
        for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
          assert(kernel_nrows * kb < scratchpad->size());
          if (m != 1) {
            PackA(kernel_nrows, kb, &A[m2 * k + k_ind], k, scratchpad->data());
            gp.A = scratchpad->data();
          } else {
            // When m == 1, it is actually vector matrix multiplication. We
            // don't need to do the transposition for packA here. Instead, we
            // can just pass the pointer of the original A matrix buffer to the
            // packed A buffer.
            gp.A = const_cast<float*>(&A[k_ind]);
          }

          int nbcol = n / Bp.blockColSize();
          gp.k = kb;
          gp.B = &(Bp(k_ind, 0));
          gp.beta = beta_;
          gp.C = &C[m2 * ldc];
          gp.ldc = ldc * sizeof(C[0]);
          gp.b_block_cols = nbcol;
          gp.b_block_size = gp.k * Bp.blockColSize() * sizeof(gp.B[0]);

          if ((n % Bp.blockColSize()) == 0) {
            int jb_begin, jb_end;
            fbgemmPartition1D(
                thread_id, num_threads, gp.b_block_cols, jb_begin, jb_end);
            gp.B += gp.k * Bp.blockColSize() * jb_begin;
            gp.C += Bp.blockColSize() * jb_begin;
            gp.b_block_cols = jb_end - jb_begin;
            if (gp.b_block_cols) {
#if defined(FBGEMM_FP16_FALLBACK_TO_REF_KERNEL)
              ref_kernel(kernel_nrows, &gp, C, m, n, use_avx512);
#else
              kernels[kernel_nrows](&gp);
#endif
            }
          } else {
            int last_blk_col = nbcol * Bp.blockColSize();
            if (nbcol) {
              int jb_begin, jb_end;
              fbgemmPartition1D(
                  thread_id, num_threads, gp.b_block_cols, jb_begin, jb_end);
              gp.B += gp.k * Bp.blockColSize() * jb_begin;
              gp.C += Bp.blockColSize() * jb_begin;
              gp.b_block_cols = jb_end - jb_begin;
              if (gp.b_block_cols) {
#if defined(FBGEMM_FP16_FALLBACK_TO_REF_KERNEL)
                ref_kernel(kernel_nrows, &gp, C, m, n, use_avx512);
#else
                kernels[kernel_nrows](&gp);
#endif
              }
            }

            // use one thread to handle the fringe cases
            if (thread_id == num_threads - 1) {
              // leftover
              int rem = n - last_blk_col;
              assert(rem < Bp.blockColSize());

              // small temporary buffer: the size should be larger than the
              // required kernel_nrow x kernel_ncols elements computed in the
              // registers.
              float c_tmp[14 * 32] = {0};
              assert(
                  sizeof(c_tmp) / sizeof(c_tmp[0]) >=
                  kernel_nrows * Bp.blockColSize());

              gp.B = &(Bp(k_ind, last_blk_col));
              gp.C = c_tmp;
              gp.ldc = Bp.blockColSize() * sizeof(C[0]);
              gp.b_block_cols = 1;
#if defined(FBGEMM_FP16_FALLBACK_TO_REF_KERNEL)
              ref_kernel(kernel_nrows, &gp, c_tmp, 14, 32, use_avx512);
#else
              kernels[kernel_nrows](&gp);
#endif
              for (int i = 0; i < kernel_nrows; i++) {
                // Todo: use assembly
                for (int j = last_blk_col; j < n; j++) {
                  assert(
                      i * Bp.blockColSize() + (j - last_blk_col) <
                      sizeof(c_tmp) / sizeof(c_tmp[0]));
                  if (beta_ == 0.f) {
                    C[(m2 + i) * ldc + j] =
                        c_tmp[i * Bp.blockColSize() + (j - last_blk_col)];
                  } else {
                    C[(m2 + i) * ldc + j] = beta_ * C[(m2 + i) * ldc + j] +
                        c_tmp[i * Bp.blockColSize() + (j - last_blk_col)];
                  }
                }
              }
            }
          }
        }
        m1 += kernel_nrows * nkernel_nrows;
      }
    }
  }
}

} // namespace fbgemm
