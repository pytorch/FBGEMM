/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/FbgemmFP16.h"

#include "fbgemm/Fbgemm.h"

#include <cpuinfo.h>
#include <array>
#include <utility>

#include "FbgemmFP16UKernelsAvx2.h"

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

struct KernelInfo {
  using knl_ptr = funcptr_fp16;
  // optimized kernels to cover all cases
  // 2 in ?x2 should be the same as kernel_ncol_blocks.
  // Here with kernel_ncol_blocks = 2, we can provide up to 6x2 kernels, due to
  // the restrictions of ymm register numbers (16).
  static constexpr knl_ptr kernel[7] = {
      nullptr,
      gemmkernel_1x2_AVX2_fA0fB0fC0,
      gemmkernel_2x2_AVX2_fA0fB0fC0,
      gemmkernel_3x2_AVX2_fA0fB0fC0,
      gemmkernel_4x2_AVX2_fA0fB0fC0,
      gemmkernel_5x2_AVX2_fA0fB0fC0,
      gemmkernel_6x2_AVX2_fA0fB0fC0
  };

  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  // clang-format off
  static constexpr int partition[121][2][2] = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
      { { 0, 0 }, { 0, 0 } }, // 0
      { { 1, 1 }, { 0, 0 } }, // 1
      { { 2, 1 }, { 0, 0 } }, // 2
      { { 3, 1 }, { 0, 0 } }, // 3
      { { 4, 1 }, { 0, 0 } }, // 4
      { { 5, 1 }, { 0, 0 } }, // 5
      { { 6, 1 }, { 0, 0 } }, // 6
      { { 5, 1 }, { 2, 1 } }, // 7
      { { 4, 2 }, { 0, 0 } }, // 8
      { { 5, 1 }, { 4, 1 } }, // 9
      { { 5, 2 }, { 0, 0 } }, // 10
      { { 6, 1 }, { 5, 1 } }, // 11
      { { 6, 2 }, { 0, 0 } }, // 12
      { { 5, 2 }, { 3, 1 } }, // 13
      { { 6, 2 }, { 2, 1 } }, // 14
      { { 5, 3 }, { 0, 0 } }, // 15
      { { 6, 2 }, { 4, 1 } }, // 16
      { { 6, 2 }, { 5, 1 } }, // 17
      { { 6, 3 }, { 0, 0 } }, // 18
      { { 5, 3 }, { 4, 1 } }, // 19
      { { 5, 4 }, { 0, 0 } }, // 20
      { { 5, 3 }, { 6, 1 } }, // 21
      { { 6, 3 }, { 4, 1 } }, // 22
      { { 6, 3 }, { 5, 1 } }, // 23
      { { 6, 4 }, { 0, 0 } }, // 24
      { { 5, 5 }, { 0, 0 } }, // 25
      { { 5, 4 }, { 6, 1 } }, // 26
      { { 6, 4 }, { 3, 1 } }, // 27
      { { 6, 4 }, { 4, 1 } }, // 28
      { { 6, 4 }, { 5, 1 } }, // 29
      { { 6, 5 }, { 0, 0 } }, // 30
      { { 6, 5 }, { 1, 1 } }, // 31
      { { 6, 5 }, { 2, 1 } }, // 32
      { { 6, 5 }, { 3, 1 } }, // 33
      { { 6, 5 }, { 4, 1 } }, // 34
      { { 6, 5 }, { 5, 1 } }, // 35
      { { 6, 6 }, { 0, 0 } }, // 36
      { { 6, 6 }, { 1, 1 } }, // 37
      { { 6, 6 }, { 2, 1 } }, // 38
      { { 6, 6 }, { 3, 1 } }, // 39
      { { 6, 6 }, { 4, 1 } }, // 40
      { { 6, 6 }, { 5, 1 } }, // 41
      { { 6, 7 }, { 0, 0 } }, // 42
      { { 6, 7 }, { 1, 1 } }, // 43
      { { 6, 7 }, { 2, 1 } }, // 44
      { { 6, 7 }, { 3, 1 } }, // 45
      { { 6, 7 }, { 4, 1 } }, // 46
      { { 6, 7 }, { 5, 1 } }, // 47
      { { 6, 8 }, { 0, 0 } }, // 48
      { { 6, 8 }, { 1, 1 } }, // 49
      { { 6, 8 }, { 2, 1 } }, // 50
      { { 6, 8 }, { 3, 1 } }, // 51
      { { 6, 8 }, { 4, 1 } }, // 52
      { { 6, 8 }, { 5, 1 } }, // 53
      { { 6, 9 }, { 0, 0 } }, // 54
      { { 6, 9 }, { 1, 1 } }, // 55
      { { 6, 9 }, { 2, 1 } }, // 56
      { { 6, 9 }, { 3, 1 } }, // 57
      { { 6, 9 }, { 4, 1 } }, // 58
      { { 6, 9 }, { 5, 1 } }, // 59
      { { 6, 10 }, { 0, 0 } }, // 60
      { { 6, 10 }, { 1, 1 } }, // 61
      { { 6, 10 }, { 2, 1 } }, // 62
      { { 6, 10 }, { 3, 1 } }, // 63
      { { 6, 10 }, { 4, 1 } }, // 64
      { { 6, 10 }, { 5, 1 } }, // 65
      { { 6, 11 }, { 0, 0 } }, // 66
      { { 6, 11 }, { 1, 1 } }, // 67
      { { 6, 11 }, { 2, 1 } }, // 68
      { { 6, 11 }, { 3, 1 } }, // 69
      { { 6, 11 }, { 4, 1 } }, // 70
      { { 6, 11 }, { 5, 1 } }, // 71
      { { 6, 12 }, { 0, 0 } }, // 72
      { { 6, 12 }, { 1, 1 } }, // 73
      { { 6, 12 }, { 2, 1 } }, // 74
      { { 6, 12 }, { 3, 1 } }, // 75
      { { 6, 12 }, { 4, 1 } }, // 76
      { { 6, 12 }, { 5, 1 } }, // 77
      { { 6, 13 }, { 0, 0 } }, // 78
      { { 6, 13 }, { 1, 1 } }, // 79
      { { 6, 13 }, { 2, 1 } }, // 80
      { { 6, 13 }, { 3, 1 } }, // 81
      { { 6, 13 }, { 4, 1 } }, // 82
      { { 6, 13 }, { 5, 1 } }, // 83
      { { 6, 14 }, { 0, 0 } }, // 84
      { { 6, 14 }, { 1, 1 } }, // 85
      { { 6, 14 }, { 2, 1 } }, // 86
      { { 6, 14 }, { 3, 1 } }, // 87
      { { 6, 14 }, { 4, 1 } }, // 88
      { { 6, 14 }, { 5, 1 } }, // 89
      { { 6, 15 }, { 0, 0 } }, // 90
      { { 6, 15 }, { 1, 1 } }, // 91
      { { 6, 15 }, { 2, 1 } }, // 92
      { { 6, 15 }, { 3, 1 } }, // 93
      { { 6, 15 }, { 4, 1 } }, // 94
      { { 6, 15 }, { 5, 1 } }, // 95
      { { 6, 16 }, { 0, 0 } }, // 96
      { { 6, 16 }, { 1, 1 } }, // 97
      { { 6, 16 }, { 2, 1 } }, // 98
      { { 6, 16 }, { 3, 1 } }, // 99
      { { 6, 16 }, { 4, 1 } }, // 100
      { { 6, 16 }, { 5, 1 } }, // 101
      { { 6, 17 }, { 0, 0 } }, // 102
      { { 6, 17 }, { 1, 1 } }, // 103
      { { 6, 17 }, { 2, 1 } }, // 104
      { { 6, 17 }, { 3, 1 } }, // 105
      { { 6, 17 }, { 4, 1 } }, // 106
      { { 6, 17 }, { 5, 1 } }, // 107
      { { 6, 18 }, { 0, 0 } }, // 108
      { { 6, 18 }, { 1, 1 } }, // 109
      { { 6, 18 }, { 2, 1 } }, // 110
      { { 6, 18 }, { 3, 1 } }, // 111
      { { 6, 18 }, { 4, 1 } }, // 112
      { { 6, 18 }, { 5, 1 } }, // 113
      { { 6, 19 }, { 0, 0 } }, // 114
      { { 6, 19 }, { 1, 1 } }, // 115
      { { 6, 19 }, { 2, 1 } }, // 116
      { { 6, 19 }, { 3, 1 } }, // 117
      { { 6, 19 }, { 4, 1 } }, // 118
      { { 6, 19 }, { 5, 1 } }, // 119
      { { 6, 20 }, { 0, 0 } }, // 120
  };
  // clang-format on
};
constexpr KernelInfo::knl_ptr KernelInfo::kernel[7];;
constexpr int KernelInfo::partition[121][2][2];

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
  constexpr int simd_width = 8;
  int kernel_ncol_blocks = Bp.kernelNumColBlocks();
  int kernel_ncols = kernel_ncol_blocks * simd_width;

  // private scratchpad storage
  static thread_local unique_ptr<std::array<float, 256 * 1024>> scratchpad(
      new std::array<float, 256 * 1024>());

  GemmParams gp;

  int i_begin, i_end;
  // fbgemmGetRange(num_threads, thread_id, m, 1, i_begin, i_end);
  i_begin = 0;
  i_end = m;
  for (auto m0 = i_begin; m0 < i_end; m0 += mb_max) {
    int mb = std::min(mb_max, i_end - m0);
    assert(mb < sizeof(KernelInfo::partition) / sizeof(KernelInfo::partition[0]));
    for (auto k_ind = 0; k_ind < k; k_ind += Bp.blockRowSize()) {
      // set up proper accumulation to avoid "Nan" problem
      float beta_;
      uint64_t accum;
      if (k_ind == 0) {
        // accumulate of beta != 0.0
        // do not!!! accumulate otherwise
        beta_ = beta;
        accum = (beta_ == 0.0f) ? 0 : 1;
      } else {
        // always accumulate with beta_ = 1.0f
        beta_ = 1.0f;
        accum = 1;
      }

      const int kb = std::min(Bp.blockRowSize(), Bp.numRows() - k_ind);

      auto m1 = m0;
      for (auto c = 0; c < 2; c++) {
        auto kernel_nrows = KernelInfo::partition[mb][c][0];
        auto nkernel_nrows = KernelInfo::partition[mb][c][1];

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
          gp.beta = &beta_;
          gp.accum = accum;
          gp.C = &C[m2 * ldc];
          gp.ldc = ldc * sizeof(C[0]);
          gp.b_block_cols = nbcol;
          gp.b_block_size = gp.k * Bp.blockColSize() * sizeof(gp.B[0]);

          if ((n % Bp.blockColSize()) == 0) {
            int jb_begin, jb_end;
            fbgemmGetRange(
                num_threads, thread_id, gp.b_block_cols, 1, jb_begin, jb_end);
            gp.B += gp.k * Bp.blockColSize() * jb_begin;
            gp.C += Bp.blockColSize() * jb_begin;
            gp.b_block_cols = jb_end - jb_begin;
            if (gp.b_block_cols) {
              KernelInfo::kernel[kernel_nrows](&gp);
            }
          } else {
            int last_blk_col = nbcol * Bp.blockColSize();
            if (nbcol) {
              int jb_begin, jb_end;
              fbgemmGetRange(
                  num_threads, thread_id, gp.b_block_cols, 1, jb_begin, jb_end);
              gp.B += gp.k * Bp.blockColSize() * jb_begin;
              gp.C += Bp.blockColSize() * jb_begin;
              gp.b_block_cols = jb_end - jb_begin;
              if (gp.b_block_cols) {
                KernelInfo::kernel[kernel_nrows](&gp);
              }
            }

            // use one thread to handle the fringe cases
            if (thread_id == num_threads - 1) {
              // leftover
              int rem = n - last_blk_col;
              assert(rem < kernel_ncols);

              if ((rem % Bp.blockColSize()) == 0) {
                gp.B = &(Bp(k_ind, last_blk_col));
                gp.C = &C[m2 * ldc + last_blk_col];
                gp.b_block_cols = 1;
                KernelInfo::kernel[kernel_nrows](&gp);
              } else {
                // small temporary buffer: the size should be larger than the
                // required kernel_nrow x kernel_ncols elements computed in the
                // registers.
                float c_tmp[16 * 24] = {0};
                assert((16 * 24) > kernel_nrows * kernel_ncols);

                gp.B = &(Bp(k_ind, last_blk_col));
                gp.C = c_tmp;
                gp.ldc = kernel_ncols * sizeof(C[0]);
                gp.b_block_cols = 1;
                KernelInfo::kernel[kernel_nrows](&gp);
                for (int i = 0; i < kernel_nrows; i++) {
                  // Todo: use assembly
                  for (int j = last_blk_col; j < n; j++) {
                    assert(
                        i * kernel_ncols + (j - last_blk_col) <
                        sizeof(c_tmp) / sizeof(c_tmp[0]));
                    if (accum == 0) {
                      C[(m2 + i) * ldc + j] =
                          c_tmp[i * kernel_ncols + (j - last_blk_col)];
                    } else {
                      C[(m2 + i) * ldc + j] = beta_ * C[(m2 + i) * ldc + j] +
                          c_tmp[i * kernel_ncols + (j - last_blk_col)];
                    }
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
