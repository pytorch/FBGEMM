/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/FbgemmFP16.h"

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
  static constexpr array<knl_ptr, 15> kernel = {
      {nullptr,
       gemmkernel_1x1_AVX2_fA0fB0fC0,
       gemmkernel_2x1_AVX2_fA0fB0fC0,
       gemmkernel_3x1_AVX2_fA0fB0fC0,
       gemmkernel_4x1_AVX2_fA0fB0fC0,
       gemmkernel_5x1_AVX2_fA0fB0fC0,
       gemmkernel_6x1_AVX2_fA0fB0fC0,
       gemmkernel_7x1_AVX2_fA0fB0fC0,
       gemmkernel_8x1_AVX2_fA0fB0fC0,
       gemmkernel_9x1_AVX2_fA0fB0fC0,
       gemmkernel_10x1_AVX2_fA0fB0fC0,
       gemmkernel_11x1_AVX2_fA0fB0fC0,
       gemmkernel_12x1_AVX2_fA0fB0fC0,
       gemmkernel_13x1_AVX2_fA0fB0fC0,
       gemmkernel_14x1_AVX2_fA0fB0fC0}};

  // autotuned kernel splits for various cases m = 1:mb_max
  // may need re-autotuning for new uarch
  static constexpr array<array<array<int, 2>, 2>, 121> partition = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    {
      {{ { 0, 0 }, { 0, 0 } } },
      {{ { 1, 1 }, { 0, 0 } } },
      {{ { 2, 1 }, { 0, 0 } } },
      {{ { 3, 1 }, { 0, 0 } } },
      {{ { 4, 1 }, { 0, 0 } } },
      {{ { 5, 1 }, { 0, 0 } } },
      {{ { 6, 1 }, { 0, 0 } } },
      {{ { 7, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 0, 0 } } },
      {{ { 9, 1 }, { 0, 0 } } },
      {{ { 10, 1 }, { 0, 0 } } },
      {{ { 11, 1 }, { 0, 0 } } },
      {{ { 12, 1 }, { 0, 0 } } },
      {{ { 13, 1 }, { 0, 0 } } },
      {{ { 14, 1 }, { 0, 0 } } },
      {{ { 8, 1 }, { 7, 1 } } },
      {{ { 10, 1 }, { 6, 1 } } },
      {{ { 11, 1 }, { 6, 1 } } },
      {{ { 12, 1 }, { 6, 1 } } },
      {{ { 11, 1 }, { 8, 1 } } },
      {{ { 11, 1 }, { 9, 1 } } },
      {{ { 12, 1 }, { 9, 1 } } },
      {{ { 11, 2 }, { 0, 0 } } },
      {{ { 12, 1 }, { 11, 1 } } },
      {{ { 12, 2 }, { 0, 0 } } },
      {{ { 13, 1 }, { 12, 1 } } },
      {{ { 13, 2 }, { 0, 0 } } },
      {{ { 14, 1 }, { 13, 1 } } },
      {{ { 14, 2 }, { 0, 0 } } },
      {{ { 11, 2 }, { 7, 1 } } },
      {{ { 10, 3 }, { 0, 0 } } },
      {{ { 12, 2 }, { 7, 1 } } },
      {{ { 12, 2 }, { 8, 1 } } },
      {{ { 11, 3 }, { 0, 0 } } },
      {{ { 13, 2 }, { 8, 1 } } },
      {{ { 13, 2 }, { 9, 1 } } },
      {{ { 13, 2 }, { 10, 1 } } },
      {{ { 13, 2 }, { 11, 1 } } },
      {{ { 13, 2 }, { 12, 1 } } },
      {{ { 13, 3 }, { 0, 0 } } },
      {{ { 14, 2 }, { 12, 1 } } },
      {{ { 14, 2 }, { 13, 1 } } },
      {{ { 11, 3 }, { 9, 1 } } },
      {{ { 11, 3 }, { 10, 1 } } },
      {{ { 11, 4 }, { 0, 0 } } },
      {{ { 12, 3 }, { 9, 1 } } },
      {{ { 12, 3 }, { 10, 1 } } },
      {{ { 13, 3 }, { 8, 1 } } },
      {{ { 13, 3 }, { 9, 1 } } },
      {{ { 13, 3 }, { 10, 1 } } },
      {{ { 13, 3 }, { 11, 1 } } },
      {{ { 13, 3 }, { 12, 1 } } },
      {{ { 13, 4 }, { 0, 0 } } },
      {{ { 14, 3 }, { 11, 1 } } },
      {{ { 11, 4 }, { 10, 1 } } },
      {{ { 12, 4 }, { 7, 1 } } },
      {{ { 14, 4 }, { 0, 0 } } },
      {{ { 12, 4 }, { 9, 1 } } },
      {{ { 12, 4 }, { 10, 1 } } },
      {{ { 12, 4 }, { 11, 1 } } },
      {{ { 13, 4 }, { 8, 1 } } },
      {{ { 13, 4 }, { 9, 1 } } },
      {{ { 13, 4 }, { 10, 1 } } },
      {{ { 13, 4 }, { 11, 1 } } },
      {{ { 11, 5 }, { 9, 1 } } },
      {{ { 13, 5 }, { 0, 0 } } },
      {{ { 14, 4 }, { 10, 1 } } },
      {{ { 12, 5 }, { 7, 1 } } },
      {{ { 12, 5 }, { 8, 1 } } },
      {{ { 14, 4 }, { 13, 1 } } },
      {{ { 14, 5 }, { 0, 0 } } },
      {{ { 12, 5 }, { 11, 1 } } },
      {{ { 13, 5 }, { 7, 1 } } },
      {{ { 11, 6 }, { 7, 1 } } },
      {{ { 13, 5 }, { 9, 1 } } },
      {{ { 13, 5 }, { 10, 1 } } },
      {{ { 13, 5 }, { 11, 1 } } },
      {{ { 13, 5 }, { 12, 1 } } },
      {{ { 13, 6 }, { 0, 0 } } },
      {{ { 12, 6 }, { 7, 1 } } },
      {{ { 12, 6 }, { 8, 1 } } },
      {{ { 12, 6 }, { 9, 1 } } },
      {{ { 12, 6 }, { 10, 1 } } },
      {{ { 12, 6 }, { 11, 1 } } },
      {{ { 12, 7 }, { 0, 0 } } },
      {{ { 13, 6 }, { 7, 1 } } },
      {{ { 13, 6 }, { 8, 1 } } },
      {{ { 13, 6 }, { 9, 1 } } },
      {{ { 13, 6 }, { 10, 1 } } },
      {{ { 13, 6 }, { 11, 1 } } },
      {{ { 13, 6 }, { 12, 1 } } },
      {{ { 13, 7 }, { 0, 0 } } },
      {{ { 12, 7 }, { 8, 1 } } },
      {{ { 12, 7 }, { 9, 1 } } },
      {{ { 14, 6 }, { 10, 1 } } },
      {{ { 12, 7 }, { 11, 1 } } },
      {{ { 13, 7 }, { 5, 1 } } },
      {{ { 13, 7 }, { 6, 1 } } },
      {{ { 13, 7 }, { 7, 1 } } },
      {{ { 13, 7 }, { 8, 1 } } },
      {{ { 13, 7 }, { 9, 1 } } },
      {{ { 13, 7 }, { 10, 1 } } },
      {{ { 13, 7 }, { 11, 1 } } },
      {{ { 13, 7 }, { 12, 1 } } },
      {{ { 12, 8 }, { 8, 1 } } },
      {{ { 12, 8 }, { 9, 1 } } },
      {{ { 12, 8 }, { 10, 1 } } },
      {{ { 12, 8 }, { 11, 1 } } },
      {{ { 12, 9 }, { 0, 0 } } },
      {{ { 11, 9 }, { 10, 1 } } },
      {{ { 13, 8 }, { 6, 1 } } },
      {{ { 13, 8 }, { 7, 1 } } },
      {{ { 13, 8 }, { 8, 1 } } },
      {{ { 13, 8 }, { 9, 1 } } },
      {{ { 13, 8 }, { 10, 1 } } },
      {{ { 13, 8 }, { 11, 1 } } },
      {{ { 12, 9 }, { 8, 1 } } },
      {{ { 13, 9 }, { 0, 0 } } },
      {{ { 12, 9 }, { 10, 1 } } },
      {{ { 12, 9 }, { 11, 1 } } },
      {{ { 12, 10 }, { 0, 0 } } }
     }
  };
};
constexpr array<KernelInfo::knl_ptr, 15> KernelInfo::kernel;
constexpr array<array<array<int, 2>, 2>, 121> KernelInfo::partition;

// autotuned kernel splits for various cases m = 1:mb_max
FBGEMM_API void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C) {
  // ground truth
  assert(cpuinfo_initialize());
  assert(cpuinfo_has_x86_fma3());
  assert(cpuinfo_has_x86_f16c());
  assert(transa == matrix_op_t::NoTranspose);

  // constants
  const int n = Bp.numCols(), k = Bp.numRows(), ldc = n;
  const int mb_max = 120;
  constexpr int simd_width = 8;
  constexpr int kernel_ncol_blocks = 1;
  constexpr int kernel_ncols = kernel_ncol_blocks * simd_width;

  // private scratchpad storage
  static thread_local unique_ptr<std::array<float, 256 * 1024>> scratchpad(
      new std::array<float, 256 * 1024>());

  GemmParams gp;
  for (auto m0 = 0; m0 < m; m0 += mb_max) {
    int mb = std::min(mb_max, m - m0);
    assert(mb < KernelInfo::partition.size());
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
          PackA(kernel_nrows, kb, &A[m2 * k + k_ind], k, scratchpad->data());

          int nbcol = n / Bp.blockColSize();
          gp.k = kb;
          gp.A = scratchpad->data();
          gp.B = &(Bp(k_ind, 0));
          gp.beta = &beta_;
          gp.accum = accum;
          gp.C = &C[m2 * ldc];
          gp.ldc = ldc * sizeof(C[0]);
          gp.b_block_cols = nbcol;
          gp.b_block_size = gp.k * Bp.blockColSize() * sizeof(gp.B[0]);
          if ((n % Bp.blockColSize()) == 0) {
            KernelInfo::kernel[kernel_nrows](&gp);
          } else {
            int last_blk_col = nbcol * Bp.blockColSize();
            if (nbcol) {
              KernelInfo::kernel[kernel_nrows](&gp);
            }

            // leftover
            int rem = n - last_blk_col;
            assert(rem < kernel_ncols);
            int b = (rem % simd_width) ? ((rem + simd_width) / simd_width)
                                       : (rem / simd_width);
            assert(b == 1);
            if ((rem % simd_width) == 0) {
              gp.B = &(Bp(k_ind, last_blk_col));
              gp.C = &C[m2 * ldc + last_blk_col];
              gp.b_block_cols = 1;
              KernelInfo::kernel[kernel_nrows](&gp);
            } else {
              // small temporary buffer
              float c_tmp[16 * 24] = {0};
              assert((16 * 24) > kernel_nrows * kernel_ncols);

              gp.B = &(Bp(k_ind, last_blk_col));
              gp.C = c_tmp;
              gp.ldc = 8 * sizeof(C[0]);
              gp.b_block_cols = 1;
              KernelInfo::kernel[kernel_nrows](&gp);
              for (int i = 0; i < kernel_nrows; i++) {
                // Todo: use assembly
                for (int j = last_blk_col; j < n; j++) {
                  assert(
                      i * 8 + (j - last_blk_col) <
                      sizeof(c_tmp) / sizeof(c_tmp[0]));
                  if (accum == 0) {
                    C[(m2 + i) * ldc + j] = c_tmp[i * 8 + (j - last_blk_col)];
                  } else {
                    C[(m2 + i) * ldc + j] = beta_ * C[(m2 + i) * ldc + j] +
                        c_tmp[i * 8 + (j - last_blk_col)];
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
