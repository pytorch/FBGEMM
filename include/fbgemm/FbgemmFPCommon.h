/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright 2024-2025 Arm Limited and/or its affiliates
 * <open-source-office@arm.com> All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fbgemm/FbgemmPackMatrixB.h>
#include <fbgemm/Types.h>
#include <fbgemm/Utils.h>
#include <array>
#include <memory>

#if defined(FBGEMM_FP16_FALLBACK_TO_REF_KERNEL) || \
    defined(FBGEMM_FP32_FALLBACK_TO_REF_KERNEL)
#define FBGEMM_USE_REF_KERNEL
#endif

namespace fbgemm {

using partition_array_t = std::array<std::array<std::array<int, 2>, 2>, 121>;
extern partition_array_t partition_avx2;
extern partition_array_t partition_avx512;
extern partition_array_t partition_sve128;
#ifdef FBGEMM_ENABLE_KLEIDIAI
extern partition_array_t partition_neon;
#endif

template <typename T>
struct GemmParams {
  uint64_t k;
  float* A;
  const T* B;
  float beta;
  float* C;
  uint64_t ldc;
  uint64_t b_block_cols;
  uint64_t b_block_size;
};

template <>
struct GemmParams<float16> {
  uint64_t k;
  float* A;
  const float16* B;
  float beta;
  float* C;
  uint64_t ldc;
  uint64_t b_block_cols;
#ifdef FBGEMM_ENABLE_KLEIDIAI
  uint64_t lda;
#else
  uint64_t b_block_size;
#endif
};

template <>
struct GemmParams<float> {
  uint64_t k;
  float* A;
  const float* B;
  float beta;
  float* C;
  uint64_t ldc;
  uint64_t b_block_cols;
#ifdef FBGEMM_ENABLE_KLEIDIAI
  uint64_t lda;
#else
  uint64_t b_block_size;
#endif
};

template <typename T>
using funcptr_t = void (*)(GemmParams<T>*);
template <typename T>
using kernel_array_t = std::array<funcptr_t<T>, 15>;
template <typename T>
using isa_descriptor = std::tuple<kernel_array_t<T>, partition_array_t>;

template <typename T>
extern const isa_descriptor<T>& getIsaHandlers(inst_set_t isa, T);

void PackA(int nrow, int ncol, const float* from, int ldim, float* to);

// define this to debug fp16 kernel using a reference C implementation
// #define FBGEMM_FP16_FALLBACK_TO_REF_KERNEL
#ifdef FBGEMM_USE_REF_KERNEL
template <typename T>
FBGEMM_API void ref_kernel(
    int kernel_nrows,
    GemmParams<T>* gp,
    const float* C_base,
    int m_total,
    int n_total,
    int vlen);
#endif

template <typename T>
FBGEMM_API void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixB<T>& Bp,
    const float beta,
    float* C,
    int thread_id = 0,
    int num_threads = 1);

#if defined(FBGEMM_EXPORTS)
// autotuned kernel splits for various cases m = 1:mb_max
template <typename T>
void cblas_gemm_compute(
    const matrix_op_t transa [[maybe_unused]],
    const int m,
    const float* A,
    const PackedGemmMatrixB<T>& Bp,
    const float beta,
    float* C,
    int thread_id,
    int num_threads) {
  // ground truth
  assert(cpuinfo_initialize());
#ifndef __aarch64__
  assert(cpuinfo_has_x86_fma3());
  assert(cpuinfo_has_x86_f16c());
#endif
  assert(transa == matrix_op_t::NoTranspose);

  const auto iset = fbgemmInstructionSet();
  // private scratchpad storage
  static thread_local std::unique_ptr<std::array<float, 256 * 1024>> scratchpad(
      new std::array<float, 256 * 1024>());

  const auto& isaHandlers = getIsaHandlers<T>(iset, T());

  const auto& kernels = std::get<0>(isaHandlers);
  const auto& partition = std::get<1>(isaHandlers);

  // constants
  const int n = Bp.numCols(), k = Bp.numRows(), ldc = n;
  const int mb_max = 120;
#ifdef FBGEMM_USE_REF_KERNEL
  const int kernel_ncol_blocks = Bp.kernelNumColBlocks();
  // By some reason, if packed B is using packing layout for avx2, we just use
  // avx2 even if avx512 is available.
  const int simd_width =
#ifndef __aarch64__
      (iset == inst_set_t::avx512 || iset == inst_set_t::avx512_vnni) &&
          (Bp.blockColSize() == 16 * kernel_ncol_blocks)
      ? simd_info<inst_set_t::avx512>::WIDTH_32BIT_ELEMS
      : simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
#else
      simd_info<inst_set_t::sve>::WIDTH_32BIT_ELEMS;
  (void)kernel_ncol_blocks;
  (void)kernels;
#endif
#endif
  GemmParams<T> gp;
  int i_begin = 0, i_end = 0;
  i_begin = 0;
  i_end = m;
  for (auto m0 = i_begin; m0 < i_end; m0 += mb_max) {
    int mb = std::min(mb_max, i_end - m0);
    assert(mb < static_cast<int64_t>(partition.size()));
    for (auto k_ind = 0; k_ind < k; k_ind += Bp.blockRowSize()) {
      // set up proper accumulation to avoid "Nan" problem
      // accumulate of beta != 0.0
      // do not!!! accumulate otherwise
      float beta_ = beta;
      if (k_ind != 0) {
        // always accumulate with beta_ = 1.0f
        beta_ = 1.0f;
      }

      const int kb = std::min(Bp.blockRowSize(), Bp.numRows() - k_ind);

      auto m1 = m0;
      auto const num_cycles = partition[mb].size();
      for (size_t c = 0; c < num_cycles; ++c) {
        auto kernel_nrows = partition[mb][c][0];
        auto nkernel_nrows = partition[mb][c][1];
        auto m_start = m1;
        auto m_end = m1 + kernel_nrows * nkernel_nrows;
        for (auto m2 = m_start; m2 < m_end; m2 += kernel_nrows) {
          assert(kernel_nrows * kb < static_cast<int64_t>(scratchpad->size()));
          if (m != 1) {
#ifdef FBGEMM_ENABLE_KLEIDIAI
            if constexpr (
                std::is_same<T, float16>::value ||
                std::is_same<T, float>::value) {
              gp.A = const_cast<float*>(&A[m2 * k + k_ind]);
            } else {
#endif
              PackA(
                  kernel_nrows, kb, &A[m2 * k + k_ind], k, scratchpad->data());
              gp.A = scratchpad->data();
#ifdef FBGEMM_ENABLE_KLEIDIAI
            }
#endif
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
#ifdef FBGEMM_ENABLE_KLEIDIAI
          if constexpr (
              std::is_same<T, float16>::value ||
              std::is_same<T, float>::value) {
            gp.lda = k * sizeof(A[0]);
          } else {
#endif
            gp.b_block_size = gp.k * Bp.blockColSize() * sizeof(gp.B[0]);
#ifdef FBGEMM_ENABLE_KLEIDIAI
          }
#endif
          if ((n % Bp.blockColSize()) == 0) {
            int64_t jb_begin = 0, jb_end = 0;
            fbgemmPartition1D(
                thread_id, num_threads, gp.b_block_cols, jb_begin, jb_end);
            gp.B += gp.k * Bp.blockColSize() * jb_begin;
            gp.C += Bp.blockColSize() * jb_begin;
            gp.b_block_cols = jb_end - jb_begin;
            if (gp.b_block_cols) {
#ifdef FBGEMM_USE_REF_KERNEL
              if constexpr (
                  std::is_same<T, float16>::value ||
                  std::is_same<T, float>::value) {
                kernels[kernel_nrows](&gp);
              } else {
                ref_kernel<T>(kernel_nrows, &gp, C, m, n, simd_width);
              }
#else
              kernels[kernel_nrows](&gp);
#endif
            }
          } else {
            int last_blk_col = nbcol * Bp.blockColSize();
            if (nbcol) {
              int64_t jb_begin = 0, jb_end = 0;
              fbgemmPartition1D(
                  thread_id, num_threads, gp.b_block_cols, jb_begin, jb_end);
              gp.B += gp.k * Bp.blockColSize() * jb_begin;
              gp.C += Bp.blockColSize() * jb_begin;
              gp.b_block_cols = jb_end - jb_begin;
              if (gp.b_block_cols) {
#ifdef FBGEMM_USE_REF_KERNEL
                if constexpr (
                    std::is_same<T, float16>::value ||
                    std::is_same<T, float>::value) {
                  kernels[kernel_nrows](&gp);
                } else {
                  ref_kernel(kernel_nrows, &gp, C, m, n, simd_width);
                }
#else
                kernels[kernel_nrows](&gp);
#endif
              }
            }

            // use one thread to handle the fringe cases
            if (thread_id == num_threads - 1) {
              // leftover
              const int rem [[maybe_unused]] = n - last_blk_col;
              assert(rem < Bp.blockColSize());

              // small temporary buffer: the size should be larger than the
              // required kernel_nrow x kernel_ncols elements computed in the
              // registers.
              std::array<float, 14 * 32> c_tmp{0.f};
              assert(
                  static_cast<int64_t>(c_tmp.size()) >=
                  kernel_nrows * Bp.blockColSize());

              gp.B = &(Bp(k_ind, last_blk_col));
              gp.C = c_tmp.data();
              gp.ldc = Bp.blockColSize() * sizeof(C[0]);
              gp.b_block_cols = 1;
#ifdef FBGEMM_USE_REF_KERNEL
              if constexpr (
                  std::is_same<T, float16>::value ||
                  std::is_same<T, float>::value) {
                kernels[kernel_nrows](&gp);
              } else {
                ref_kernel<T>(
                    kernel_nrows, &gp, c_tmp.data(), 14, 32, simd_width);
              }
#else
              kernels[kernel_nrows](&gp);
#endif
              for (int i = 0; i < kernel_nrows; i++) {
                // Todo: use assembly
                for (int j = last_blk_col; j < n; j++) {
                  assert(
                      i * Bp.blockColSize() + (j - last_blk_col) <
                      static_cast<int64_t>(sizeof(c_tmp) / sizeof(c_tmp[0])));
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
#endif

#undef FBGEMM_USE_REF_KERNEL
} // namespace fbgemm
