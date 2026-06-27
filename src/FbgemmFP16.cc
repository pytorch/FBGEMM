/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include <array>
#include <cmath>
#include <utility>

#ifndef __aarch64__
#include "./GenerateKernelFP16FP32.h" // @manual
#endif
#ifdef __aarch64__
#include "./FbgemmFP16UKernelsSve128.h" // @manual
#endif
#ifdef FBGEMM_ENABLE_KLEIDIAI
#include "./KleidiAIFP16UKernelsNeon.h" // @manual
#endif
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmFPCommon.h"
#ifdef FBGEMM_FP16_FALLBACK_TO_REF_KERNEL
#include "fbgemm/FloatConversion.h"
#endif

namespace fbgemm {

namespace {
// optimized kernels to cover all cases
// 2 in ?x2 should be the same as kernel_ncol_blocks.
// Here with kernel_ncol_blocks = 2, we can provide up to 6x2 kernels, due to
// the restrictions of ymm register numbers (16).
#if !defined(__aarch64__)
const auto kernel_fp16_avx2 =
    makeKernelArray<float16, inst_set_t::avx2>(1, 6);
#else
constexpr kernel_array_t<float16> kernel_fp16_avx2 = {nullptr};
#endif

#if defined(__aarch64__) && defined(FBGEMM_ENABLE_FP16_SVE128)
constexpr kernel_array_t<float16> kernel_fp16_sve128 = {
    nullptr,
    gemmkernel_1x2_Sve128_fp16_fA0fB0fC0,
    gemmkernel_2x2_Sve128_fp16_fA0fB0fC0,
    gemmkernel_3x2_Sve128_fp16_fA0fB0fC0,
    gemmkernel_4x2_Sve128_fp16_fA0fB0fC0,
    gemmkernel_5x2_Sve128_fp16_fA0fB0fC0,
    gemmkernel_6x2_Sve128_fp16_fA0fB0fC0,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
#endif

#ifdef FBGEMM_ENABLE_KLEIDIAI
constexpr kernel_array_t<float16> kernel_fp16_neon = {
    nullptr,
    kleidiai::gemmkernel_1x1_Neon_fp16_fA0fB0fC0,
    kleidiai::gemmkernel_2x1_Neon_fp16_fA0fB0fC0,
    kleidiai::gemmkernel_3x1_Neon_fp16_fA0fB0fC0,
    kleidiai::gemmkernel_4x1_Neon_fp16_fA0fB0fC0,
    kleidiai::gemmkernel_5x1_Neon_fp16_fA0fB0fC0,
    kleidiai::gemmkernel_6x1_Neon_fp16_fA0fB0fC0,
    kleidiai::gemmkernel_7x1_Neon_fp16_fA0fB0fC0,
    kleidiai::gemmkernel_8x1_Neon_fp16_fA0fB0fC0,
};
#endif

#if !defined(__aarch64__)
const kernel_array_t<float16> kernel_fp16_avx512_256 = []() {
  auto k = makeKernelArray<float16, inst_set_t::avx2>(1, 6);
  for (int n = 7; n <= 14; n++)
    k[n] = generateGemmKernel<float16, inst_set_t::avx512_ymm>(n);
  return k;
}();

const auto kernel_fp16_avx512 =
    makeKernelArray<float16, inst_set_t::avx512>(1, 14);
#else
constexpr kernel_array_t<float16> kernel_fp16_avx512_256 = {nullptr};
constexpr kernel_array_t<float16> kernel_fp16_avx512 = {nullptr};
#endif

} // namespace

template <>
const isa_descriptor<float16>& getIsaHandlers(inst_set_t isa) {
  static isa_descriptor<float16> avx2_descriptor{
      kernel_fp16_avx2, partition_avx2};
  static isa_descriptor<float16> avx512_descriptor{
      kernel_fp16_avx512, partition_avx512};
  static isa_descriptor<float16> avx512_256_descriptor{
      kernel_fp16_avx512_256, partition_avx512};
#ifdef __aarch64__
#ifdef FBGEMM_ENABLE_KLEIDIAI
  static isa_descriptor<float16> neon_descriptor{
      kernel_fp16_neon, partition_neon};
#endif
#ifdef FBGEMM_ENABLE_FP16_SVE128
  static isa_descriptor<float16> sve128_descriptor{
      kernel_fp16_sve128, partition_sve128};
#endif
#endif

  switch (isa) {
    case inst_set_t::sve:
#ifdef __aarch64__
#ifdef FBGEMM_ENABLE_FP16_SVE128
      return sve128_descriptor;
#endif
    case inst_set_t::anyarch:
#ifdef FBGEMM_ENABLE_KLEIDIAI
      return neon_descriptor;
#else
      throw std::runtime_error("Unsupported uArch");
#endif
#else
    case inst_set_t::anyarch:
#endif
    case inst_set_t::avx2:
      return avx2_descriptor;

    case inst_set_t::avx512:
    case inst_set_t::avx512_vnni:
      return avx512_descriptor;

    case inst_set_t::avx512_ymm:
    case inst_set_t::avx512_vnni_ymm:
      return avx512_256_descriptor;
  }

  throw std::runtime_error("Unsupported uArch");
}

#ifdef FBGEMM_FP16_FALLBACK_TO_REF_KERNEL
template <>
FBGEMM_API void ref_kernel<float16>(
    int kernel_nrows,
    GemmParams<float16>* gp,
    const float* C_base [[maybe_unused]],
    int m_total [[maybe_unused]],
    int n_total [[maybe_unused]],
    int simd_len) {
  int kernel_ncol_blocks = 2;
  int block_col_size = simd_len * kernel_ncol_blocks;
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
#endif // FBGEMM_FP16_FALLBACK_TO_REF_KERNEL

template FBGEMM_API void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixB<float16>& Bp,
    const float beta,
    float* C,
    int thread_id,
    int num_threads);

} // namespace fbgemm
