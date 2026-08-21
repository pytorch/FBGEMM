/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>
#endif
#include "./FbgemmFP16UKernelsAvx512.h" // @manual
#include "./fp32/FbgemmFP32UKernelsAvx512.h" // @manual

namespace fbgemm {

namespace {

inline __m512 load_b_avx512(const float16* addr) {
  return _mm512_cvtph_ps(
      _mm256_load_si256(reinterpret_cast<const __m256i*>(addr)));
}
inline __m512 load_b_avx512(const float* addr) {
  return _mm512_load_ps(addr);
}

// Intrinsic kernel for MSVC - shared between FP16 and FP32
template <typename T>
void gemmkernel_Avx512_fA0fB0fC0(
    GemmParams<T>* gp,
    const size_t kernel_nrows) {
  // register buffer
  __m512 zmmSum[28];
  size_t idxA = 0, idxB = 0, idxC = 0;
  // ldc in float size
  size_t ldc_floatsize = gp->ldc / sizeof(float);
  // load beta
  __m512 zmmBeta;
  if (gp->beta != 0)
    zmmBeta = _mm512_broadcastss_ps(_mm_broadcast_ss(&gp->beta));

  // outer loop - block columns
  for (uint64_t ii = 0; ii < gp->b_block_cols; ii++) {
    // reset index
    idxA = 0;
    // inner loop - k
    for (uint64_t kk = 0; kk < gp->k; kk++) {
      // load B
      __m512 zmmB0 = load_b_avx512(gp->B + idxB);
      __m512 zmmB1 = load_b_avx512(gp->B + idxB + 16);
      idxB += 32;

      // first element
      if (kk == 0) {
        if (gp->beta != 0) { // accumulate
          for (size_t jj = 0; jj < kernel_nrows; jj++) {
            // load A
            __m512 zmmA = _mm512_broadcastss_ps(
                _mm_broadcast_ss((float const*)(gp->A + idxA + jj)));
            // C = A * B + beta * C
            zmmSum[2 * jj] = _mm512_fmadd_ps(
                zmmA,
                zmmB0,
                _mm512_mul_ps(
                    zmmBeta,
                    _mm512_loadu_ps(gp->C + idxC + jj * ldc_floatsize)));
            zmmSum[2 * jj + 1] = _mm512_fmadd_ps(
                zmmA,
                zmmB1,
                _mm512_mul_ps(
                    zmmBeta,
                    _mm512_loadu_ps(gp->C + idxC + 16 + jj * ldc_floatsize)));
          }
          idxA += kernel_nrows;
        } else { // set zero
          for (size_t jj = 0; jj < kernel_nrows; jj++) {
            // load A
            __m512 zmmA = _mm512_broadcastss_ps(
                _mm_broadcast_ss((float const*)(gp->A + idxA + jj)));
            // C = A * B
            zmmSum[2 * jj] = _mm512_mul_ps(zmmA, zmmB0);
            zmmSum[2 * jj + 1] = _mm512_mul_ps(zmmA, zmmB1);
          }
          idxA += kernel_nrows;
        }
      } else {
        for (size_t jj = 0; jj < kernel_nrows; jj++) {
          // load A
          __m512 zmmA = _mm512_broadcastss_ps(
              _mm_broadcast_ss((float const*)(gp->A + idxA + jj)));
          // C = A * B + C
          zmmSum[2 * jj] = _mm512_fmadd_ps(zmmA, zmmB0, zmmSum[2 * jj]);
          zmmSum[2 * jj + 1] = _mm512_fmadd_ps(zmmA, zmmB1, zmmSum[2 * jj + 1]);
        }
        idxA += kernel_nrows;
      }
    }
    // store C
    for (size_t jj = 0; jj < kernel_nrows; jj++) {
      _mm512_storeu_ps(gp->C + idxC + jj * ldc_floatsize, zmmSum[2 * jj]);
      _mm512_storeu_ps(
          gp->C + idxC + 16 + jj * ldc_floatsize, zmmSum[2 * jj + 1]);
    }
    idxC += 32;
  }
}

} // anonymous namespace

// FP16 kernels
void NOINLINE gemmkernel_1x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 1);
}
void NOINLINE gemmkernel_2x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 2);
}
void NOINLINE gemmkernel_3x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 3);
}
void NOINLINE gemmkernel_4x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 4);
}
void NOINLINE gemmkernel_5x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 5);
}
void NOINLINE gemmkernel_6x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 6);
}
void NOINLINE gemmkernel_7x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 7);
}
void NOINLINE gemmkernel_8x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 8);
}
void NOINLINE gemmkernel_9x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 9);
}
void NOINLINE gemmkernel_10x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 10);
}
void NOINLINE gemmkernel_11x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 11);
}
void NOINLINE gemmkernel_12x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 12);
}
void NOINLINE gemmkernel_13x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 13);
}
void NOINLINE gemmkernel_14x2_Avx512_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 14);
}

// FP32 kernels
void NOINLINE gemmkernel_1x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 1);
}
void NOINLINE gemmkernel_2x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 2);
}
void NOINLINE gemmkernel_3x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 3);
}
void NOINLINE gemmkernel_4x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 4);
}
void NOINLINE gemmkernel_5x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 5);
}
void NOINLINE gemmkernel_6x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 6);
}
void NOINLINE gemmkernel_7x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 7);
}
void NOINLINE gemmkernel_8x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 8);
}
void NOINLINE gemmkernel_9x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 9);
}
void NOINLINE gemmkernel_10x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 10);
}
void NOINLINE gemmkernel_11x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 11);
}
void NOINLINE gemmkernel_12x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 12);
}
void NOINLINE gemmkernel_13x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 13);
}
void NOINLINE gemmkernel_14x2_Avx512_fp32_fA0fB0fC0(GemmParamsFP32* gp) {
  gemmkernel_Avx512_fA0fB0fC0(gp, 14);
}

} // namespace fbgemm
