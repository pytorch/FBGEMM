/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/FbgemmFakeFP16.h"
#include "fbgemm/FbgemmConvert.h"

#include "src/RefImplementations.h"

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_BLAS
#include <cblas.h>
#endif

#include <cpuinfo.h>
#include <memory>
#include <utility>
#include <vector>

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double malloc_time = 0.0;
double A_fp16_to_fp32_time = 0.0;
double B_fp16_to_fp32_time = 0.0;
double C_fp16_to_fp32_time = 0.0;
double computing_time = 0.0;
double C_fp32_to_fp16_time = 0.0;
double run_time = 0.0;
#endif

using namespace std;

namespace fbgemm {

void RoundToFloat16(
    const float* input,
    float* output,
    int size,
    bool clamp,
    bool clamp_denorms) {
  std::vector<fbgemm::float16> data_fp16(size);
  fbgemm::FloatToFloat16_simd(input, &(data_fp16[0]), size);
  fbgemm::Float16ToFloat_simd(&(data_fp16[0]), output, size);

  if (clamp) {
    // TODO: Use intrinsics to optimize clamping performance.
    for (int i = 0; i < size; ++i) {
      output[i] = std::max(std::min(output[i], 65504.0f), -65504.0f);
    }
  }

  if (clamp_denorms) {
    union epsilon_t {
      float f;
      uint32_t i;
    };

    union epsilon_t epsilon;
    epsilon.i = 0x38800000u; // 1 / 16384

    for (int i = 0; i < size; ++i) {
      if (std::abs(output[i]) < epsilon.f) {
        output[i] = 0.0;
      }
    }
  }
}

void fbgemmFakeFP16(
    const matrix_op_t transa,
    const matrix_op_t transb,
    int m,
    int n,
    int k,
    const float16* A_float16,
    const float16* B_float16,
    float beta,
    float16* C_float16) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_very_start,
      t_start, t_end;
  double dt;
  t_start = std::chrono::high_resolution_clock::now();
  t_very_start = std::chrono::high_resolution_clock::now();
#endif

  // float16 -> fp32
  float* A_fp32 =
      static_cast<float*>(fbgemmAlignedAlloc(64, m * k * sizeof(float)));
  float* B_fp32 =
      static_cast<float*>(fbgemmAlignedAlloc(64, k * n * sizeof(float)));
  float* C_fp32 =
      static_cast<float*>(fbgemmAlignedAlloc(64, m * n * sizeof(float)));

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
           .count();
  malloc_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif

  int lda = (transa == matrix_op_t::NoTranspose ? k : m);
  int ldb = (transb == matrix_op_t::NoTranspose ? n : k);
  int ldc = n;

  // Float16ToFloat_ref(A_float16, A_fp32, m * k);
  Float16ToFloat_simd(A_float16, A_fp32, m * k);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
           .count();
  A_fp16_to_fp32_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif

  // Float16ToFloat_ref(B_float16, B_fp32, k * n);
  Float16ToFloat_simd(B_float16, B_fp32, k * n);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
           .count();
  B_fp16_to_fp32_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif

  if (beta != 0.0f) {
    // Float16ToFloat_ref(C_float16, C_fp32, m * n);
    Float16ToFloat_simd(C_float16, C_fp32, m * n);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    C_fp16_to_fp32_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif
  }

  // fp32 GEMM
  float alpha = 1.f;
#if defined(USE_MKL) || defined(USE_BLAS)
  cblas_sgemm(
      CblasRowMajor,
      transa == matrix_op_t::NoTranspose ? CblasNoTrans : CblasTrans,
      transb == matrix_op_t::NoTranspose ? CblasNoTrans : CblasTrans,
      m,
      n,
      k,
      alpha,
      A_fp32,
      lda,
      B_fp32,
      ldb,
      beta,
      C_fp32,
      ldc);
#else
  cblas_sgemm_ref(
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A_fp32,
      lda,
      B_fp32,
      ldb,
      beta,
      C_fp32,
      ldc);
#endif

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
           .count();
  computing_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif

  // fp32 -> float16
  // FloatToFloat16_ref(C_fp32, C_float16, m * n);
  FloatToFloat16_simd(C_fp32, C_float16, m * n);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
           .count();
  C_fp32_to_fp16_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif

  free(A_fp32);
  free(B_fp32);
  free(C_fp32);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_very_start)
          .count();
  run_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif
}
} // namespace fbgemm
