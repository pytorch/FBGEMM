/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"

#include "./RefImplementations.h"

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_BLAS
#if __APPLE__
// not sure whether need to differentiate TARGET_OS_MAC or TARGET_OS_IPHONE,
// etc.
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#include <cpuinfo.h>
#include <memory>
#include <utility>
#include <vector>

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double naive_malloc_time = 0.0;
double naive_A_bf16_to_fp32_time = 0.0;
double naive_B_bf16_to_fp32_time = 0.0;
double naive_C_bf16_to_fp32_time = 0.0;
double naive_computing_time = 0.0;
double naive_C_fp32_to_bf16_time = 0.0;
double naive_run_time = 0.0;
#endif

using namespace std;

namespace fbgemm {

void FloatToBfloat16_ref(const float* src, bfloat16* dst, int size) {
  for (int i = 0; i < size; i++) {
    // Add 2^15 and right shift 16 to do round-nearest
    dst[i] = (*reinterpret_cast<const uint32_t*>(src + i) + (1 << 15)) >> 16;
  }
}

void Bfloat16ToFloat_ref(const bfloat16* src, float* dst, int size) {
  for (int i = 0; i < size; i++) {
    uint32_t val_fp32 =
        static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(src)[i]) << 16;
    reinterpret_cast<uint32_t*>(dst)[i] = val_fp32;
  }
}

void FloatToBfloat16_simd(const float* src, bfloat16* dst, int size) {
  // Run time CPU detection
  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512Support()) {
      FloatToBfloat16_avx512(src, dst, size);
    } else if (fbgemmHasAvx2Support()) {
      FloatToBfloat16_avx2(src, dst, size);
    } else {
      FloatToBfloat16_ref(src, dst, size);
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

void Bfloat16ToFloat_simd(const bfloat16* src, float* dst, int size) {
  // Run time CPU detection
  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512Support()) {
      Bfloat16ToFloat_avx512(src, dst, size);
    } else if (fbgemmHasAvx2Support()) {
      Bfloat16ToFloat_avx2(src, dst, size);
    } else {
      Bfloat16ToFloat_ref(src, dst, size);
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

} // namespace fbgemm
