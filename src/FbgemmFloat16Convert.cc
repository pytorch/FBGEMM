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

using namespace std;

namespace fbgemm {

void FloatToFloat16_ref(
    const float* src,
    float16* dst,
    int size,
    bool do_clip) {
  constexpr float FP16_MAX = 65504.f;
  if (do_clip) {
    for (int i = 0; i < size; i++) {
      float cur_src = std::max(-FP16_MAX, std::min(src[i], FP16_MAX));
      dst[i] = cpu_float2half_rn(cur_src);
    }
  } else {
    for (int i = 0; i < size; i++) {
      dst[i] = cpu_float2half_rn(src[i]);
    }
  }
}

void Float16ToFloat_ref(const float16* src, float* dst, int size) {
  for (int i = 0; i < size; i++) {
    dst[i] = cpu_half2float(src[i]);
  }
}

void FloatToFloat16_simd(
    const float* src,
    float16* dst,
    int size,
    bool do_clip) {
  // Run time CPU detection
  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512Support()) {
      FloatToFloat16_avx512(src, dst, size, do_clip);
    } else if (fbgemmHasAvx2Support()) {
      FloatToFloat16_avx2(src, dst, size, do_clip);
    } else {
      FloatToFloat16_ref(src, dst, size, do_clip);
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

void Float16ToFloat_simd(const float16* src, float* dst, int size) {
  // Run time CPU detection
  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512Support()) {
      Float16ToFloat_avx512(src, dst, size);
    } else if (fbgemmHasAvx2Support()) {
      Float16ToFloat_avx2(src, dst, size);
    } else {
      Float16ToFloat_ref(src, dst, size);
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

} // namespace fbgemm
