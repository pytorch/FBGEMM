/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"

#include "./RefImplementations.h"

#include "xmmintrin.h"
#include "pmmintrin.h"

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

void FloatToFloat16_simd(
    const float* src,
    float16* dst,
    size_t size,
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

void Float16ToFloat_simd(const float16* src, float* dst, size_t size) {
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

void RoundToFloat16(
    const float* input,
    float* output,
    size_t size,
    bool clamp,
    bool clamp_denorms) {
  std::vector<fbgemm::float16> data_fp16(size);
  auto flush_mode = _MM_GET_DENORMALS_ZERO_MODE();
  auto denormal_mode = _MM_GET_DENORMALS_ZERO_MODE();
  if (clamp_denorms) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }
  // clamp_denorms is always true, since we use FloatToFloat16_simd function
  // with _mm256_cvtps_ph.
  FloatToFloat16_simd(input, &(data_fp16[0]), size, /*do_clip=*/clamp);
  Float16ToFloat_simd(&(data_fp16[0]), output, size);
  if (clamp_denorms) {
    _MM_SET_FLUSH_ZERO_MODE(flush_mode);
    _MM_SET_DENORMALS_ZERO_MODE(denormal_mode);
    union epsilon_t {
      float f;
      uint32_t i;
    };
    union epsilon_t epsilon;
    epsilon.i = 0x38800000u; // 1 / 16384
    for (size_t i = 0; i < size; ++i) {
      if (std::abs(output[i]) < epsilon.f) {
        output[i] = 0.0;
      }
    }
  }
}

} // namespace fbgemm
