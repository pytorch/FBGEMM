/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

namespace fbgemm {
void FloatRelu_simd(const float* src, float* dst, size_t size) {
  // Run time CPU detection
  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512Support()) {
      FloatRelu_avx512(src, dst, size);
    } else if (fbgemmHasAvx2Support()) {
      FloatRelu_avx2(src, dst, size);
    } else {
      FloatRelu_ref(src, dst, size);
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}
} // namespace fbgemm
