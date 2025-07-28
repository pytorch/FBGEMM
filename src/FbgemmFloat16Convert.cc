/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/Utils.h"

#include <cpuinfo.h>
#include <memory>
#include <vector>

namespace fbgemm {

void FloatToFloat16_simd(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  // Run time CPU detection
  if (cpuinfo_initialize()) {
#if defined(FBGEMM_FBCODE) || !defined(__aarch64__)
    if (fbgemmHasAvx512Support()) {
      FloatToFloat16_avx512(src, dst, size, do_clip);
    } else if (fbgemmHasAvx2Support()) {
      FloatToFloat16_avx2(src, dst, size, do_clip);
    } else
#endif
    {
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
#if defined(FBGEMM_FBCODE) || !defined(__aarch64__)
    if (fbgemmHasAvx512Support()) {
      Float16ToFloat_avx512(src, dst, size);
    } else if (fbgemmHasAvx2Support()) {
      Float16ToFloat_avx2(src, dst, size);
    } else
#endif
    {
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
  FloatToFloat16_simd(input, data_fp16.data(), size, /*do_clip=*/clamp);
  Float16ToFloat_simd(data_fp16.data(), output, size);
  if (clamp_denorms) {
    // FloatToFloat16_simd always preserve fp16 denorm, so we need to manually
    // clamp.
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
