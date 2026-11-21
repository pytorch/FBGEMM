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

#include <memory>
#include <vector>

namespace fbgemm {

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
