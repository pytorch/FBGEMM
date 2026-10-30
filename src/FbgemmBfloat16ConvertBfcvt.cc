/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"

#include <algorithm>
#include <bit>

namespace fbgemm::detail {

// Compiled separately with +bf16 and called only after CPU feature detection.
void float_to_bfloat16_bfcvt(const float* src, bfloat16* dst, size_t size) {
  uint64_t fpcr;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  // Fall back for nondefault rounding/denormal handling or enabled FP traps.
  if (fpcr & ((3ull << 22) | (1ull << 24) | 0x9f03ull)) {
    FloatToBfloat16_ref(src, dst, size);
    return;
  }
  for (size_t i = 0; i < size;) {
    const size_t count = std::min(size - i, size_t{32});
    bool has_nan = false;
    for (size_t j = 0; j < count; ++j) {
      has_nan |= (std::bit_cast<uint32_t>(src[i + j]) & 0x7fffffffu) >
          0x7f800000u;
    }
    // Preserve reference NaN behavior without software rounding in the hot loop.
    if (has_nan) {
      FloatToBfloat16_ref(src + i, dst + i, count);
    } else {
      for (size_t j = 0; j < count; ++j) {
        dst[i + j].val =
            std::bit_cast<uint16_t>(static_cast<__bf16>(src[i + j]));
      }
    }
    i += count;
  }
}

} // namespace fbgemm::detail
