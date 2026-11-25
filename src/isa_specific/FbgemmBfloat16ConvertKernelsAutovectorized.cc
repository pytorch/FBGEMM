/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./PlatformSpecificKernelsIsa.h"

#include <bit>
#include <cstdint>

namespace fbgemm::FBGEMM_SIMD_NAMESPACE {

#if FBGEMM_SIMD_IN(ref, avx2, avx512)
void Kernels::BFloat16ToFloat32(const bfloat16* src, float* dst, size_t size)
    const noexcept {
  for (size_t i = 0; i != size; ++i) {
    dst[i] = std::bit_cast<float>(uint32_t{src[i].val} << 16);
  }
}
#endif

#if FBGEMM_SIMD_IN(ref)
void Kernels::Float32ToBFloat16(const float* src, bfloat16* dst, size_t size)
    const noexcept {
  for (size_t i = 0; i != size; ++i) {
    const uint32_t temp = std::bit_cast<uint32_t>(src[i]);
    if ((temp & 0x7fffffffu) > 0x7f800000u) {
      dst[i].val = static_cast<uint16_t>((temp >> 16) | 0x0040u);
    } else {
      dst[i].val = static_cast<uint16_t>((temp + (1u << 15)) >> 16);
    }
  }
}
#endif

} // namespace fbgemm::FBGEMM_SIMD_NAMESPACE
