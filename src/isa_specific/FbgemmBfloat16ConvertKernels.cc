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
void Kernels::Bfloat16ToFloat(const bfloat16* src, float* dst, size_t size)
    const noexcept {
  for (size_t i = 0; i != size; ++i) {
    dst[i] = std::bit_cast<float>(uint32_t{src[i].val} << 16);
  }
}
#endif

} // namespace fbgemm::FBGEMM_SIMD_NAMESPACE
