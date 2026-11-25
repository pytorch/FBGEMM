/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FBGEMM_SIMD_BASE
#error \
    "Internal: PlatformSpecificKernelsIsa.h includes this once per platform. To call platform kernels, include PlatformSpecificKernels.h and use fbgemm::platformKernels()."
#endif

namespace fbgemm::FBGEMM_SIMD_NAMESPACE {

struct Kernels : FBGEMM_SIMD_BASE {
  static const Kernels instance;

  Kernels(const Kernels&) = delete;
  Kernels(Kernels&&) = delete;
  Kernels& operator=(const Kernels&) = delete;
  Kernels& operator=(Kernels&&) = delete;

  std::string_view name() const noexcept override;

#if FBGEMM_OVERRIDE_IN(avx2, avx512)
  void BFloat16ToFloat32(const bfloat16* src, float* dst, size_t size)
      const noexcept override;
#endif

#if FBGEMM_OVERRIDE_IN(avx2, avx512, neon)
  void Float32ToBFloat16(const float* src, bfloat16* dst, size_t size)
      const noexcept override;
#endif

#if FBGEMM_OVERRIDE_IN(avx2, avx512, neon)
  void Float16ToFloat32(const float16* src, float* dst, size_t size)
      const noexcept override;
#endif

#if FBGEMM_OVERRIDE_IN(avx2, avx512, neon)
  void Float32ToFloat16(const float* src, float16* dst, size_t size)
      const noexcept override;
#endif

#if FBGEMM_OVERRIDE_IN(avx2, avx512, neon)
  void Float32ToFloat16Clipping(const float* src, float16* dst, size_t size)
      const noexcept override;
#endif

 protected:
  constexpr Kernels() = default;
  ~Kernels() = default;
};

} // namespace fbgemm::FBGEMM_SIMD_NAMESPACE
