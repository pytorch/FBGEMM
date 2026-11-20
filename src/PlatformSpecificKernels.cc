/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "./PlatformSpecificKernels.h"

#include <stdexcept>

#include <cpuinfo.h>

#include "fbgemm/Utils.h"

namespace fbgemm {

#if defined(FBGEMM_PLATFORM_KERNELS_AVX512_256) && \
    !defined(FBGEMM_PLATFORM_KERNELS_AVX2)
#error "avx512_256 derives from avx2, which isn't built"
#endif
#if defined(FBGEMM_PLATFORM_KERNELS_AVX512_256_BF16) && \
    !defined(FBGEMM_PLATFORM_KERNELS_AVX512_256)
#error "avx512_256_bf16 derives from avx512_256, which isn't built"
#endif
#if defined(FBGEMM_PLATFORM_KERNELS_AVX512) && \
    !defined(FBGEMM_PLATFORM_KERNELS_AVX512_256)
#error "avx512 derives from avx512_256, which isn't built"
#endif
#if defined(FBGEMM_PLATFORM_KERNELS_AVX512_BF16) && \
    !defined(FBGEMM_PLATFORM_KERNELS_AVX512)
#error "avx512_bf16 derives from avx512, which isn't built"
#endif
#if defined(FBGEMM_PLATFORM_KERNELS_SVE128) && \
    !defined(FBGEMM_PLATFORM_KERNELS_NEON)
#error "sve128 derives from neon, which isn't built"
#endif

std::vector<const PlatformSpecificKernels*> supportedPlatformKernels() {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  std::vector<const PlatformSpecificKernels*> res{&ref::kernels()};
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  using enum inst_set_t;
  const inst_set_t isa = fbgemmInstructionSet();
  const bool allow_avx512_bf16 = isa == avx512_vnni;
  const bool allow_avx512 = allow_avx512_bf16 || isa == avx512;
  const bool allow_avx512_256_bf16 =
      allow_avx512_bf16 || isa == avx512_vnni_ymm;
  const bool allow_avx512_256 =
      allow_avx512 || allow_avx512_256_bf16 || isa == avx512_ymm;
  [[maybe_unused]] const bool allow_avx2 = allow_avx512_256 || isa == avx2;
#ifdef FBGEMM_PLATFORM_KERNELS_AVX2
  if (allow_avx2 && fbgemmHasAvx2Support()) {
    res.push_back(&avx2::kernels());
  }
#endif
#ifdef FBGEMM_PLATFORM_KERNELS_AVX512_256
  if (allow_avx512_256 && fbgemmHasAvx512Support()) {
    res.push_back(&avx512_256::kernels());
  }
#endif
#ifdef FBGEMM_PLATFORM_KERNELS_AVX512_256_BF16
  if (allow_avx512_256_bf16 && fbgemmHasAvx512Support() &&
      fbgemmHasAvx512Bf16Support()) {
    res.push_back(&avx512_256_bf16::kernels());
  }
#endif
#ifdef FBGEMM_PLATFORM_KERNELS_AVX512
  if (allow_avx512 && fbgemmHasAvx512Support()) {
    res.push_back(&avx512::kernels());
  }
#endif
#ifdef FBGEMM_PLATFORM_KERNELS_AVX512_BF16
  if (allow_avx512_bf16 && fbgemmHasAvx512Support() &&
      fbgemmHasAvx512Bf16Support()) {
    res.push_back(&avx512_bf16::kernels());
  }
#endif
#elif CPUINFO_ARCH_ARM64
#ifdef FBGEMM_PLATFORM_KERNELS_NEON
  if (cpuinfo_has_arm_neon()) {
    res.push_back(&neon::kernels());
  }
#endif
#ifdef FBGEMM_PLATFORM_KERNELS_SVE128
  if (cpuinfo_has_arm_sve2() && cpuinfo_get_max_arm_sve_length() == 128) {
    res.push_back(&sve128::kernels());
  }
#endif
#endif
  return res;
}

namespace {

struct SelectingKernels final : PlatformSpecificKernels {
  std::string_view name() const noexcept override {
    initializePlatformDispatch();
    return platformKernels().name();
  }

  void BFloat16ToFloat32(const bfloat16* src, float* dst, size_t size)
      const noexcept override {
    initializePlatformDispatch();
    platformKernels().BFloat16ToFloat32(src, dst, size);
  }

  void Float32ToBFloat16(const float* src, bfloat16* dst, size_t size)
      const noexcept override {
    initializePlatformDispatch();
    platformKernels().Float32ToBFloat16(src, dst, size);
  }

  void Float16ToFloat32(const float16* src, float* dst, size_t size)
      const noexcept override {
    initializePlatformDispatch();
    platformKernels().Float16ToFloat32(src, dst, size);
  }

  void Float32ToFloat16(const float* src, float16* dst, size_t size)
      const noexcept override {
    initializePlatformDispatch();
    platformKernels().Float32ToFloat16(src, dst, size);
  }

  void Float32ToFloat16Clipping(const float* src, float16* dst, size_t size)
      const noexcept override {
    initializePlatformDispatch();
    platformKernels().Float32ToFloat16Clipping(src, dst, size);
  }
};

constinit SelectingKernels defaultKernels;

} // namespace

constinit std::atomic<const PlatformSpecificKernels*> platformSpecificKernels{
    &defaultKernels};

void initializePlatformDispatch(bool force) noexcept {
  const PlatformSpecificKernels* selected =
      []() -> const PlatformSpecificKernels* {
    try {
      return supportedPlatformKernels().back();
    } catch (...) {
      return &ref::kernels();
    }
  }();
  if (force) {
    platformSpecificKernels.store(selected, std::memory_order_relaxed);
    return;
  }
  const PlatformSpecificKernels* expected = &defaultKernels;
  platformSpecificKernels.compare_exchange_strong(
      expected, selected, std::memory_order_relaxed);
}

} // namespace fbgemm
