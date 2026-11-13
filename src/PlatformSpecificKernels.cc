/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./PlatformSpecificKernels.h"

#include <stdexcept>

#include <cpuinfo.h>

#include "fbgemm/Utils.h"

namespace fbgemm {

std::vector<const PlatformSpecificKernels*> supportedPlatformKernels() {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  std::vector<const PlatformSpecificKernels*> res{&ref::kernels()};
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
#ifndef FBGEMM_PLATFORM_KERNELS_DISABLE_AVX2
  if (fbgemmHasAvx2Support()) {
    res.push_back(&avx2::kernels());
  }
#endif
#ifndef FBGEMM_PLATFORM_KERNELS_DISABLE_AVX512
  if (fbgemmHasAvx512Support() && fbgemmHasAvx512Bf16Support()) {
    res.push_back(&avx512::kernels());
  }
#endif
#elif CPUINFO_ARCH_ARM64
#ifndef FBGEMM_PLATFORM_KERNELS_DISABLE_NEON
  if (cpuinfo_has_arm_neon()) {
    res.push_back(&neon::kernels());
  }
#endif
#ifndef FBGEMM_PLATFORM_KERNELS_DISABLE_SVE128
  if (cpuinfo_has_arm_sve2() && cpuinfo_get_max_arm_sve_length() == 128) {
    res.push_back(&sve128::kernels());
  }
#endif
#endif
  return res;
}

void initializePlatformDispatch() noexcept {
  const PlatformSpecificKernels* selected = &ref::kernels();
  try {
    selected = supportedPlatformKernels().back();
  } catch (...) {
  }
  platformSpecificKernels.store(selected, std::memory_order_relaxed);
}

namespace {

struct SelectingKernels final : PlatformSpecificKernels {
  std::string_view name() const noexcept override {
    initializePlatformDispatch();
    return platformKernels().name();
  }

  void Bfloat16ToFloat(const bfloat16* src, float* dst, size_t size)
      const noexcept override {
    initializePlatformDispatch();
    platformKernels().Bfloat16ToFloat(src, dst, size);
  }
};

constinit SelectingKernels defaultKernels;

} // namespace

constinit std::atomic<const PlatformSpecificKernels*> platformSpecificKernels{
    &defaultKernels};

} // namespace fbgemm
