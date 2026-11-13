/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <string_view>
#include <vector>

#include "fbgemm/Types.h"

namespace fbgemm {

struct PlatformSpecificKernels {
  PlatformSpecificKernels(const PlatformSpecificKernels&) = delete;
  PlatformSpecificKernels(PlatformSpecificKernels&&) = delete;
  PlatformSpecificKernels& operator=(const PlatformSpecificKernels&) = delete;
  PlatformSpecificKernels& operator=(PlatformSpecificKernels&&) = delete;

  virtual std::string_view name() const noexcept = 0;

  virtual void Bfloat16ToFloat(const bfloat16* src, float* dst, size_t size)
      const noexcept = 0;

 protected:
  constexpr PlatformSpecificKernels() = default;
  ~PlatformSpecificKernels() = default;
};

namespace ref {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace ref
namespace avx2 {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace avx2
namespace avx512 {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace avx512
namespace neon {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace neon
namespace sve128 {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace sve128

std::vector<const PlatformSpecificKernels*> supportedPlatformKernels();

void initializePlatformDispatch() noexcept;

extern constinit std::atomic<const PlatformSpecificKernels*>
    platformSpecificKernels;

inline const PlatformSpecificKernels& platformKernels() noexcept {
  return *platformSpecificKernels.load(std::memory_order_relaxed);
}

} // namespace fbgemm
