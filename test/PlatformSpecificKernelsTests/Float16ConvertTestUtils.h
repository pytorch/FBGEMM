/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <span>
#include <vector>

#include "fbgemm/Types.h"
#include "src/PlatformSpecificKernels.h" // @manual

namespace fbgemm {

struct Float16ConvertTestUtils {
  std::vector<const PlatformSpecificKernels*> kernels =
      supportedPlatformKernels();

  void f16ToF32(std::span<const float16> in, std::span<float> out) const;
  void f32ToF16(std::span<const float> in, std::span<float16> out, bool doClip)
      const;

  void f16ToF32Test(std::span<const float16> in, std::span<float> out) const;
  void f32ToF16Test(
      std::span<const float> in,
      std::span<float16> out,
      bool doClip) const;
  void f16ToF32ToF16Test(
      std::span<const float16> in,
      std::span<float> f32Out,
      std::span<float16> f16Out) const;
};

} // namespace fbgemm
