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

struct Bfloat16ConvertTestUtils {
  std::vector<const PlatformSpecificKernels*> kernels =
      supportedPlatformKernels();

  void b16ToF32(std::span<const bfloat16> in, std::span<float> out) const;
  void f32ToB16(std::span<const float> in, std::span<bfloat16> out) const;

  void b16ToF32Test(std::span<const bfloat16> in, std::span<float> out) const;
  void f32ToB16Test(std::span<const float> in, std::span<bfloat16> out) const;
  void b16ToF32ToB16Test(
      std::span<const bfloat16> in,
      std::span<float> f32Out,
      std::span<bfloat16> b16Out) const;
};

} // namespace fbgemm
