/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test/PlatformSpecificKernelsTests/Bfloat16ConvertTestUtils.h" // @manual

#include <algorithm>
#include <vector>

#include "fbgemm/FloatConversion.h"
#include "test/PlatformSpecificKernelsTests/PlatformSpecificKernelsTestUtils.h" // @manual

namespace fbgemm {

void Bfloat16ConvertTestUtils::b16ToF32(
    std::span<const bfloat16> in,
    std::span<float> out) const {
  runAllKernels(
      kernels,
      [](const PlatformSpecificKernels& k, auto... args) {
        k.BFloat16ToFloat32(args...);
      },
      in,
      out);
}

void Bfloat16ConvertTestUtils::f32ToB16(
    std::span<const float> in,
    std::span<bfloat16> out) const {
  runAllKernels(
      kernels,
      [](const PlatformSpecificKernels& k, auto... args) {
        k.Float32ToBFloat16(args...);
      },
      in,
      out);
}

void Bfloat16ConvertTestUtils::b16ToF32Test(
    std::span<const bfloat16> in,
    std::span<float> out) const {
  b16ToF32(in, out);
  std::vector<float> expected(in.size());
  std::ranges::transform(in, expected.begin(), cpu_bf162float);
  checkSame(in, out, expected, "BFloat16ToFloat32 vs cpu_bf162float");
}

void Bfloat16ConvertTestUtils::f32ToB16Test(
    std::span<const float> in,
    std::span<bfloat16> out) const {
  f32ToB16(in, out);
  std::vector<bfloat16> expected(in.size());
  std::ranges::transform(in, expected.begin(), cpu_float2bfloat16);
  checkSame(in, out, expected, "Float32ToBFloat16 vs cpu_float2bfloat16");
}

void Bfloat16ConvertTestUtils::b16ToF32ToB16Test(
    std::span<const bfloat16> in,
    std::span<float> f32Out,
    std::span<bfloat16> b16Out) const {
  b16ToF32(in, f32Out);
  f32ToB16(f32Out, b16Out);
  std::vector<bfloat16> expected(in.size());
  std::ranges::transform(in, expected.begin(), [](bfloat16 x) {
    return cpu_float2bfloat16(cpu_bf162float(x));
  });
  checkSame(in, b16Out, expected, "bfloat16 -> float -> bfloat16");
}

} // namespace fbgemm
