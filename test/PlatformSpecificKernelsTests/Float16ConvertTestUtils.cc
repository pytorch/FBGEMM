/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test/PlatformSpecificKernelsTests/Float16ConvertTestUtils.h" // @manual

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "fbgemm/FloatConversion.h"
#include "test/PlatformSpecificKernelsTests/PlatformSpecificKernelsTestUtils.h" // @manual

namespace fbgemm {

namespace {

// Depending on the platform, clipping a NaN gives either -65504 or a NaN.
void normalizeNanResults(std::span<const float> in, std::span<float16> out) {
  constexpr float16 kClippedNan = cpu_float2half_rn(-65504.f);
  for (size_t i = 0; i != in.size(); ++i) {
    if (std::isnan(in[i]) && std::isnan(cpu_half2float(out[i]))) {
      out[i] = kClippedNan;
    }
  }
}

} // namespace

void Float16ConvertTestUtils::f16ToF32(
    std::span<const float16> in,
    std::span<float> out) const {
  runAllKernels(
      kernels,
      [](const PlatformSpecificKernels& k, auto... args) {
        k.Float16ToFloat32(args...);
      },
      in,
      out);
}

void Float16ConvertTestUtils::f32ToF16(
    std::span<const float> in,
    std::span<float16> out,
    bool doClip) const {
  runAllKernels(
      kernels,
      [doClip](
          const PlatformSpecificKernels& k,
          const float* src,
          float16* dst,
          size_t size) {
        if (doClip) {
          k.Float32ToFloat16Clipping(src, dst, size);
          normalizeNanResults(std::span(src, size), std::span(dst, size));
        } else {
          k.Float32ToFloat16(src, dst, size);
        }
      },
      in,
      out);
}

void Float16ConvertTestUtils::f16ToF32Test(
    std::span<const float16> in,
    std::span<float> out) const {
  f16ToF32(in, out);
  std::vector<float> expected(in.size());
  std::ranges::transform(in, expected.begin(), cpu_half2float_ref);
  checkSame(in, out, expected, "Float16ToFloat32 vs cpu_half2float_ref");
}

void Float16ConvertTestUtils::f32ToF16Test(
    std::span<const float> in,
    std::span<float16> out,
    bool doClip) const {
  f32ToF16(in, out, doClip);
  std::vector<float16> expected(in.size());
  std::ranges::transform(in, expected.begin(), [doClip](float x) {
    return cpu_float2half_rn(doClip ? clipToFloat16Range(x) : x);
  });
  if (doClip) {
    normalizeNanResults(in, expected);
  }
  checkSame(
      in,
      out,
      expected,
      "Float32ToFloat16 vs cpu_float2half_rn, do_clip ",
      doClip);
}

void Float16ConvertTestUtils::f16ToF32ToF16Test(
    std::span<const float16> in,
    std::span<float> f32Out,
    std::span<float16> f16Out) const {
  f16ToF32(in, f32Out);
  f32ToF16(f32Out, f16Out, /*doClip=*/false);
  std::vector<float16> expected(in.begin(), in.end());
  // Widening quiets signaling NaNs (sets the 0x0200 bit), as F16C and AArch64
  // do, so they come back quiet.
  for (float16& h : expected) {
    if ((h.val & 0x7c00) == 0x7c00 && (h.val & 0x03ff) != 0) {
      h.val |= 0x0200;
    }
  }
  checkSame(in, f16Out, expected, "float16 -> float -> float16");
}

} // namespace fbgemm
