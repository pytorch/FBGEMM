/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
#endif

#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {

#if defined(__ARM_FEATURE_SVE2)

namespace {

// Load two vectors, convert them from fp32 to fp16, store one vector.
void FloatToFloat16KernelSve2_TwoVecs(const float* src, float16* dst) {
  const svbool_t pt = svptrue_b16();
  svfloat32x2_t srcVecs = svld2_f32(pt, src);
  svfloat16_t even = svcvt_f16_f32_x(pt, svget2(srcVecs, 0));
  svfloat16_t result = svcvtnt_f16_f32_x(even, pt, svget2(srcVecs, 1));
  svst1_f16(pt, reinterpret_cast<float16_t*>(dst), result);
}

// Load and clip two vectors, convert them from fp32 to fp16, store one
// vector.
void FloatToFloat16KernelSve2_TwoVecs_WithClip(const float* src, float16* dst) {
  const svbool_t pt = svptrue_b16();
  constexpr float FP16_MAX = 65504.f;

  // Load two vectors
  const svfloat32x2_t srcVecs = svld2_f32(pt, src);
  svfloat32_t src0 = svget2(srcVecs, 0);
  svfloat32_t src1 = svget2(srcVecs, 1);

  // Do the clipping
  src0 = svmin_n_f32_x(pt, src0, FP16_MAX);
  src0 = svmax_n_f32_x(pt, src0, -FP16_MAX);
  src1 = svmin_n_f32_x(pt, src1, FP16_MAX);
  src1 = svmax_n_f32_x(pt, src1, -FP16_MAX);

  // Convert fp32 -> fp16
  const svfloat16_t even = svcvt_f16_f32_x(pt, src0);
  const svfloat16_t result = svcvtnt_f16_f32_x(even, pt, src1);

  // Store one vector
  svst1_f16(pt, reinterpret_cast<float16_t*>(dst), result);
}

} // namespace

void FloatToFloat16_sve2(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  const size_t chunkSize = svcntw() * 2;

  // Note: we don't use predicates here, because then we can't use svld2. This
  // is not optimal for small buffers, but we already have high overhead on
  // small buffers because we have to set fp rounding mode, so I don't care.
  if (do_clip) {
    size_t i;
    for (i = 0; i + chunkSize < size; i += chunkSize) {
      FloatToFloat16KernelSve2_TwoVecs_WithClip(src + i, dst + i);
    }
    FloatToFloat16_ref(src + i, dst + i, size - i, do_clip);
  } else {
    size_t i;
    for (i = 0; i + chunkSize < size; i += chunkSize) {
      FloatToFloat16KernelSve2_TwoVecs(src + i, dst + i);
    }
    FloatToFloat16_ref(src + i, dst + i, size - i, do_clip);
  }
}

#else

void FloatToFloat16_sve2(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  throw std::runtime_error{
      "CPU supports SVE2 instructions, but you didn't enable SVE2 in your build command. Fix your build!"};
}

#endif // defined(__ARM_FEATURE_SVE2)

} // namespace fbgemm
