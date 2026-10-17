/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif

#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"

#include <stdexcept>

namespace fbgemm {

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace {

// Load two vectors, convert them from fp32 to fp16, store one vector.
void FloatToFloat16KernelNeon_TwoVecs(const float* src, float16* dst) {
  const float32x4_t src0 = vld1q_f32(src);
  const float32x4_t src1 = vld1q_f32(src + 4);
  vst1q_f16(
      reinterpret_cast<float16_t*>(dst),
      vcombine_f16(vcvt_f16_f32(src0), vcvt_f16_f32(src1)));
}

// Load and clip two vectors, convert them from fp32 to fp16, store one
// vector.
void FloatToFloat16KernelNeon_TwoVecs_WithClip(
    const float* src,
    float16* dst) {
  constexpr float FP16_MAX = 65504.f;
  const float32x4_t vmax = vdupq_n_f32(FP16_MAX);
  const float32x4_t vmin = vdupq_n_f32(-FP16_MAX);
  float32x4_t src0 = vld1q_f32(src);
  float32x4_t src1 = vld1q_f32(src + 4);
  src0 = vmaxq_f32(vminq_f32(src0, vmax), vmin);
  src1 = vmaxq_f32(vminq_f32(src1, vmax), vmin);
  vst1q_f16(
      reinterpret_cast<float16_t*>(dst),
      vcombine_f16(vcvt_f16_f32(src0), vcvt_f16_f32(src1)));
}

// Load one vector, convert it from fp16 to fp32, store two vectors.
void Float16ToFloatKernelNeon_OneVec(const float16* src, float* dst) {
  const float16x8_t srcVec = vld1q_f16(reinterpret_cast<const float16_t*>(src));
  vst1q_f32(dst, vcvt_f32_f16(vget_low_f16(srcVec)));
  vst1q_f32(dst + 4, vcvt_f32_f16(vget_high_f16(srcVec)));
}

} // namespace

void FloatToFloat16_neon(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  constexpr size_t chunkSize = 8;

  size_t i = 0;
  if (do_clip) {
    for (; i + chunkSize <= size; i += chunkSize) {
      FloatToFloat16KernelNeon_TwoVecs_WithClip(src + i, dst + i);
    }
  } else {
    for (; i + chunkSize <= size; i += chunkSize) {
      FloatToFloat16KernelNeon_TwoVecs(src + i, dst + i);
    }
  }
  FloatToFloat16_ref(src + i, dst + i, size - i, do_clip);
}

void Float16ToFloat_neon(const float16* src, float* dst, size_t size) {
  constexpr size_t chunkSize = 8;

  size_t i = 0;
  for (; i + chunkSize <= size; i += chunkSize) {
    Float16ToFloatKernelNeon_OneVec(src + i, dst + i);
  }
  Float16ToFloat_ref(src + i, dst + i, size - i);
}

#else

void FloatToFloat16_neon(
    const float* src [[maybe_unused]],
    float16* dst [[maybe_unused]],
    size_t size [[maybe_unused]],
    bool do_clip [[maybe_unused]]) {
  throw std::runtime_error{
      "CPU supports NEON FP16 instructions, but you didn't enable them in your build command. Fix your build!"};
}

void Float16ToFloat_neon(
    const float16* src [[maybe_unused]],
    float* dst [[maybe_unused]],
    size_t size [[maybe_unused]]) {
  throw std::runtime_error{
      "CPU supports NEON FP16 instructions, but you didn't enable them in your build command. Fix your build!"};
}

#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

} // namespace fbgemm
