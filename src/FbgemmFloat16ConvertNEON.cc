/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <arm_neon.h>
#endif

#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/FloatConversion.h"

#include <algorithm>
#include <array>

namespace fbgemm {

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

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
void FloatToFloat16KernelNeon_TwoVecs_WithClip(const float* src, float16* dst) {
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
  if (i < size) {
    std::array<float, chunkSize> src_tail{};
    std::array<float16, chunkSize> dst_tail;
    std::copy_n(src + i, size - i, src_tail.begin());
    if (do_clip) {
      FloatToFloat16KernelNeon_TwoVecs_WithClip(
          src_tail.data(), dst_tail.data());
    } else {
      FloatToFloat16KernelNeon_TwoVecs(src_tail.data(), dst_tail.data());
    }
    std::copy_n(dst_tail.begin(), size - i, dst + i);
  }
}

void Float16ToFloat_neon(const float16* src, float* dst, size_t size) {
  constexpr size_t chunkSize = 8;

  size_t i = 0;
  for (; i + chunkSize <= size; i += chunkSize) {
    Float16ToFloatKernelNeon_OneVec(src + i, dst + i);
  }
  if (i < size) {
    std::array<float16, chunkSize> src_tail{};
    std::array<float, chunkSize> dst_tail;
    std::copy_n(src + i, size - i, src_tail.begin());
    Float16ToFloatKernelNeon_OneVec(src_tail.data(), dst_tail.data());
    std::copy_n(dst_tail.begin(), size - i, dst + i);
  }
}

#else

void FloatToFloat16_neon(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  constexpr float FP16_MAX = 65504.f;
  for (size_t i = 0; i < size; ++i) {
    const float value =
        do_clip ? std::max(-FP16_MAX, std::min(src[i], FP16_MAX)) : src[i];
    dst[i] = cpu_float2half(value);
  }
}

void Float16ToFloat_neon(const float16* src, float* dst, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    dst[i] = cpu_half2float(src[i]);
  }
}

#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

} // namespace fbgemm
