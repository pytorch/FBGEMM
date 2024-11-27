/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliate <open-source-office@arm.com>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <arm_neon.h>
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {

void FloatToFloat16_neon(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  if (do_clip) {
    constexpr float FP16_MAX = 65504.f;
    auto vpos = vdupq_n_f32(FP16_MAX);
    auto vneg = vdupq_n_f32(-FP16_MAX);
    size_t i = 0;
    for (; i + 16 < size; i += 16) {
      auto f32_vec1 = vld1q_f32(src + i);
      auto f32_vec2 = vld1q_f32(src + i + 4);
      auto f32_vec3 = vld1q_f32(src + i + 8);
      auto f32_vec4 = vld1q_f32(src + i + 12);
      f32_vec1 = vmaxq_f32(vminq_f32(f32_vec1, vpos), vneg);
      f32_vec2 = vmaxq_f32(vminq_f32(f32_vec2, vpos), vneg);
      f32_vec3 = vmaxq_f32(vminq_f32(f32_vec3, vpos), vneg);
      f32_vec4 = vmaxq_f32(vminq_f32(f32_vec4, vpos), vneg);
      auto f16_vec1 = vcvt_f16_f32(f32_vec1);
      auto f16_vec2 = vcvt_f16_f32(f32_vec2);
      auto f16_vec3 = vcvt_f16_f32(f32_vec3);
      auto f16_vec4 = vcvt_f16_f32(f32_vec4);
      vst1_f16((__fp16*)dst + i, f16_vec1);
      vst1_f16((__fp16*)dst + i + 4, f16_vec2);
      vst1_f16((__fp16*)dst + i + 8, f16_vec3);
      vst1_f16((__fp16*)dst + i + 12, f16_vec4);
    }
    for (; i + 8 < size; i += 8) {
      auto f32_vec1 = vld1q_f32(src + i);
      auto f32_vec2 = vld1q_f32(src + i + 4);
      f32_vec1 = vmaxq_f32(vminq_f32(f32_vec1, vpos), vneg);
      f32_vec2 = vmaxq_f32(vminq_f32(f32_vec2, vpos), vneg);
      auto f16_vec1 = vcvt_f16_f32(f32_vec1);
      auto f16_vec2 = vcvt_f16_f32(f32_vec2);
      vst1_f16((__fp16*)dst + i, f16_vec1);
      vst1_f16((__fp16*)dst + i + 4, f16_vec2);
    }
    for (; i + 4 < size; i += 4) {
      auto f32_vec = vld1q_f32(src + i);
      f32_vec = vmaxq_f32(vminq_f32(f32_vec, vpos), vneg);
      auto f16_vec = vcvt_f16_f32(f32_vec);
      vst1_f16((__fp16*)dst + i, f16_vec);
    }
    FloatToFloat16_ref(src + i, dst + i, size - i, do_clip);
  } else {
    size_t i = 0;
    for (; i + 16 < size; i += 16) {
      auto f32_vec1 = vld1q_f32(src + i);
      auto f32_vec2 = vld1q_f32(src + i + 4);
      auto f32_vec3 = vld1q_f32(src + i + 8);
      auto f32_vec4 = vld1q_f32(src + i + 12);
      auto f16_vec1 = vcvt_f16_f32(f32_vec1);
      auto f16_vec2 = vcvt_f16_f32(f32_vec2);
      auto f16_vec3 = vcvt_f16_f32(f32_vec3);
      auto f16_vec4 = vcvt_f16_f32(f32_vec4);
      vst1_f16((__fp16*)dst + i, f16_vec1);
      vst1_f16((__fp16*)dst + i + 4, f16_vec2);
      vst1_f16((__fp16*)dst + i + 8, f16_vec3);
      vst1_f16((__fp16*)dst + i + 12, f16_vec4);
    }
    for (; i + 8 < size; i += 8) {
      auto f32_vec1 = vld1q_f32(src + i);
      auto f32_vec2 = vld1q_f32(src + i + 4);
      auto f16_vec1 = vcvt_f16_f32(f32_vec1);
      auto f16_vec2 = vcvt_f16_f32(f32_vec2);
      vst1_f16((__fp16*)dst + i, f16_vec1);
      vst1_f16((__fp16*)dst + i + 4, f16_vec2);
    }
    for (; i + 4 < size; i += 4) {
      auto f32_vec = vld1q_f32(src + i);
      auto f16_vec = vcvt_f16_f32(f32_vec);
      vst1_f16((__fp16*)dst + i, f16_vec);
    }
    FloatToFloat16_ref(src + i, dst + i, size - i);
  }
}

void Float16ToFloat_neon(const float16* src, float* dst, size_t size) {
  size_t i = 0;
  for (; i + 16 < size; i += 16) {
    auto f16_vec1 = vld1_f16((__fp16*)src + i);
    auto f16_vec2 = vld1_f16((__fp16*)src + i + 4);
    auto f16_vec3 = vld1_f16((__fp16*)src + i + 8);
    auto f16_vec4 = vld1_f16((__fp16*)src + i + 12);
    auto f32_vec1 = vcvt_f32_f16(f16_vec1);
    auto f32_vec2 = vcvt_f32_f16(f16_vec2);
    auto f32_vec3 = vcvt_f32_f16(f16_vec3);
    auto f32_vec4 = vcvt_f32_f16(f16_vec4);
    vst1q_f32(dst + i, f32_vec1);
    vst1q_f32(dst + i + 4, f32_vec2);
    vst1q_f32(dst + i + 8, f32_vec3);
    vst1q_f32(dst + i + 12, f32_vec4);
  }
  for (; i + 8 < size; i += 8) {
    auto f16_vec1 = vld1_f16((__fp16*)src + i);
    auto f16_vec2 = vld1_f16((__fp16*)src + i + 4);
    auto f32_vec1 = vcvt_f32_f16(f16_vec1);
    auto f32_vec2 = vcvt_f32_f16(f16_vec2);
    vst1q_f32(dst + i, f32_vec1);
    vst1q_f32(dst + i + 4, f32_vec2);
  }
  for (; i + 4 < size; i += 4) {
    auto f16_vec = vld1_f16((__fp16*)src + i);
    auto f32_vec = vcvt_f32_f16(f16_vec);
    vst1q_f32(dst + i, f32_vec);
  }
  Float16ToFloat_ref(src + i, dst + i, size - i);
}

} // namespace fbgemm
