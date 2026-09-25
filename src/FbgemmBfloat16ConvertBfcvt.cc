/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Compiled with -march=...+bf16 by the fbgemm_bfcvt target; BFCVT must not
// leak into code runnable without FEAT_BF16 (dispatch: cpuinfo_has_arm_bf16).
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)

#include <arm_neon.h>

#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {

namespace {

inline void FloatToBfloat16KernelBfcvt(const float* src, bfloat16* dst) {
  const float32x4_t src_reg0 = vld1q_f32(src);
  const float32x4_t src_reg1 = vld1q_f32(src + 4);
  bfloat16x8_t dst_reg = vcvtq_low_bf16_f32(src_reg0);
  dst_reg = vcvtq_high_bf16_f32(dst_reg, src_reg1);
  vst1q_u16(reinterpret_cast<uint16_t*>(dst), vreinterpretq_u16_bf16(dst_reg));
}

inline void Bfloat16ToFloatKernelBfcvt(const bfloat16* src, float* dst) {
  const bfloat16x8_t src_reg = vreinterpretq_bf16_u16(
      vld1q_u16(reinterpret_cast<const uint16_t*>(src)));
  vst1q_f32(dst, vcvtq_low_f32_bf16(src_reg));
  vst1q_f32(dst + 4, vcvtq_high_f32_bf16(src_reg));
}

} // namespace

void FloatToBfloat16_bfcvt(const float* src, bfloat16* dst, size_t size) {
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    FloatToBfloat16KernelBfcvt(src + i, dst + i);
  }
  FloatToBfloat16_ref(src + i, dst + i, size - i);
}

void Bfloat16ToFloat_bfcvt(const bfloat16* src, float* dst, size_t size) {
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    Bfloat16ToFloatKernelBfcvt(src + i, dst + i);
  }
  Bfloat16ToFloat_ref(src + i, dst + i, size - i);
}

} // namespace fbgemm

#endif
