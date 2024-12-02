/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliate <open-source-office@arm.com>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <arm_sve.h>
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {

void FloatToFloat16_sve(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  if (do_clip) {
    constexpr float FP16_MAX = 65504.f;
    size_t i = 0;
    int lanes = svcntw();
    auto p_32 = svptrue_b32();
    auto p_16 = svptrue_b16();
    auto pfalse = svpfalse();
    auto p_16_half = svuzp1_b16(p_16, pfalse);
    while (i + 2 * lanes < size) {
      auto f32_vec1 = svld1_f32(p_32, src + i);
      auto f32_vec2 = svld1_f32(p_32, src + i + lanes);
      f32_vec1 = svmax_n_f32_x(p_32, svmin_n_f32_x(p_32, f32_vec1, FP16_MAX), -FP16_MAX);
      f32_vec2 = svmax_n_f32_x(p_32, svmin_n_f32_x(p_32, f32_vec2, FP16_MAX), -FP16_MAX);
      auto f16_vec1 = svcvt_f16_f32_x(p_32, f32_vec1);
      auto f16_vec2 = svcvt_f16_f32_x(p_32, f32_vec2);
      auto f16_vec = svuzp1_f16(f16_vec1, f16_vec2);
      svst1_f16(p_16, (__fp16*)dst + i, f16_vec);
      i += 2 * lanes;
    }
    while (i + lanes < size) {
      auto f32_vec = svld1_f32(p_32, src + i);
      f32_vec = svmax_n_f32_x(p_32, svmin_n_f32_x(p_32, f32_vec, FP16_MAX), -FP16_MAX);
      auto f16_vec = svcvt_f16_f32_x(p_16, f32_vec);
      f16_vec = svuzp1_f16(f16_vec, f16_vec);
      svst1_f16(p_16_half, (__fp16*)dst + i, f16_vec);
      i += lanes;
    }
    FloatToFloat16_ref(src + i, dst + i, size - i, do_clip);
  } else {
    size_t i = 0;
    int lanes = svcntw();
    auto p_32 = svptrue_b32();
    auto p_16 = svptrue_b16();
    auto pfalse = svpfalse();
    auto p_16_half = svuzp1_b16(p_16, pfalse);
    while (i + 2 * lanes < size) {
      auto f32_vec1 = svld1_f32(p_32, src + i);
      auto f32_vec2 = svld1_f32(p_32, src + i + lanes);
      auto f16_vec1 = svcvt_f16_f32_x(p_32, f32_vec1);
      auto f16_vec2 = svcvt_f16_f32_x(p_32, f32_vec2);
      auto f16_vec = svuzp1_f16(f16_vec1, f16_vec2);
      svst1_f16(p_16, (__fp16*)dst + i, f16_vec);
      i += 2 * lanes;
    }
    while (i + lanes < size) {
      auto f32_vec = svld1_f32(p_32, src + i);
      auto f16_vec = svcvt_f16_f32_x(p_32, f32_vec);
      f16_vec = svuzp1_f16(f16_vec, f16_vec);
      svst1_f16(p_16_half, (__fp16*)dst + i, f16_vec);
      i += lanes;
    }
    FloatToFloat16_ref(src + i, dst + i, size - i);
  }
}

void Float16ToFloat_sve(const float16* src, float* dst, size_t size) {
  size_t i = 0;
  int lanes = svcntw();
  auto p_32 = svptrue_b32();
  auto p_16 = svptrue_b16();
  auto pfalse = svpfalse();
  auto p_16_half = svuzp1_b16(p_16, pfalse);
  while (i + 2 * lanes < size) {
    auto f16_vec = svld1_f16(p_16, (__fp16*)src + i);
    auto f16_vec1 = svzip1(f16_vec, f16_vec);
    auto f16_vec2 = svzip2(f16_vec, f16_vec);
    auto f32_vec1 = svcvt_f32_f16_x(p_16, f16_vec1);
    auto f32_vec2 = svcvt_f32_f16_x(p_16, f16_vec2);
    svst1_f32(p_32, dst + i, f32_vec1);
    svst1_f32(p_32, dst + i + lanes, f32_vec2);
    i += 2 * lanes;
  }
  while (i + lanes < size) {
    auto f16_vec = svld1_f16(p_16_half, (__fp16*)src + i);
    f16_vec = svzip1_f16(f16_vec, f16_vec);
    auto f32_vec = svcvt_f32_f16_x(p_32, f16_vec);
    svst1_f32(p_32, dst + i, f32_vec);
    i += lanes;
  }
  Float16ToFloat_ref(src + i, dst + i, size - i);
}

} // namespace fbgemm
