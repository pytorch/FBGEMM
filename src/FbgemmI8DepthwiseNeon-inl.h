/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__aarch64__) || (defined(_MSC_VER) && defined(_M_ARM64))

#include <cassert>
#include <cfenv>
#include <cmath> // for lrintf and sqrt
#include <cstdint>
#include <type_traits> // for is_same

#include <arm_neon.h>

#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/UtilsAvx2.h"

namespace fbgemm {

template <
    bool FUSE_RELU,
    bool HAS_BIAS,
    QuantizationGranularity Q_GRAN,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    int K_PER_G,
    typename BIAS_TYPE>
static ALWAYS_INLINE void requantize_(
    std::int32_t A_zero_point,
    const std::int32_t* B_zero_point,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    const std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    int n,
    const std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias [[maybe_unused]],
    const float* act_times_w_scale = nullptr) {
  float32x4_t multiplier_v = vdupq_n_f32(0.0f);
  // Broadcasted reciprocal of act_times_w_scale
  float32x4_t act_times_w_rcp_v [[maybe_unused]] = vdupq_n_f32(0.0f);
  int32x4_t B_zero_point_v = vdupq_n_s32(0);
  if constexpr (Q_GRAN == QuantizationGranularity::TENSOR) {
    multiplier_v = vdupq_n_f32(*C_multiplier);
    if constexpr (std::is_same_v<BIAS_TYPE, float>) {
      act_times_w_rcp_v = vdupq_n_f32(1.0 / (*act_times_w_scale));
    }
    B_zero_point_v = vdupq_n_s32(B_zero_point[0]);
  }

  uint8x16_t min_v = vdupq_n_u8(0);

  if constexpr (A_SYMMETRIC) {
    assert(A_zero_point == 0 || col_offsets == nullptr);
  }
  int32x4_t A_zero_point_v = vdupq_n_s32(A_zero_point);
  int16x8_t C_zero_point_epi16_v = vdupq_n_s16(C_zero_point);
  int8x16_t C_zero_point_epi8_v = vdupq_n_s8(C_zero_point);

  constexpr int VLEN = 4;
  int j = 0;
  for (; j < n / (VLEN * 4) * (VLEN * 4); j += (VLEN * 4)) {
    int32x4_t x_v = vld1q_s32(C_int32 + j);
    int32x4_t y_v = vld1q_s32(C_int32 + j + VLEN);
    int32x4_t z_v = vld1q_s32(C_int32 + j + 2 * VLEN);
    int32x4_t w_v = vld1q_s32(C_int32 + j + 3 * VLEN);

    int32x4_t row_offset_v;
    if constexpr (!B_SYMMETRIC) {
      if constexpr (K_PER_G == 1) {
        row_offset_v = vld1q_s32(row_offsets + j);
      } else {
        static_assert(K_PER_G == 2);
        // Load row_offsets for 2 groups and broadcast by 2 times.
        row_offset_v =
            vcombine_s32(vld1_s32(row_offsets + j / 2), vdup_n_s32(0));
        row_offset_v = vzip1q_u32(row_offset_v, row_offset_v);
      }
      if constexpr (
          Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = vld1q_s32(B_zero_point + j);
      } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
        static_assert(K_PER_G == 2);
        B_zero_point_v =
            vcombine_s32(vld1_s32(B_zero_point + j / 2), vdup_n_s32(0));
        B_zero_point_v = vzip1q_u32(B_zero_point_v, B_zero_point_v);
      }
      x_v = vmlsq_s32(x_v, row_offset_v, B_zero_point_v);
    }
    int32x4_t col_off_v;
    if constexpr (!A_SYMMETRIC) {
      x_v = vmlsq_s32(x_v, A_zero_point_v, vld1q_s32(col_offsets + j));
    }

    if constexpr (!B_SYMMETRIC) {
      if constexpr (K_PER_G == 1) {
        row_offset_v = vld1q_s32(row_offsets + j + VLEN);
      } else {
        row_offset_v =
            vcombine_s32(vld1_s32(row_offsets + (j + VLEN) / 2), vdup_n_s32(0));
        row_offset_v = vzip1q_u32(row_offset_v, row_offset_v);
      }
      if constexpr (
          Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = vld1q_s32(B_zero_point + j + VLEN);
      } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
        B_zero_point_v = vcombine_s32(
            vld1_s32(B_zero_point + (j + VLEN) / 2), vdup_n_s32(0));
        B_zero_point_v = vzip1q_u32(B_zero_point_v, B_zero_point_v);
      }
      y_v = vmlsq_s32(y_v, row_offset_v, B_zero_point_v);
    }
    if constexpr (!A_SYMMETRIC) {
      y_v = vmlsq_s32(y_v, A_zero_point_v, vld1q_s32(col_offsets + j + VLEN));
    }

    if constexpr (!B_SYMMETRIC) {
      if constexpr (K_PER_G == 1) {
        row_offset_v = vld1q_s32(row_offsets + j + 2 * VLEN);
      } else {
        row_offset_v = vcombine_s32(
            vld1_s32(row_offsets + (j + 2 * VLEN) / 2), vdup_n_s32(0));
        row_offset_v = vzip1q_u32(row_offset_v, row_offset_v);
      }
      if constexpr (
          Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = vld1q_s32(B_zero_point + j + 2 * VLEN);
      } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
        B_zero_point_v = vcombine_s32(
            vld1_s32(B_zero_point + (j + 2 * VLEN) / 2), vdup_n_s32(0));
        B_zero_point_v = vzip1q_u32(B_zero_point_v, B_zero_point_v);
      }
      z_v = vmlsq_s32(z_v, row_offset_v, B_zero_point_v);
    }
    if constexpr (!A_SYMMETRIC) {
      z_v =
          vmlsq_s32(z_v, A_zero_point_v, vld1q_s32(col_offsets + j + 2 * VLEN));
    }

    if constexpr (!B_SYMMETRIC) {
      if constexpr (K_PER_G == 1) {
        row_offset_v = vld1q_s32(row_offsets + j + 3 * VLEN);
      } else {
        row_offset_v = vcombine_s32(
            vld1_s32(row_offsets + (j + 3 * VLEN) / 2), vdup_n_s32(0));
        row_offset_v = vzip1q_u32(row_offset_v, row_offset_v);
      }
      if constexpr (
          Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = vld1q_s32(B_zero_point + j + 3 * VLEN);
      } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
        B_zero_point_v = vcombine_s32(
            vld1_s32(B_zero_point + (j + 3 * VLEN) / 2), vdup_n_s32(0));
        B_zero_point_v = vzip1q_u32(B_zero_point_v, B_zero_point_v);
      }
      w_v = vmlsq_s32(w_v, row_offset_v, B_zero_point_v);
    }
    if constexpr (!A_SYMMETRIC) {
      w_v =
          vmlsq_s32(w_v, A_zero_point_v, vld1q_s32(col_offsets + j + 3 * VLEN));
    }

    // convert to float
    float32x4_t xf_v, yf_v, zf_v, wf_v;
    if constexpr (HAS_BIAS) { // static if
      if constexpr (std::is_same_v<BIAS_TYPE, float>) {
        float32x4_t x_bias_v, y_bias_v, z_bias_v, w_bias_v;
        if constexpr (
            Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
            (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
          x_bias_v = vdivq_f32(
              vld1q_f32(bias + j + 0 * VLEN),
              vld1q_f32(act_times_w_scale + j + 0 * VLEN));
          y_bias_v = vdivq_f32(
              vld1q_f32(bias + j + 1 * VLEN),
              vld1q_f32(act_times_w_scale + j + 1 * VLEN));
          z_bias_v = vdivq_f32(
              vld1q_f32(bias + j + 2 * VLEN),
              vld1q_f32(act_times_w_scale + j + 2 * VLEN));
          w_bias_v = vdivq_f32(
              vld1q_f32(bias + j + 3 * VLEN),
              vld1q_f32(act_times_w_scale + j + 3 * VLEN));
        } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
          static_assert(K_PER_G == 2);
          auto tmp = vcombine_f32(
              vld1_f32(act_times_w_scale + (j + 0 * VLEN) / 2),
              vdup_n_f32(0.0f));

          x_bias_v =
              vdivq_f32(vld1q_f32(bias + j + 0 * VLEN), vzip1q_f32(tmp, tmp));

          tmp = vcombine_f32(
              vld1_f32(act_times_w_scale + (j + 1 * VLEN) / 2),
              vdup_n_f32(0.0f));
          y_bias_v =
              vdivq_f32(vld1q_f32(bias + j + 1 * VLEN), vzip1q_f32(tmp, tmp));

          tmp = vcombine_f32(
              vld1_f32(act_times_w_scale + (j + 2 * VLEN) / 2),
              vdup_n_f32(0.0f));
          z_bias_v =
              vdivq_f32(vld1q_f32(bias + j + 2 * VLEN), vzip1q_f32(tmp, tmp));

          tmp = vcombine_f32(
              vld1_f32(act_times_w_scale + (j + 3 * VLEN) / 2),
              vdup_n_f32(0.0f));
          w_bias_v =
              vdivq_f32(vld1q_f32(bias + j + 3 * VLEN), vzip1q_f32(tmp, tmp));

        } else {
          x_bias_v =
              vmulq_f32(vld1q_f32(bias + j + 0 * VLEN), act_times_w_rcp_v);
          y_bias_v =
              vmulq_f32(vld1q_f32(bias + j + 1 * VLEN), act_times_w_rcp_v);
          z_bias_v =
              vmulq_f32(vld1q_f32(bias + j + 2 * VLEN), act_times_w_rcp_v);
          w_bias_v =
              vmulq_f32(vld1q_f32(bias + j + 3 * VLEN), act_times_w_rcp_v);
        }
        xf_v = vaddq_f32(vcvtq_f32_s32(x_v), x_bias_v);
        yf_v = vaddq_f32(vcvtq_f32_s32(y_v), y_bias_v);
        zf_v = vaddq_f32(vcvtq_f32_s32(z_v), z_bias_v);
        wf_v = vaddq_f32(vcvtq_f32_s32(w_v), w_bias_v);
      } else {
        x_v = vaddq_s32(x_v, vld1q_s32(bias + j + 0 * VLEN));
        y_v = vaddq_s32(y_v, vld1q_s32(bias + j + 1 * VLEN));
        z_v = vaddq_s32(z_v, vld1q_s32(bias + j + 2 * VLEN));
        w_v = vaddq_s32(w_v, vld1q_s32(bias + j + 3 * VLEN));
        xf_v = vcvtq_f32_s32(x_v);
        yf_v = vcvtq_f32_s32(y_v);
        zf_v = vcvtq_f32_s32(z_v);
        wf_v = vcvtq_f32_s32(w_v);
      }
    } else {
      xf_v = vcvtq_f32_s32(x_v);
      yf_v = vcvtq_f32_s32(y_v);
      zf_v = vcvtq_f32_s32(z_v);
      wf_v = vcvtq_f32_s32(w_v);
    }

    if constexpr (
        Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = vld1q_f32(C_multiplier + j + 0 * VLEN);
    } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v =
          vcombine_f32(vld1_f32(C_multiplier + j / 2), vdup_n_f32(0.0f));
      multiplier_v = vzip1q_u32(multiplier_v, multiplier_v);
    }
    float32x4_t x_scaled_v = vmulq_f32(xf_v, multiplier_v);
    if constexpr (
        Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = vld1q_f32(C_multiplier + j + 1 * VLEN);
    } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v = vcombine_f32(
          vld1_f32(C_multiplier + (j + VLEN) / 2), vdup_n_f32(0.0f));
      multiplier_v = vzip1q_u32(multiplier_v, multiplier_v);
    }
    float32x4_t y_scaled_v = vmulq_f32(yf_v, multiplier_v);
    if constexpr (
        Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = vld1q_f32(C_multiplier + j + 2 * VLEN);
    } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v = vcombine_f32(
          vld1_f32(C_multiplier + (j + 2 * VLEN) / 2), vdup_n_f32(0.0f));
      multiplier_v = vzip1q_u32(multiplier_v, multiplier_v);
    }
    float32x4_t z_scaled_v = vmulq_f32(zf_v, multiplier_v);
    if constexpr (
        Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = vld1q_f32(C_multiplier + j + 3 * VLEN);
    } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v = vcombine_f32(
          vld1_f32(C_multiplier + (j + 3 * VLEN) / 2), vdup_n_f32(0.0f));
      multiplier_v = vzip1q_u32(multiplier_v, multiplier_v);
    }
    float32x4_t w_scaled_v = vmulq_f32(wf_v, multiplier_v);

    // vcvtnq_s32_f32 always rounds to nearest, which is slightly different
    // from x86's _mm256_cvtps_epi32 which rounds according to the current
    // rounding mode, which may not be round to nearest. To help catch issues
    // and debug, we add an assertion here.
    assert(fegetround() == FE_TONEAREST);
    int32x4_t x_rounded_v = vcvtnq_s32_f32(x_scaled_v);
    int32x4_t y_rounded_v = vcvtnq_s32_f32(y_scaled_v);
    int32x4_t z_rounded_v = vcvtnq_s32_f32(z_scaled_v);
    int32x4_t w_rounded_v = vcvtnq_s32_f32(w_scaled_v);

    int16x8_t xy_packed_v = vqaddq_s16(
        vcombine_s16(vqmovn_s32(x_rounded_v), vqmovn_s32(y_rounded_v)),
        C_zero_point_epi16_v);
    int16x8_t zw_packed_v = vqaddq_s16(
        vcombine_s16(vqmovn_s32(z_rounded_v), vqmovn_s32(w_rounded_v)),
        C_zero_point_epi16_v);
    uint8x16_t xyzw_packed_v =
        vcombine_u8(vqmovun_s16(xy_packed_v), vqmovun_s16(zw_packed_v));
    uint8x16_t xyzw_clamped_v =
        vmaxq_u8(FUSE_RELU ? C_zero_point_epi8_v : min_v, xyzw_packed_v);

    vst1q_u8(C_uint8 + j, xyzw_clamped_v);
  } // j loop vectorized and unrolled 4x

vec_tail:
  for (; j < n / VLEN * VLEN; j += VLEN) {
    int32x4_t x_v = vld1q_s32(C_int32 + j);

    if constexpr (!B_SYMMETRIC) {
      int32x4_t row_offset_v;
      if constexpr (K_PER_G == 1) {
        row_offset_v = vld1q_s32(row_offsets + j);
      } else {
        static_assert(K_PER_G == 2);
        // Load row_offsets for 2 groups and broadcast by 2 times.
        row_offset_v =
            vcombine_s32(vld1_s32(row_offsets + j / 2), vdup_n_s32(0));
        row_offset_v = vzip1q_u32(row_offset_v, row_offset_v);
      }
      if constexpr (
          Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
          (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
        B_zero_point_v = vld1q_s32(B_zero_point + j);
      } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
        static_assert(K_PER_G == 2);
        B_zero_point_v =
            vcombine_s32(vld1_s32(B_zero_point + j / 2), vdup_n_s32(0));
        B_zero_point_v = vzip1q_u32(B_zero_point_v, B_zero_point_v);
      }
      x_v = vmlsq_s32(x_v, row_offset_v, B_zero_point_v);
    }
    if constexpr (!A_SYMMETRIC) {
      x_v = vmlsq_s32(x_v, A_zero_point_v, vld1q_s32(col_offsets + j));
    }

    // Convert to float
    float32x4_t xf_v;
    if constexpr (HAS_BIAS) { // static if
      if constexpr (std::is_same_v<BIAS_TYPE, float>) {
        float32x4_t x_bias_v;
        if constexpr (
            Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
            (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
          x_bias_v =
              vdivq_f32(vld1q_f32(bias + j), vld1q_f32(act_times_w_scale + j));
        } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
          auto tmp = vcombine_f32(
              vld1_f32(act_times_w_scale + j / 2), vdup_n_f32(0.0f));

          x_bias_v = vdivq_f32(vld1q_f32(bias + j), vzip1q_f32(tmp, tmp));
        } else {
          x_bias_v = vmulq_f32(vld1q_f32(bias + j), act_times_w_rcp_v);
        }
        xf_v = vaddq_f32(vcvtq_f32_s32(x_v), x_bias_v);
      } else {
        x_v = vaddq_s32(
            x_v, vld1q_s32(reinterpret_cast<const int32_t*>(bias + j)));
        xf_v = vcvtq_f32_s32(x_v);
      }
    } else {
      xf_v = vcvtq_f32_s32(x_v);
    }

    if constexpr (
        Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      multiplier_v = vld1q_f32(C_multiplier + j);
    } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
      multiplier_v =
          vcombine_f32(vld1_f32(C_multiplier + j / 2), vdup_n_f32(0.0f));
      multiplier_v = vzip1q_u32(multiplier_v, multiplier_v);
    }
    float32x4_t x_scaled_v = vmulq_f32(xf_v, multiplier_v);
    // vcvtnq_s32_f32 always rounds to nearest, which is slightly different
    // from x86's _mm256_cvtps_epi32 which rounds according to the current
    // rounding mode, which may not be round to nearest. To help catch issues
    // and debug, we add an assertion here.
    assert(fegetround() == FE_TONEAREST);
    int32x4_t x_rounded_v = vcvtnq_s32_f32(x_scaled_v);

    int16x8_t x_packed_v_s16 = vqaddq_s16(
        vcombine_s16(vqmovn_s32(x_rounded_v), vdup_n_s16(0)),
        C_zero_point_epi16_v);
    uint8x8_t x_packed_v_u8 = vqmovun_s16(x_packed_v_s16);
    uint8x8_t x_clamped_v = vmax_u8(
        FUSE_RELU ? vget_low_u8(C_zero_point_epi8_v) : vget_low_u8(min_v),
        x_packed_v_u8);

    vst1_lane_u32(C_uint8 + j, vreinterpret_u32_u8(x_clamped_v), 0);
  } // j loop vectorized

  // There are some leftovers that cannot fit in one full vector. Instead of
  // doing a scalar loop, we prepare j to be n - VLEN and jump back to the
  // above loop for one extra iteration. Compared to a scalar loop, this reuses
  // vector loop code so code size bloat is minimal. Another alternative is
  // to use a partial vector register, but that also bloats code size more than
  // reusing the above loop body.
  if (j < n) {
    j = n - VLEN;
    goto vec_tail;
  }
}

} // namespace fbgemm

#endif
