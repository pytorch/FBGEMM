/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__aarch64__)

#include "fbgemm/Utils.h"

#define FBGEMM_EXPORTS
#include <arm_fp16.h> // @manual
#include <arm_neon.h> // @manual
#if HAVE_SVE
#include <arm_neon_sve_bridge.h> // @manual
#include <arm_sve.h> // @manual
#endif

#include <algorithm> //for std::min/std::max
#include <cassert> //for assert
#include <cfloat> // for FLT_MAX
#include <cmath> //for nearbyint
#include <cstring> //for memcpy
#include <limits> //for numeric_limits
#include "fbgemm/FloatConversion.h"
#include "fbgemm/QuantUtilsNeon.h"
#include "fbgemm/Types.h"

namespace fbgemm {

using namespace std;
////////////////////////////////////////////////////////////////////////////////
// Utility functions
static inline void
FindMinMaxImpl_f32(const float* m, float* min, float* max, uint64_t count) {
  float first = *m;

  float tmp_min_s = first;
  float tmp_max_s = first;

  float32x4_t temp_min_0 = vdupq_n_f32(first);
  float32x4_t temp_min_1 = vdupq_n_f32(first);
  float32x4_t temp_max_0 = vdupq_n_f32(first);
  float32x4_t temp_max_1 = vdupq_n_f32(first);
  constexpr uint64_t kItemsPerIter = 8;
  uint64_t loopIters = count / kItemsPerIter;
  uint64_t loopRemainder = count % kItemsPerIter;

  if (__builtin_expect(loopIters > 0, 1)) {
    do {
      float32x4_t v0 = vld1q_f32(m);
      float32x4_t v1 = vld1q_f32(m + 4);
      m += kItemsPerIter;
      loopIters -= 1;
      temp_min_0 = vminq_f32(temp_min_0, v0);
      temp_min_1 = vminq_f32(temp_min_1, v1);
      temp_max_0 = vmaxq_f32(temp_max_0, v0);
      temp_max_1 = vmaxq_f32(temp_max_1, v1);
    } while (loopIters > 0);

    temp_min_0 = vminq_f32(temp_min_0, temp_min_1);
    temp_max_0 = vmaxq_f32(temp_max_0, temp_max_1);

    tmp_min_s = vminvq_f32(temp_min_0);
    tmp_max_s = vmaxvq_f32(temp_max_0);
  }

#ifdef __clang__
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#elif defined(__GNUC__) && __GNUC__ >= 14
#pragma GCC novector unroll 0
#endif
  while (loopRemainder > 0) {
    float tmp = *m++;
    loopRemainder -= 1;
    tmp_min_s = std::min(tmp_min_s, tmp);
    tmp_max_s = std::max(tmp_max_s, tmp);
  }

  *min = tmp_min_s;
  *max = tmp_max_s;
}

void FindMinMax(const float* m, float* min, float* max, int64_t len) {
  if (__builtin_expect(len <= 0, 0)) {
    *min = 0.0f;
    *max = 0.0f;
    return;
  }

  FindMinMaxImpl_f32(m, min, max, static_cast<uint64_t>(len));
}

#if HAVE_SVE

template <typename OutType>
static inline void FindMinMaxImpl_f16(
    const float16_t* m,
    OutType* min,
    OutType* max,
    uint64_t count) {
  float16_t first = *m;

  float16_t tmp_min_s = first;
  float16_t tmp_max_s = first;

  float16x8_t temp_min_0 = vdupq_n_f16(first);
  float16x8_t temp_min_1 = vdupq_n_f16(first);
  float16x8_t temp_max_0 = vdupq_n_f16(first);
  float16x8_t temp_max_1 = vdupq_n_f16(first);
  constexpr uint64_t kItemsPerIter = 16;
  uint64_t loopIters = count / kItemsPerIter;
  uint64_t loopRemainder = count % kItemsPerIter;

  if (__builtin_expect(loopIters > 0, 1)) {
    do {
      float16x8_t v0 = vld1q_f16(m);
      float16x8_t v1 = vld1q_f16(m + 8);
      m += kItemsPerIter;
      loopIters -= 1;
      temp_min_0 = vminq_f16(temp_min_0, v0);
      temp_min_1 = vminq_f16(temp_min_1, v1);
      temp_max_0 = vmaxq_f16(temp_max_0, v0);
      temp_max_1 = vmaxq_f16(temp_max_1, v1);
    } while (loopIters > 0);

    temp_min_0 = vminq_f16(temp_min_0, temp_min_1);
    temp_max_0 = vmaxq_f16(temp_max_0, temp_max_1);

    tmp_min_s = vminvq_f16(temp_min_0);
    tmp_max_s = vmaxvq_f16(temp_max_0);
  }

#ifdef __clang__
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#elif defined(__GNUC__) && __GNUC__ >= 14
#pragma GCC novector unroll 0
#endif
  while (loopRemainder > 0) {
    float16_t tmp = *m++;
    loopRemainder -= 1;
    tmp_min_s = vminh_f16(tmp_min_s, tmp);
    tmp_max_s = vmaxh_f16(tmp_max_s, tmp);
  }

  *min = static_cast<OutType>(tmp_min_s);
  *max = static_cast<OutType>(tmp_max_s);
}

template <typename InputType>
void FloatOrHalfToFused8BitRowwiseQuantizedSBFloatNeon(
    const InputType* input,
    size_t input_rows,
    int input_columns,
    uint8_t* output) {
  constexpr float kEpsilon = 1e-8f;

  if (input_rows == 0 || input_columns <= 0) {
    return;
  }

  uint64_t column_count = static_cast<uint64_t>(input_columns);

  const uint64_t output_columns = column_count + 2 * sizeof(float);

  for (size_t row = 0; __builtin_expect(row < input_rows, 1); ++row) {
    const InputType* input_row = input + row * column_count;
    uint8_t* output_row = output + row * output_columns;

    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + column_count);

    float minimum_element;
    float maximum_element;
    if constexpr (std::is_same_v<InputType, float>) {
      FindMinMaxImpl_f32(
          input_row, &minimum_element, &maximum_element, column_count);
    } else {
      FindMinMaxImpl_f16(
          reinterpret_cast<const float16_t*>(input_row),
          &minimum_element,
          &maximum_element,
          column_count);
    }
    float range = maximum_element - minimum_element;

    const auto inverse_scale = 255.0f / (range + kEpsilon);

    float32x4_t inverse_scale_v = vdupq_n_f32(inverse_scale);
    float32x4_t min_v = vdupq_n_f32(minimum_element);

    constexpr uint64_t kItemsPerIter = 8;
    uint64_t loopIters = column_count / kItemsPerIter;
    uint64_t loopRemainder = column_count % kItemsPerIter;

    output_row_scale_bias[0] = range / 255.0f;
    output_row_scale_bias[1] = minimum_element;

    while (__builtin_expect(loopIters > 0, 1)) {
      float32x4_t v0;
      float32x4_t v1;

      if constexpr (std::is_same_v<InputType, float>) {
        v0 = vld1q_f32(input_row);
        v1 = vld1q_f32(input_row + 4);
      } else {
        float16x8_t h0 =
            vld1q_f16(reinterpret_cast<const float16_t*>(input_row));
        v0 = vcvt_f32_f16(vget_low_f16(h0));
        v1 = vcvt_high_f32_f16(h0);
      }

      input_row += kItemsPerIter;
      loopIters -= 1;

      v0 = vsubq_f32(v0, min_v);
      v1 = vsubq_f32(v1, min_v);

      v0 = vmulq_f32(v0, inverse_scale_v);
      v1 = vmulq_f32(v1, inverse_scale_v);

      int32x4_t i0 = vcvtnq_s32_f32(v0);
      int32x4_t i1 = vcvtnq_s32_f32(v1);

      svst1b_s32(
          svptrue_b8(),
          reinterpret_cast<int8_t*>(output_row),
          svset_neonq_s32(svundef_s32(), i0));
      svst1b_s32(
          svptrue_b8(),
          reinterpret_cast<int8_t*>(output_row + 4),
          svset_neonq_s32(svundef_s32(), i1));

      output_row += kItemsPerIter;
    }

#ifdef __clang__
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#elif defined(__GNUC__) && __GNUC__ >= 14
#pragma GCC novector unroll 0
#endif
    while (loopRemainder > 0) {
      float32x4_t v0;
      if constexpr (std::is_same_v<InputType, float>) {
        v0[0] = *input_row++;
      } else {
        v0[0] =
            static_cast<float>(*reinterpret_cast<const float16_t*>(input_row));
        input_row += 1;
      }
      loopRemainder -= 1;
      v0 = vsubq_f32(v0, min_v);
      v0 = vmulq_f32(v0, inverse_scale_v);
      int32x4_t i0 = vcvtnq_s32_f32(v0);
      *output_row = i0[0];
      output_row += 1;
    }

  } // for each row
}

template <typename InputType, int BIT_RATE>
void FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfNeon(
    const InputType* input,
    size_t input_rows,
    int input_columns,
    std::uint8_t* output) {
  if (input_rows == 0 || input_columns <= 0) {
    return;
  }

  static_assert(
      std::is_same_v<InputType, float> || std::is_same_v<InputType, float16>,
      "Only float and float16 types are allowed.");

  static_assert(
      (BIT_RATE == 8) || (BIT_RATE == 4) || (BIT_RATE == 2),
      "Only bit rates of 8, 4 and 2 are allowed.");

  constexpr uint64_t num_elem_per_byte = 8 / BIT_RATE;
  uint64_t column_count = static_cast<uint64_t>(input_columns);
  const int output_columns =
      (column_count + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(float16);

  for (size_t row = 0; __builtin_expect(row < input_rows, 1); ++row) {
    const InputType* input_row = input + row * column_count;
    std::uint8_t* output_row = output + row * output_columns;
    float16_t* output_row_scale_bias = reinterpret_cast<float16_t*>(
        output_row +
        (column_count + num_elem_per_byte - 1) / num_elem_per_byte);

    float minimum_element;
    float maximum_element;
    float16_t minimum_element_fp16;
    if constexpr (std::is_same_v<InputType, float>) {
      FindMinMaxImpl_f32(
          input_row, &minimum_element, &maximum_element, column_count);
      minimum_element_fp16 = static_cast<float16_t>(minimum_element);
      minimum_element = static_cast<float>(minimum_element_fp16);
    } else {
      float16_t maximum_element_fp16;
      FindMinMaxImpl_f16(
          reinterpret_cast<const float16_t*>(input_row),
          &minimum_element_fp16,
          &maximum_element_fp16,
          column_count);
      minimum_element = static_cast<float>(minimum_element_fp16);
      maximum_element = static_cast<float>(maximum_element_fp16);
    }

    const float range = maximum_element - minimum_element;

    float scale = range == 0 ? 1.0f : range / ((1 << BIT_RATE) - 1);
    float16_t scale_fp16 = static_cast<float16_t>(scale);
    scale = static_cast<float>(scale_fp16);
    svfloat32_t inverse_scale_sv;
    if (scale != 0.0f) {
      float inverse_scale = 1.0f / scale;
      inverse_scale_sv = svdup_n_f32(inverse_scale);
      bool isInf = svptest_any(
          svptrue_b8(),
          svcmpuo_f32(
              svptrue_b8(),
              svsub_f32_x(svptrue_b8(), inverse_scale_sv, inverse_scale_sv),
              svdup_n_f32(0.0)));
      if (isInf) {
        scale_fp16 = static_cast<float16_t>(1.0f);
        scale = 1.0f;
        inverse_scale_sv = svdup_n_f32(1.0f);
      }
    } else {
      // Corner case handling when maximum_element == minimum_element
      // Any scale would work because X - minimum_element will be 0 for all X
      scale_fp16 = static_cast<float16_t>(1.0f);
      scale = 1.0f;
      inverse_scale_sv = svdup_n_f32(1.0f);
    }

    constexpr uint64_t kItemsPerIter = 8;
    uint64_t loopIters = column_count / kItemsPerIter;
    uint64_t loopRemainder = column_count % kItemsPerIter;

    output_row_scale_bias[0] = scale_fp16;
    output_row_scale_bias[1] = minimum_element_fp16;

    float32x4_t inverse_scale_v = svget_neonq(inverse_scale_sv);
    float32x4_t min_v = vdupq_n_f32(minimum_element);

    constexpr unsigned int maxValPerBitRate = (1ul << BIT_RATE) - 1;
    uint32x4_t maxval_v = vdupq_n_u32(maxValPerBitRate);

    svbool_t lastPredA = svwhilelt_b32_u64(0, loopRemainder);
    svbool_t lastPredB = svwhilelt_b32_u64(4, loopRemainder);

    while (__builtin_expect(loopIters > 0, 1)) {
      float32x4_t v0;
      float32x4_t v1;

      if constexpr (std::is_same_v<InputType, float>) {
        v0 = vld1q_f32(input_row);
        v1 = vld1q_f32(input_row + 4);
      } else {
        float16x8_t h0 =
            vld1q_f16(reinterpret_cast<const float16_t*>(input_row));
        v0 = vcvt_f32_f16(vget_low_f16(h0));
        v1 = vcvt_high_f32_f16(h0);
      }

      input_row += kItemsPerIter;
      loopIters -= 1;

      v0 = vsubq_f32(v0, min_v);
      v1 = vsubq_f32(v1, min_v);

      v0 = vmulq_f32(v0, inverse_scale_v);
      v1 = vmulq_f32(v1, inverse_scale_v);

      int32x4_t i0 = vcvtnq_s32_f32(v0);
      int32x4_t i1 = vcvtnq_s32_f32(v1);

      uint32x4_t u0 = vminq_u32(vreinterpretq_u32_s32(i0), maxval_v);
      uint32x4_t u1 = vminq_u32(vreinterpretq_u32_s32(i1), maxval_v);

      if constexpr (num_elem_per_byte == 1) {
        svst1b_u32(
            svptrue_b8(), output_row, svset_neonq_u32(svundef_u32(), u0));
        svst1b_u32(
            svptrue_b8(), output_row + 4, svset_neonq_u32(svundef_u32(), u1));
      } else {
        constexpr uint64_t shiftVar = num_elem_per_byte == 2 ? 28 : 30;

        uint64x2_t u2 = vreinterpretq_u64_u32(u0) >> shiftVar;
        uint64x2_t u3 = vreinterpretq_u64_u32(u1) >> shiftVar;

        u2 = veorq_u64(u2, vreinterpretq_u64_u32(u0));
        u3 = veorq_u64(u3, vreinterpretq_u64_u32(u1));

        if constexpr (num_elem_per_byte == 2) {
          svst1b_u64(
              svptrue_b8(), output_row, svset_neonq_u64(svundef_u64(), u2));
          svst1b_u64(
              svptrue_b8(), output_row + 2, svset_neonq_u64(svundef_u64(), u3));

        } else if constexpr (num_elem_per_byte == 4) {
          auto u4 = vdup_laneq_u8(vreinterpretq_u8_u64(u2), 8);
          auto u5 = vdup_laneq_u8(vreinterpretq_u8_u64(u3), 8);

          u4 = u4 << 4;
          u5 = u5 << 4;

          u4 = veor_u8(u4, vget_low_u8(u2));
          u5 = veor_u8(u5, vget_low_u8(u3));

          vst1_lane_u8(output_row, u4, 0);
          vst1_lane_u8(output_row + 1, u5, 0);
        }
      }

      constexpr uint64_t bytesStored = kItemsPerIter / num_elem_per_byte;
      output_row += bytesStored;
    }

    if (loopRemainder > 0) {
      float32x4_t v0;
      float32x4_t v1;

      if constexpr (std::is_same_v<InputType, float>) {
        v0 = svget_neonq(svld1_f32(lastPredA, input_row));
        v1 = svget_neonq(svld1_f32(lastPredB, input_row + 4));
      } else {
        auto h0 = svld1uh_u32(
            lastPredA, reinterpret_cast<const uint16_t*>(input_row));
        auto h1 = svld1uh_u32(
            lastPredB, reinterpret_cast<const uint16_t*>(input_row + 4));
        v0 = svget_neonq(
            svcvt_f32_f16_x(svptrue_b8(), svreinterpret_f16_u32(h0)));
        v1 = svget_neonq(
            svcvt_f32_f16_x(svptrue_b8(), svreinterpret_f16_u32(h1)));
      }

      v0 = vsubq_f32(v0, min_v);
      v1 = vsubq_f32(v1, min_v);

      v0 = vmulq_f32(v0, inverse_scale_v);
      v1 = vmulq_f32(v1, inverse_scale_v);

      int32x4_t i0 = vcvtnq_s32_f32(v0);
      int32x4_t i1 = vcvtnq_s32_f32(v1);

      uint32x4_t u0 = vminq_u32(vreinterpretq_u32_s32(i0), maxval_v);
      uint32x4_t u1 = vminq_u32(vreinterpretq_u32_s32(i1), maxval_v);

      if constexpr (num_elem_per_byte == 1) {
        svst1b_u32(lastPredA, output_row, svset_neonq_u32(svundef_u32(), u0));
        svst1b_u32(
            lastPredB, output_row + 4, svset_neonq_u32(svundef_u32(), u1));
      } else {
        constexpr uint64_t shiftVar = num_elem_per_byte == 2 ? 28 : 30;

        uint64x2_t u2 = vreinterpretq_u64_u32(u0) >> shiftVar;
        uint64x2_t u3 = vreinterpretq_u64_u32(u1) >> shiftVar;

        u2 = veorq_u64(u2, vreinterpretq_u64_u32(u0));
        u3 = veorq_u64(u3, vreinterpretq_u64_u32(u1));

        if constexpr (num_elem_per_byte == 2) {
          svst1b_u64(lastPredA, output_row, svset_neonq_u64(svundef_u64(), u2));
          svst1b_u64(
              lastPredB, output_row + 2, svset_neonq_u64(svundef_u64(), u3));

        } else if constexpr (num_elem_per_byte == 4) {
          auto u4 = vdup_laneq_u8(vreinterpretq_u8_u64(u2), 8);
          auto u5 = vdup_laneq_u8(vreinterpretq_u8_u64(u3), 8);

          u4 = u4 << 4;
          u5 = u5 << 4;

          u4 = veor_u8(u4, vget_low_u8(u2));
          u5 = veor_u8(u5, vget_low_u8(u3));

          vst1_lane_u8(output_row, u4, 0);
          if (loopRemainder > 4) {
            vst1_lane_u8(output_row + 1, u5, 0);
          }
        }
      }
    }
  } // for each row
}

template <typename OutputType>
void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfNeon(
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    OutputType* output) {
  size_t output_columns = std::max<int>(input_columns - 2 * sizeof(float), 0);

  svbool_t allTruePred = svptrue_b32();
  size_t output_columns_mod = output_columns % 8;
  svbool_t lastPredA = svwhilelt_b32_u64(0, output_columns_mod);
  svbool_t lastPredB = svwhilelt_b32_u64(4, output_columns_mod);
  svbool_t lastPredC = svwhilelt_b16_u64(0, output_columns_mod);

  const uint64_t* input_row_v_0 = reinterpret_cast<const uint64_t*>(input);
  const uint64_t* input_row_v_1 = reinterpret_cast<const uint64_t*>(input + 4);
  OutputType* output_row = output;

  for (; input_rows > 0; --input_rows) {
    const float* input_row_scale_bias = reinterpret_cast<const float*>(
        reinterpret_cast<const uint8_t*>(input_row_v_0) + output_columns);

    float scale = input_row_scale_bias[0];
    float bias = input_row_scale_bias[1];
    svfloat32_t scale_v = svdup_n_f32(scale);
    svfloat32_t bias_v = svdup_n_f32(bias);

    float32x4x2_t* output_row_v = reinterpret_cast<float32x4x2_t*>(output_row);
    float16x4x2_t* output_row_v_half =
        reinterpret_cast<float16x4x2_t*>(output_row);

    size_t colIndex = 0;
    for (size_t colMax = output_columns / 8;
         __builtin_expect(colIndex < colMax, 1);
         ++colIndex) {
      svuint32_t in_v_0 = svld1ub_u32(
          allTruePred,
          reinterpret_cast<const uint8_t*>(input_row_v_0 + colIndex));
      svuint32_t in_v_1 = svld1ub_u32(
          allTruePred,
          reinterpret_cast<const uint8_t*>(input_row_v_1 + colIndex));
      svfloat32_t in_v_0_f = svcvt_f32_u32_x(allTruePred, in_v_0);
      svfloat32_t in_v_1_f = svcvt_f32_u32_x(allTruePred, in_v_1);

      in_v_0_f = svmad_f32_m(allTruePred, in_v_0_f, scale_v, bias_v);
      in_v_1_f = svmad_f32_m(allTruePred, in_v_1_f, scale_v, bias_v);

      if constexpr (std::is_same_v<OutputType, float>) {
        output_row_v[colIndex].val[0] = svget_neonq(in_v_0_f);
        output_row_v[colIndex].val[1] = svget_neonq(in_v_1_f);
      } else {
        float16x4_t dequantzed_v_half_low = vcvt_f16_f32(svget_neonq(in_v_0_f));
        float16x4_t dequantzed_v_half_high =
            vcvt_f16_f32(svget_neonq(in_v_1_f));
        output_row_v_half[colIndex].val[0] = dequantzed_v_half_low;
        output_row_v_half[colIndex].val[1] = dequantzed_v_half_high;
      }
    }

    if (output_columns_mod != 0) {
      svuint32_t in_v_0 = svld1ub_u32(
          lastPredA,
          reinterpret_cast<const uint8_t*>(input_row_v_0 + colIndex));
      svuint32_t in_v_1 = svld1ub_u32(
          lastPredB,
          reinterpret_cast<const uint8_t*>(input_row_v_1 + colIndex));
      svfloat32_t in_v_0_f = svcvt_f32_u32_x(lastPredA, in_v_0);
      svfloat32_t in_v_1_f = svcvt_f32_u32_x(lastPredB, in_v_1);

      in_v_0_f = svmad_f32_m(lastPredA, in_v_0_f, scale_v, bias_v);
      in_v_1_f = svmad_f32_m(lastPredB, in_v_1_f, scale_v, bias_v);

      if constexpr (std::is_same_v<OutputType, float>) {
        svst1_f32(lastPredA, (float32_t*)&(output_row_v[colIndex]), in_v_0_f);
        svst1_f32(
            lastPredB, (float32_t*)&(output_row_v[colIndex].val[1]), in_v_1_f);
      } else {
        float16x4_t dequantzed_v_half_low_low =
            vcvt_f16_f32(svget_neonq(in_v_0_f));
        float16x8_t dequantzed_v_half_low =
            vcvt_high_f16_f32(dequantzed_v_half_low_low, svget_neonq(in_v_1_f));
        svst1_f16(
            lastPredC,
            (float16_t*)&(output_row_v_half[colIndex]),
            svset_neonq_f16(svundef_f16(), dequantzed_v_half_low));
      }
    }

    input_row_v_0 = reinterpret_cast<const uint64_t*>(
        reinterpret_cast<const uint8_t*>(input_row_v_0) + input_columns);
    input_row_v_1 = reinterpret_cast<const uint64_t*>(
        reinterpret_cast<const uint8_t*>(input_row_v_1) + input_columns);
    output_row += output_columns;

  } // for each row
}

template <typename OutputType, int BIT_RATE>
void FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfNeon(
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    OutputType* output) {
  svbool_t allTruePred = svptrue_b8();
  constexpr size_t kNumElemsPerIter = 8;
  constexpr size_t kNumBytesPerIter = BIT_RATE;
  constexpr size_t kNumElemsPerByte = 8 / BIT_RATE;

  size_t bytesPerRow = std::max<int>(input_columns - 2 * sizeof(uint16_t), 0);
  size_t output_columns = bytesPerRow * kNumElemsPerByte;

  size_t input_columns_mod = bytesPerRow % kNumBytesPerIter;
  size_t output_columns_mod = output_columns % kNumElemsPerIter;
  svbool_t lastPredA = svwhilelt_b32_u64(0, output_columns_mod);
  svbool_t lastPredB = svwhilelt_b32_u64(4, output_columns_mod);
  svbool_t lastPredC = svwhilelt_b16_u64(0, output_columns_mod);
  svbool_t lastPredD = svwhilelt_b64_u64(0, input_columns_mod);
  svbool_t lastPredE = svwhilelt_b64_u64(2, input_columns_mod);

  svuint32_t shift = svindex_u32(0, 2); // {0, 2, 4, 6};
  svuint64_t multiplier = svdup_n_u64((1ULL << 28) + 1);

  for (; input_rows > 0; --input_rows) {
    const float* input_row_scale_bias =
        reinterpret_cast<const float*>(input + bytesPerRow);

    svfloat16_t scale_bias_v =
        svreinterpret_f16_f32(svdup_n_f32(*input_row_scale_bias));

    svfloat32_t scale_v =
        svcvt_f32_f16_m(svundef_f32(), allTruePred, scale_bias_v);
    svfloat32_t bias_v =
        svcvtlt_f32_f16_m(svundef_f32(), allTruePred, scale_bias_v);

    for (size_t iters = bytesPerRow / kNumBytesPerIter;
         __builtin_expect(iters > 0, 1);
         --iters) {
      svuint32_t in_v_0;
      svuint32_t in_v_1;
      if constexpr (BIT_RATE == 8) {
        in_v_0 = svld1ub_u32(allTruePred, input);
        in_v_1 = svld1ub_u32(allTruePred, input + 4);

        input += 8;
      } else if constexpr (BIT_RATE == 4) {
        in_v_0 = svreinterpret_u32_u64(svld1ub_u64(allTruePred, input));
        in_v_1 = svreinterpret_u32_u64(svld1ub_u64(allTruePred, input + 2));

        input += 4;

        in_v_0 =
            svreinterpret_u32_u64(svreinterpret_u64_u32(in_v_0) * multiplier);
        in_v_1 =
            svreinterpret_u32_u64(svreinterpret_u64_u32(in_v_1) * multiplier);

        in_v_0 &= 15;
        in_v_1 &= 15;
      } else if constexpr (BIT_RATE == 2) {
        in_v_0 = svreinterpret_u32_u8(svdup_n_u8(input[0]));
        in_v_1 = svreinterpret_u32_u8(svdup_n_u8(input[1]));

        input += 2;

        in_v_0 = in_v_0 >> shift;
        in_v_1 = in_v_1 >> shift;

        in_v_0 &= 3;
        in_v_1 &= 3;
      }

      svfloat32_t in_v_0_f = svcvt_f32_u32_x(allTruePred, in_v_0);
      svfloat32_t in_v_1_f = svcvt_f32_u32_x(allTruePred, in_v_1);

      in_v_0_f = svmad_f32_m(allTruePred, in_v_0_f, scale_v, bias_v);
      in_v_1_f = svmad_f32_m(allTruePred, in_v_1_f, scale_v, bias_v);

      if constexpr (std::is_same_v<OutputType, float>) {
        vst1q_f32(output, svget_neonq(in_v_0_f));
        vst1q_f32(output + 4, svget_neonq(in_v_1_f));
      } else {
        float16x4_t dequantzed_v_half_low = vcvt_f16_f32(svget_neonq(in_v_0_f));
        float16x4_t dequantzed_v_half_high =
            vcvt_f16_f32(svget_neonq(in_v_1_f));
        vst1_f16(reinterpret_cast<float16_t*>(output), dequantzed_v_half_low);
        vst1_f16(
            reinterpret_cast<float16_t*>(output + 4), dequantzed_v_half_high);
      }

      output += 8;
    }

    if (output_columns_mod != 0) {
      svuint32_t in_v_0;
      svuint32_t in_v_1;
      if constexpr (BIT_RATE == 8) {
        in_v_0 = svld1ub_u32(lastPredA, input);
        in_v_1 = svld1ub_u32(lastPredB, input + 4);
      } else if constexpr (BIT_RATE == 4) {
        in_v_0 = svreinterpret_u32_u64(svld1ub_u64(lastPredD, input));
        in_v_1 = svreinterpret_u32_u64(svld1ub_u64(lastPredE, input + 2));

        in_v_0 =
            svreinterpret_u32_u64(svreinterpret_u64_u32(in_v_0) * multiplier);
        in_v_1 =
            svreinterpret_u32_u64(svreinterpret_u64_u32(in_v_1) * multiplier);

        in_v_0 &= 15;
        in_v_1 &= 15;
      } else if constexpr (BIT_RATE == 2) {
        in_v_0 = svreinterpret_u32_u8(svdup_n_u8(input[0]));

        in_v_0 = in_v_0 >> shift;

        in_v_0 &= 3;
      }

      input += input_columns_mod;

      svfloat32_t in_v_0_f = svcvt_f32_u32_x(lastPredA, in_v_0);
      svfloat32_t in_v_1_f = svcvt_f32_u32_x(lastPredB, in_v_1);

      in_v_0_f = svmad_f32_m(lastPredA, in_v_0_f, scale_v, bias_v);
      in_v_1_f = svmad_f32_m(lastPredB, in_v_1_f, scale_v, bias_v);

      if constexpr (std::is_same_v<OutputType, float>) {
        svst1_f32(lastPredA, output, in_v_0_f);
        svst1_f32(lastPredB, output + 4, in_v_1_f);
      } else {
        float16x4_t dequantzed_v_half_low_low =
            vcvt_f16_f32(svget_neonq(in_v_0_f));
        float16x8_t dequantzed_v_half_low =
            vcvt_high_f16_f32(dequantzed_v_half_low_low, svget_neonq(in_v_1_f));
        svst1_f16(
            lastPredC,
            reinterpret_cast<float16_t*>(output),
            svset_neonq_f16(svundef_f16(), dequantzed_v_half_low));
      }

      output += output_columns_mod;
    }

    input += 2 * sizeof(uint16_t);
  } // for each row
}

#define INSTANTIATE_QuantizationNeonFunctions8Bits(type)                 \
  template void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfNeon<type>( \
      const std::uint8_t* input,                                         \
      size_t input_rows,                                                 \
      int input_columns,                                                 \
      type* output);                                                     \
  template void FloatOrHalfToFused8BitRowwiseQuantizedSBFloatNeon<type>( \
      const type* input,                                                 \
      size_t input_rows,                                                 \
      int input_columns,                                                 \
      uint8_t* output);

// clang-format off
INSTANTIATE_QuantizationNeonFunctions8Bits(float)
INSTANTIATE_QuantizationNeonFunctions8Bits(float16)
// clang-format on
#undef INSTANTIATE_QuantizationNeonFunctions8Bits

#define INSTANTIATE_QuantizationNeonFunctionsNBits(type, bit_rate)  \
  template void                                                     \
  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfNeon<type, bit_rate>( \
      const type* input,                                            \
      size_t input_rows,                                            \
      int input_columns,                                            \
      std::uint8_t* output);                                        \
  template void                                                     \
  FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfNeon<type, bit_rate>( \
      const std::uint8_t* input,                                    \
      size_t input_rows,                                            \
      int input_columns,                                            \
      type* output);

    // clang-format off
INSTANTIATE_QuantizationNeonFunctionsNBits(float, 2)
INSTANTIATE_QuantizationNeonFunctionsNBits(float, 4)
INSTANTIATE_QuantizationNeonFunctionsNBits(float, 8)
INSTANTIATE_QuantizationNeonFunctionsNBits(float16, 2)
INSTANTIATE_QuantizationNeonFunctionsNBits(float16, 4)
INSTANTIATE_QuantizationNeonFunctionsNBits(float16, 8)
// clang-format on
#undef INSTANTIATE_QuantizationNeonFunctionsNBits

#endif // HAVE_SVE

} // namespace fbgemm

#endif // __aarch64__
