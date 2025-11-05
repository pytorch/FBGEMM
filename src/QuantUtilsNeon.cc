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
#elif defined(__GNUC__)
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

static inline void
FindMinMaxImpl_f16(const float16_t* m, float* min, float* max, uint64_t count) {
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
#elif defined(__GNUC__)
#pragma GCC novector unroll 0
#endif
  while (loopRemainder > 0) {
    float16_t tmp = *m++;
    loopRemainder -= 1;
    tmp_min_s = vminh_f16(tmp_min_s, tmp);
    tmp_max_s = vmaxh_f16(tmp_max_s, tmp);
  }

  *min = static_cast<float>(tmp_min_s);
  *max = static_cast<float>(tmp_max_s);
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
    if constexpr (std::is_same<InputType, float>()) {
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

      if constexpr (std::is_same<InputType, float>()) {
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
#elif defined(__GNUC__)
#pragma GCC novector unroll 0
#endif
    while (loopRemainder > 0) {
      float32x4_t v0;
      if constexpr (std::is_same<InputType, float>()) {
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

      if constexpr (std::is_same<OutputType, float>()) {
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

      if constexpr (std::is_same<OutputType, float>()) {
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

#endif // HAVE_SVE

} // namespace fbgemm

#endif // __aarch64__
