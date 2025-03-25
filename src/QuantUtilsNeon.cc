/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm/Utils.h"

#if HAVE_SVE

#define FBGEMM_EXPORTS
#include <arm_neon.h>
#include <arm_sve.h>

#include <arm_neon_sve_bridge.h>
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
    float16x8_t* output_row_v_half = reinterpret_cast<float16x8_t*>(output_row);

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
        float16x4_t dequantzed_v_half_low_low =
            vcvt_f16_f32(svget_neonq(in_v_0_f));
        float16x8_t dequantzed_v_half_low =
            vcvt_high_f16_f32(dequantzed_v_half_low_low, svget_neonq(in_v_1_f));
        output_row_v_half[colIndex] = dequantzed_v_half_low;
      }
    }

    svuint32_t in_v_0 = svld1ub_u32(
        lastPredA, reinterpret_cast<const uint8_t*>(input_row_v_0 + colIndex));
    svuint32_t in_v_1 = svld1ub_u32(
        lastPredB, reinterpret_cast<const uint8_t*>(input_row_v_1 + colIndex));
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
      type* output);

// clang-format off
INSTANTIATE_QuantizationNeonFunctions8Bits(float)
INSTANTIATE_QuantizationNeonFunctions8Bits(float16)
// clang-format on
#undef INSTANTIATE_QuantizationNeonFunctions8Bits

} // namespace fbgemm

#endif // __aarch64__
