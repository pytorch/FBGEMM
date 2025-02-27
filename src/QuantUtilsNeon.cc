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
  int output_columns = input_columns - 2 * sizeof(float);

  for (size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float* input_row_scale_bias =
        reinterpret_cast<const float*>(input_row + output_columns);
    OutputType* output_row = output + row * output_columns;

    svbool_t pred = svptrue_b32();

    float scale = input_row_scale_bias[0];
    float bias = input_row_scale_bias[1];
    svfloat32_t scale_v = svdup_n_f32(scale);
    svfloat32_t bias_v = svdup_n_f32(bias);

    const uint64_t* input_row_v_0 =
        reinterpret_cast<const uint64_t*>(input_row);
    const uint64_t* input_row_v_1 =
        reinterpret_cast<const uint64_t*>(input_row + 4);
    float32x4x2_t* output_row_v = reinterpret_cast<float32x4x2_t*>(output_row);
    float16x8_t* output_row_v_half = reinterpret_cast<float16x8_t*>(output_row);

    int colIndex = 0;
    for (int colMax = output_columns / 8; colIndex < colMax; ++colIndex) {
      svuint32_t in_v_0 = svld1ub_u32(
          pred, reinterpret_cast<const uint8_t*>(input_row_v_0 + colIndex));
      svuint32_t in_v_1 = svld1ub_u32(
          pred, reinterpret_cast<const uint8_t*>(input_row_v_1 + colIndex));
      svfloat32_t in_v_0_f = svcvt_f32_u32_x(pred, in_v_0);
      svfloat32_t in_v_1_f = svcvt_f32_u32_x(pred, in_v_1);

      in_v_0_f = svmad_f32_m(pred, in_v_0_f, scale_v, bias_v);
      in_v_1_f = svmad_f32_m(pred, in_v_1_f, scale_v, bias_v);

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

#pragma clang loop vectorize(disable)
#pragma clang loop unroll(disable)
    for (colIndex *= 8; colIndex < output_columns; ++colIndex) {
      float output_value = input_row[colIndex] * input_row_scale_bias[0] +
          input_row_scale_bias[1];
      if (std::is_same<OutputType, float>()) {
        output_row[colIndex] = output_value;
      } else {
        output_row[colIndex] = cpu_float2half_rn(output_value);
      }
    }
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
