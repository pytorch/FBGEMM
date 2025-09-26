/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>
#define FBGEMM_EXPORTS
#include "fbgemm/QuantUtilsAvx512.h"
#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>
#endif
#include <cpuinfo.h>
#include <fbgemm/FloatConversion.h>
#include <cassert>

namespace fbgemm {

using namespace std;
template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    int C_PER_G,
    typename BIAS_TYPE>
void requantizeOutputProcessingGConvAvx512(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t<BIAS_TYPE>& r) {
  // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
  // using AVX2 instructions
  int quant_param_idx = 0;
  if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
    int ncol_per_group = r.ncols / r.groups;
    int g = block.col_start / ncol_per_group;
    quant_param_idx = g;
  }
  __m512 multiplier_v = _mm512_set1_ps(r.C_multiplier[quant_param_idx]);
  // Broadcasted reciprocal of act_times_w_scale
  __m512 act_times_w_rcp_v;

  if constexpr (!(Q_GRAN == QuantizationGranularity::OUT_CHANNEL)) {
    if constexpr (is_same_v<BIAS_TYPE, float>) {
      act_times_w_rcp_v =
          _mm512_set1_ps(1.0 / r.act_times_w_scale[quant_param_idx]);
    }
  }
  __m512i min_v = _mm512_set1_epi8(static_cast<uint8_t>(0));
  __m512i max_v = _mm512_set1_epi8(static_cast<uint8_t>(255));

  assert(
      (A_SYMMETRIC == (r.A_zero_point == 0)) &&
      "A_SYMMETRIC == true if and only if A_zero_point == 0");
  assert(
      (B_SYMMETRIC ==
       ((Q_GRAN == QuantizationGranularity::TENSOR && r.B_zero_point[0] == 0) ||
        r.row_offsets == nullptr)) &&
      "B_SYMMETRIC == true if and only if B_zero_point == 0 "
      "or r.row_offsets == nullptr");
  assert(
      (HAS_BIAS == (r.bias != nullptr)) &&
      "HAS_BIAS == true if and only if bias != nullptr");

  __m512i A_zero_point_v = _mm512_set1_epi32(r.A_zero_point);
  __m512i C_zero_point_epi16_v = _mm512_set1_epi16(r.C_zero_point);
  __m512i C_zero_point_epi8_v = _mm512_set1_epi8(r.C_zero_point);
  __m512i permute_mask_v_g8 = _mm512_set_epi32(
      0x0f,
      0x07,
      0x0e,
      0x06,
      0x0d,
      0x05,
      0x0c,
      0x04,
      0x0b,
      0x03,
      0x0a,
      0x02,
      0x09,
      0x01,
      0x08,
      0x00);

  __m512i permute_mask_v_g4 = _mm512_set_epi32(
      0x0f,
      0x0b,
      0x07,
      0x03,
      0x0e,
      0x0a,
      0x06,
      0x02,
      0x0d,
      0x09,
      0x05,
      0x01,
      0x0c,
      0x08,
      0x04,
      0x00);
  // vector lane width  16 * 32 = 512 bits
  constexpr int VLEN = 16;
  const __mmask16 mask = 0x00ff;

  for (int64_t i = block.row_start; i < block.row_start + block.row_size; ++i) {
    int64_t j = block.col_start;
    // changed the iteration termination criteria for C_per_g = 8
    // for avx512 currently all 4 cases supported will only run one iteration of
    // inner loop
    // for C_per_g == 8, we only have 8 outputs while the other cases have 16.
    // thus, we do masked load for all col quantization scheme under C_per_g ==
    // 8
    for (; j < block.col_start + ((block.col_size + VLEN - 1) / VLEN * VLEN);
         j += VLEN) {
      __m512i x_v;
      if constexpr (C_PER_G != 8) {
        x_v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(
            inp + (i - block.row_start) * ld_in + (j - block.col_start)));
      } else {
        // as of now we only have C_per_G = 2,4,8,16 thus this j loop all only
        // execute one iteration, the following point will be wrong if run more
        // than one iter
        x_v = _mm512_maskz_loadu_epi32(
            mask, inp + (i - block.row_start) * ld_in + (j - block.col_start));
      }

      if constexpr (!A_SYMMETRIC) {
        __m512i col_off_raw_v;
        if constexpr (C_PER_G != 8) {
          col_off_raw_v = _mm512_loadu_si512(
              reinterpret_cast<const __m512i*>(r.col_offsets + j));
        } else {
          col_off_raw_v = _mm512_maskz_loadu_epi32(mask, r.col_offsets + j);
        }

        __m512i col_off_v = _mm512_mullo_epi32(A_zero_point_v, col_off_raw_v);
        x_v = _mm512_sub_epi32(x_v, col_off_v);
      }

      if constexpr (!B_SYMMETRIC) {
        __m512i row_offset_v;

        if constexpr (C_PER_G == 2) {
          // When C_PER_G == 2, we need to handle 8 groups at a time to fully
          // utilize 64B AVX12 vector register (C_PER_G * 8 * sizeof(int32_t) ==
          // 64B)
          // Load row_offsets for 8 groups and broadcast by 2 times.
          row_offset_v =
              _mm512_castps_si512(_mm512_moveldup_ps(_mm512_permutexvar_ps(
                  permute_mask_v_g8,
                  _mm512_castps256_ps512(
                      _mm256_loadu_ps(reinterpret_cast<const float*>(
                          r.row_offsets + (i - block.row_start) * 8))))));

        }
        // When C_PER_G == 4, we need to handle 4 groups at a time to fully
        // utilize 32B AVX2 vector register (C_PER_G * 4 * sizeof(int32_t) ==
        // 32B)
        // When C_PER_G == 8, we just need 1 group at a time on the other hand.

        // Groups 0,1,2,3 when C_PER_G == 4
        // Group 0 when C_PER_G == 8
        else if constexpr (C_PER_G == 4) {
          // Load row_offsets for 4 groups and broadcast by 4 times each because
          // we have 4 channels per group.
          // groups 0,1,2,3
          row_offset_v = _mm512_permutexvar_epi32(
              permute_mask_v_g4,
              _mm512_broadcast_i32x4(
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                      r.row_offsets + (i - block.row_start) * 4))));
        } else if constexpr (C_PER_G == 8) {
          row_offset_v =
              _mm512_set1_epi32(r.row_offsets[(i - block.row_start)]);
        } else {
          static_assert(C_PER_G == 16);
          row_offset_v =
              _mm512_set1_epi32(r.row_offsets[(i - block.row_start)]);
        }

        __m512i B_zero_point_v = _mm512_set1_epi32(r.B_zero_point[0]);
        if constexpr (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          if constexpr (C_PER_G != 8) {
            B_zero_point_v = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(r.B_zero_point + j));
          } else {
            B_zero_point_v = _mm512_maskz_loadu_epi32(mask, r.B_zero_point + j);
          }
        } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
          if constexpr (C_PER_G == 2) {
            B_zero_point_v =
                _mm512_castps_si512(_mm512_moveldup_ps(_mm512_permutexvar_ps(
                    permute_mask_v_g8,
                    _mm512_castps256_ps512(
                        _mm256_loadu_ps(reinterpret_cast<const float*>(
                            r.B_zero_point + quant_param_idx))))));
          } else if constexpr (C_PER_G == 4) {
            B_zero_point_v = _mm512_permutexvar_epi32(
                permute_mask_v_g4,
                _mm512_broadcast_i32x4(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        r.B_zero_point + quant_param_idx))));
          } else {
            B_zero_point_v = _mm512_set1_epi32(r.B_zero_point[quant_param_idx]);
          }
        }
        row_offset_v = _mm512_mullo_epi32(row_offset_v, B_zero_point_v);
        x_v = _mm512_sub_epi32(x_v, row_offset_v);
      }
      __m512 xf_v;
      if constexpr (HAS_BIAS) {
        if constexpr (is_same_v<BIAS_TYPE, float>) {
          __m512 x_bias_v;
          if constexpr (C_PER_G != 8) {
            x_bias_v =
                _mm512_loadu_ps(reinterpret_cast<const float*>(r.bias + j));
          } else {
            x_bias_v = _mm512_maskz_loadu_ps(
                mask, reinterpret_cast<const float*>(r.bias + j));
          }

          if constexpr (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
            __m512 act_times_w_scale_v;
            if constexpr (C_PER_G != 8) {
              act_times_w_scale_v = _mm512_loadu_ps(r.act_times_w_scale + j);
            } else {
              act_times_w_scale_v =
                  _mm512_maskz_loadu_ps(mask, r.act_times_w_scale + j);
            }
            x_bias_v = _mm512_div_ps(x_bias_v, act_times_w_scale_v);
          } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
            __m512 diviser_v;
            if constexpr (C_PER_G == 2) {
              diviser_v = _mm512_moveldup_ps(_mm512_permutexvar_ps(
                  permute_mask_v_g8,
                  _mm512_castps256_ps512(
                      _mm256_loadu_ps(r.act_times_w_scale + quant_param_idx))));
            } else if constexpr (C_PER_G == 4) {
              diviser_v = _mm512_permutexvar_ps(
                  permute_mask_v_g4,
                  _mm512_broadcast_f32x4(

                      _mm_loadu_ps(r.act_times_w_scale + quant_param_idx)));
            } else if constexpr (C_PER_G == 8) {
              diviser_v = _mm512_set1_ps(r.act_times_w_scale[quant_param_idx]);
            } else {
              static_assert(C_PER_G == 16);
              diviser_v = _mm512_set1_ps(r.act_times_w_scale[quant_param_idx]);
            }
            x_bias_v = _mm512_div_ps(x_bias_v, diviser_v);
          } else {
            x_bias_v = _mm512_mul_ps(x_bias_v, act_times_w_rcp_v);
          }
          xf_v = _mm512_add_ps(_mm512_cvtepi32_ps(x_v), x_bias_v);
        } else {
          x_v = _mm512_add_epi32(
              x_v,
              _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r.bias + j)));
          xf_v = _mm512_cvtepi32_ps(x_v);
        }
      } else {
        xf_v = _mm512_cvtepi32_ps(x_v);
      }

      /*
       * Convert int32_t input to FP32 and multiply by FP32 scale.
       * Both operations involve statistically unbiased roundings (with
       * default MXCSR rounding mode):
       * - Large int32_t values can't be exactly represented as FP32.
       * CVTDQ2PS instruction on x86 would round it according to nearest
       * FP32 value with ties to even (assuming default MXCSR rounding
       * mode).
       * - Product of two FP32 values is generally not exactly
       * representation as an FP32 value, and will be rounded to nearest
       * FP32 value with ties to even with default MXCSR rounding mode.
       */
      __m512 x_scaled_v;
      if constexpr (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        __m512 C_multiplier_v;
        if constexpr (C_PER_G != 8) {
          C_multiplier_v = _mm512_loadu_ps(r.C_multiplier + j);
        } else {
          C_multiplier_v = _mm512_maskz_loadu_ps(mask, r.C_multiplier + j);
        }
        x_scaled_v = _mm512_mul_ps(xf_v, C_multiplier_v);
      } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
        if constexpr (C_PER_G == 2) {
          multiplier_v = _mm512_moveldup_ps(_mm512_permutexvar_ps(
              permute_mask_v_g8,
              _mm512_castps256_ps512(
                  _mm256_loadu_ps(r.C_multiplier + quant_param_idx))));
        } else if constexpr (C_PER_G == 4) {
          multiplier_v = _mm512_permutexvar_ps(
              permute_mask_v_g4,
              _mm512_broadcast_f32x4(
                  _mm_loadu_ps(r.C_multiplier + quant_param_idx)));
        } else {
          multiplier_v = _mm512_set1_ps(r.C_multiplier[quant_param_idx]);
        }
        x_scaled_v = _mm512_mul_ps(xf_v, multiplier_v);
      } else {
        x_scaled_v = _mm512_mul_ps(xf_v, multiplier_v);
      }

      /*
       * Convert scaled FP32 result to int32_t using CVTPS2DQ instruction.
       * CVTPS2DQ instruction rounds result according to nearest FP32 value
       * with ties to even (assuming default MXCSR rounding mode). However,
       * when conversion overflows, it produces INT32_MIN as a result. For
       * large positive inputs the result of conversion can become negative,
       * which affects the final requantization result. Note that on x86
       * SSE2 we have e.g. int32_t(float(INT32_MAX)) == INT32_MIN! This
       * happens because float(INT32_MAX) rounds to 2**31, which overflows
       * int32_t when it is converted back to integer.
       *
       * Thankfully, we can prove that overflow never happens in this
       * requantization scheme. The largest positive input is INT32_MAX
       * (2**31 - 1), which turns into 2**31 when converted to float. The
       * largest scale value is 0x1.FFFFFEp-1. When multiplied together, the
       * result is 2147483520 (compare to INT32_MAX = 2147483647), which
       * fits into int32_t without overflow.
       */
      __m512i x_rounded_v = _mm512_cvtps_epi32(x_scaled_v);

      /*
       * Standard final sequence on x86 AVX512:
       * - Pack to int16_t and saturate
       * - Add zero point
       * - Pack to uint8_t and saturate
       * - Clamp between qmin and qmax
       */
      __m512i x_packed_v = _mm512_adds_epi16(
          _mm512_packs_epi32(x_rounded_v, _mm512_setzero_si512()),
          C_zero_point_epi16_v);
      x_packed_v = _mm512_packus_epi16(x_packed_v, _mm512_setzero_si512());
      __m512i x_clamped_v = _mm512_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm512_min_epu8(x_packed_v, max_v));

      /*
       * x_clamped_v has results in the following layout so we need to
       * permute: x0-3 garbage0-11 x4-7 garbage12-23 x8-11 garbage24-35 x12-15
       * garbage36-47
       */
      x_clamped_v = _mm512_permutexvar_epi32(permute_mask_v_g4, x_clamped_v);

      /*
       * 1x CVTDQ2PS
       * 1x MULPS
       * 1x CVTPS2DQ
       * 1x PACKSSDW
       * 1x PACKUSWB
       * 1x PADDW
       * 1x PMAXUB
       * 1x PMINUB
       * 1x PERMD
       * ---------------------
       * 9 instructions total
       */
      if constexpr (C_PER_G != 8) {
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(out + i * ld_out + j),
            _mm512_castsi512_si128(x_clamped_v));
      } else {
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(out + i * ld_out + j),
            _mm512_castsi512_si128(x_clamped_v));
      }
    } // j loop vectorized

#ifndef NDEBUG
    const int64_t remainder = block.col_start + block.col_size - j;
    assert(remainder == 0 || C_PER_G == 8);
#endif
  } // i loop
}

void Fused8BitRowwiseQuantizedSBFloatToBfloat16Avx512(
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    bfloat16* output) {
#if (CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64) && defined(FBGEMM_FBCODE)
  constexpr int VLEN = 8;
  int output_columns = input_columns - 2 * sizeof(float);

  for (size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float* input_row_scale_bias =
        reinterpret_cast<const float*>(input_row + output_columns);
    bfloat16* output_row = output + row * output_columns;

    __m256 scale_v = _mm256_set1_ps(input_row_scale_bias[0]);
    __m256 bias_v = _mm256_set1_ps(input_row_scale_bias[1]);

    int col = 0;
    for (col = 0; col < output_columns / VLEN * VLEN; col += VLEN) {
      __m256 in_v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_row + col))));
#ifdef __FMA__
      __m256 dequantzed_v = _mm256_fmadd_ps(in_v, scale_v, bias_v);
#else
      __m256 dequantzed_v = _mm256_add_ps(_mm256_mul_ps(in_v, scale_v), bias_v);
#endif
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(output_row + col),
          (__m128i)(_mm256_cvtneps_pbh(dequantzed_v)));
    }

    for (; col < output_columns; ++col) {
      float output_value =
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
      output_row[col] = cpu_float2bfloat16(output_value);
    }
  } // for each row
#else
  throw std::runtime_error(
      "Fused8BitRowwiseQuantizedSBFloatToBfloat16Avx512 not implemented for non x86");
#endif
}

#define INSTANTIATE_REQUANTIZE_BIAS_TYPE(              \
    A_SYM, B_SYM, Q_GRAN, BIAS, RELU, BIAS_TYPE)       \
  template void requantizeOutputProcessingGConvAvx512< \
      A_SYM,                                           \
      B_SYM,                                           \
      Q_GRAN,                                          \
      BIAS,                                            \
      RELU,                                            \
      2,                                               \
      BIAS_TYPE>(                                      \
      uint8_t * out,                                   \
      const int32_t* inp,                              \
      const block_type_t& block,                       \
      int ld_out,                                      \
      int ld_in,                                       \
      const requantizationParams_t<BIAS_TYPE>& r);     \
  template void requantizeOutputProcessingGConvAvx512< \
      A_SYM,                                           \
      B_SYM,                                           \
      Q_GRAN,                                          \
      BIAS,                                            \
      RELU,                                            \
      4,                                               \
      BIAS_TYPE>(                                      \
      uint8_t * out,                                   \
      const int32_t* inp,                              \
      const block_type_t& block,                       \
      int ld_out,                                      \
      int ld_in,                                       \
      const requantizationParams_t<BIAS_TYPE>& r);     \
  template void requantizeOutputProcessingGConvAvx512< \
      A_SYM,                                           \
      B_SYM,                                           \
      Q_GRAN,                                          \
      BIAS,                                            \
      RELU,                                            \
      8,                                               \
      BIAS_TYPE>(                                      \
      uint8_t * out,                                   \
      const int32_t* inp,                              \
      const block_type_t& block,                       \
      int ld_out,                                      \
      int ld_in,                                       \
      const requantizationParams_t<BIAS_TYPE>& r);     \
  template void requantizeOutputProcessingGConvAvx512< \
      A_SYM,                                           \
      B_SYM,                                           \
      Q_GRAN,                                          \
      BIAS,                                            \
      RELU,                                            \
      16,                                              \
      BIAS_TYPE>(                                      \
      uint8_t * out,                                   \
      const int32_t* inp,                              \
      const block_type_t& block,                       \
      int ld_out,                                      \
      int ld_in,                                       \
      const requantizationParams_t<BIAS_TYPE>& r);

#define INSTANTIATE_REQUANTIZE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU)            \
  INSTANTIATE_REQUANTIZE_BIAS_TYPE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU, float) \
  INSTANTIATE_REQUANTIZE_BIAS_TYPE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU, int32_t)

#define INSTANTIATE_A_SYM(B_SYM, Q_GRAN, BIAS, RELU)      \
  INSTANTIATE_REQUANTIZE(true, B_SYM, Q_GRAN, BIAS, RELU) \
  INSTANTIATE_REQUANTIZE(false, B_SYM, Q_GRAN, BIAS, RELU)

#define INSTANTIATE_B_SYM(Q_GRAN, BIAS, RELU) \
  INSTANTIATE_A_SYM(true, Q_GRAN, BIAS, RELU) \
  INSTANTIATE_A_SYM(false, Q_GRAN, BIAS, RELU)

#define INSTANTIATE_Q_GRANS(BIAS, RELU)                          \
  INSTANTIATE_B_SYM(QuantizationGranularity::TENSOR, BIAS, RELU) \
  INSTANTIATE_B_SYM(QuantizationGranularity::GROUP, BIAS, RELU)  \
  INSTANTIATE_B_SYM(QuantizationGranularity::OUT_CHANNEL, BIAS, RELU)

#define INSTANTIATE_BIAS(RELU)    \
  INSTANTIATE_Q_GRANS(true, RELU) \
  INSTANTIATE_Q_GRANS(false, RELU)

INSTANTIATE_BIAS(true)
INSTANTIATE_BIAS(false)

#undef INSTANTIATE_A_SYM
#undef INSTANTIATE_B_SYM
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BIAS

} // namespace fbgemm
