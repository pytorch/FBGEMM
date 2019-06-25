/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "GroupwiseConvAcc32Intrinsic.h"

namespace fbgemm {

using namespace std;

template <bool TOP, bool BOTTOM, int SPATIAL_DIM>
void groupConvAvx512(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf) {
  assert(SPATIAL_DIM == 2 && "3D conv not supported yet");
  gconv_kernel_<TOP, BOTTOM>(conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
}

template void groupConvAvx512<true, false, 2>(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx512<false, false, 2>(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx512<false, true, 2>(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx512<true, false, 3>(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx512<false, false, 3>(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx512<false, true, 3>(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    int C_PER_G>
void requantizeOutputProcessingGConvAvx512(
    uint8_t* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t& r) {
  // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
  // using AVX512 instructions
  int quant_param_idx = 0;
  if (Q_GRAN == QuantizationGranularity::GROUP) {
    int ncol_per_group = r.ncols / r.groups;
    int g = block.col_start / ncol_per_group;
    quant_param_idx = g;
  }
  constexpr int SIMD_WIDTH = 512;
  constexpr int SIMD_WIDTH_I32 = SIMD_WIDTH / 32;

  __m512 multiplier_v = _mm512_set1_ps(r.C_multiplier[quant_param_idx]);

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

  __m512i permute_mask_v =
      _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);

  constexpr int VLEN = SIMD_WIDTH_I32;
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    int j = block.col_start;
    for (; j < block.col_start + (block.col_size / (VLEN * 4) * (VLEN * 4));
         j += (VLEN * 4)) {
      __m512i x_v = _mm512_loadu_si512(
          inp + (i - block.row_start) * ld_in + (j - block.col_start));
      __m512i y_v = _mm512_loadu_si512(
          inp + (i - block.row_start) * ld_in + (j - block.col_start) +
          1 * VLEN);
      __m512i z_v = _mm512_loadu_si512(
          inp + (i - block.row_start) * ld_in + (j - block.col_start) +
          2 * VLEN);
      __m512i w_v = _mm512_loadu_si512(
          inp + (i - block.row_start) * ld_in + (j - block.col_start) +
          3 * VLEN);

      if (!A_SYMMETRIC) {
        __m512i col_off_v = _mm512_mullo_epi32(
            A_zero_point_v, _mm512_loadu_si512(r.col_offsets + j));
        x_v = _mm512_sub_epi32(x_v, col_off_v);
        col_off_v = _mm512_mullo_epi32(
            A_zero_point_v, _mm512_loadu_si512(r.col_offsets + j + VLEN));
        y_v = _mm512_sub_epi32(y_v, col_off_v);
        col_off_v = _mm512_mullo_epi32(
            A_zero_point_v, _mm512_loadu_si512(r.col_offsets + j + 2 * VLEN));
        z_v = _mm512_sub_epi32(z_v, col_off_v);
        col_off_v = _mm512_mullo_epi32(
            A_zero_point_v, _mm512_loadu_si512(r.col_offsets + j + 3 * VLEN));
        w_v = _mm512_sub_epi32(w_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        __m512i row_offset_v;

        // When C_PER_G == 4, we need to handle 4 groups at a time to fully
        // utilize 64B AVX512 vector register (C_PER_G * 4 * sizeof(int32_t) ==
        // 64B)
        // When C_PER_G == 8, we need 2 groups at a time.
        // When C_PER_G == 16, we just need 1 group.

        // Groups 0-3 when C_PER_G == 4
        // Groups 0-1 when C_PER_G == 8
        if (C_PER_G == 4) {
          // Load row_offsets for 4 groups and broadcast by 4 times each because
          // we have 4 channels per group.
          // groups 0-3
          row_offset_v = _mm512_castps_si512(_mm512_permutevar_ps(
              _mm512_broadcast_f32x4(
                  _mm_loadu_ps(reinterpret_cast<const float*>(
                      r.row_offsets + (i - block.row_start) * VLEN))),
              _mm512_set_epi32(
                  3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0)));
        } else if (C_PER_G == 8) {
          // Load row_offsets for 2 groups and broadcast by 8 times
          // groups 0-1
          row_offset_v = _mm512_inserti32x8(
              _mm512_castsi256_si512(_mm256_set1_epi32(
                  r.row_offsets
                      [(i - block.row_start) * VLEN +
                       (j - block.col_start) / (VLEN * 4) * 8 + 0])),
              _mm256_set1_epi32(
                  r.row_offsets
                      [(i - block.row_start) * VLEN +
                       (j - block.col_start) / (VLEN * 4) * 8 + 1]),
              1);
        } else {
          assert(C_PER_G == 16);
          row_offset_v =
              _mm512_set1_epi32(r.row_offsets
                                    [(i - block.row_start) * VLEN +
                                     (j - block.col_start) / (VLEN * 4) * 4]);
        }
        __m512i B_zero_point_v = _mm512_set1_epi32(r.B_zero_point[0]);
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          B_zero_point_v = _mm512_loadu_si512(r.B_zero_point + j);
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          if (C_PER_G == 4) {
            B_zero_point_v = _mm512_castps_si512(_mm512_permutevar_ps(
                _mm512_broadcast_f32x4(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        r.B_zero_point + quant_param_idx))),
                _mm512_set_epi32(
                    3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0)));
          } else if (C_PER_G == 8) {
            B_zero_point_v = _mm512_inserti32x8(
                _mm512_castsi256_si512(_mm256_set1_epi32(
                    r.B_zero_point
                        [quant_param_idx +
                         (j - block.col_start) / (VLEN * 4) * 8 + 0])),
                _mm256_set1_epi32(
                    r.B_zero_point
                        [quant_param_idx +
                         (j - block.col_start) / (VLEN * 4) * 8 + 1]),
                1);
          } else {
            B_zero_point_v = _mm512_set1_epi32(
                r.B_zero_point
                    [quant_param_idx + (j - block.col_start) / (VLEN * 4) * 4]);
          }
        }
        row_offset_v = _mm512_mullo_epi32(row_offset_v, B_zero_point_v);
        x_v = _mm512_sub_epi32(x_v, row_offset_v);

        // Groups 4-7 when C_PER_G == 4
        // Groups 2-3 when C_PER_G == 8
        if (C_PER_G == 4) {
          // groups 4-7
          row_offset_v = _mm512_castps_si512(_mm512_permutevar_ps(
              _mm512_broadcast_f32x4(
                  _mm_loadu_ps(reinterpret_cast<const float*>(
                      r.row_offsets + (i - block.row_start) * VLEN + 4))),
              _mm512_set_epi32(
                  3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0)));
        } else if (C_PER_G == 8) {
          row_offset_v = _mm512_inserti32x8(
              _mm512_castsi256_si512(_mm256_set1_epi32(
                  r.row_offsets
                      [(i - block.row_start) * VLEN +
                       (j - block.col_start) / (VLEN * 4) * 8 + 2])),
              _mm256_set1_epi32(
                  r.row_offsets
                      [(i - block.row_start) * VLEN +
                       (j - block.col_start) / (VLEN * 4) * 8 + 3]),
              1);
        } else {
          row_offset_v = _mm512_set1_epi32(
              r.row_offsets
                  [(i - block.row_start) * VLEN +
                   (j - block.col_start) / (VLEN * 4) * 4 + 1]);
        }
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          B_zero_point_v = _mm512_loadu_si512(r.B_zero_point + j + VLEN);
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          if (C_PER_G == 4) {
            B_zero_point_v = _mm512_castps_si512(_mm512_permutevar_ps(
                _mm512_broadcast_f32x4(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        r.B_zero_point + quant_param_idx + 4))),
                _mm512_set_epi32(
                    3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0)));
          } else if (C_PER_G == 8) {
            B_zero_point_v = _mm512_inserti32x8(
                _mm512_castsi256_si512(_mm256_set1_epi32(
                    r.B_zero_point
                        [quant_param_idx +
                         (j - block.col_start) / (VLEN * 4) * 8 + 2])),
                _mm256_set1_epi32(
                    r.B_zero_point
                        [quant_param_idx +
                         (j - block.col_start) / (VLEN * 4) * 8 + 3]),
                1);
          } else {
            B_zero_point_v = _mm512_set1_epi32(
                r.B_zero_point
                    [quant_param_idx + (j - block.col_start) / (VLEN * 4) * 4 +
                     1]);
          }
        }
        row_offset_v = _mm512_mullo_epi32(row_offset_v, B_zero_point_v);
        y_v = _mm512_sub_epi32(y_v, row_offset_v);

        // Groups 8-11 when C_PER_G == 4
        // Groups 4-5 when C_PER_G == 8
        if (C_PER_G == 4) {
          row_offset_v = _mm512_castps_si512(_mm512_permutevar_ps(
              _mm512_broadcast_f32x4(
                  _mm_loadu_ps(reinterpret_cast<const float*>(
                      r.row_offsets + (i - block.row_start) * VLEN + 8))),
              _mm512_set_epi32(
                  3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0)));
        } else if (C_PER_G == 8) {
          row_offset_v = _mm512_inserti32x8(
              _mm512_castsi256_si512(_mm256_set1_epi32(
                  r.row_offsets
                      [(i - block.row_start) * VLEN +
                       (j - block.col_start) / (VLEN * 4) * 8 + 4])),
              _mm256_set1_epi32(
                  r.row_offsets
                      [(i - block.row_start) * VLEN +
                       (j - block.col_start) / (VLEN * 4) * 8 + 5]),
              1);
        } else {
          row_offset_v = _mm512_set1_epi32(
              r.row_offsets
                  [(i - block.row_start) * VLEN +
                   (j - block.col_start) / (VLEN * 4) * 4 + 2]);
        }
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          B_zero_point_v = _mm512_loadu_si512(r.B_zero_point + j + 2 * VLEN);
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          if (C_PER_G == 4) {
            B_zero_point_v = _mm512_castps_si512(_mm512_permutevar_ps(
                _mm512_broadcast_f32x4(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        r.B_zero_point + quant_param_idx + 8))),
                _mm512_set_epi32(
                    3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0)));
          } else if (C_PER_G == 8) {
            B_zero_point_v = _mm512_inserti32x8(
                _mm512_castsi256_si512(_mm256_set1_epi32(
                    r.B_zero_point
                        [quant_param_idx +
                         (j - block.col_start) / (VLEN * 4) * 8 + 4])),
                _mm256_set1_epi32(
                    r.B_zero_point
                        [quant_param_idx +
                         (j - block.col_start) / (VLEN * 4) * 8 + 5]),
                1);
          } else {
            B_zero_point_v = _mm512_set1_epi32(
                r.B_zero_point
                    [quant_param_idx + (j - block.col_start) / (VLEN * 4) * 4 +
                     2]);
          }
        }
        row_offset_v = _mm512_mullo_epi32(row_offset_v, B_zero_point_v);
        z_v = _mm512_sub_epi32(z_v, row_offset_v);

        // Groups 12-15 when C_PER_G == 4
        // Groups 6-7 when C_PER_G == 8
        if (C_PER_G == 4) {
          row_offset_v = _mm512_castps_si512(_mm512_permutevar_ps(
              _mm512_broadcast_f32x4(
                  _mm_loadu_ps(reinterpret_cast<const float*>(
                      r.row_offsets + (i - block.row_start) * VLEN + 12))),
              _mm512_set_epi32(
                  3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0)));
        } else if (C_PER_G == 8) {
          row_offset_v = _mm512_inserti32x8(
              _mm512_castsi256_si512(_mm256_set1_epi32(
                  r.row_offsets
                      [(i - block.row_start) * VLEN +
                       (j - block.col_start) / (VLEN * 4) * 8 + 6])),
              _mm256_set1_epi32(
                  r.row_offsets
                      [(i - block.row_start) * VLEN +
                       (j - block.col_start) / (VLEN * 4) * 8 + 7]),
              1);
        } else {
          row_offset_v = _mm512_set1_epi32(
              r.row_offsets
                  [(i - block.row_start) * VLEN +
                   (j - block.col_start) / (VLEN * 4) * 4 + 3]);
        }
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          B_zero_point_v = _mm512_loadu_si512(r.B_zero_point + j + 3 * VLEN);
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          if (C_PER_G == 4) {
            B_zero_point_v = _mm512_castps_si512(_mm512_permutevar_ps(
                _mm512_broadcast_f32x4(
                    _mm_loadu_ps(reinterpret_cast<const float*>(
                        r.B_zero_point + quant_param_idx + 12))),
                _mm512_set_epi32(
                    3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0)));
          } else if (C_PER_G == 8) {
            B_zero_point_v = _mm512_inserti32x8(
                _mm512_castsi256_si512(_mm256_set1_epi32(
                    r.B_zero_point
                        [quant_param_idx +
                         (j - block.col_start) / (VLEN * 4) * 8 + 6])),
                _mm256_set1_epi32(
                    r.B_zero_point
                        [quant_param_idx +
                         (j - block.col_start) / (VLEN * 4) * 8 + 7]),
                1);
          } else {
            B_zero_point_v = _mm512_set1_epi32(
                r.B_zero_point
                    [quant_param_idx + (j - block.col_start) / (VLEN * 4) * 4 +
                     3]);
          }
        }
        row_offset_v = _mm512_mullo_epi32(row_offset_v, B_zero_point_v);
        w_v = _mm512_sub_epi32(w_v, row_offset_v);
      }
      if (HAS_BIAS) {
        x_v = _mm512_add_epi32(x_v, _mm512_loadu_si512(r.bias + j));
        y_v = _mm512_add_epi32(y_v, _mm512_loadu_si512(r.bias + j + VLEN));
        z_v = _mm512_add_epi32(z_v, _mm512_loadu_si512(r.bias + j + 2 * VLEN));
        w_v = _mm512_add_epi32(w_v, _mm512_loadu_si512(r.bias + j + 3 * VLEN));
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
      __m512 x_scaled_v, y_scaled_v, z_scaled_v, w_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v = _mm512_mul_ps(
            _mm512_cvtepi32_ps(x_v), _mm512_loadu_ps(r.C_multiplier + j));
        y_scaled_v = _mm512_mul_ps(
            _mm512_cvtepi32_ps(y_v),
            _mm512_loadu_ps(r.C_multiplier + j + VLEN));
        z_scaled_v = _mm512_mul_ps(
            _mm512_cvtepi32_ps(z_v),
            _mm512_loadu_ps(r.C_multiplier + j + 2 * VLEN));
        w_scaled_v = _mm512_mul_ps(
            _mm512_cvtepi32_ps(w_v),
            _mm512_loadu_ps(r.C_multiplier + j + 3 * VLEN));
      } else if (Q_GRAN == QuantizationGranularity::GROUP) {
        if (C_PER_G == 4) {
          multiplier_v = _mm512_permutevar_ps(
              _mm512_broadcast_f32x4(
                  _mm_loadu_ps(r.C_multiplier + quant_param_idx + 0)),
              _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0));
          x_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(x_v), multiplier_v);

          multiplier_v = _mm512_permutevar_ps(
              _mm512_broadcast_f32x4(
                  _mm_loadu_ps(r.C_multiplier + quant_param_idx + 4)),
              _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0));
          y_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(y_v), multiplier_v);

          multiplier_v = _mm512_permutevar_ps(
              _mm512_broadcast_f32x4(
                  _mm_loadu_ps(r.C_multiplier + quant_param_idx + 8)),
              _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0));
          z_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(z_v), multiplier_v);

          multiplier_v = _mm512_permutevar_ps(
              _mm512_broadcast_f32x4(
                  _mm_loadu_ps(r.C_multiplier + quant_param_idx + 12)),
              _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0));
          w_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(w_v), multiplier_v);
        } else if (C_PER_G == 8) {
          multiplier_v = _mm512_insertf32x8(
              _mm512_castps256_ps512(_mm256_set1_ps(
                  r.C_multiplier
                      [quant_param_idx +
                       (j - block.col_start) / (VLEN * 4) * 8 + 0])),
              _mm256_set1_ps(
                  r.C_multiplier
                      [quant_param_idx +
                       (j - block.col_start) / (VLEN * 4) * 8 + 1]),
              1);
          x_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(x_v), multiplier_v);

          multiplier_v = _mm512_insertf32x8(
              _mm512_castps256_ps512(_mm256_set1_ps(
                  r.C_multiplier
                      [quant_param_idx +
                       (j - block.col_start) / (VLEN * 4) * 8 + 2])),
              _mm256_set1_ps(
                  r.C_multiplier
                      [quant_param_idx +
                       (j - block.col_start) / (VLEN * 4) * 8 + 3]),
              1);
          y_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(y_v), multiplier_v);

          multiplier_v = _mm512_insertf32x8(
              _mm512_castps256_ps512(_mm256_set1_ps(
                  r.C_multiplier
                      [quant_param_idx +
                       (j - block.col_start) / (VLEN * 4) * 8 + 4])),
              _mm256_set1_ps(
                  r.C_multiplier
                      [quant_param_idx +
                       (j - block.col_start) / (VLEN * 4) * 8 + 5]),
              1);
          z_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(z_v), multiplier_v);

          multiplier_v = _mm512_insertf32x8(
              _mm512_castps256_ps512(_mm256_set1_ps(
                  r.C_multiplier
                      [quant_param_idx +
                       (j - block.col_start) / (VLEN * 4) * 8 + 6])),
              _mm256_set1_ps(
                  r.C_multiplier
                      [quant_param_idx +
                       (j - block.col_start) / (VLEN * 4) * 8 + 7]),
              1);
          w_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(w_v), multiplier_v);
        } else {
          multiplier_v = _mm512_set1_ps(
              r.C_multiplier
                  [quant_param_idx + (j - block.col_start) / (VLEN * 4) * 4]);
          x_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(x_v), multiplier_v);

          multiplier_v =
              _mm512_set1_ps(r.C_multiplier
                                 [quant_param_idx +
                                  (j - block.col_start) / (VLEN * 4) * 4 + 1]);
          y_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(y_v), multiplier_v);

          multiplier_v =
              _mm512_set1_ps(r.C_multiplier
                                 [quant_param_idx +
                                  (j - block.col_start) / (VLEN * 4) * 4 + 2]);
          z_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(z_v), multiplier_v);

          multiplier_v =
              _mm512_set1_ps(r.C_multiplier
                                 [quant_param_idx +
                                  (j - block.col_start) / (VLEN * 4) * 4 + 3]);
          w_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(w_v), multiplier_v);
        }
      } else {
        x_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(x_v), multiplier_v);
        y_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(y_v), multiplier_v);
        z_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(z_v), multiplier_v);
        w_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(w_v), multiplier_v);
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
      __m512i y_rounded_v = _mm512_cvtps_epi32(y_scaled_v);
      __m512i z_rounded_v = _mm512_cvtps_epi32(z_scaled_v);
      __m512i w_rounded_v = _mm512_cvtps_epi32(w_scaled_v);

      /*
       * Standard final sequence on x86 AVX512:
       * - Pack to int16_t and saturate
       * - Add zero point
       * - Pack to uint8_t and saturate
       * - Clamp between qmin and qmax
       */
      __m512i xy_packed_v = _mm512_adds_epi16(
          _mm512_packs_epi32(x_rounded_v, y_rounded_v), C_zero_point_epi16_v);
      __m512i zw_packed_v = _mm512_adds_epi16(
          _mm512_packs_epi32(z_rounded_v, w_rounded_v), C_zero_point_epi16_v);
      __m512i xyzw_packed_v = _mm512_packus_epi16(xy_packed_v, zw_packed_v);
      __m512i xyzw_clamped_v = _mm512_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm512_min_epu8(xyzw_packed_v, max_v));

      /*
       * xyzw_clamped_v has results in the following layout so we need to
       * permute: x0-3  y0-3  z0-3  w0-3  x4-7   y4-7   z4-7   w4-7
       *          x8-11 y8-11 z8-11 w8-11 x12-15 y12-15 z12-15 w12-15
       */
      xyzw_clamped_v = _mm512_permutexvar_epi32(permute_mask_v, xyzw_clamped_v);

      /*
       * 4x CVTDQ2PS
       * 4x MULPS
       * 4x CVTPS2DQ
       * 2x PACKSSDW
       * 1x PACKUSWB
       * 2x PADDW
       * 1x PMAXUB
       * 1x PMINUB
       * 1x PERMD
       * ---------------------
       * 20 instructions total
       */
      _mm512_storeu_si512(out + i * ld_out + j, xyzw_clamped_v);
    } // j loop vectorized and unrolled 4x

    int remainder = block.col_start + block.col_size - j;
    assert(remainder == 0);
  } // i loop
}

#define INSTANTIATE_REQUANTIZE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU)               \
  template void                                                                \
  requantizeOutputProcessingGConvAvx512<A_SYM, B_SYM, Q_GRAN, BIAS, RELU, 4>(  \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t& r);                                        \
  template void                                                                \
  requantizeOutputProcessingGConvAvx512<A_SYM, B_SYM, Q_GRAN, BIAS, RELU, 8>(  \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t& r);                                        \
  template void                                                                \
  requantizeOutputProcessingGConvAvx512<A_SYM, B_SYM, Q_GRAN, BIAS, RELU, 16>( \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t& r);

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
