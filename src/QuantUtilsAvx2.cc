/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/QuantUtilsAvx2.h"
#include <immintrin.h>
#include <algorithm> //for std::min/std::max
#include <cmath> //for nearbyint
#include <limits> //for numeric_limits
#include "./MaskAvx2.h"
#include "fbgemm/Fbgemm.h" //for ReQuantizeOutput

namespace fbgemm {

using namespace std;
////////////////////////////////////////////////////////////////////////////////
// Utility functions

// ASAN seems to have a false-positive for _mm_maskmoveu_si128
template <typename T, bool LEGACY>
void NO_SANITIZE("address") QuantizeAvx2(
    const float* src,
    T* dst,
    int len,
    const TensorQuantizationParams& qparams) {
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
  constexpr int VLEN = 8;
  constexpr int32_t min_val = std::numeric_limits<T>::min();
  constexpr int32_t max_val = std::numeric_limits<T>::max();
  // This is the largest int32 value less than int32_max
  // that is exactly representable in float
  constexpr int32_t int32_float_max_val =
      std::numeric_limits<int32_t>::max() - 127;
  std::size_t i = 0;
  float inverse_scale = 1.f / qparams.scale;
  __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
  // clang-format off
  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00);
  // clang-format on
  __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);
  for (; i < len / VLEN * VLEN; i += VLEN) {
    __m256 src_v = _mm256_loadu_ps(src + i);
    __m256 transformed_v;
    if (LEGACY) { // static if
      transformed_v = _mm256_fmadd_ps(
          src_v, inverse_scale_v, _mm256_set1_ps(qparams.zero_point));
    } else {
      transformed_v = _mm256_mul_ps(src_v, inverse_scale_v);
    }
    // If the floating point value is greater than int32_max,
    // _mm256_cvtps_epi32 converts them to negative. Clip at int32_float_max_val
    // to avoid this.
    transformed_v =
        _mm256_min_ps(transformed_v, _mm256_set1_ps(int32_float_max_val));

    __m256i rounded_v = _mm256_cvtps_epi32(transformed_v);
    if (!LEGACY) {
      rounded_v =
          _mm256_add_epi32(rounded_v, _mm256_set1_epi32(qparams.zero_point));
    }
    __m256i clipped_v = _mm256_min_epi32(
        _mm256_max_epi32(rounded_v, _mm256_set1_epi32(min_val)),
        _mm256_set1_epi32(max_val));

    // An instruction sequence to save 8 32-bit integers as 8 8-bit integers
    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + i), _mm256_castsi256_si128(clipped_v));
  }

  // Handle remainder using mask instructions so that
  // the main loop and remainder loop have the same behavior
  int rem = len - i;
  if (rem > 0) {
    __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_masks[rem]));
    __m128i store_mask_v = _mm_load_si128(
        reinterpret_cast<const __m128i*>(internal::sse_epi8_masks[rem]));
    __m256 src_v = _mm256_maskload_ps(src + i, mask_v);
    __m256 transformed_v;
    if (LEGACY) {
      transformed_v = _mm256_fmadd_ps(
          src_v, inverse_scale_v, _mm256_set1_ps(qparams.zero_point));
    } else {
      transformed_v = _mm256_mul_ps(src_v, inverse_scale_v);
    }
    transformed_v =
        _mm256_min_ps(transformed_v, _mm256_set1_ps(int32_float_max_val));

    __m256i rounded_v = _mm256_cvtps_epi32(transformed_v);
    if (!LEGACY) {
      rounded_v =
          _mm256_add_epi32(rounded_v, _mm256_set1_epi32(qparams.zero_point));
    }
    __m256i clipped_v = _mm256_min_epi32(
        _mm256_max_epi32(rounded_v, _mm256_set1_epi32(min_val)),
        _mm256_set1_epi32(max_val));

    // An instruction sequence to save "rem" number of 32-bit integers
    // as "rem" number of 8-bit integers
    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);
    _mm_maskmoveu_si128(
        _mm256_castsi256_si128(clipped_v),
        store_mask_v,
        reinterpret_cast<char*>(dst + i));
  }
#endif
}

// Instantiate QuantizeAvx2 for known datatypes
#define SPECIALIZE_QUANTIZEAVX2(T, LEGACY) \
  template void QuantizeAvx2<T, LEGACY>(   \
      const float* src,                    \
      T* dst,                              \
      int len,                             \
      const TensorQuantizationParams& qparams);
SPECIALIZE_QUANTIZEAVX2(uint8_t, true)
SPECIALIZE_QUANTIZEAVX2(int8_t, true)
SPECIALIZE_QUANTIZEAVX2(uint8_t, false)
SPECIALIZE_QUANTIZEAVX2(int8_t, false)
#undef SPECIALIZE_QUANTIZEAVX2

template <typename T>
void NO_SANITIZE("address") FusedQuantizeDequantizeAvx2(
    const float* src,
    float* dst,
    int len,
    const TensorQuantizationParams& qparams) {
  float inverse_scale = 1.f / qparams.scale;
  constexpr int32_t min_val = std::numeric_limits<T>::min();
  constexpr int32_t max_val = std::numeric_limits<T>::max();
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))

  constexpr int VLEN = 8;
  // This is the largest int32 value less than int32_max
  // that is exactly representable in float
  constexpr int32_t int32_float_max_val =
      std::numeric_limits<int32_t>::max() - 127;
  std::size_t i = 0;
  __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
  __m256 scale_v = _mm256_set1_ps(qparams.scale);
  __m256 zp_v = _mm256_set1_ps(qparams.zero_point);

  for (; i < len / VLEN * VLEN; i += VLEN) {
    // prefetch src and dst
    _mm_prefetch(reinterpret_cast<const char*>(src + i + VLEN), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(dst + i + VLEN), _MM_HINT_T0);

    __m256 src_v = _mm256_loadu_ps(src + i);
    __m256 transformed_v;

    transformed_v = _mm256_mul_ps(src_v, inverse_scale_v);
    // If the floating point value is greater than int32_max,
    // _mm256_cvtps_epi32 converts them to negative. Clip at int32_float_max_val
    // to avoid this.
    transformed_v =
        _mm256_min_ps(transformed_v, _mm256_set1_ps(int32_float_max_val));

    __m256i rounded_v = _mm256_cvtps_epi32(transformed_v);
    rounded_v =
        _mm256_add_epi32(rounded_v, _mm256_set1_epi32(qparams.zero_point));
    __m256i clipped_v = _mm256_min_epi32(
        _mm256_max_epi32(rounded_v, _mm256_set1_epi32(min_val)),
        _mm256_set1_epi32(max_val));

    // convert int32 to float32
    __m256 fp32_clipped_v = _mm256_cvtepi32_ps(clipped_v);
    // minus zero point, multiply by scale
    __m256 fp32_dq_sub = _mm256_sub_ps(fp32_clipped_v, zp_v);
    __m256 fp32_dq = _mm256_mul_ps(fp32_dq_sub, scale_v);

    // save fusued quantize-dequantize fp32 values into dst
    _mm256_storeu_ps(dst + i, fp32_dq);
  }

  // Handle remainder using mask instructions so that
  // the main loop and remainder loop have the same behavior
  int rem = len - i;
  if (rem > 0) {
    __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_masks[rem]));

    __m256 src_v = _mm256_maskload_ps(src + i, mask_v);
    __m256 transformed_v;

    transformed_v = _mm256_mul_ps(src_v, inverse_scale_v);
    // If the floating point value is greater than int32_max,
    // _mm256_cvtps_epi32 converts them to negative. Clip at int32_float_max_val
    // to avoid this.
    transformed_v =
        _mm256_min_ps(transformed_v, _mm256_set1_ps(int32_float_max_val));

    __m256i rounded_v = _mm256_cvtps_epi32(transformed_v);
    rounded_v =
        _mm256_add_epi32(rounded_v, _mm256_set1_epi32(qparams.zero_point));

    __m256i clipped_v = _mm256_min_epi32(
        _mm256_max_epi32(rounded_v, _mm256_set1_epi32(min_val)),
        _mm256_set1_epi32(max_val));

    // convert int32 to float32
    __m256 fp32_clipped_v = _mm256_cvtepi32_ps(clipped_v);
    // minus zero point, multiply by scale
    __m256 fp32_dq_sub =
        _mm256_sub_ps(fp32_clipped_v, _mm256_set1_ps(qparams.zero_point));
    __m256 fp32_dq = _mm256_mul_ps(fp32_dq_sub, _mm256_set1_ps(qparams.scale));

    // store fp32 values with mask
    _mm256_maskstore_ps(dst + i, mask_v, fp32_dq);
  }
#endif
}

// Instantiate QuantizeAvx2 for known datatypes
#define SPECIALIZE_FUSEDDQAVX2(T)               \
  template void FusedQuantizeDequantizeAvx2<T>( \
      const float* src,                         \
      float* dst,                               \
      int len,                                  \
      const TensorQuantizationParams& qparams);
SPECIALIZE_FUSEDDQAVX2(uint8_t)
SPECIALIZE_FUSEDDQAVX2(int8_t)

#undef SPECIALIZE_FUSEDDQAVX2

void FindMinMax(const float* a, float* min, float* max, int len) {
  if (len <= 0) {
    *min = 0.0f;
    *max = 0.0f;
    return;
  }

  float temp_min = *a, temp_max = *a;
  int i = 0;

#ifdef __AVX__
  __m256 min_v = _mm256_set1_ps(*a), max_v = _mm256_set1_ps(*a);
  constexpr int VLEN = 8;
  if (len >= VLEN) {
    for (; i < len / VLEN * VLEN; i += VLEN) {
      min_v = _mm256_min_ps(min_v, _mm256_loadu_ps(a + i));
      max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(a + i));
    }

    float min_buf[VLEN], max_buf[VLEN];
    _mm256_storeu_ps(min_buf, min_v);
    _mm256_storeu_ps(max_buf, max_v);
    for (int j = 0; j < VLEN; ++j) {
      temp_min = std::min(temp_min, min_buf[j]);
      temp_max = std::max(temp_max, max_buf[j]);
    }
  }
#endif

  for (; i < len; i++) {
    temp_min = std::min(temp_min, a[i]);
    temp_max = std::max(temp_max, a[i]);
  }
  *min = temp_min;
  *max = temp_max;
}

////////////////////////////////////////////////////////////////////////////////
// Requantization (with floats)

#ifdef __AVX2__
void RequantizeAvx2(
    const int32_t* src,
    uint8_t* dst,
    int len,
    const RequantizationParams& params) {
  DoNothing<> doNothingObj{};
  int32_t Bq_zero_point[] = {0};
  ReQuantizeOutput<false /* FUSE_RELU */> requantizeObj(
      doNothingObj,
      &params.real_multiplier,
      params.target_qparams.zero_point,
      0, // Aq_zero_point
      Bq_zero_point, // Bq_zero_point
      nullptr, // row_offsets
      nullptr, // col_offsets
      nullptr, // bias
      len); // ncol
  requantizeObj.f<inst_set_t::avx2>(dst, src, {0, 1, 0, len}, 0, 0);
}

void RequantizeFixedPointAvx2(
    const int32_t* src,
    uint8_t* dst,
    int len,
    const RequantizationParams& params) {
  constexpr int VLEN = 8;

  __m256i b = _mm256_set1_epi32(params.multiplier);

  // AVX2 doesn't support arithmetic right shift.
  // As a work around, we convert 64-bit multiplied results to uint64_t by
  // adding 0x8000000000000000ULL, logical right shift, and subtract by
  // (0x8000000000000000ULL >> right_shift).
  __m256i pre_shift_nudge = _mm256_set1_epi64x(
      (1ll << (params.right_shift - 1)) + 0x8000000000000000ULL);
  __m256i post_shift_nudge = _mm256_set1_epi64x(
      params.target_qparams.zero_point -
      (0x8000000000000000ULL >> params.right_shift));

  __m256i min_v = _mm256_set1_epi32(numeric_limits<uint8_t>::min());
  __m256i max_v = _mm256_set1_epi32(numeric_limits<uint8_t>::max());

  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  int i = 0;
  for (; i < len / VLEN * VLEN; i += VLEN) {
    __m256i a_v = _mm256_loadu_si256((const __m256i*)(src + i));

    // a = a0 | a1 | a2 | a3 | a4 | a5 | a6 | a7
    // b = b0 | b1 | b3 | b3 | b4 | b5 | b6 | b7
    __m256i a_even_v = a_v;
    __m256i a_odd_v = _mm256_srli_si256(a_v, 4);

    __m256i ab_even_v = _mm256_mul_epi32(a_even_v, b);
    __m256i ab_odd_v = _mm256_mul_epi32(a_odd_v, b);

    __m256i even_rounded_v = _mm256_add_epi64(ab_even_v, pre_shift_nudge);
    __m256i odd_rounded_v = _mm256_add_epi64(ab_odd_v, pre_shift_nudge);

    __m256i even_result_v = _mm256_add_epi64(
        _mm256_srli_epi64(even_rounded_v, params.right_shift),
        post_shift_nudge);
    __m256i odd_result_v = _mm256_add_epi64(
        _mm256_srli_epi64(odd_rounded_v, params.right_shift), post_shift_nudge);
    odd_result_v = _mm256_slli_si256(odd_result_v, 4);

    // even_result_v has numbers we want in its even 32-bit SIMD lanes, and
    // odd_result_v has numbers we want in its odd 32-bit SIMD lanes.
    // Use blend to combine them.
    __m256i result_v = _mm256_blend_epi32(even_result_v, odd_result_v, 0xaa);
    __m256i clipped_v =
        _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, result_v));

    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);
    *(int64_t*)(dst + i) = _mm256_extract_epi64(clipped_v, 0);
  }

  for (; i < len; ++i) {
    int64_t ab_64 =
        static_cast<int64_t>(src[i]) * static_cast<int64_t>(params.multiplier);
    int64_t nudge = 1ll << std::max(0, params.right_shift - 1);
    int64_t quantized_down = params.target_qparams.zero_point +
        ((ab_64 + nudge) >> params.right_shift);
    dst[i] = std::min<int64_t>(std::max<int64_t>(quantized_down, 0l), 255l);
  }
}
#endif

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    typename BIAS_TYPE>
void requantizeOutputProcessingAvx2(
    uint8_t* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t<BIAS_TYPE>& r) {
  // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
  // using AVX2 instructions
  int quant_param_idx = 0;
  if (Q_GRAN == QuantizationGranularity::GROUP) {
    int ncol_per_group = r.ncols / r.groups;
    int g = block.col_start / ncol_per_group;
    quant_param_idx = g;
  }
  __m256 multiplier_v = _mm256_set1_ps(r.C_multiplier[quant_param_idx]);

  // Broadcasted reciprocal of act_times_w_scale
  __m256 act_times_w_rcp_v;
  if (!(Q_GRAN == QuantizationGranularity::OUT_CHANNEL)) {
    if (is_same<BIAS_TYPE, float>::value) {
      act_times_w_rcp_v =
          _mm256_set1_ps(1.0 / r.act_times_w_scale[quant_param_idx]);
    }
  }

  __m256i min_v = _mm256_set1_epi8(static_cast<uint8_t>(0));
  __m256i max_v = _mm256_set1_epi8(static_cast<uint8_t>(255));

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

  __m256i A_zero_point_v = _mm256_set1_epi32(r.A_zero_point);
  __m256i C_zero_point_epi16_v = _mm256_set1_epi16(r.C_zero_point);
  __m256i C_zero_point_epi8_v = _mm256_set1_epi8(r.C_zero_point);

  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  constexpr int VLEN = 8;
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    // Scale row_offset with Bq_zero_point
    int32_t row_offset = 0;
    if (B_SYMMETRIC) {
      row_offset = 0;
    } else if (
        Q_GRAN == QuantizationGranularity::TENSOR ||
        Q_GRAN == QuantizationGranularity::GROUP) {
      row_offset =
          r.row_offsets[i - block.row_start] * r.B_zero_point[quant_param_idx];
    } else {
      assert(
          Q_GRAN == QuantizationGranularity::OUT_CHANNEL &&
          "unknown quantization granularity");
    }
    __m256i row_offset_v = _mm256_set1_epi32(row_offset);

    int j = block.col_start;
    for (; j < block.col_start + (block.col_size / (VLEN * 4) * (VLEN * 4));
         j += (VLEN * 4)) {
      __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start)));
      __m256i y_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start) +
          1 * VLEN));
      __m256i z_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start) +
          2 * VLEN));
      __m256i w_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start) +
          3 * VLEN));

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j)));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
        col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j + VLEN)));
        y_v = _mm256_sub_epi32(y_v, col_off_v);
        col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                r.col_offsets + j + 2 * VLEN)));
        z_v = _mm256_sub_epi32(z_v, col_off_v);
        col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                r.col_offsets + j + 3 * VLEN)));
        w_v = _mm256_sub_epi32(w_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.B_zero_point + j)));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.B_zero_point + j + VLEN)));
        }
        y_v = _mm256_sub_epi32(y_v, row_offset_v);
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                  r.B_zero_point + j + 2 * VLEN)));
        }
        z_v = _mm256_sub_epi32(z_v, row_offset_v);
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                  r.B_zero_point + j + 3 * VLEN)));
        }
        w_v = _mm256_sub_epi32(w_v, row_offset_v);
      }
      __m256 xf_v, yf_v, zf_v, wf_v;
      if (HAS_BIAS) {
        if (is_same<BIAS_TYPE, float>::value) {
          __m256 x_bias_v, y_bias_v, z_bias_v, w_bias_v;
          if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
            x_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 0 * VLEN)),
                _mm256_loadu_ps(r.act_times_w_scale + j + 0 * VLEN));
            y_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 1 * VLEN)),
                _mm256_loadu_ps(r.act_times_w_scale + j + 1 * VLEN));
            z_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 2 * VLEN)),
                _mm256_loadu_ps(r.act_times_w_scale + j + 2 * VLEN));
            w_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 3 * VLEN)),
                _mm256_loadu_ps(r.act_times_w_scale + j + 3 * VLEN));
          } else {
            x_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 0 * VLEN)),
                act_times_w_rcp_v);
            y_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 1 * VLEN)),
                act_times_w_rcp_v);
            z_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 2 * VLEN)),
                act_times_w_rcp_v);
            w_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(
                    reinterpret_cast<const float*>(r.bias + j + 3 * VLEN)),
                act_times_w_rcp_v);
          }
          xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
          yf_v = _mm256_add_ps(_mm256_cvtepi32_ps(y_v), y_bias_v);
          zf_v = _mm256_add_ps(_mm256_cvtepi32_ps(z_v), z_bias_v);
          wf_v = _mm256_add_ps(_mm256_cvtepi32_ps(w_v), w_bias_v);
        } else {
          x_v = _mm256_add_epi32(
              x_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.bias + j + 0 * VLEN)));
          y_v = _mm256_add_epi32(
              y_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.bias + j + 1 * VLEN)));
          z_v = _mm256_add_epi32(
              z_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.bias + j + 2 * VLEN)));
          w_v = _mm256_add_epi32(
              w_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.bias + j + 3 * VLEN)));
          xf_v = _mm256_cvtepi32_ps(x_v);
          yf_v = _mm256_cvtepi32_ps(y_v);
          zf_v = _mm256_cvtepi32_ps(z_v);
          wf_v = _mm256_cvtepi32_ps(w_v);
        }
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
        yf_v = _mm256_cvtepi32_ps(y_v);
        zf_v = _mm256_cvtepi32_ps(z_v);
        wf_v = _mm256_cvtepi32_ps(w_v);
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
      __m256 x_scaled_v, y_scaled_v, z_scaled_v, w_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v =
            _mm256_mul_ps(xf_v, _mm256_loadu_ps(r.C_multiplier + j + 0 * VLEN));
        y_scaled_v =
            _mm256_mul_ps(yf_v, _mm256_loadu_ps(r.C_multiplier + j + 1 * VLEN));
        z_scaled_v =
            _mm256_mul_ps(zf_v, _mm256_loadu_ps(r.C_multiplier + j + 2 * VLEN));
        w_scaled_v =
            _mm256_mul_ps(wf_v, _mm256_loadu_ps(r.C_multiplier + j + 3 * VLEN));
      } else {
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
        y_scaled_v = _mm256_mul_ps(yf_v, multiplier_v);
        z_scaled_v = _mm256_mul_ps(zf_v, multiplier_v);
        w_scaled_v = _mm256_mul_ps(wf_v, multiplier_v);
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
      __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);
      __m256i y_rounded_v = _mm256_cvtps_epi32(y_scaled_v);
      __m256i z_rounded_v = _mm256_cvtps_epi32(z_scaled_v);
      __m256i w_rounded_v = _mm256_cvtps_epi32(w_scaled_v);

      /*
       * Standard final sequence on x86 AVX2:
       * - Pack to int16_t and saturate
       * - Add zero point
       * - Pack to uint8_t and saturate
       * - Clamp between qmin and qmax
       */
      __m256i xy_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(x_rounded_v, y_rounded_v), C_zero_point_epi16_v);
      __m256i zw_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(z_rounded_v, w_rounded_v), C_zero_point_epi16_v);
      __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
      __m256i xyzw_clamped_v = _mm256_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm256_min_epu8(xyzw_packed_v, max_v));

      /*
       * xyzw_clamped_v has results in the following layout so we need to
       * permute: x0-3 y0-3 z0-3 w0-3 x4-7 y4-7 z4-7 w4-7
       */
      xyzw_clamped_v =
          _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);

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
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(out + i * ld_out + j), xyzw_clamped_v);
    } // j loop vectorized and unrolled 4x

    for (; j < block.col_start + (block.col_size / VLEN * VLEN); j += VLEN) {
      __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start)));

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j)));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.B_zero_point + j)));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }
      __m256 xf_v;
      if (HAS_BIAS) {
        if (is_same<BIAS_TYPE, float>::value) {
          __m256 x_bias_v;
          if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
            x_bias_v = _mm256_div_ps(
                _mm256_loadu_ps(reinterpret_cast<const float*>(r.bias + j)),
                _mm256_loadu_ps(r.act_times_w_scale + j));
          } else {
            x_bias_v = _mm256_mul_ps(
                _mm256_loadu_ps(reinterpret_cast<const float*>(r.bias + j)),
                act_times_w_rcp_v);
          }
          xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
        } else {
          x_v = _mm256_add_epi32(
              x_v,
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r.bias + j)));
          xf_v = _mm256_cvtepi32_ps(x_v);
        }
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
      }

      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v = _mm256_mul_ps(xf_v, _mm256_loadu_ps(r.C_multiplier + j));
      } else {
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
      }
      __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);

      __m256i x_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(x_rounded_v, _mm256_setzero_si256()),
          C_zero_point_epi16_v);
      x_packed_v = _mm256_packus_epi16(x_packed_v, _mm256_setzero_si256());
      __m256i x_clamped_v = _mm256_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm256_min_epu8(x_packed_v, max_v));

      /*
       * x_clamped_v has results in the following layout so we need to
       * permute: x0-3 garbage0-11 x4-7 garbage12-23
       */
      x_clamped_v = _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

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
      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(out + i * ld_out + j),
          _mm256_castsi256_si128(x_clamped_v));
    } // j loop vectorized

    int remainder = block.col_start + block.col_size - j;
    if (remainder > 0) {
      __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
          internal::avx2_ps_or_epi32_masks[remainder]));

      __m256i x_v = _mm256_maskload_epi32(
          inp + (i - block.row_start) * ld_in + (j - block.col_start), mask_v);

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v, _mm256_maskload_epi32(r.col_offsets + j, mask_v));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_maskload_epi32(r.B_zero_point + j, mask_v));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }

      __m256 xf_v;
      if (HAS_BIAS) {
        if (is_same<BIAS_TYPE, float>::value) {
          __m256 x_bias_v;
          if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
            x_bias_v = _mm256_div_ps(
                _mm256_maskload_ps(
                    reinterpret_cast<const float*>(r.bias + j), mask_v),
                _mm256_maskload_ps(r.act_times_w_scale + j, mask_v));
          } else {
            x_bias_v = _mm256_mul_ps(
                _mm256_maskload_ps(
                    reinterpret_cast<const float*>(r.bias + j), mask_v),
                act_times_w_rcp_v);
          }
          xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
        } else {
          x_v = _mm256_add_epi32(
              x_v,
              _mm256_maskload_epi32(
                  reinterpret_cast<const int*>(r.bias + j), mask_v));
          xf_v = _mm256_cvtepi32_ps(x_v);
        }
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
      }

      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v =
            _mm256_mul_ps(xf_v, _mm256_maskload_ps(r.C_multiplier + j, mask_v));
      } else {
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
      }
      __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);

      __m256i x_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(x_rounded_v, _mm256_setzero_si256()),
          C_zero_point_epi16_v);
      x_packed_v = _mm256_packus_epi16(x_packed_v, _mm256_setzero_si256());
      __m256i x_clamped_v = _mm256_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm256_min_epu8(x_packed_v, max_v));

      /*
       * x_clamped_v has results in the following layout so we need to
       * permute: x0-3 garbage0-11 x4-7 garbage12-23
       */
      x_clamped_v = _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

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
      alignas(64) uint8_t x_clamped_buffer[32];
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(x_clamped_buffer), x_clamped_v);
      for (int k = 0; k < remainder; ++k) {
        out[i * ld_out + j + k] = x_clamped_buffer[k];
      }
    } // j loop remainder
  } // i loop
}

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU>
void requantizeForFloatAvx2(
    float* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationForFloatParams_t& r) {
  // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
  // using AVX2 instructions
  int quant_param_idx = 0;
  if (Q_GRAN == QuantizationGranularity::GROUP) {
    int ncol_per_group = r.ncols / r.groups;
    int g = block.col_start / ncol_per_group;
    quant_param_idx = g;
  }
  __m256 multiplier_v = _mm256_set1_ps(r.A_scale * r.B_scale[quant_param_idx]);

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

  __m256i A_zero_point_v = _mm256_set1_epi32(r.A_zero_point);

  constexpr int VLEN = 8;
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    // Scale row_offset with Bq_zero_point
    int32_t row_offset = 0;
    if (B_SYMMETRIC) {
      row_offset = 0;
    } else if (
        Q_GRAN == QuantizationGranularity::TENSOR ||
        Q_GRAN == QuantizationGranularity::GROUP) {
      row_offset =
          r.row_offsets[i - block.row_start] * r.B_zero_point[quant_param_idx];
    } else {
      assert(
          Q_GRAN == QuantizationGranularity::OUT_CHANNEL &&
          "unknown quantization granularity");
    }
    __m256i row_offset_v = _mm256_set1_epi32(row_offset);

    int j = block.col_start;
    for (; j < block.col_start + (block.col_size / VLEN * VLEN); j += VLEN) {
      __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start)));

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j)));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(r.B_zero_point + j)));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }

      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(x_v),
            _mm256_mul_ps(
                _mm256_set1_ps(r.A_scale), _mm256_loadu_ps(r.B_scale + j)));
      } else {
        x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
      }

      if (HAS_BIAS) {
        x_scaled_v = _mm256_add_ps(x_scaled_v, _mm256_loadu_ps(r.bias + j));
      }
      if (FUSE_RELU) {
        x_scaled_v = _mm256_max_ps(_mm256_setzero_ps(), x_scaled_v);
      }

      _mm256_storeu_ps(out + i * ld_out + j, x_scaled_v);
    } // j loop vectorized

    int remainder = block.col_start + block.col_size - j;
    if (remainder > 0) {
      __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
          internal::avx2_ps_or_epi32_masks[remainder]));

      __m256i x_v = _mm256_maskload_epi32(
          inp + (i - block.row_start) * ld_in + (j - block.col_start), mask_v);

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v, _mm256_maskload_epi32(r.col_offsets + j, mask_v));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          row_offset_v = _mm256_mullo_epi32(
              _mm256_set1_epi32(r.row_offsets[i - block.row_start]),
              _mm256_maskload_epi32(r.B_zero_point + j, mask_v));
        }
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }

      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v = _mm256_mul_ps(
            _mm256_cvtepi32_ps(x_v),
            _mm256_mul_ps(
                _mm256_set1_ps(r.A_scale),
                _mm256_maskload_ps(r.B_scale + j, mask_v)));
      } else {
        x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
      }

      if (HAS_BIAS) {
        x_scaled_v =
            _mm256_add_ps(x_scaled_v, _mm256_maskload_ps(r.bias + j, mask_v));
      }
      if (FUSE_RELU) {
        x_scaled_v = _mm256_max_ps(_mm256_setzero_ps(), x_scaled_v);
      }

      _mm256_maskstore_ps(out + i * ld_out + j, mask_v, x_scaled_v);
    } // j loop remainder
  } // i loop
}

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    int C_PER_G,
    typename BIAS_TYPE>
void requantizeOutputProcessingGConvAvx2(
    uint8_t* out,
    const int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t<BIAS_TYPE>& r) {
  // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
  // using AVX2 instructions
  int quant_param_idx = 0;
  if (Q_GRAN == QuantizationGranularity::GROUP) {
    int ncol_per_group = r.ncols / r.groups;
    int g = block.col_start / ncol_per_group;
    quant_param_idx = g;
  }
  __m256 multiplier_v = _mm256_set1_ps(r.C_multiplier[quant_param_idx]);

  // Broadcasted reciprocal of act_times_w_scale
  __m256 act_times_w_rcp_v;
  if (!(Q_GRAN == QuantizationGranularity::OUT_CHANNEL)) {
    if (is_same<BIAS_TYPE, float>::value) {
      act_times_w_rcp_v =
          _mm256_set1_ps(1.0 / r.act_times_w_scale[quant_param_idx]);
    }
  }
  __m256i min_v = _mm256_set1_epi8(static_cast<uint8_t>(0));
  __m256i max_v = _mm256_set1_epi8(static_cast<uint8_t>(255));

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

  __m256i A_zero_point_v = _mm256_set1_epi32(r.A_zero_point);
  __m256i C_zero_point_epi16_v = _mm256_set1_epi16(r.C_zero_point);
  __m256i C_zero_point_epi8_v = _mm256_set1_epi8(r.C_zero_point);

  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  constexpr int VLEN = 8;
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    int j = block.col_start;
    for (; j < block.col_start + (block.col_size / VLEN * VLEN); j += VLEN) {
      __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          inp + (i - block.row_start) * ld_in + (j - block.col_start)));

      if (!A_SYMMETRIC) {
        __m256i col_off_v = _mm256_mullo_epi32(
            A_zero_point_v,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(r.col_offsets + j)));
        x_v = _mm256_sub_epi32(x_v, col_off_v);
      }

      if (!B_SYMMETRIC) {
        __m256i row_offset_v;

        if (C_PER_G == 2) {
          // When C_PER_G == 2, we need to handle 4 groups at a time to fully
          // utilize 32B AVX2 vector register (C_PER_G * 4 * sizeof(int32_t) ==
          // 32B)
          // Load row_offsets for 4 groups and broadcast by 2 times.
          row_offset_v =
              _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                  _mm256_castps128_ps256(
                      _mm_loadu_ps(reinterpret_cast<const float*>(
                          r.row_offsets + (i - block.row_start) * 4))),
                  permute_mask_v)));

        }
        // When C_PER_G == 4, we need to handle 2 groups at a time to fully
        // utilize 32B AVX2 vector register (C_PER_G * 2 * sizeof(int32_t) ==
        // 32B)
        // When C_PER_G == 8, we just need 1 group at a time on the other hand.

        // Groups 0 and 1 when C_PER_G == 4
        // Group 0 when C_PER_G == 8
        else if (C_PER_G == 4) {
          // Load row_offsets for 2 groups and broadcast by 4 times each because
          // we have 4 channels per group.
          // groups 0 and 1
          row_offset_v = _mm256_insertf128_si256(
              _mm256_castsi128_si256(
                  _mm_set1_epi32(r.row_offsets[(i - block.row_start) * 2 + 0])),
              _mm_set1_epi32(r.row_offsets[(i - block.row_start) * 2 + 1]),
              1);
        } else if (C_PER_G == 8) {
          row_offset_v =
              _mm256_set1_epi32(r.row_offsets[(i - block.row_start)]);
        } else {
          assert(C_PER_G == 16);
          row_offset_v =
              _mm256_set1_epi32(r.row_offsets[(i - block.row_start)]);
        }

        __m256i B_zero_point_v = _mm256_set1_epi32(r.B_zero_point[0]);
        if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
          B_zero_point_v = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(r.B_zero_point + j));
        } else if (Q_GRAN == QuantizationGranularity::GROUP) {
          if (C_PER_G == 2) {
            B_zero_point_v =
                _mm256_castps_si256(_mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                    _mm256_castps128_ps256(
                        _mm_loadu_ps(reinterpret_cast<const float*>(
                            r.B_zero_point + quant_param_idx))),
                    permute_mask_v)));
          } else if (C_PER_G == 4) {
            B_zero_point_v = _mm256_insertf128_si256(
                _mm256_castsi128_si256(
                    _mm_set1_epi32(r.B_zero_point[quant_param_idx])),
                _mm_set1_epi32(r.B_zero_point[quant_param_idx + 1]),
                1);
          } else if (C_PER_G == 8) {
            B_zero_point_v = _mm256_set1_epi32(r.B_zero_point[quant_param_idx]);
          } else {
            B_zero_point_v = _mm256_set1_epi32(r.B_zero_point[quant_param_idx]);
          }
        }
        row_offset_v = _mm256_mullo_epi32(row_offset_v, B_zero_point_v);
        x_v = _mm256_sub_epi32(x_v, row_offset_v);
      }
      __m256 xf_v;
      if (HAS_BIAS) {
        if (is_same<BIAS_TYPE, float>::value) {
          __m256 x_bias_v =
              _mm256_loadu_ps(reinterpret_cast<const float*>(r.bias + j));
          if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
            x_bias_v = _mm256_div_ps(
                x_bias_v, _mm256_loadu_ps(r.act_times_w_scale + j));
          } else if (Q_GRAN == QuantizationGranularity::GROUP) {
            __m256 diviser_v;
            if (C_PER_G == 2) {
              diviser_v = _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
                  _mm256_castps128_ps256(
                      _mm_loadu_ps(r.act_times_w_scale + quant_param_idx)),
                  permute_mask_v));
            } else if (C_PER_G == 4) {
              diviser_v = _mm256_insertf128_ps(
                  _mm256_castps128_ps256(
                      _mm_set1_ps(r.act_times_w_scale[quant_param_idx + 0])),
                  _mm_set1_ps(r.act_times_w_scale[quant_param_idx + 1]),
                  1);
            } else if (C_PER_G == 8) {
              diviser_v = _mm256_set1_ps(r.act_times_w_scale[quant_param_idx]);
            } else {
              assert(C_PER_G == 16);
              diviser_v = _mm256_set1_ps(r.act_times_w_scale[quant_param_idx]);
            }
            x_bias_v = _mm256_div_ps(x_bias_v, diviser_v);
          } else {
            x_bias_v = _mm256_mul_ps(x_bias_v, act_times_w_rcp_v);
          }
          xf_v = _mm256_add_ps(_mm256_cvtepi32_ps(x_v), x_bias_v);
        } else {
          x_v = _mm256_add_epi32(
              x_v,
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r.bias + j)));
          xf_v = _mm256_cvtepi32_ps(x_v);
        }
      } else {
        xf_v = _mm256_cvtepi32_ps(x_v);
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
      __m256 x_scaled_v;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        x_scaled_v = _mm256_mul_ps(xf_v, _mm256_loadu_ps(r.C_multiplier + j));
      } else if (Q_GRAN == QuantizationGranularity::GROUP) {
        if (C_PER_G == 2) {
          multiplier_v = _mm256_moveldup_ps(_mm256_permutevar8x32_ps(
              _mm256_castps128_ps256(
                  _mm_loadu_ps(r.C_multiplier + quant_param_idx)),
              permute_mask_v));
        } else if (C_PER_G == 4) {
          multiplier_v = _mm256_insertf128_ps(
              _mm256_castps128_ps256(
                  _mm_set1_ps(r.C_multiplier[quant_param_idx])),
              _mm_set1_ps(r.C_multiplier[quant_param_idx + 1]),
              1);
        } else if (C_PER_G == 8) {
          multiplier_v = _mm256_set1_ps(r.C_multiplier[quant_param_idx]);
        } else {
          multiplier_v = _mm256_set1_ps(r.C_multiplier[quant_param_idx]);
        }
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
      } else {
        x_scaled_v = _mm256_mul_ps(xf_v, multiplier_v);
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
      __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);

      /*
       * Standard final sequence on x86 AVX2:
       * - Pack to int16_t and saturate
       * - Add zero point
       * - Pack to uint8_t and saturate
       * - Clamp between qmin and qmax
       */
      __m256i x_packed_v = _mm256_adds_epi16(
          _mm256_packs_epi32(x_rounded_v, _mm256_setzero_si256()),
          C_zero_point_epi16_v);
      x_packed_v = _mm256_packus_epi16(x_packed_v, _mm256_setzero_si256());
      __m256i x_clamped_v = _mm256_max_epu8(
          FUSE_RELU ? C_zero_point_epi8_v : min_v,
          _mm256_min_epu8(x_packed_v, max_v));

      /*
       * x_clamped_v has results in the following layout so we need to
       * permute: x0-3 garbage0-11 x4-7 garbage12-23
       */
      x_clamped_v = _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

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

      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(out + i * ld_out + j),
          _mm256_castsi256_si128(x_clamped_v));
    } // j loop vectorized

    int remainder = block.col_start + block.col_size - j;
    assert(remainder == 0);
  } // i loop
}

#define INSTANTIATE_REQUANTIZE_BIAS_TYPE(                                      \
    A_SYM, B_SYM, Q_GRAN, BIAS, RELU, BIAS_TYPE)                               \
  template void FBGEMM_API                                                     \
  requantizeOutputProcessingAvx2<A_SYM, B_SYM, Q_GRAN, BIAS, RELU, BIAS_TYPE>( \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);                             \
  template void requantizeOutputProcessingGConvAvx2<                           \
      A_SYM,                                                                   \
      B_SYM,                                                                   \
      Q_GRAN,                                                                  \
      BIAS,                                                                    \
      RELU,                                                                    \
      2,                                                                       \
      BIAS_TYPE>(                                                              \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);                             \
  template void requantizeOutputProcessingGConvAvx2<                           \
      A_SYM,                                                                   \
      B_SYM,                                                                   \
      Q_GRAN,                                                                  \
      BIAS,                                                                    \
      RELU,                                                                    \
      4,                                                                       \
      BIAS_TYPE>(                                                              \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);                             \
  template void requantizeOutputProcessingGConvAvx2<                           \
      A_SYM,                                                                   \
      B_SYM,                                                                   \
      Q_GRAN,                                                                  \
      BIAS,                                                                    \
      RELU,                                                                    \
      8,                                                                       \
      BIAS_TYPE>(                                                              \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);                             \
  template void requantizeOutputProcessingGConvAvx2<                           \
      A_SYM,                                                                   \
      B_SYM,                                                                   \
      Q_GRAN,                                                                  \
      BIAS,                                                                    \
      RELU,                                                                    \
      16,                                                                      \
      BIAS_TYPE>(                                                              \
      uint8_t * out,                                                           \
      const int32_t* inp,                                                      \
      const block_type_t& block,                                               \
      int ld_out,                                                              \
      int ld_in,                                                               \
      const requantizationParams_t<BIAS_TYPE>& r);

#define INSTANTIATE_REQUANTIZE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU)              \
  INSTANTIATE_REQUANTIZE_BIAS_TYPE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU, float)   \
  INSTANTIATE_REQUANTIZE_BIAS_TYPE(A_SYM, B_SYM, Q_GRAN, BIAS, RELU, int32_t) \
  template void requantizeForFloatAvx2<A_SYM, B_SYM, Q_GRAN, BIAS, RELU>(     \
      float* out,                                                             \
      const int32_t* inp,                                                     \
      const block_type_t& block,                                              \
      int ld_out,                                                             \
      int ld_in,                                                              \
      const requantizationForFloatParams_t& r);

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
