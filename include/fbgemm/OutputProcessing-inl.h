/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

template <typename outT, typename inT, typename nextOPType>
template <inst_set_t instSet>
inline int memCopy<outT, inT, nextOPType>::f(
    outT* out,
    inT* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in) const {
  static_assert(
      std::is_same<outT, inT>::value,
      "input and output data type must be of same type");
  // only copy if destination is not the same as source
  if (out + block.row_start * ld_out + block.col_start != inp) {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      memcpy(
          out + block.col_start + i * ld_out,
          inp + (i - block.row_start) * ld_in,
          block.col_size * sizeof(inT));
    }
  }
  return nextop_.template f<instSet>(out, out, block, ld_out, ld_out);
}

template <typename outT, typename inT, typename nextOPType>
template <inst_set_t instSet>
inline int DoSpmdmOnInpBuffer<outT, inT, nextOPType>::f(
    outT* out,
    inT* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in) const {
  assert(B_csc_.NumOfCols() % groups_ == 0);
  int n_per_group = B_csc_.NumOfCols() / groups_;
  int g = block.col_start / n_per_group;
  B_csc_.SpMDM(block, A_ + g * B_csc_.NumOfRows(), lda_, true, inp, ld_in);
  return nextop_.template f<instSet>(out, inp, block, ld_out, ld_in);
}

template <bool FUSE_RELU, typename outT, typename inT, typename nextOPType>
template <inst_set_t instSet>
inline int ReQuantizeOutput<FUSE_RELU, outT, inT, nextOPType>::f(
    outT* out,
    inT* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in) const {
  static_assert(
      std::is_same<inT, int32_t>::value,
      "input data type must be of int32_t type");
  if (instSet == inst_set_t::anyarch) {
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
        inT raw = inp[(i - block.row_start) * ld_in + (j - block.col_start)];
        raw -= Aq_zero_point_ * q_col_offsets_[j];
        if (q_row_offsets_) {
          raw -= q_row_offsets_[i - block.row_start] * Bq_zero_point_;
        }
        if (bias_) {
          raw += bias_[j];
        }

        float ab = raw * C_multiplier_;
        long rounded = std::lrintf(ab) + C_zero_point_;

        out[i * ld_out + j] = std::max(
            FUSE_RELU ? static_cast<long>(C_zero_point_) : 0l,
            std::min(255l, rounded));
      }
    }
  } else if (instSet == inst_set_t::avx2) {
    if (std::is_same<outT, uint8_t>::value) {
      // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
      // using AVX2 instructions
      __m256 multiplier_v = _mm256_set1_ps(C_multiplier_);

      __m256i min_v = _mm256_set1_epi8(std::numeric_limits<uint8_t>::min());
      __m256i max_v = _mm256_set1_epi8(std::numeric_limits<uint8_t>::max());

      __m256i A_zero_point_v = _mm256_set1_epi32(Aq_zero_point_);
      __m256i C_zero_point_epi16_v = _mm256_set1_epi16(C_zero_point_);
      __m256i C_zero_point_epi8_v = _mm256_set1_epi8(C_zero_point_);

      __m256i permute_mask_v =
          _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

      constexpr int VLEN = 8;
      for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
        std::int32_t row_offset = q_row_offsets_
            ? q_row_offsets_[i - block.row_start] * Bq_zero_point_
            : 0;
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

          // if (A_zero_pt != 0) {
          __m256i col_off_v = _mm256_mullo_epi32(
              A_zero_point_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(q_col_offsets_ + j)));
          x_v = _mm256_sub_epi32(x_v, col_off_v);
          col_off_v = _mm256_mullo_epi32(
              A_zero_point_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(q_col_offsets_ + j + VLEN)));
          y_v = _mm256_sub_epi32(y_v, col_off_v);
          col_off_v = _mm256_mullo_epi32(
              A_zero_point_v,
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                  q_col_offsets_ + j + 2 * VLEN)));
          z_v = _mm256_sub_epi32(z_v, col_off_v);
          col_off_v = _mm256_mullo_epi32(
              A_zero_point_v,
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                  q_col_offsets_ + j + 3 * VLEN)));
          w_v = _mm256_sub_epi32(w_v, col_off_v);
          //}

          // if (row_offset != 0) {
          x_v = _mm256_sub_epi32(x_v, row_offset_v);
          y_v = _mm256_sub_epi32(y_v, row_offset_v);
          z_v = _mm256_sub_epi32(z_v, row_offset_v);
          w_v = _mm256_sub_epi32(w_v, row_offset_v);
          //}
          if (bias_) {
            x_v = _mm256_add_epi32(
                x_v,
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(bias_ + j)));
            y_v = _mm256_add_epi32(
                y_v,
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(bias_ + j + VLEN)));
            z_v = _mm256_add_epi32(
                z_v,
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(bias_ + j + 2 * VLEN)));
            w_v = _mm256_add_epi32(
                w_v,
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(bias_ + j + 3 * VLEN)));
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
          __m256 x_scaled_v =
              _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
          __m256 y_scaled_v =
              _mm256_mul_ps(_mm256_cvtepi32_ps(y_v), multiplier_v);
          __m256 z_scaled_v =
              _mm256_mul_ps(_mm256_cvtepi32_ps(z_v), multiplier_v);
          __m256 w_scaled_v =
              _mm256_mul_ps(_mm256_cvtepi32_ps(w_v), multiplier_v);

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
              _mm256_packs_epi32(x_rounded_v, y_rounded_v),
              C_zero_point_epi16_v);
          __m256i zw_packed_v = _mm256_adds_epi16(
              _mm256_packs_epi32(z_rounded_v, w_rounded_v),
              C_zero_point_epi16_v);
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

        for (; j < block.col_start + (block.col_size / VLEN * VLEN);
             j += VLEN) {
          __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
              inp + (i - block.row_start) * ld_in + (j - block.col_start)));

          // if (A_zero_pt != 0) {
          __m256i col_off_v = _mm256_mullo_epi32(
              A_zero_point_v,
              _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(q_col_offsets_ + j)));
          x_v = _mm256_sub_epi32(x_v, col_off_v);
          //}

          // if (row_offset != 0) {
          x_v = _mm256_sub_epi32(x_v, row_offset_v);
          //}
          if (bias_) {
            x_v = _mm256_add_epi32(
                x_v,
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(bias_ + j)));
          }

          __m256 x_scaled_v =
              _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
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
          x_clamped_v =
              _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

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

        for (; j < block.col_start + block.col_size; ++j) {
          int32_t raw =
              inp[(i - block.row_start) * ld_in + (j - block.col_start)];
          // if (A_zero_pt != 0) {
          raw -= Aq_zero_point_ * q_col_offsets_[j];
          //}
          raw -= row_offset;
          if (bias_) {
            raw += bias_[j];
          }

          float ab = raw * C_multiplier_;
          long rounded = std::lrintf(ab) + C_zero_point_;

          out[i * ld_out + j] = std::max(
              FUSE_RELU ? static_cast<long>(C_zero_point_) : 0l,
              std::min(255l, rounded));
        } // j loop remainder
      } // i loop
    } else {
      assert(0 && "Not supported yet");
    }
  } else {
    assert(0 && "Not supported yet");
  }
  return nextop_.template f<instSet>(out, out, block, ld_out, ld_out);
}

template <bool FUSE_RELU, typename outT, typename inT, typename nextOPType>
template <inst_set_t instSet>
inline int ReQuantizeForFloat<FUSE_RELU, outT, inT, nextOPType>::f(
    outT* out,
    inT* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in) const {
  static_assert(
      std::is_same<int32_t, inT>::value,
      "input data type is of not expected type");
  static_assert(
      std::is_same<float, outT>::value,
      "output data type is of not expected type");
  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
      inT raw = inp[(i - block.row_start) * ld_in + j - block.col_start];
      raw -= Aq_zero_point_ * q_col_offsets_[j];
      raw -= q_row_offsets_[i - block.row_start] * Bq_zero_point_;
      float res = raw * Aq_scale_ * Bq_scale_;
      if (bias_) {
        res += bias_[j];
      }
      out[i * ld_out + j] = res;
      if (FUSE_RELU) {
        out[i * ld_out + j] = std::max<outT>(0.0f, out[i * ld_out + j]);
      }
    }
  }

  return nextop_.template f<instSet>(out, out, block, ld_out, ld_out);
}
