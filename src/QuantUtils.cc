#include "fbgemm/QuantUtils.h"

#include <cpuinfo.h>
#include <immintrin.h>

#include "fbgemm/Fbgemm.h"

namespace fbgemm {

using namespace std;

float TensorQuantizationParams::Min() const {
  return Dequantize(0, *this);
}

float TensorQuantizationParams::Max() const {
  return Dequantize((1 << precision) - 1, *this);
}

TensorQuantizationParams ChooseQuantizationParams(
    float min,
    float max,
    int32_t qmin,
    int32_t qmax,
    bool preserve_sparsity,
    bool force_scale_power_of_two) {
  if (min < 0 && max > 0 && preserve_sparsity) {
    int symmetric_qmin = -((qmax - qmin) / 2 + 1);
    int symmetric_qmax = (qmax - qmin) / 2;
    double max_scale =
        std::max(fabs(min / symmetric_qmin), fabs(max / symmetric_qmax));
    min = max_scale * symmetric_qmin;
    max = max_scale * symmetric_qmax;
  }

  double scale =
      (std::max(max, 0.f) - std::min(min, 0.f)) / ((double)qmax - qmin);
  if (scale == 0) {
    scale = 0.1;
  }
  // If scale is 0, we arbitrary adjust the scale to 0.1
  assert(scale > 0);

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  if (force_scale_power_of_two) {
    if (scale < 1) {
      scale = 1. / (1 << (int)floor(log2(1 / scale)));
    } else {
      scale = 1 << (int)ceil(log2(scale));
    }
  }

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zero_point_from_min = qmin - min / scale;
  double zero_point_from_max = qmax - max / scale;
  double zero_point_from_min_error = std::abs(qmin) + std::abs(min / scale);
  double zero_point_from_max_error = std::abs(qmax) + std::abs(max / scale);
  double initial_zero_point =
      zero_point_from_min_error < zero_point_from_max_error
      ? zero_point_from_min
      : zero_point_from_max;

  // for symmetric quantization (preserve_sparsity == true), we force zero_point
  // to be a middle value between qmin and qmax.
  // If either min or max is 0, then we just use 0 as zero_point.
  if (min < 0 && max > 0 && preserve_sparsity) {
    initial_zero_point = (qmin + qmax) / 2 + 1;
  }

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with zero
  // padding).
  int32_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = nearbyint(initial_zero_point);
  }

  TensorQuantizationParams result;
  result.scale = scale;
  result.zero_point = nudged_zero_point;
  return result;
}

void ChooseRequantizationMultiplier(
    float real_multiplier,
    int32_t* quantized_multiplier,
    int* right_shift,
    int requantization_multiplier_precision) {
  assert(real_multiplier != 0.f);

  // Assuming requantization_multiplier_precision_ = 31,
  // the default right shift is 31 when the real multiplier is already
  // in interval [1/2, 1).
  // Multiplying a 32-bit signed integer with all 31 bits except the sign bit
  // is used followed by 31-bit right shift implements multiplying with a real
  // number in [1/2, 1).
  // We want to utilize all 31 bits except the sign bit in the 32-bit signed
  // integer to get the best accuracy.
  int s = 31;

  // We want to bring the real multiplier into the interval [1/2, 1).
  // We can do so by multiplying it by two, and recording how many times
  // we multiplied by two so that we can compensate that by a right
  // shift by the same amount.
  if (real_multiplier > 0.f) {
    while (real_multiplier < 0.5f) {
      real_multiplier *= 2.f;
      s++;
    }
    while (real_multiplier > 1.f) {
      real_multiplier /= 2.f;
      s--;
    }
  }
  // Now that the real multiplier is in [1/2, 1), we convert it
  // into a fixed-point number.
  int64_t q = nearbyint(
      real_multiplier * (1ll << (requantization_multiplier_precision - 1)));
  assert(q <= (1ll << (requantization_multiplier_precision - 1)));
  // Handle the special case when the real multiplier was so close to 1
  // that its fixed-point approximation was undistinguishable from 1.
  // We handle this by dividing it by two, and remembering to decrement
  // the right shift amount.
  if (q == (1ll << (requantization_multiplier_precision - 1))) {
    q /= 2;
    s--;
  }
  assert(s >= 0);
  assert(q >= 0);
  assert(q <= numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q);
  *right_shift = s;
  assert(s < 64);
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions

// FIXME: code duplication with PackAWithQuantRowOffset
template <typename T>
void Quantize(
    const float* src,
    T* dst,
    int len,
    const TensorQuantizationParams& qparams) {
#if defined(__AVX2__) && defined(__FMA__)
  bool avx2_support = cpuinfo_has_x86_avx2();
  bool fma_support = cpuinfo_has_x86_fma3();
  if (avx2_support && fma_support && qparams.precision == 8 &&
      std::is_same<T, uint8_t>::value) {
    // fast path
    constexpr int VLEN = 8;
    std::size_t i = 0;
    __m256 inverse_scale_v = _mm256_set1_ps(1.f / qparams.scale);
    for (; i < len / VLEN * VLEN; i += VLEN) {
      __m256 src_v = _mm256_loadu_ps(src + i);
      __m256 transformed_v = _mm256_fmadd_ps(
          src_v, inverse_scale_v, _mm256_set1_ps(qparams.zero_point));
      __m256 clipped_v = _mm256_min_ps(
          _mm256_max_ps(transformed_v, _mm256_set1_ps(0.f)),
          _mm256_set1_ps(255.f));
      __m256i rounded_v = _mm256_cvtps_epi32(clipped_v);
      alignas(64) std::int32_t temp_int32[VLEN];
      _mm256_store_si256((__m256i*)temp_int32, rounded_v);
      for (int j = 0; j < VLEN; ++j) {
        dst[i + j] = temp_int32[j];
      }
    }

    for (; i < len; ++i) {
      float transformed = qparams.zero_point + src[i] / qparams.scale;
      float clipped = std::min(std::max(transformed, 0.f), 255.f);
      // Not exactly the same behavior as the vectorized code.
      // The vectorized code above always rounds to even in halfway cases
      // (https://software.intel.com/en-us/node/523819), but std::nearbyint
      // does the same only when the current rounding mode is FE_TONEAREST.
      // However, in practice, this should not be a problem because most cases
      // use the default rounding mode FE_TONEAREST.
      // Note that we cannot implement the same behavior as the vectorized code
      // using std::round because it does rounding away from zero in halfway
      // cases.
      dst[i] = nearbyint(clipped);
    }
  } else
#endif
  {
    for (std::size_t i = 0; i < len; ++i) {
      dst[i] = Quantize<T>(src[i], qparams);
    }
  }
}

template void Quantize<uint8_t>(
    const float* src,
    uint8_t* dst,
    int len,
    const TensorQuantizationParams& qparams);

template void Quantize<int8_t>(
    const float* src,
    int8_t* dst,
    int len,
    const TensorQuantizationParams& qparams);

template void Quantize<uint16_t>(
    const float* src,
    uint16_t* dst,
    int len,
    const TensorQuantizationParams& qparams);

template void Quantize<int16_t>(
    const float* src,
    int16_t* dst,
    int len,
    const TensorQuantizationParams& qparams);

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
// Requantization (pure fixed-point)

int64_t SaturatingRoundingMulWithShift(int32_t a, int32_t b, int right_shift) {
  int64_t a_64(a);
  int64_t b_64(b);
  int64_t ab_64 = a_64 * b_64;

  int64_t nudge = 1ll << (right_shift - 1);
  return (ab_64 + nudge) >> right_shift;
}

#ifdef __AVX2__
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
    dst[i] = RequantizeFixedPoint<uint8_t>(src[i], params);
  }
}

template <typename T>
void RequantizeFixedPoint(
    const std::int32_t* src,
    T* dst,
    int len,
    const RequantizationParams& params) {
  if (std::is_same<T, uint8_t>::value && params.target_qparams.precision == 8 &&
      cpuinfo_has_x86_avx2()) {
    RequantizeFixedPointAvx2(src, dst, len, params);
  } else {
    for (int i = 0; i < len; ++i) {
      dst[i] = RequantizeFixedPoint<T>(src[i], params);
    }
  }
}

#define FBGEMM_SPECIALIZED_REQUANTIZE(T)                \
  template <>                                           \
  void RequantizeFixedPoint<T>(                         \
      const int32_t* src,                               \
      T* dst,                                           \
      const int len,                                    \
      const RequantizationParams& params) {             \
    for (int i = 0; i < len; ++i) {                     \
      dst[i] = RequantizeFixedPoint<T>(src[i], params); \
    }                                                   \
  }
FBGEMM_SPECIALIZED_REQUANTIZE(uint16_t)
FBGEMM_SPECIALIZED_REQUANTIZE(int32_t)
#undef FBGEMM_SPECIALIZED_REQUANTIZE

template <>
void RequantizeFixedPoint<uint8_t>(
    const int32_t* src,
    uint8_t* dst,
    const int len,
    const RequantizationParams& params) {
  if (params.target_qparams.precision == 8 && cpuinfo_has_x86_avx2()) {
    RequantizeFixedPointAvx2(src, dst, len, params);
  } else {
    for (int i = 0; i < len; ++i) {
      dst[i] = RequantizeFixedPoint<uint8_t>(src[i], params);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Requantization (with floats)

void RequantizeAvx2(
    const int32_t* src,
    uint8_t* dst,
    int len,
    const RequantizationParams& params) {
  DoNothing<> doNothingObj{};
  ReQuantizeOutput<false /* FUSE_RELU */> requantizeObj(
    doNothingObj,
    &params.real_multiplier,
    params.target_qparams.zero_point,
    0,
    0,
    nullptr,
    nullptr,
    nullptr,
    len);
  requantizeObj.f<inst_set_t::avx2>(dst, src, {0, 1, 0, len}, 0, 0);
}
#endif

#define FBGEMM_SPECIALIZED_REQUANTIZE(T)      \
  template <>                                 \
  void Requantize<T>(                         \
      const int32_t* src,                     \
      T* dst,                                 \
      const int len,                          \
      const RequantizationParams& params) {   \
    for (int i = 0; i < len; ++i) {           \
      dst[i] = Requantize<T>(src[i], params); \
    }                                         \
  }
FBGEMM_SPECIALIZED_REQUANTIZE(uint16_t)
FBGEMM_SPECIALIZED_REQUANTIZE(int32_t)
#undef FBGEMM_SPECIALIZED_REQUANTIZE

template <>
void Requantize<uint8_t>(
    const int32_t* src,
    uint8_t* dst,
    const int len,
    const RequantizationParams& params) {
  if (params.target_qparams.precision == 8 && cpuinfo_has_x86_avx2()) {
    RequantizeAvx2(src, dst, len, params);
  } else {
    for (int i = 0; i < len; ++i) {
      dst[i] = Requantize<uint8_t>(src[i], params);
    }
  }
}

} // namespace fbgemm
