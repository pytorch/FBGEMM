#include "fbgemm/QuantUtils.h"

#include <cpuinfo.h>

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

#define FBGEMM_SPECIALIZED_QUANTIZE(T)           \
  template <>                                    \
  void Quantize<T>(                              \
      const float* src,                          \
      T* dst,                                    \
      const int len,                             \
      const TensorQuantizationParams& qparams) { \
    for (int i = 0; i < len; ++i) {              \
      dst[i] = Quantize<T>(src[i], qparams);     \
    }                                            \
  }
FBGEMM_SPECIALIZED_QUANTIZE(int8_t)
FBGEMM_SPECIALIZED_QUANTIZE(uint16_t)
FBGEMM_SPECIALIZED_QUANTIZE(int16_t)
FBGEMM_SPECIALIZED_QUANTIZE(int32_t)
#undef FBGEMM_SPECIALIZED_QUANTIZE

template <>
void Quantize<uint8_t>(
    const float* src,
    uint8_t* dst,
    int len,
    const TensorQuantizationParams& qparams) {
  bool avx2_support = cpuinfo_initialize() && fbgemmHasAvx2Support();
  bool fma_support = cpuinfo_has_x86_fma3();
  if (avx2_support && fma_support && qparams.precision == 8) {
    // fast path
    QuantizeAvx2(src, dst, len, qparams);
  } else {
    for (std::size_t i = 0; i < len; ++i) {
      dst[i] = Quantize<uint8_t>(src[i], qparams);
    }
  }
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
  if (params.target_qparams.precision == 8 && cpuinfo_initialize() &&
      fbgemmHasAvx2Support()) {
    RequantizeAvx2(src, dst, len, params);
  } else {
    for (int i = 0; i < len; ++i) {
      dst[i] = Requantize<uint8_t>(src[i], params);
    }
  }
}

template <typename T>
void RequantizeFixedPoint(
    const std::int32_t* src,
    T* dst,
    int len,
    const RequantizationParams& params) {
  if (std::is_same<T, uint8_t>::value && params.target_qparams.precision == 8 &&
      cpuinfo_initialize() && fbgemmHasAvx2Support()) {
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
  if (params.target_qparams.precision == 8 && cpuinfo_initialize() &&
      fbgemmHasAvx2Support()) {
    RequantizeFixedPointAvx2(src, dst, len, params);
  } else {
    for (int i = 0; i < len; ++i) {
      dst[i] = RequantizeFixedPoint<uint8_t>(src[i], params);
    }
  }
}

} // namespace fbgemm
