/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

namespace fbgemm::detail {

static_assert(
    sizeof(double) == sizeof(uint64_t) &&
    std::numeric_limits<double>::is_iec559 &&
    std::numeric_limits<double>::digits == 53);
static_assert(
    sizeof(float) == sizeof(uint32_t) &&
    std::numeric_limits<float>::is_iec559 &&
    std::numeric_limits<float>::digits == 24);

inline uint64_t roundHalfSignificand(uint64_t significand, int shift) {
  const uint64_t retained = significand >> shift;
  const uint64_t remainder = significand & ((uint64_t{1} << shift) - 1);
  const uint64_t midpoint = uint64_t{1} << (shift - 1);
  return retained +
      (remainder > midpoint || (remainder == midpoint && (retained & 1)));
}

inline uint16_t doubleToHalfRne(double value) {
  uint64_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = (bits >> 48) & 0x8000;
  const int exponent = (bits >> 52) & 0x7ff;
  const uint64_t fraction = bits & ((uint64_t{1} << 52) - 1);
  if (exponent == 0x7ff) {
    return sign | (fraction ? 0x7e00 : 0x7c00);
  }
  if (exponent == 0) {
    return sign;
  }

  int halfExponent = exponent - 1023 + 15;
  const uint64_t significand = fraction | (uint64_t{1} << 52);
  if (halfExponent <= 0) {
    const int shift = 43 - halfExponent;
    if (shift > 53) {
      return sign;
    }
    return sign | roundHalfSignificand(significand, shift);
  }
  if (halfExponent >= 31) {
    return sign | 0x7c00;
  }

  uint64_t rounded = roundHalfSignificand(significand, 42);
  if (rounded == 0x800) {
    rounded = 0x400;
    ++halfExponent;
    if (halfExponent == 31) {
      return sign | 0x7c00;
    }
  }
  return sign | (halfExponent << 10) | (rounded & 0x3ff);
}

inline float halfToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  const int exponent = (value >> 10) & 0x1f;
  uint32_t fraction = value & 0x3ff;
  uint32_t bits{};
  if (exponent == 0) {
    if (fraction == 0) {
      bits = sign;
    } else {
      int normalizedExponent = -14;
      while ((fraction & 0x400) == 0) {
        fraction <<= 1;
        --normalizedExponent;
      }
      bits = sign | ((normalizedExponent + 127) << 23) |
          ((fraction & 0x3ff) << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000 | (fraction << 13);
    if (fraction != 0) {
      bits |= 0x00400000;
    }
  } else {
    bits = sign | ((exponent + 112) << 23) | (fraction << 13);
  }
  float result{};
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline float gpuCompatibleInt4Value(
    uint16_t coefficient,
    uint16_t bias,
    int quantized,
    double multiplier) {
  const uint16_t coefficientMagnitude = coefficient & 0x7fff;
  const uint16_t biasMagnitude = bias & 0x7fff;
  if (coefficientMagnitude > 0x7c00 || biasMagnitude > 0x7c00) {
    return halfToFloat(0x7e00);
  }
  if (coefficientMagnitude == 0x7c00) {
    if (quantized == 0 ||
        (biasMagnitude == 0x7c00 && ((coefficient ^ bias) & 0x8000))) {
      return halfToFloat(0x7e00);
    }
    return halfToFloat(coefficient);
  }
  if (biasMagnitude == 0x7c00) {
    return halfToFloat(bias);
  }

  // Finite INT4 operands make this affine value exact in binary64. Its
  // smallest nonzero unit is 2^-24 and its magnitude is less than 2^17.
  const double value = (quantized / multiplier) * halfToFloat(coefficient) +
      static_cast<double>(halfToFloat(bias));
  if (value == 0) {
    const uint16_t sign =
        (quantized == 0 || coefficientMagnitude == 0) && biasMagnitude == 0
        ? coefficient & bias & 0x8000
        : 0;
    return halfToFloat(sign);
  }
  return halfToFloat(doubleToHalfRne(value));
}

// Entries [0, 16) decode low nibbles; entries [16, 32) decode high nibbles.
// This mirrors dequantize_permuted_int4's Half multiply and fused Half FMA.
inline std::array<float, 32> makeGpuCompatibleInt4Lut(
    uint16_t scale,
    uint16_t bias) {
  std::array<float, 32> result{};
  const double scaleValue = halfToFloat(scale);
  const uint16_t lowCoefficient = doubleToHalfRne(scaleValue * 512.0);
  for (int q = 0; q < 16; ++q) {
    result[q] = gpuCompatibleInt4Value(lowCoefficient, bias, q, 512.0);
  }
  // Power-of-two coefficient scaling is exact for Half until overflow,
  // including subnormals. Both finite paths therefore evaluate q*scale+bias.
  if ((lowCoefficient & 0x7fff) < 0x7c00) {
    std::copy_n(result.begin(), 16, result.begin() + 16);
    return result;
  }
  const uint16_t highCoefficient = doubleToHalfRne(scaleValue * 32.0);
  for (int q = 0; q < 16; ++q) {
    result[16 + q] = gpuCompatibleInt4Value(highCoefficient, bias, q, 32.0);
  }
  return result;
}

} // namespace fbgemm::detail
