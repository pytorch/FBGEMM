/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "./Types.h"

#ifndef __is_identifier
#define __is_identifier(x) 1
#endif

#define __has_keyword(__x) !(__is_identifier(__x))

// TODO: we're disabling native fp16 on Windows to workaround test failures
// due to "undefined symbol __gnu_h2f_ieee" error. We should follup on this
// later.
#if __has_keyword(__fp16) && !defined(_WIN32)
#define HAS_NATIVE_FP16_TYPE
using native_fp16_t = __fp16;
#elif __has_keyword(_Float16) && !defined(_WIN32)
#define HAS_NATIVE_FP16_TYPE
using native_fp16_t = _Float16;
#else
using native_fp16_t = void;
#endif

namespace fbgemm {

namespace detail {

template <typename T, int ExponentBits, bool HasInfinity = true>
struct FloatFormat {
  using value_type = T;
  static constexpr int bits = sizeof(T) * CHAR_BIT;
  static constexpr int exponent_bits = ExponentBits;
  static constexpr int mantissa_bits = bits - exponent_bits - 1;
  static constexpr int sign_bit_pos = bits - 1;
  static constexpr int exponent_bias = (1 << (exponent_bits - 1)) - 1;
  static constexpr int unbiased_exponent_min = -exponent_bias + 1;
  static constexpr int unbiased_exponent_max =
      HasInfinity ? exponent_bias : (exponent_bias + 1);
  static constexpr T sign_bit = T{1} << sign_bit_pos;
  static constexpr T exponent_mask = ((T{1} << exponent_bits) - 1)
      << mantissa_bits;
  static constexpr T mantissa_mask = (T{1} << mantissa_bits) - 1;
  // signaling/quiet encoding is unspecified by IEEE754. This mirrors x86/ARM.
  static constexpr T quiet_nan_bit = T{1} << (mantissa_bits - 1);

  static constexpr T nan = exponent_mask | mantissa_mask;
  static constexpr T overflow_value = HasInfinity ? exponent_mask : nan;
  static constexpr bool has_infinity = HasInfinity;
  static constexpr bool has_nan_payload = HasInfinity;
};

using IEEE754Single = FloatFormat</*T=*/uint32_t, /*ExponentBits=*/8>;
using IEEE754Half = FloatFormat</*T=*/uint16_t, /*ExponentBits=*/5>;
// See https://arxiv.org/abs/1905.12322v3
using BFloat16 = FloatFormat</*T=*/uint16_t, /*ExponentBits=*/8>;
// See https://doi.org/10.48550/arXiv.2209.05433
using FP8_E5M2 = FloatFormat</*T=*/uint8_t, /*ExponentBits=*/5>;
// See https://doi.org/10.48550/arXiv.2209.05433
using FP8_E4M3FN = FloatFormat<
    /*T=*/uint8_t,
    /*ExponentBits=*/4,
    /*HasInfinity=*/false>;

enum class RoundingMode {
  ToNearestTiesToEven,
  ToZero,
};

// Generic IEEE754 truncation algorithm.
template <typename Src, typename Tgt, RoundingMode RoundingMode>
[[gnu::always_inline]] inline typename Tgt::value_type ieee754_trunc(
    typename Src::value_type value) {
  static_assert(Src::exponent_bits >= Tgt::exponent_bits);
  static_assert(Src::mantissa_bits > Tgt::mantissa_bits);
  using ST = typename Src::value_type;
  using TT = typename Tgt::value_type;

  ST src_exponent = value & Src::exponent_mask;
  ST src_mantissa = value & Src::mantissa_mask;
  // Fast-path: If there is no difference in exponent sizes (e.g. fp32 -> bf16)
  // and we round toward zero, then we can just drop the least significant bits.
  if constexpr (
      Src::exponent_bits == Tgt::exponent_bits && Src::has_infinity &&
      Tgt::has_infinity && RoundingMode == RoundingMode::ToZero) {
    TT result = value >> (Src::bits - Tgt::bits);
    // Turn signaling NaN into quiet NaN. This also avoids that the mantissa
    // is completely zero after truncation (which would be misinterpreted as
    // INF).
    if (src_exponent == Src::exponent_mask && src_mantissa != 0) {
      result |= Tgt::quiet_nan_bit;
    }
    return result;
  }

  ST tgt_sign =
      (value & Src::sign_bit) >> (Src::sign_bit_pos - Tgt::sign_bit_pos);
  constexpr bool denormal_becomes_zero =
      Tgt::unbiased_exponent_min - Src::unbiased_exponent_min >
      Src::mantissa_bits - Tgt::mantissa_bits;
  if constexpr (denormal_becomes_zero) {
    // Fast-path for zero exponentbits: This means the number was zero or a
    // denormal number that will turn into zero in the Tgt format.
    if (src_exponent == 0) {
      return tgt_sign; // tgt_exponent == 0, tgt_mantissa == 0
    }
  }

  int unbiased_exponent =
      (src_exponent >> Src::mantissa_bits) - Src::exponent_bias;
  if (unbiased_exponent < Tgt::unbiased_exponent_min) {
    int shift = Tgt::unbiased_exponent_min - unbiased_exponent;
    if (shift <= Tgt::mantissa_bits + 1) {
      // Result is denormal.
      ST src_mantissa_one = src_mantissa;
      // Add explicit one if the source was not denormal.
      if (denormal_becomes_zero || src_exponent != 0) {
        src_mantissa_one |= TT{1} << Src::mantissa_bits;
      } else {
        shift--;
      }
      TT tgt_mantissa =
          src_mantissa_one >> (Src::mantissa_bits - Tgt::mantissa_bits + shift);

      if constexpr (RoundingMode == RoundingMode::ToNearestTiesToEven) {
        int half_pos = Src::mantissa_bits - Tgt::mantissa_bits + shift - 1;
        ST half = 1 << half_pos;
        ST remainder = src_mantissa_one & ((half << 1) - 1);
        if (remainder > half ||
            (remainder == half && (tgt_mantissa & 1) != 0)) {
          tgt_mantissa += 1;
        }
      } else {
        assert(RoundingMode == RoundingMode::ToZero);
      }
      return tgt_sign | tgt_mantissa; // tgt_exponent == 0
    } else {
      // Result is +/- zero
      return tgt_sign; // tgt_exponent == 0, tgt_mantissa == 0
    }
  }

  if (unbiased_exponent > Tgt::unbiased_exponent_max) {
    if (unbiased_exponent == Src::exponent_bias + 1 && src_mantissa != 0) {
      TT tgt_mantissa;
      if constexpr (Tgt::has_nan_payload) {
        // NaN; not a number
        tgt_mantissa =
            src_mantissa >> (Src::mantissa_bits - Tgt::mantissa_bits);
        tgt_mantissa |= Tgt::quiet_nan_bit;
      } else {
        tgt_mantissa = Tgt::mantissa_mask;
      }
      return tgt_sign | Tgt::exponent_mask | tgt_mantissa;
    } else {
      if (RoundingMode == RoundingMode::ToZero &&
          (!Src::has_infinity || src_exponent != Src::exponent_mask)) {
        // Return largest finite number.
        return tgt_sign | (Tgt::exponent_mask - Tgt::has_infinity) |
            Tgt::mantissa_mask;
      }
      // Infinity or NaN for formats without infinity.
      return tgt_sign | Tgt::overflow_value;
    }
  }

  // Normal number.
  TT tgt_mantissa = src_mantissa >> (Src::mantissa_bits - Tgt::mantissa_bits);
  TT tgt_exponent = (unbiased_exponent + Tgt::exponent_bias)
      << Tgt::mantissa_bits;
  if constexpr (RoundingMode == RoundingMode::ToNearestTiesToEven) {
    ST half = 1 << (Src::mantissa_bits - Tgt::mantissa_bits - 1);
    ST remainder = src_mantissa & ((half << 1) - 1);
    if (remainder > half || (remainder == half && (tgt_mantissa & 1) != 0)) {
      if (tgt_mantissa < Tgt::mantissa_mask) {
        tgt_mantissa += 1;
      } else {
        // Mantissa overflowed, increment exponent.

        // Normally we can just add to the exponent and will naturally end up
        // on infinity on overflow. But we need special treatments for formats
        // without infinity.
        if (Tgt::has_infinity || tgt_exponent != Tgt::exponent_mask) {
          tgt_mantissa = 0;
          tgt_exponent += TT{1} << Tgt::mantissa_bits;
        } else {
          // Return NaN.
          tgt_mantissa = Tgt::mantissa_mask;
        }
      }
    }
  } else {
    assert(RoundingMode == RoundingMode::ToZero);
  }
  return tgt_sign | tgt_exponent | tgt_mantissa;
}

} // namespace detail

inline float16 cpu_float2half_rn(float f) {
  uint32_t f_u32;
  std::memcpy(&f_u32, &f, sizeof(f_u32));
  return detail::ieee754_trunc<
      /*Src=*/detail::IEEE754Single,
      /*Tgt=*/detail::IEEE754Half,
      detail::RoundingMode::ToNearestTiesToEven>(f_u32);
}

inline float16 cpu_float2half_rz(float f) {
  uint32_t f_u32;
  std::memcpy(&f_u32, &f, sizeof(f_u32));
  return detail::ieee754_trunc<
      /*Src=*/detail::IEEE754Single,
      /*Tgt=*/detail::IEEE754Half,
      detail::RoundingMode::ToZero>(f_u32);
};

// Converts a 16-bit unsigned integer representation of a IEEE754 half-precision
// float into an IEEE754 32-bit single-precision float
inline float cpu_half2float_ref(const float16 h) {
  constexpr uint32_t f16_num_exponent_bits = 5;
  constexpr uint32_t f16_num_mantissa_bits = 10;
  constexpr uint32_t f16_num_non_sign_bits =
      f16_num_exponent_bits + f16_num_mantissa_bits;
  constexpr uint32_t f16_exponent_bias = 15;
  constexpr uint32_t f16_exponent_mask = 0b1'1111;
  constexpr uint32_t f16_mantissa_mask = 0b11'1111'1111;

  constexpr uint32_t f32_num_exponent_bits = 8;
  constexpr uint32_t f32_num_mantissa_bits = 23;
  constexpr uint32_t f32_num_non_sign_bits =
      f32_num_exponent_bits + f32_num_mantissa_bits;
  constexpr uint32_t f32_exponent_bias = 127;
  constexpr uint32_t f32_exponent_mask = 0b1111'1111;
  constexpr uint32_t f32_mantissa_mask = 0x7F'FF'FF;
  constexpr uint32_t f32_most_significant_bit = 1u << 22;

  // Get sign and exponent alone by themselves
  uint32_t sign_bit = (h >> f16_num_non_sign_bits) & 1;
  uint32_t exponent = (h >> f16_num_mantissa_bits) & f16_exponent_mask;
  // Shift mantissa so that it fills the most significant bits of a float32
  uint32_t mantissa = (h & f16_mantissa_mask)
      << (f32_num_mantissa_bits - f16_num_mantissa_bits);

  if (exponent == f16_exponent_mask) { // NaN or Inf
    if (mantissa) {
      mantissa = f32_mantissa_mask;
      sign_bit = 0;
    }
    exponent = f32_exponent_mask;
  } else if (!exponent) { // Denorm or Zero
    if (mantissa) {
      uint32_t msb;
      exponent = f32_exponent_bias - f16_exponent_bias + 1;
      do {
        msb = mantissa & f32_most_significant_bit;
        mantissa <<= 1; // normalize
        --exponent;
      } while (!msb);
      mantissa &= f32_mantissa_mask; // 1.mantissa is implicit
    }
  } else {
    exponent += f32_exponent_bias - f16_exponent_bias;
  }

  const uint32_t i = (sign_bit << f32_num_non_sign_bits) |
      (exponent << f32_num_mantissa_bits) | mantissa;

  float ret;
  std::memcpy(&ret, &i, sizeof(float));
  return ret;
}

// Same as the previous function, but use the built-in fp16 to fp32
// conversion provided by the compiler
inline float cpu_half2float(const float16 h) {
#ifdef HAS_NATIVE_FP16_TYPE
  __fp16 h_fp16;
  std::memcpy(&h_fp16, &h, sizeof(__fp16));
  return h_fp16;
#else
  return cpu_half2float_ref(h);
#endif
}

inline float16 cpu_float2half(const float f) {
#ifdef HAS_NATIVE_FP16_TYPE
  __fp16 h = f;
  float16 res;
  std::memcpy(&res, &h, sizeof(__fp16));
  return res;
#else
  return cpu_float2half_rn(f);
#endif
}

inline float cpu_bf162float(bfloat16 src) {
  float ret;
  uint32_t val_fp32 =
      static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(&src)[0]) << 16;
  std::memcpy(&ret, &val_fp32, sizeof(float));
  return ret;
}

inline bfloat16 cpu_float2bfloat16(float src) {
  uint32_t temp;
  std::memcpy(&temp, &src, sizeof(uint32_t));
  return (temp + (1u << 15)) >> 16;
}

} // namespace fbgemm
