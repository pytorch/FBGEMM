/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifndef __is_identifier
#define __is_identifier(x) 1
#endif

#define __has_keyword(__x) !(__is_identifier(__x))

// TODO: we're disabling native fp16 on Windows to workaround test failures
// due to "undefined symbol __gnu_h2f_ieee" error. We should follup on this
// later.
#if __has_keyword(__fp16) && !defined(_WIN32)
#define HAS_NATIVE_FP16_TYPE
typedef __fp16 native_fp16_t;
#elif __has_keyword(_Float16) && !defined(_WIN32)
#define HAS_NATIVE_FP16_TYPE
typedef _Float16 native_fp16_t;
#else
typedef void native_fp16_t;
#endif

namespace fbgemm {

using float16 = std::uint16_t;
using bfloat16 = std::uint16_t;

// The IEEE754 standard species a binary16 as having the following format:
// SEEEEEMMMMMMMMMM
// 0432109876543210
// That is:
//  *  1 sign bit
//  *  5 exponent bits
//  * 10 mantissa/significand bits (an 11th bit is implicit)
constexpr uint32_t f16_num_bits = 16;
constexpr uint32_t f16_num_exponent_bits = 5;
constexpr uint32_t f16_num_mantissa_bits = 10;
constexpr uint32_t f16_num_non_sign_bits =
    f16_num_exponent_bits + f16_num_mantissa_bits;
constexpr uint32_t f16_exponent_mask = 0b1'1111; // 5 bits
constexpr uint32_t f16_sign_bit = 1u
    << (f16_num_exponent_bits + f16_num_mantissa_bits);
constexpr uint32_t f16_exponent_bits = f16_exponent_mask
    << f16_num_mantissa_bits;
constexpr uint32_t f16_mantissa_mask = 0b11'1111'1111; // 10 bits
constexpr uint32_t f16_exponent_bias = 15;
constexpr uint32_t f16_nan = 0x7F'FF;

// The IEEE754 standard specifies a binary32 as having:
// SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM
// That is:
//  *  1 sign bit
//  *  8 exponent bits
//  * 23 mantissa/significand bits (a 24th bit is implicit)
constexpr uint32_t f32_num_exponent_bits = 8;
constexpr uint32_t f32_num_mantissa_bits = 23;
constexpr uint32_t f32_exponent_mask = 0b1111'1111; // 8 bits
constexpr uint32_t f32_mantissa_mask = 0x7F'FF'FF; // 23 bits
constexpr uint32_t f32_exponent_bias = 127;
constexpr uint32_t f32_all_non_sign_mask = 0x7F'FF'FF'FF; // 31 bits
constexpr uint32_t f32_most_significant_bit = 1u << 22; // Turn on 23rd bit
constexpr uint32_t f32_num_non_sign_bits =
    f32_num_exponent_bits + f32_num_mantissa_bits;

// Round to nearest even
static inline float16 cpu_float2half_rn(float f) {
  static_assert(
      sizeof(uint32_t) == sizeof(float),
      "Programming error sizeof(uint32_t) != sizeof(float)");

  uint32_t* xp = reinterpret_cast<uint32_t*>(&f);
  uint32_t x = *xp;
  uint32_t u = (x & f32_all_non_sign_mask);

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    return static_cast<float16>(f16_nan);
  }

  uint32_t sign = ((x >> f16_num_bits) & f16_sign_bit);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    return static_cast<float16>(sign | f16_exponent_bits);
  }
  if (u < 0x33000001) {
    return static_cast<float16>(sign | 0x0000);
  }

  uint32_t exponent = ((u >> f32_num_mantissa_bits) & f32_exponent_mask);
  uint32_t mantissa = (u & f32_mantissa_mask);

  uint32_t shift;
  if (exponent > f32_exponent_bias - f16_exponent_bias) {
    shift = f32_num_mantissa_bits - f16_num_mantissa_bits;
    exponent -= f32_exponent_bias - f16_exponent_bias;
  } else {
    shift = (f32_exponent_bias - 1) - exponent;
    exponent = 0;
    mantissa |=
        (1u
         << f32_num_mantissa_bits); // Bump the least significant exponent bit
  }
  const uint32_t lsb = (1u << shift);
  const uint32_t lsb_s1 = (lsb >> 1);
  const uint32_t lsb_m1 = (lsb - 1);

  // Round to nearest even.
  const uint32_t remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & f16_mantissa_mask)) {
      ++exponent;
      mantissa = 0;
    }
  }

  return static_cast<float16>(
      sign | (exponent << f16_num_mantissa_bits) | mantissa);
}

// Round to zero
static inline float16 cpu_float2half_rz(float f) {
  static_assert(
      sizeof(uint32_t) == sizeof(float),
      "Programming error sizeof(uint32_t) != sizeof(float)");

  const uint32_t* xp = reinterpret_cast<uint32_t*>(&f);
  const uint32_t x = *xp;
  const uint32_t u = (x & f32_all_non_sign_mask);

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    return static_cast<float16>(f16_nan);
  }

  uint32_t sign = ((x >> f16_num_bits) & f16_sign_bit);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    return static_cast<float16>(sign | f16_exponent_bits);
  }
  if (u < 0x33000001) {
    return static_cast<float16>(sign | 0x0000);
  }

  uint32_t exponent = ((u >> f32_num_mantissa_bits) & f32_exponent_mask);
  uint32_t mantissa = (u & f32_mantissa_mask);

  uint32_t shift;
  if (exponent > f32_exponent_bias - f16_exponent_bias) {
    shift = f32_num_mantissa_bits - f16_num_mantissa_bits;
    exponent -= f32_exponent_bias - f16_exponent_bias;
  } else {
    shift = (f32_exponent_bias - 1) - exponent;
    exponent = 0;
    mantissa |=
        (1u
         << f32_num_mantissa_bits); // Bump the least significant exponent bit
  }

  // Round to zero.
  mantissa >>= shift;

  return static_cast<float16>(
      sign | (exponent << f16_num_mantissa_bits) | mantissa);
}

// Converts a 16-bit unsigned integer representation of a IEEE754 half-precision
// float into an IEEE754 32-bit single-precision float
inline float cpu_half2float_ref(const float16 h) {
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

static inline float cpu_bf162float(bfloat16 src) {
  float ret;
  uint32_t val_fp32 =
      static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(&src)[0]) << 16;
  memcpy(&ret, &val_fp32, sizeof(float));
  return ret;
}

static inline bfloat16 cpu_float2bfloat16(float src) {
  uint32_t temp;
  memcpy(&temp, &src, sizeof(uint32_t));
  return (temp + (1u << 15)) >> 16;
}

inline int64_t round_up(int64_t val, int64_t unit) {
  return (val + unit - 1) / unit * unit;
}

inline int64_t div_up(int64_t val, int64_t unit) {
  return (val + unit - 1) / unit;
}

} // namespace fbgemm
