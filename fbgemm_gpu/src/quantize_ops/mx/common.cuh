// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Microsoft Corporation.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PYT_MX_COMMON_CUH
#define PYT_MX_COMMON_CUH

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

// Max threads per block for CUDA compute capability 2.x - 7.5 is 1024
// Max threads for some CUDA random number generators is 256
#define MAX_THREADS 1024
#define WARP_SIZE 32

#define FLOAT32_EXP_BIAS 127
#define FLOAT32_EXP_MAX 255
#define FLOAT32_TRAILING_MBITS 23
#define FLOAT32_IMPLIED1 (1 << FLOAT32_TRAILING_MBITS)
#define FLOAT32_FULL_MBITS (FLOAT32_TRAILING_MBITS + 1)
#define FLOAT32_INF 0x7fe00000
#define FLOAT32_EXP_OFFSET 23
#define FLOAT32_SIGN_OFFSET 31
#define FLOAT32_EXP_MASK 0x7f800000
#define FLOAT32_MANTISSA_MASK 0x007fffff

#define FLOAT16_MIN_NORMAL_EXP -14
#define FLOAT16_MAX_EXP 15
#define FLOAT16_EXP_BIAS 15

//---------------------------------------------------------
// Helper types/structs
//---------------------------------------------------------
typedef union {
  unsigned int i;
  float f;
} u_float_int;

typedef enum _RoundingMode {
  rd_away = 0, // round nearest, ties to away
  rd_floor = 1, // floor
  rd_even = 2 // round nearest, ties to even
} RoundingMode;

//-----------------------------------------------------------------------
// Bound the shared_exp based on ebits
//-----------------------------------------------------------------------
__host__ __device__ __forceinline__ int clamp_shared_exp(
    int shared_exp,
    const int ebits) {
  // Set overflowing shared exps to NaN and
  // bound underflowing shared exps to -emax
  // Note that (for 8 bits) the min scale is -127, not -126
  int emax = ebits != 0 ? (1 << (ebits - 1)) - 1 : FLOAT32_EXP_MAX;
  int shared_ub = shared_exp - FLOAT32_EXP_BIAS;
  shared_exp = shared_ub > emax ? FLOAT32_EXP_MAX : shared_exp;
  shared_exp = shared_ub < -emax ? FLOAT32_EXP_BIAS - emax : shared_exp;
  return shared_exp;
}

//---------------------------------------------------------
// Helper functions for quantization
//---------------------------------------------------------
__host__ __device__ __forceinline__ int get_sign(const u_float_int input) {
  int sign = input.i >> FLOAT32_SIGN_OFFSET;
  return sign;
}

__host__ __device__ __forceinline__ int get_biased_exponent(
    const u_float_int input) {
  // Mask only exponent bits
  int exp = input.i & FLOAT32_EXP_MASK;
  // Shift down to lowest bits
  exp = exp >> FLOAT32_EXP_OFFSET;
  return exp;
}

__host__ __device__ __forceinline__ int get_biased_exponent(const float input) {
  u_float_int u;
  u.f = input;
  return get_biased_exponent(u);
}

// get_unbiased_exponent supports denorms
__host__ __device__ __forceinline__ int get_unbiased_exponent(
    const float input) {
  u_float_int u;
  u.f = input;
  int exp = get_biased_exponent(u);
  if (exp == 0) {
    // Denorm
    return 1 - FLOAT32_EXP_BIAS;
  } else {
    return exp - FLOAT32_EXP_BIAS;
  }
}

__host__ __device__ __forceinline__ int get_biased_exponent(
    const __half input) {
  u_float_int u;
  u.f = __half2float(input);
  return get_biased_exponent(u);
}

__host__ __device__ __forceinline__ int get_trailing_mantissa(
    const u_float_int input) {
  return input.i & FLOAT32_MANTISSA_MASK;
}

// Construct float from sign, biased exponent, and mantissa
__host__ __device__ __forceinline__ float
construct_float(int sign, int biased_exp, int trailing_mantissa) {
  u_float_int x;
  x.i = trailing_mantissa | (biased_exp << FLOAT32_EXP_OFFSET) |
      (sign << FLOAT32_SIGN_OFFSET);
  return x.f;
}

//---------------------------------------------------------
// Shift right and round a float32 mantissa
// Example of "allow_overflow". Say we are rounding 11111 to 4 bits
// If allow_overflow is False, it will floor the result to 1111
// If allow_overflow is True,  it will round up to 10000, overflowing 4 bits
//---------------------------------------------------------
__host__ __device__ __forceinline__ void shift_right_round_mantissa_mx4(
    int& mantissa, // 23-bit float32 trailing mantissa
    const bool is_subnorm, // is the input a subnorm?
    const int exp_diff // extra right shifts
) {
  // Implied 1
  mantissa = is_subnorm ? mantissa : mantissa + FLOAT32_IMPLIED1;
  const int fp32_sig_bits = is_subnorm ? 23 : 24;

  constexpr int mbits = 2;

  // Adjust for shared exponent and Shift down to target bit width + 1
  mantissa = mantissa >> (exp_diff + fp32_sig_bits - mbits - 1);
  // Rounding using floor(x+1), with overflow check
  mantissa = mantissa + 1;

  // Shift last bit away
  mantissa = mantissa >> 1;
}

__host__ __device__ __forceinline__ void shift_right_round_mantissa(
    int& mantissa, // 23-bit float32 trailing mantissa
    const bool is_subnorm, // is the input a subnorm?
    const int mbits, // number to bits to round to
    const int exp_diff, // extra right shifts
    const RoundingMode rounding_mode,
    const bool allow_overflow = false) {
  // Implied 1
  mantissa = is_subnorm ? mantissa : mantissa + FLOAT32_IMPLIED1;
  int fp32_sig_bits = is_subnorm ? 23 : 24;

  // RNE logic
  bool tie = false;
  bool even = false;
  if (rounding_mode == rd_even) {
    // tbits is the no. of bits that will be removed
    int tbits = exp_diff + (fp32_sig_bits - mbits);
    // 1 at all truncation locations except the first truncation location
    int mask = (1 << (tbits - 1)) - 1;
    // We have a tie only if all the truncation bits except the first
    // one are zero. If the first truncation bit is 1, we have an
    // actual tie. For rounding, we don't care if the first bits
    // is 1 or 0. If it is 0, no rounding happens.
    tie = !(mantissa & mask);
    mask = (1 << tbits); // 1 at the first non-truncated location
    even =
        !(mantissa &
          mask); // True if the last bit before truncation location is 0
  }

  // Adjust for shared exponent
  mantissa = mantissa >> exp_diff;
  // Shift down to target bit width + 1
  mantissa = mantissa >> (fp32_sig_bits - mbits - 1);
  // Rounding using floor(x+1), with overflow check
  if ((rounding_mode == rd_away || rounding_mode == rd_even) &&
      (allow_overflow || mantissa != ((1 << (mbits + 1)) - 1))) {
    if (!(tie && even))
      mantissa = mantissa + 1;
  }
  // Shift last bit away
  mantissa = mantissa >> 1;
}

//---------------------------------------------------------
// Shift back up to restore a float32 mantissa
// Use in pair with shift_right_round_mantissa to check for
// overflows caused by rounding
//---------------------------------------------------------
__host__ __device__ __forceinline__ bool shift_left_mantissa(
    int& mantissa, // From shift_right_round_mantissa
    const bool is_subnorm, // is the input a subnorm?
    const int mbits,
    const int exp_diff) {
  int fp32_sig_bits = is_subnorm ? 23 : 24;
  mantissa = mantissa << (fp32_sig_bits - mbits + exp_diff);
  // Handle overflow.
  // When a subnorm overflows (into a normal) we don't rshift
  const bool overflow = (mantissa >= (1 << fp32_sig_bits));
  mantissa = (overflow && !is_subnorm) ? mantissa >> 1 : mantissa;
  // Remove implied 1
  mantissa = mantissa & (FLOAT32_IMPLIED1 - 1);
  return overflow;
}

#endif
