/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include "fbgemm_gpu/utils/float.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "mx/common.cuh"

#define FULL_WARP_MASK 0xff'ff'ff'ff

//-----------------------------------------------------------------------
// MX4-Float mapping
//-----------------------------------------------------------------------

__constant__ float MX4_values[16] = {
    0.0f,
    0.5f,
    1.0f,
    1.5f,
    2.0f,
    3.0f,
    4.0f,
    6.0f,
    -0.0f,
    -0.5f,
    -1.0f,
    -1.5f,
    -2.0f,
    -3.0f,
    -4.0f,
    -6.0f};

//---------------------------------------------------------
// Helper functions for quantization
//---------------------------------------------------------

__host__ __device__ __forceinline__ uint8_t
// construct fp4 and store the 4 bit at the end
construct_fp4(
    const uint32_t sign,
    const uint32_t new_biased_exp,
    const uint32_t trailing_mantissa) {
  const uint32_t f_4bit =
      (trailing_mantissa) | (new_biased_exp << 1) | (sign << 3);
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&f_4bit);
  return *ptr;
}

__device__ __forceinline__ uint8_t quantize_elemwise_4bit(
    const float input,
    const int bits, // bits = mantissa bits + sign bit
    const int exp_bits, // exp_bits == 0 indicates integer dtype
    const float max_norm,
    const RoundingMode rounding_mode = rd_away,
    const bool saturate_normals = false,
    const bool allow_denorm = true) {
  u_float_int input_;
  input_.f = input;

  // TODO: Refactor to return unsigned data
  int biased_exp = get_biased_exponent(input_);
  int sign = get_sign(input_);
  int tmant = get_trailing_mantissa(input_);

  // Mantissa bits to quantize to (remove sign)
  const int mbits = bits - 1;
  const bool is_int = exp_bits == 0;

  // Integers can be treated has having exp bias of 1
  const int new_bias = is_int ? 1 : (1 << (exp_bits - 1)) - 1;
  int new_biased_exp = biased_exp - FLOAT32_EXP_BIAS + new_bias;

  // Skip denorms
  if ((!is_int) && (!allow_denorm) && (new_biased_exp < 1)) {
    return 0.0;
  }

  // Use exp_diff to truncate additional bits for subnorms
  // mbits includes implicit 1, so when new_biased_exp==0
  // we want exp_diff = 1 to truncate away 1 bit
  int exp_diff = (new_biased_exp <= 0) ? 1 - new_biased_exp : 0;
  exp_diff = (exp_diff > FLOAT32_FULL_MBITS) ? FLOAT32_FULL_MBITS : exp_diff;

  // Shift down and round mantissa, allow overflow except for integers
  // This converts tmant into a full mantissa
  shift_right_round_mantissa(
      tmant, biased_exp == 0, mbits, exp_diff, rounding_mode, !is_int);

  if (tmant == 0) {
    return 0.0;
  }

  // Shift back up to restore mantissa
  // This converts back to a trailing mantissa
  const bool overflow =
      shift_left_mantissa(tmant, biased_exp == 0, mbits, exp_diff);
  if (overflow) {
    biased_exp = biased_exp + 1;
    new_biased_exp = new_biased_exp + 1;
  }

  // Reconstruct float number
  const float output = construct_float(sign, biased_exp, tmant);

  /* Convert float to MX4 encodings:
    bits  FP4     [int4 lookup]
                    +  - (sign)
    S000 = 0    <=> 0  8
    S001 = 0.5  <=> 1  9
    S010 = 1    <=> 2  10
    S011 = 1.5  <=> 3  11
    S100 = 2.0  <=> 4  12
    S101 = 3.0  <=> 5  13
    S110 = 4.0  <=> 6  14
    S111 = 6.0  <=> 7  15
  */

  // construct the 4 bit using 1-bit sign, 2-bit new_exp 1-bit tmant
  // |0.5f| is the exception since it has tmant of 0 instead of 1
  // return the lookup value
  if (output == 0.5f) {
    return 1; // bits 0001
  } else if (output == -0.5f) {
    return 9; // bits 1001
  }

  // Return Inf if rounded value is out of bounds,
  // unless target format is integer or saturate_normals==True
  if (abs(output) > max_norm) {
    if (is_int || saturate_normals) {
      // max norm = 6.0f => bias=3, tmant = 1, sign remains the same
      new_biased_exp = 3;
      tmant = 4194304; // bit 10000000000000000000000
    } else {
      // TODO: set Inf for 4 bit for other patterns
      new_biased_exp = 0xFF;
      tmant = 0;
      // e2m1 has no inf
      CUDA_KERNEL_ASSERT(false);
    }
  }
  CUDA_KERNEL_ASSERT(new_biased_exp >= 0 && new_biased_exp <= 3);
  return construct_fp4(sign, new_biased_exp, tmant);
}
