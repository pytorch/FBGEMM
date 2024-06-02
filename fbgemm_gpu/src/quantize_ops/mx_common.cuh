/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "mx/common.cuh"

//-----------------------------------------------------------------------
// MX4-Float mapping
//-----------------------------------------------------------------------

__device__ const float MX4_values[16] = {
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

//-----------------------------------------------------------------------
// Misc. helper functions
//-----------------------------------------------------------------------

inline uint32_t align(int a, int b) {
  return (a + b - 1) / b * b;
}

// Refactor to use FBGEMM's
__host__ __device__ __forceinline__ uint32_t round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b);
}

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
      (trailing_mantissa >> 22) | (new_biased_exp << 1) | (sign << 3);
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&f_4bit);
  return *ptr;
}
