/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "mx/common.cuh"

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
      (trailing_mantissa >> 22) | (new_biased_exp << 1) | (sign << 3);
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&f_4bit);
  return *ptr;
}
