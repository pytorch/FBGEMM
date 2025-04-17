
/*******************************************************************************
 * Copyright (c) 2016 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 ******************************************************************************/

#pragma once

#include <hip/hip_fp16.h>

#include <ATen/ATen.h>

#include "fbgemm_gpu/utils/rocm/half2.h"
#include "fbgemm_gpu/utils/rocm/vec2.h"
#include "fbgemm_gpu/utils/stochastic_rounding.cuh"
#include "fbgemm_gpu/utils/types.h"

namespace fbgemm_gpu::rocm {
template <typename dst_t, typename src_t>
DEVICE_INLINE void stochastic_rounding_vector(
    dst_t* output,
    const Vec2T<src_t>& value,
    StochasticRoundingRNGState& state,
    const float2 /* not used */) {
  value.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    at::Half* output,
    const Vec2T<at::Half>& value,
    StochasticRoundingRNGState& state,
    const float2 /* not used */) {
  const uint4 random_bits = stochastic_rounding_rand4(&state);
  Half2 v;
  v.a = __halves2half2(
      stochastic_rounding_scalar(value.acc.x, random_bits.x),
      stochastic_rounding_scalar(value.acc.y, random_bits.y));

  v.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    at::Half* output,
    const Vec2T<float>& value,
    StochasticRoundingRNGState& state,
    const float2 /* not used */) {
  const uint4 random_bits = stochastic_rounding_rand4(&state);
  Half2 v;
  v.a = __halves2half2(
      stochastic_rounding_scalar(value.acc.x, random_bits.x),
      stochastic_rounding_scalar(value.acc.y, random_bits.y));

  v.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    uint8_t* output,
    const Vec2T<float>& value,
    StochasticRoundingRNGState& state,
    const float2 qparams) {
  const uint4 random_bits = stochastic_rounding_rand4(&state);
  const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = stochastic_rounding_scalar_uint8(
      (value.acc.x - qparams.y) * inv_scale, random_bits.x);
  output[1] = stochastic_rounding_scalar_uint8(
      (value.acc.y - qparams.y) * inv_scale, random_bits.y);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    uint8_t* output,
    const Vec2T<at::Half>& value,
    StochasticRoundingRNGState& state,
    const float2 qparams) {
  const uint4 random_bits = stochastic_rounding_rand4(&state);
  const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = stochastic_rounding_scalar_uint8(
      (value.acc.x - qparams.y) * inv_scale, random_bits.x);
  output[1] = stochastic_rounding_scalar_uint8(
      (value.acc.y - qparams.y) * inv_scale, random_bits.y);
}

template <typename dst_t, typename src_t>
DEVICE_INLINE void nearest_rounding_vector(
    dst_t* output,
    const Vec2T<src_t>& value,
    const float2 /* not used */) {
  value.store(output);
}

template <>
DEVICE_INLINE void nearest_rounding_vector(
    uint8_t* output,
    const Vec2T<float>& value,
    const float2 qparams) {
  const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = lrintf((value.acc.x - qparams.y) * inv_scale);
  output[1] = lrintf((value.acc.y - qparams.y) * inv_scale);
}

template <>
DEVICE_INLINE void nearest_rounding_vector(
    uint8_t* output,
    const Vec2T<at::Half>& value,
    const float2 qparams) {
  const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = lrintf((value.acc.x - qparams.y) * inv_scale);
  output[1] = lrintf((value.acc.y - qparams.y) * inv_scale);
}

} // namespace fbgemm_gpu::rocm
