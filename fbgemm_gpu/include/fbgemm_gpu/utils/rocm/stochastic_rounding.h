/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
  const auto random_bits = state.rand4();
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
  const auto random_bits = state.rand4();
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
  const auto random_bits = state.rand4();
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
  const auto random_bits = state.rand4();
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

template <>
DEVICE_INLINE void nearest_rounding_vector(
    at::Float8_e4m3fnuz* output,
    const Vec2T<at::Half>& value,
    const float2 /* Not used yet */) {
  __nv_fp8x2_e4m3* fp8_ptr = reinterpret_cast<__nv_fp8x2_e4m3*>(output);
  fp8_ptr[0] = static_cast<__nv_fp8x2_e4m3>(value.acc);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    at::Float8_e4m3fnuz* output,
    const Vec2T<float>& value,
    StochasticRoundingRNGState& /* state */,
    const float2 /* qparams */) {
  // TODO, make this actually stochastic later.
  __nv_fp8x2_e4m3* fp8_ptr = reinterpret_cast<__nv_fp8x2_e4m3*>(output);
  fp8_ptr[0] = static_cast<__nv_fp8x2_e4m3>(value.acc);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    at::Float8_e4m3fnuz* output,
    const Vec2T<at::Half>& value,
    StochasticRoundingRNGState& /* state */,
    const float2 /* qparams */) {
  // TODO, make this stochastic later.
  __nv_fp8x2_e4m3* fp8_ptr = reinterpret_cast<__nv_fp8x2_e4m3*>(output);
  fp8_ptr[0] = static_cast<__nv_fp8x2_e4m3>(value.acc);
}

// Scalar-array weight write for the optimized HIP optimizer update path.
//
// Writes `thread_length` accumulator values (`value`, in cache_t/fp32) into the
// reduced-precision `output` storage applying stochastic rounding (SR). SR is
// only representable for at::Half on this scalar path (it relies on
// stochastic_rounding_scalar, which yields __half); for any other emb_t this
// falls back to a plain round-to-nearest cast. The RNG state is taken by
// reference (the on/off decision is made by the caller); pass a register-local
// copy so rand4() operates on registers.
//
// Reusable by any optimizer's update() that writes a per-lane scalar array.
template <typename emb_t, typename cache_t, int32_t thread_length>
DEVICE_INLINE void stochastic_rounding_store_vector(
    emb_t* output,
    const cache_t* value,
    StochasticRoundingRNGState& state) {
  if constexpr (std::is_same_v<emb_t, at::Half>) {
#pragma unroll
    for (int32_t i = 0; i < thread_length; i += 4) {
      const uint4 random_bits = state.rand4();
      const uint32_t bits[4] = {
          random_bits.x, random_bits.y, random_bits.z, random_bits.w};
#pragma unroll
      for (int32_t j = 0; j < 4; j++) {
        if (i + j < thread_length) {
          output[i + j] = stochastic_rounding_scalar(
              static_cast<float>(value[i + j]), bits[j]);
        }
      }
    }
  } else {
    // SR is not representable for non-Half on this scalar path: plain cast.
#pragma unroll
    for (int32_t i = 0; i < thread_length; i++) {
      output[i] = static_cast<emb_t>(value[i]);
    }
  }
}

// Scalar-array weight write with plain round-to-nearest (no SR). Used for the
// SR-disabled case and for fp32 weights (where SR is a no-op).
template <typename emb_t, typename cache_t, int32_t thread_length>
DEVICE_INLINE void nearest_rounding_store_vector(
    emb_t* output,
    const cache_t* value) {
#pragma unroll
  for (int32_t i = 0; i < thread_length; i++) {
    output[i] = static_cast<emb_t>(value[i]);
  }
}

} // namespace fbgemm_gpu::rocm
