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

#include <ATen/ATen.h>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/rocm/stochastic_rounding.h"
#include "fbgemm_gpu/utils/rocm/vec2.h"
#include "fbgemm_gpu/utils/stochastic_rounding.cuh"

namespace fbgemm_gpu::rocm {
template <typename dst_t, typename src_t>
DEVICE_INLINE void quantize_store(
    dst_t* output,
    const Vec2T<src_t>& value,
    StochasticRoundingRNGState* state,
    const float2 qparams) {
  if (!state) {
    nearest_rounding_vector<dst_t, src_t>(output, value, qparams);
  } else {
    stochastic_rounding_vector<dst_t, src_t>(output, value, *state, qparams);
  }
}

template <typename dst_t, typename src_t>
DEVICE_INLINE Vec2T<dst_t> dequantize_load(
    const src_t* value,
    const float2 /* unused */) {
  return Vec2T<dst_t>(value);
}

template <>
DEVICE_INLINE Vec2T<float> dequantize_load(
    const uint8_t* value,
    const float2 qparams) {
  Vec2T<float> out;
  out.acc.x = value[0] * qparams.x + qparams.y;
  out.acc.y = value[1] * qparams.x + qparams.y;

  return out;
}

template <>
DEVICE_INLINE Vec2T<at::Half> dequantize_load(
    const uint8_t* value,
    const float2 qparams) {
  Vec2T<at::Half> out;
  out.acc.x = value[0] * qparams.x + qparams.y;
  out.acc.y = value[1] * qparams.x + qparams.y;

  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Weight Row Accessor for Vec2T
////////////////////////////////////////////////////////////////////////////////

template <typename emb_t, typename cache_t, typename dst_t, bool uses_cache>
struct WeightRowAccessorVec2 {
  const emb_t* row_;
  const cache_t* cache_row_;
  const int dim_;

  DEVICE_INLINE
  WeightRowAccessorVec2(
      const emb_t* row,
      const cache_t* cache_row,
      const int dim)
      : row_(row), cache_row_(cache_row), dim_(dim) {}

  DEVICE_INLINE Vec2T<dst_t> load(const int32_t d, const float2 qparams) const {
    if constexpr (uses_cache) {
      return rocm::dequantize_load<dst_t, cache_t>(cache_row_ + d, qparams);
    } else {
      return rocm::dequantize_load<dst_t, emb_t>(row_ + d, qparams);
    }
  }

  DEVICE_INLINE float2 load_qparams() const {
    if constexpr (std::is_same_v<emb_t, uint8_t>) {
      return load_qparams_from_row<emb_t>(row_ + dim_);
    } else {
      return make_float2(0.0f, 0.0f);
    }
  }
};

} // namespace fbgemm_gpu::rocm
