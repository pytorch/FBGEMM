/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm> // for min and max
#include <cassert>
#include <cmath>
#include <cmath> // for lrintf and sqrt
#include <cstdint>
#include <type_traits> // for is_same

#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/UtilsAvx2.h"

namespace fbgemm {

// Almost same as ReQuantizeOutput in OutputProcessing-inh.h but different
// row_offsets for each row because of depth-wise convolution

template <
    bool FUSE_RELU,
    bool HAS_BIAS,
    QuantizationGranularity Q_GRAN,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    int K_PER_G,
    typename BIAS_TYPE>
static ALWAYS_INLINE void requantize_i8dw_ref_(
    std::int32_t A_zero_point,
    const std::int32_t* B_zero_point,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    const std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    int n,
    int j, // starting index
    const std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias [[maybe_unused]],
    const float* act_times_w_scale = nullptr) {
  for (; j < n; ++j) {
    std::int32_t raw = C_int32[j];
    int quant_param_idx = 0;
    if constexpr (
        Q_GRAN == QuantizationGranularity::OUT_CHANNEL ||
        (Q_GRAN == QuantizationGranularity::GROUP && K_PER_G == 1)) {
      quant_param_idx = j;
    } else if constexpr (Q_GRAN == QuantizationGranularity::GROUP) {
      quant_param_idx = j / 2;
    }
    if constexpr (!B_SYMMETRIC) {
      raw -= B_zero_point[quant_param_idx] * row_offsets[j / K_PER_G];
    }
    if constexpr (!A_SYMMETRIC) {
      raw -= A_zero_point * col_offsets[j];
    }
    float raw_f = NAN;
    if constexpr (HAS_BIAS) { // static if
      if constexpr (std::is_same_v<BIAS_TYPE, float>) {
        raw_f = raw;
        raw_f += bias[j] / act_times_w_scale[quant_param_idx];
      } else {
        raw += bias[j];
        raw_f = raw;
      }
    } else {
      raw_f = raw;
    }

    float ab = raw_f * C_multiplier[quant_param_idx];
    long rounded = lrintf(ab) + C_zero_point;

    C_uint8[j] = std::max(
        FUSE_RELU ? static_cast<long>(C_zero_point) : 0l,
        std::min(255l, rounded));
  }
}

static inline std::pair<int, int> closest_factors_(int n) {
  int a = static_cast<int>(std::sqrt(n));
  while (n % a != 0) {
    a--;
  }
  return {a, n / a}; // a <= n / a
}

} // namespace fbgemm

#include "FbgemmI8DepthwiseAvx2-inl.h"
#include "FbgemmI8DepthwiseNeon-inl.h"
