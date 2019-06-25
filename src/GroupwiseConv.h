/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>

#include "fbgemm/ConvUtils.h"
#include "fbgemm/QuantUtilsAvx2.h"

namespace fbgemm {

// Compute group-wise convolution for h-th row across 8 groups
template <bool TOP, bool BOTTOM, int SPATIAL_DIM = 2>
void groupConvAvx2(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    int h,
    const std::int8_t* B,
    std::int32_t* C,
    std::int32_t* rowOffsetBuf);

// Compute group-wise convolution for h-th row across 16 groups
template <bool TOP, bool BOTTOM, int SPATIAL_DIM = 2>
void groupConvAvx512(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    int h,
    const std::int8_t* B,
    std::int32_t* C,
    std::int32_t* rowOffsetBuf);

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    int C_PER_G>
void requantizeOutputProcessingGConvAvx512(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t& r);

} // namespace fbgemm
