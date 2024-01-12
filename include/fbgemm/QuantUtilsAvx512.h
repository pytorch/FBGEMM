/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include "./FbgemmBuild.h"
#include "./UtilsAvx2.h"

/// @defgroup fbgemm-quant-utils-avx512 Quantization Utilities (AVX512)
///

namespace fbgemm {

/// @ingroup fbgemm-quant-utils-avx512
///
/// Requantize with AVX512.
template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    int C_PER_G,
    typename BIAS_TYPE = std::int32_t>
FBGEMM_API void requantizeOutputProcessingGConvAvx512(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t<BIAS_TYPE>& r);
} // namespace fbgemm
