/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __aarch64__

#include <cstdint>
#include "./FbgemmBuild.h" // @manual

/// @defgroup fbgemm-quant-utils-avx2 Quantization Utilities (AVX2)
///

namespace fbgemm {

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

template <typename InputType>
void FloatOrHalfToFused8BitRowwiseQuantizedSBFloatNeon(
    const InputType* input,
    size_t input_rows,
    int input_columns,
    uint8_t* output);

template <typename OutputType>
void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfNeon(
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    OutputType* output);

template <typename InputType, int BIT_RATE>
void FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfNeon(
    const InputType* input,
    size_t input_rows,
    int input_columns,
    std::uint8_t* output);

template <typename OutputType, int BIT_RATE>
void FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfNeon(
    const std::uint8_t* input,
    size_t input_rows,
    int input_columns,
    OutputType* output);

} // namespace fbgemm

#endif // __aarch64__
