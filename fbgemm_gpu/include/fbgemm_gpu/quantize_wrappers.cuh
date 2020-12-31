/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

namespace fbgemm_gpu_test {
// FP32 -> UINT8 RowWise
void FloatToFused8BitRowwiseQuantized(
    const int32_t nrows,
    const int32_t ncols,
    const float* __restrict__ input,
    uint8_t* __restrict__ output);
// UINT8 RowWise -> FP32
void Fused8BitRowwiseQuantizedToFloat(
    const int32_t nrows,
    const int32_t ncols,
    const uint8_t* __restrict__ input,
    float* __restrict__ output);
// FP32 -> UINT4/2 RowWise
void FloatToFusedNBitRowwiseQuantizedSBHalf(
    const int32_t nrows,
    const int32_t ncols,
    const int32_t bit_rate,
    const float* __restrict__ input,
    uint8_t* __restrict__ output);
// UINT4/2 RowWise -> FP32
void FusedNBitRowwiseQuantizedSBHalfToFloat(
    const int32_t nrows,
    const int32_t ncols,
    const int32_t bit_rate,
    const uint8_t* __restrict__ input,
    float* __restrict__ output);
} // namespace fbgemm_gpu_test
