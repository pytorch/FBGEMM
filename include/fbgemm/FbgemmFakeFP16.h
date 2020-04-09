/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/Types.h"
#include "fbgemm/Utils.h"

// Turning on this option will print out time breakdown of each stage (e.g.,
// pre-processing fp32 -> fp16 conversion, the main GEMM kernel, post-processing
// fp16 -> fp32 conversion). Please note that currently this option won't report
// accurate timing if multiple threads are used. #define
// FBGEMM_MEASURE_TIME_BREAKDOWN

// #define FBGEMM_MEASURE_TIME_BREAKDOWN

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
#include <chrono>
#include <iostream>
extern double malloc_time;
extern double A_fp16_to_fp32_time;
extern double B_fp16_to_fp32_time;
extern double C_fp16_to_fp32_time;
extern double computing_time;
extern double C_fp32_to_fp16_time;
extern double run_time;
#endif

namespace fbgemm {

// typedef uint16_t float16;

/**
 * @ Transform all entries in a matrix from fp32 to float16 and back to fp32
 *
 */
FBGEMM_API void RoundToFloat16(
    const float* input,
    float* output,
    int len,
    bool clamp = false,
    bool clamp_denorms = false);

FBGEMM_API void fbgemmFakeFP16(
    const matrix_op_t transa,
    const matrix_op_t transb,
    int m,
    int n,
    int k,
    const float16* A_float16,
    const float16* B_float16,
    float beta,
    float16* C_float16);

}; // namespace fbgemm
