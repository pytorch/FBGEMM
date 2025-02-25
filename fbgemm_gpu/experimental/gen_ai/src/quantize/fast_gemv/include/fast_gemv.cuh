// @lint-ignore-every LICENSELINT

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// MIT License

// Copyright (c) 2023 Siping Wang

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
 * The source code contained in this file is pulled from original
 * github repo: https://github.com/wangsiping97/FastGEMV.
 */

#ifndef FAST_GEMV_CUH_
#define FAST_GEMV_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/float8.h>

#include "utility.cuh"

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

///////////////////////////// GEMV //////////////////////////////
__global__ void gemv_bf16(
    __nv_bfloat16* mat,
    __nv_bfloat16* vec,
    __nv_bfloat16* res,
    unsigned int n,
    unsigned int num_per_thread);

__global__ void gemv_quantized_bf16_fp8(
    cutlass::float_e4m3_t* mat,
    __nv_bfloat16* vec,
    __nv_bfloat16* res,
    unsigned int n,
    float const* scale,
    unsigned int num_per_thread);

__global__ void gemv_quantized_fp8_fp8(
    cutlass::float_e4m3_t* mat,
    cutlass::float_e4m3_t* vec,
    __nv_bfloat16* res,
    unsigned int n,
    float const* scale,
    unsigned int num_per_thread);

__global__ void gemv_quantized_int4(
    uint4_2* mat,
    half* vec,
    half* res,
    unsigned int n,
    half scale,
    half zero_point,
    unsigned int num_per_thread);

///////////////////////////// REDUCE SUM //////////////////////////////
__device__ __forceinline__ float warpReduceSum(
    float sum,
    unsigned int threadNum);

#endif // FAST_GEMV_CUH_
