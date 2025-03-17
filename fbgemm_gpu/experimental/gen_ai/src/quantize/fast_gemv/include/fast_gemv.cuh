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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cutlass/cutlass.h>
#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>
#include <driver_functions.h>

#include "utility.cuh"

#define MAX_M_SIZE 4
#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

using SizeType32 = std::size_t;

///////////////////////////// REDUCE_SUM /////////////////////////
__device__ __forceinline__ float warpReduceSum(
    float sum,
    unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
  return sum;
}

///////////////////////////// GEMV //////////////////////////////
__global__ void gemv_bf16(
    __nv_bfloat16* mat,
    __nv_bfloat16* vec,
    __nv_bfloat16* res,
    const unsigned int k,
    const unsigned int m,
    const unsigned int n,
    unsigned int num_per_thread);

__global__ void gemv_quantized_bf16_fp8(
    cutlass::float_e4m3_t* mat,
    __nv_bfloat16* vec,
    __nv_bfloat16* res,
    const unsigned int k,
    const unsigned int m,
    const unsigned int n,
    float const* scale,
    unsigned int num_per_thread);

///////////////////////////// QUANTIZED-FLOAT8 //////////////////////////////
template <SizeType32 TILE_M, SizeType32 TILE_N, SizeType32 BLOCK_DIM_X>
__global__ void gemv_quantized_fp8_fp8(
    cutlass::float_e4m3_t* mat,
    cutlass::float_e4m3_t* vec,
    __nv_bfloat16* res,
    const unsigned int k,
    const unsigned int m,
    const unsigned int n,
    float const* scale,
    unsigned int num_iter_per_thread) {
  float sum[TILE_N][TILE_M] = {{0.0f}, {0.0f}};
  const auto tid = threadIdx.x;
  float4* mat8 = reinterpret_cast<float4*>(mat);
  float4* vec8 = reinterpret_cast<float4*>(vec);
  cutlass::NumericArrayConverter<float, cutlass::float_e4m3_t, 4> converter;
  cutlass::Array<float, 4> mat_elements[TILE_N][4];

#pragma unroll
  for (SizeType32 iter = 0; iter < num_iter_per_thread; iter++) {
    unsigned int j = tid + iter * BLOCK_DIM_X;
    if (j < k >> 4) {
#pragma unroll
      for (SizeType32 i = 0; i < TILE_N; i++) {
        auto row = TILE_N * blockIdx.y + i;
        auto mat_val = mat8[row * (k >> 4) + j]; // float4
        mat_elements[i][0] = converter(
            reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>&>(
                mat_val.x));
        mat_elements[i][1] = converter(
            reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>&>(
                mat_val.y));
        mat_elements[i][2] = converter(
            reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>&>(
                mat_val.z));
        mat_elements[i][3] = converter(
            reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>&>(
                mat_val.w));
      }

#pragma unroll
      for (SizeType32 col = 0; col < TILE_M; col++) {
        auto vec_val = vec8[col * (k >> 4) + j]; // float4
        cutlass::Array<float, 4> vec_elements[4];
        vec_elements[0] = converter(
            reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>&>(
                vec_val.x));
        vec_elements[1] = converter(
            reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>&>(
                vec_val.y));
        vec_elements[2] = converter(
            reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>&>(
                vec_val.z));
        vec_elements[3] = converter(
            reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>&>(
                vec_val.w));
#pragma unroll
        for (SizeType32 i = 0; i < TILE_N; i++) {
#pragma unroll
          for (SizeType32 idx = 0; idx < 16; idx++) {
            SizeType32 c = idx / 4, l = idx % 4;
            sum[i][col] =
                fma(vec_elements[c][l], mat_elements[i][c][l], sum[i][col]);
          }
        }
      }
    }
  }

  static constexpr SizeType32 numWarps = BLOCK_DIM_X / WARP_SIZE;
  // Shared mem for partial sums (one per warp in the block)
  __shared__ float warpLevelSums[TILE_M * TILE_N][numWarps];
  SizeType32 laneId = tid % WARP_SIZE;
  SizeType32 warpId = tid / WARP_SIZE;

#pragma unroll
  for (SizeType32 i = 0; i < TILE_N; i++) {
#pragma unroll
    for (SizeType32 col = 0; col < TILE_M; col++) {
      sum[i][col] *= (*scale);
      sum[i][col] = warpReduceSum(sum[i][col], BLOCK_DIM_X);
      if (laneId == 0)
        warpLevelSums[i * TILE_M + col][warpId] = sum[i][col];
    }
  }

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
#pragma unroll
      for (SizeType32 ni = 0; ni < TILE_N; ni++) {
        auto row = TILE_N * blockIdx.y + ni;
#pragma unroll
        for (SizeType32 mi = 0; mi < TILE_M; mi++) {
          res[row + mi * n] = __float2bfloat16(sum[ni][mi]);
        }
      }
      return;
    }
  }

  __syncthreads();

  assert(TILE_M * TILE_N < BLOCK_DIM_X);

  if (tid < TILE_M * TILE_N) {
    SizeType32 row = tid / TILE_M + TILE_N * blockIdx.y;
    SizeType32 col = tid % TILE_M;
    float val = 0;
#pragma unroll
    for (SizeType32 s = 0; s < numWarps; s++) {
      val += warpLevelSums[tid][s];
    }
    res[row + col * n] = __float2bfloat16(val);
  }
}

__global__ void gemv_quantized_int4(
    uint4_2* mat,
    half* vec,
    half* res,
    unsigned int n,
    half scale,
    half zero_point,
    unsigned int num_per_thread);

#endif // FAST_GEMV_CUH_
