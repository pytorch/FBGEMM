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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>
#include <driver_functions.h>

#include "fast_gemv.cuh"
#include "utility.cuh"

using SizeType32 = std::size_t;

///////////////////////////// NORMAL //////////////////////////////
// thread_per_block = blockDim.x
// blockDim.y <= SHARED_MEM_MAX_ROWS
__global__ void gemv_bf16(
    __nv_bfloat16* mat,
    __nv_bfloat16* vec,
    __nv_bfloat16* res,
    const unsigned int k,
    const unsigned int m,
    const unsigned int n,
    unsigned int num_per_thread) {
  float sum[MAX_M_SIZE] = {0.0f};
  const auto tid = threadIdx.x;
  const auto row = blockIdx.y * blockDim.y + threadIdx.y;
  const auto start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < k >> 3) {
      const auto mat_val = mat4[row * (k >> 3) + j];
      const bfloat16_2* mat_h1 = (bfloat16_2*)&mat_val.x;
      const bfloat16_2* mat_h2 = (bfloat16_2*)&mat_val.y;
      const bfloat16_2* mat_h3 = (bfloat16_2*)&mat_val.z;
      const bfloat16_2* mat_h4 = (bfloat16_2*)&mat_val.w;
#pragma unroll
      for (int col = 0; col < m; col++) {
        const auto vec_val = vec4[col * (k >> 3) + j];
        const bfloat16_2* vec_h1 = (bfloat16_2*)&vec_val.x;
        const bfloat16_2* vec_h2 = (bfloat16_2*)&vec_val.y;
        const bfloat16_2* vec_h3 = (bfloat16_2*)&vec_val.z;
        const bfloat16_2* vec_h4 = (bfloat16_2*)&vec_val.w;
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h1->x),
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(mat_h1->x),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h1->y),
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(mat_h1->y),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h2->x),
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(mat_h2->x),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h2->y),
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(mat_h2->y),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h3->x),
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(mat_h3->x),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h3->y),
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(mat_h3->y),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h4->x),
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(mat_h4->x),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h4->y),
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(mat_h4->y),
            sum[col]);
      }
    }
  }
#pragma unroll
  for (int col = 0; col < m; col++) {
    sum[col] = warpReduceSum(sum[col], blockDim.x);
  }

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      for (int col = 0; col < m; col++) {
        res[row + col * n] = __float2bfloat16(sum[col]);
      }
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
#pragma unroll
  for (int col = 0; col < m; col++) {
    if (laneId == 0)
      warpLevelSums[threadIdx.y][warpId] = sum[col];
    __syncthreads();
    // read from shared memory only if that warp existed
    sum[col] = (threadIdx.x < blockDim.x / WARP_SIZE)
        ? warpLevelSums[threadIdx.y][laneId]
        : 0.0;
    // Final reduce using first warp
    if (warpId == 0)
      sum[col] = warpReduceSum(sum[col], blockDim.x / WARP_SIZE);
    if (tid == 0) {
      res[row + col * n] = __float2bfloat16(sum[col]);
    }
  }
}

///////////////////////////// QUANTIZED-FLOAT8-MIXED
/////////////////////////////////

__global__ void gemv_quantized_bf16_fp8(
    cutlass::float_e4m3_t* mat,
    __nv_bfloat16* vec,
    __nv_bfloat16* res,
    const unsigned int k,
    const unsigned int m,
    const unsigned int n,
    float const* scale,
    unsigned int num_per_thread) {
  float sum[MAX_M_SIZE] = {0.0f};
  // each thread load num_per_thread elements from global
  const auto tid = threadIdx.x;
  const auto row = blockIdx.y * blockDim.y + threadIdx.y;
  const auto start_idx = threadIdx.x;
  half4* mat4 = reinterpret_cast<half4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < k >> 3) {
      const auto mat_val = mat4[row * (k >> 3) + j];
      const fp8_2* mat_h1 = (fp8_2*)&mat_val.x;
      const fp8_2* mat_h2 = (fp8_2*)&mat_val.y;
      const fp8_2* mat_h3 = (fp8_2*)&mat_val.z;
      const fp8_2* mat_h4 = (fp8_2*)&mat_val.w;
#pragma unroll
      for (int col = 0; col < m; col++) {
        const auto vec_val = vec4[col * (k >> 3) + j];
        const bfloat16_2* vec_h1 = (bfloat16_2*)&vec_val.x;
        const bfloat16_2* vec_h2 = (bfloat16_2*)&vec_val.y;
        const bfloat16_2* vec_h3 = (bfloat16_2*)&vec_val.z;
        const bfloat16_2* vec_h4 = (bfloat16_2*)&vec_val.w;
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h1->x),
            cutlass::NumericConverter<float, cutlass::float_e4m3_t>::convert(
                mat_h1->x),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h1->y),
            cutlass::NumericConverter<float, cutlass::float_e4m3_t>::convert(
                mat_h1->y),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h2->x),
            cutlass::NumericConverter<float, cutlass::float_e4m3_t>::convert(
                mat_h2->x),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h2->y),
            cutlass::NumericConverter<float, cutlass::float_e4m3_t>::convert(
                mat_h2->y),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h3->x),
            cutlass::NumericConverter<float, cutlass::float_e4m3_t>::convert(
                mat_h3->x),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h3->y),
            cutlass::NumericConverter<float, cutlass::float_e4m3_t>::convert(
                mat_h3->y),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h4->x),
            cutlass::NumericConverter<float, cutlass::float_e4m3_t>::convert(
                mat_h4->x),
            sum[col]);
        sum[col] = fma(
            cutlass::NumericConverter<float, __nv_bfloat16>::convert(vec_h4->y),
            cutlass::NumericConverter<float, cutlass::float_e4m3_t>::convert(
                mat_h4->y),
            sum[col]);
      }
    }
  }
#pragma unroll
  for (int col = 0; col < m; col++) {
    sum[col] *= (*scale);
    sum[col] = warpReduceSum(sum[col], blockDim.x);
  }

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
#pragma unroll
      for (int col = 0; col < m; col++) {
        res[row + col * n] = __float2bfloat16(sum[col]);
      }
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
#pragma unroll
  for (int col = 0; col < m; col++) {
    if (laneId == 0)
      warpLevelSums[threadIdx.y][warpId] = sum[col];
    __syncthreads();
    // read from shared memory only if that warp existed
    sum[col] = (threadIdx.x < blockDim.x / WARP_SIZE)
        ? warpLevelSums[threadIdx.y][laneId]
        : 0.0;
    // Final reduce using first warp
    if (warpId == 0)
      sum[col] = warpReduceSum(sum[col], blockDim.x / WARP_SIZE);
    if (tid == 0) {
      res[row + col * n] = __float2bfloat16(sum[col]);
    }
  }
}

///////////////////////////// QUANTIZED-INT4 //////////////////////////////

// based on previous experiments, num_per_thread can >= 16
__global__ void gemv_quantized_int4(
    uint4_2* mat,
    half* vec,
    half* res,
    unsigned int n,
    half scale,
    half zero_point,
    unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  uint4_2_4* mat4 = reinterpret_cast<uint4_2_4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 4; iter++) {
    unsigned int j = 2 * (start_idx + iter * blockDim.x);
    if (j < n >> 3) {
      float4 vec_val_1 = vec4[j]; // 8 half
      float4 vec_val_2 = vec4[j + 1];
      const bfloat16_2* vec_h1 = (bfloat16_2*)&vec_val_1.x;
      const bfloat16_2* vec_h2 = (bfloat16_2*)&vec_val_1.y;
      const bfloat16_2* vec_h3 = (bfloat16_2*)&vec_val_1.z;
      const bfloat16_2* vec_h4 = (bfloat16_2*)&vec_val_1.w;
      const bfloat16_2* vec_h5 = (bfloat16_2*)&vec_val_2.x;
      const bfloat16_2* vec_h6 = (bfloat16_2*)&vec_val_2.y;
      const bfloat16_2* vec_h7 = (bfloat16_2*)&vec_val_2.z;
      const bfloat16_2* vec_h8 = (bfloat16_2*)&vec_val_2.w;

      uint4_2_4 mat_val_1 = mat4[row * (n >> 3) + j];
      uint4_2_4 mat_val_2 = mat4[row * (n >> 3) + j + 1];
      const uint4_2* mat_h1 = (uint4_2*)&mat_val_1.x;
      const uint4_2* mat_h2 = (uint4_2*)&mat_val_1.y;
      const uint4_2* mat_h3 = (uint4_2*)&mat_val_1.z;
      const uint4_2* mat_h4 = (uint4_2*)&mat_val_1.w;
      const uint4_2* mat_h5 = (uint4_2*)&mat_val_2.x;
      const uint4_2* mat_h6 = (uint4_2*)&mat_val_2.y;
      const uint4_2* mat_h7 = (uint4_2*)&mat_val_2.z;
      const uint4_2* mat_h8 = (uint4_2*)&mat_val_2.w;

      sum += static_cast<float>(vec_h1->x) *
          (static_cast<float>(mat_h1->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h1->y) *
          (static_cast<float>(mat_h1->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h2->x) *
          (static_cast<float>(mat_h2->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h2->y) *
          (static_cast<float>(mat_h2->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h3->x) *
          (static_cast<float>(mat_h3->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h3->y) *
          (static_cast<float>(mat_h3->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h4->x) *
          (static_cast<float>(mat_h4->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h4->y) *
          (static_cast<float>(mat_h4->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h5->x) *
          (static_cast<float>(mat_h5->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h5->y) *
          (static_cast<float>(mat_h5->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h6->x) *
          (static_cast<float>(mat_h6->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h6->y) *
          (static_cast<float>(mat_h6->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h7->x) *
          (static_cast<float>(mat_h7->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h7->y) *
          (static_cast<float>(mat_h7->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h8->x) *
          (static_cast<float>(mat_h8->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h8->y) *
          (static_cast<float>(mat_h8->getY()) - zero_point_f);
    }
  }

  sum *= scale_f;

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0)
    warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
      ? warpLevelSums[threadIdx.y][laneId]
      : 0.0;
  // Final reduce using first warp
  if (warpId == 0)
    sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}
