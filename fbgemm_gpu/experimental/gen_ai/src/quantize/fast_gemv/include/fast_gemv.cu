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
    unsigned int num_per_thread) {
  float sum[TILE_N][TILE_M] = {{0.0f}, {0.0f}};
  // each thread load num_per_thread elements from global
  const auto tid = threadIdx.x;
  float4* mat8 = reinterpret_cast<float4*>(mat);
  float4* vec8 = reinterpret_cast<float4*>(vec);
  cutlass::NumericArrayConverter<float, cutlass::float_e4m3_t, 4> converter;
  cutlass::Array<float, 4> mat_elements[TILE_N][4];

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 4; iter++) {
    unsigned int j = tid + iter * BLOCK_DIM_X;
    if (j < k >> 4) {
#pragma unroll
      for (int i = 0; i < TILE_N; i++) {
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
      for (int col = 0; col < TILE_M; col++) {
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
        for (int i = 0; i < TILE_N; i++) {
#pragma unroll
          for (int idx = 0; idx < 16; idx++) {
            int c = idx / 4;
            int l = idx % 4;
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
  for (int i = 0; i < TILE_N; i++) {
#pragma unroll
    for (int col = 0; col < TILE_M; col++) {
      sum[i][col] *= (*scale);
      sum[i][col] = warpReduceSum(sum[i][col], BLOCK_DIM_X);
      if (laneId == 0)
        warpLevelSums[i * TILE_M + col][warpId] = sum[i][col];
    }
  }

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
#pragma unroll
      for (int i = 0; i < TILE_N; i++) {
        auto row = TILE_N * blockIdx.y + i;
#pragma unroll
        for (int col = 0; col < TILE_M; col++) {
          res[row + col * n] = __float2bfloat16(sum[i][col]);
        }
      }
      return;
    }
  }

  __syncthreads();

  if (tid < TILE_M * TILE_N) {
    SizeType32 row = tid / TILE_M + TILE_N * blockIdx.y;
    SizeType32 col = tid % TILE_M;
    float val = 0;
#pragma unroll
    for (int s = 0; s < numWarps; s++) {
      val += warpLevelSums[tid][s];
    }
    res[row + col * n] = __float2bfloat16(val);
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

///////////////////////////// REDUCE SUM //////////////////////////////

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

template __global__ void gemv_quantized_fp8_fp8<1ul, 2ul, 128ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<2ul, 2ul, 128ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<3ul, 2ul, 128ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<4ul, 2ul, 128ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<1ul, 2ul, 64ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<2ul, 2ul, 64ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<3ul, 2ul, 64ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<4ul, 2ul, 64ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<1ul, 2ul, 32ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<2ul, 2ul, 32ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<3ul, 2ul, 32ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
template __global__ void gemv_quantized_fp8_fp8<4ul, 2ul, 32ul>(
    cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*,
    __nv_bfloat16*,
    unsigned int,
    unsigned int,
    unsigned int,
    float const*,
    unsigned int);
