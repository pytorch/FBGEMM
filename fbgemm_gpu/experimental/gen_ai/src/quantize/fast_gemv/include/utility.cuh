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
 * The kernel source code contained in this file is pulled from original
 * github repo: https://github.com/wangsiping97/FastGEMV.
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/float8.h>
#include <stdio.h>

#include <cstdint>

///////////////////////////// DATA TYPES //////////////////////////////

struct uint4_2 {
  uint8_t data;

  uint4_2(uint8_t x = 0, uint8_t y = 0) {
    setX(x);
    setY(y);
  }

  __host__ __device__ uint8_t getX() const {
    return data & 0x0F; // get the lower 4 bits
  }

  __host__ __device__ uint8_t getY() const {
    return (data >> 4) & 0x0F; // get the upper 4 bits
  }

  __host__ __device__ void setX(uint8_t x) {
    data = (data & 0xF0) | (x & 0x0F); // set the lower 4 bits
  }

  __host__ __device__ void setY(uint8_t y) {
    data = (data & 0x0F) | ((y & 0x0F) << 4); // set the upper 4 bits
  }
};

struct half4 {
  half x, y, z, w;
};
struct bfloat16_2 {
  __nv_bfloat16 x, y;
};
struct int8_2 {
  int8_t x, y;
};
struct fp8_2 {
  cutlass::float_e4m3_t x, y;
};
struct uint4_2_4 {
  uint4_2 x, y, z, w;
};

///////////////////////////// CUDA UTILITIES //////////////////////////////

void print_cuda_info();

// Define the error checking function
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void check(
    cudaError_t result,
    char const* const func,
    const char* const file,
    int const line);

__global__ void generate_random_numbers(half* numbers, int Np);
__global__ void generate_random_int8_numbers(int8_t* numbers, int Np);
__global__ void generate_random_int4_numbers(uint4_2* numbers, int Np);

#endif // UTILITY_H_
