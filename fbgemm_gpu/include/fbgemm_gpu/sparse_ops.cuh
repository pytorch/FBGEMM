/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#ifdef __HIP_PLATFORM_HCC__
#define HIPCUB_ARCH 1
#endif

#include <cuda.h>

// clang-format off
#include "./cub_namespace_prefix.cuh"
#include "cub/block/block_reduce.cuh"
#include "./cub_namespace_postfix.cuh"
// clang-format on

// Kernel for calculating the offsets ranges
template <typename scalar_t>
__global__ void _offsets_range_cuda_kernel(
    int64_t N,
    int64_t range_size,
    const scalar_t* __restrict__ offsets_data,
    scalar_t* __restrict__ range_data) {
  int start_row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int row_idx = start_row_idx; row_idx < N; row_idx += stride) {
    scalar_t row_start = offsets_data[row_idx];
    scalar_t row_end =
        (row_idx < N - 1 ? offsets_data[row_idx + 1] : range_size);
    if (blockDim.x == 32) {
      scalar_t i = row_start - (row_start & 31) + threadIdx.x;
      // unaligned part
      if (i >= row_start && i < row_end) {
        range_data[i] = i - row_start;
      }
      // aligned part
      for (i += 32; i < row_end; i += 32) {
        range_data[i] = i - row_start;
      }
    } else {
      for (scalar_t i = row_start + threadIdx.x; i < row_end; i += blockDim.x) {
        range_data[i] = i - row_start;
      }
    }
  }
}

// Kernel for permuting the lengths. Used for permutation of sparse features.
template <typename index_t>
__global__ void permute_lengths_kernel(
    int32_t T,
    int32_t B,
    const index_t* __restrict__ lengths,
    const int32_t* __restrict__ permute,
    index_t* __restrict__ permuted_lengths) {
  int32_t b_t_start = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    permuted_lengths[b_t] = lengths[permute[t] * B + b];
  }
}
