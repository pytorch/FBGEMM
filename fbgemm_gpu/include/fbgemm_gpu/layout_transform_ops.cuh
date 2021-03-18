/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cuda.h>

#include "./fbgemm_cuda_utils.cuh"


// Kernel for recat the embedding gradient output with the mixed dimension
// support
template <typename scalar_t>
__global__ void recat_copy_async_kernel(
    const int64_t* __restrict__ dim_sum_per_rank, // 1D, T
    const int64_t* __restrict__ cum_dim_sum_per_rank, // 1D, T
    const scalar_t* __restrict__ go, // 2D, B x sum(mixed_D)
    scalar_t* __restrict__ sgo, // 1D, B * sum(mixed_D)
    const int64_t T,
    const int64_t B,
    const int64_t dim_sum) {
  auto b_t = blockIdx.x * blockDim.y + threadIdx.y;
  auto b = b_t % B;
  auto t = b_t / B;

  if (b_t >= B * T) {
    return;
  }
  auto dim_current = dim_sum_per_rank[t];
  const auto tgt_base_addr = B * cum_dim_sum_per_rank[t];
  const auto src_base_addr = cum_dim_sum_per_rank[t];

  if (fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(
          &sgo[tgt_base_addr + b * dim_current]) &&
      fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(
          &go[src_base_addr + b * dim_sum])) {
    int32_t d_base = dim_current / 4 * 4;
    for (int32_t d = threadIdx.x * 4; d < d_base; d += blockDim.x * 4) {
      fbgemm_gpu::Vec4T<scalar_t>::copy(
          &go[src_base_addr + b * dim_sum + d],
          &sgo[tgt_base_addr + b * dim_current + d]);
    }
    // Use elementwise access for the last incomplete vector.
    for (int32_t d_left = threadIdx.x; d_base + d_left < dim_current;
         d_left += blockDim.x) {
      sgo[tgt_base_addr + b * dim_current + d_base + d_left] =
          go[src_base_addr + b * dim_sum + d_base + d_left];
    }
  } else {
    for (int32_t d = threadIdx.x; d < dim_current; d += blockDim.x) {
      sgo[tgt_base_addr + b * dim_current + d] =
          go[src_base_addr + b * dim_sum + d];
    }
  }
}
