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

#include <ATen/ATen.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <cuda.h>

// clang-format off
#include "./cub_namespace_prefix.cuh"
#include "cub/block/block_reduce.cuh"
#include "./cub_namespace_postfix.cuh"
// clang-format on

namespace fbgemm_gpu {

// Kernel for calculating the offsets ranges
template <typename scalar_t>
__global__ void _offsets_range_cuda_kernel(
    int64_t N,
    int64_t range_size,
    const scalar_t* __restrict__ offsets_data,
    scalar_t* __restrict__ range_data);

// Kernel for permuting the lengths. Used for permutation of sparse features.
template <typename index_t>
__global__ void permute_2D_lengths_kernel(
    int32_t T,
    int32_t B,
    const index_t* __restrict__ lengths,
    const int32_t* __restrict__ permute,
    index_t* __restrict__ permuted_lengths);

} // namespace fbgemm_gpu
