/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mutex>

#include "dispatch_macros.h"
#include "embedding_common.h"
#include "fbgemm_cuda_utils.cuh"
#include "sparse_ops_utils.h"

#define SHFL_SYNC(val, srcLane) \
  shfl_sync(val, srcLane, kThreadGroupSize, shfl_sync_mask)

constexpr size_t kBackwardMaxThreads = 512;
constexpr int32_t kCacheLocationMissing = -1;

DEVICE_INLINE int64_t gpuAtomicIncrement(int64_t* p) {
  static_assert(
      sizeof(int64_t) == sizeof(unsigned long long),
      "expected int64_t to be unsigned long long");
  return static_cast<int64_t>(atomicAdd(
      reinterpret_cast<unsigned long long int*>(p),
      static_cast<unsigned long long int>(1)));
}

namespace fbgemm_gpu {
namespace {

// Based on the empirical study, max grid size that is 64x larger than the
// number of SMs gives good performance across the board
constexpr int MAX_THREAD_BLOCKS_FACTOR = 64;

int get_max_thread_blocks_() {
  return MAX_THREAD_BLOCKS_FACTOR *
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}
} // namespace
} // namespace fbgemm_gpu

__global__
    __launch_bounds__(fbgemm_gpu::kMaxThreads) void split_embedding_backward_codegen_find_long_segments(
        const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            sorted_linear_indices_num_runs,
        const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            sorted_linear_indices_run_lengths,
        at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            long_run_ids,
        at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            num_long_run_ids,
        at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            long_run_id_to_really_long_run_ids,
        at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            num_really_long_run_ids,
        at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            grad_accum_counter,
        const int32_t max_segment_length_per_warp,
        const int32_t max_segment_length_per_cta,
        const bool use_deterministic_algorithms);

template <typename grad_t>
__global__ __launch_bounds__(fbgemm_gpu::kMaxThreads) void grad_mean_kernel(
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,

    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output_mean);
