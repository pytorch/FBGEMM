/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
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

#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"

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
