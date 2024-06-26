/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

#include <cuda.h>
#if !(                                                  \
    defined(USE_ROCM) ||                                \
    ((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#elif (defined(USE_ROCM))
#include <hip/hip_bfloat16.h>
#endif
#include <cuda_fp16.h>

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 9000
#define FBGEMM_USE_SUBWARP_SHUFFLE
#endif

namespace {

int get_device_sm_cnt_() {
  cudaDeviceProp* deviceProp =
      at::cuda::getDeviceProperties(c10::cuda::current_device());
  return deviceProp->multiProcessorCount;
}

} // namespace

namespace fbgemm_gpu {

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

#define CUDA_DEVICE_GUARD(TENSOR)           \
  at::cuda::OptionalCUDAGuard device_guard; \
  device_guard.set_index(TENSOR.get_device())

// Warp size
#ifdef USE_ROCM
static constexpr int32_t kWarpSize = 64;
#else
static constexpr int32_t kWarpSize = 32;
#endif

// Max thread num in one thread block
static constexpr int32_t kMaxThreads = 1024;

// Max block size in Y dimension of a grid
static constexpr int32_t kMaxBlockYDim = 65535;

// Max block size in Z dimension of a grid
static constexpr int32_t kMaxBlockZDim = 65535;

// Full warp mask
static constexpr uint32_t kFullWarpMask = 0xff'ff'ff'ff;

static constexpr float kQParamEps = 1e-8f;

/* For rowwise int8 quantization, two quantization parameters (qparams)
will be stored at the end of each row in FP32 formats, appending a total of
8 bytes to each row.
*/
static constexpr float kINT8QparamsBytes = 8;

// Customized Half4 data types with two half2 (64-bit in total)
struct Half4 {
  half2 a;
  half2 b;

  __device__ inline void store(at::Half* p) {
#ifdef USE_ROCM
    p[0] = __low2half(a);
    p[1] = __high2half(a);
    p[2] = __low2half(b);
    p[3] = __high2half(b);
#elif CUDA_VERSION >= 9000

#ifndef __HALF2_TO_UI
// cuda_fp16.hpp doesn't export this
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif

    asm("st.v2.u32 [%0], {%1, %2};"
        :
        : "l"(p), "r"(__HALF2_TO_UI(a)), "r"(__HALF2_TO_UI(b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(a.x), "r"(b.x));
#endif
  }
};

template <typename T>
DEVICE_INLINE T shfl_xor(
    const T val,
    int laneMask,
    int width = kWarpSize,
    unsigned shfl_sync_mask = kFullWarpMask) {
#if defined(USE_ROCM) || CUDA_VERSION < 9000
  return __shfl_xor(val, laneMask, width);
#else
  return __shfl_xor_sync(shfl_sync_mask, val, laneMask, width);
#endif
}

template <typename T>
DEVICE_INLINE T shfl_sync(
    const T val,
    int srcLane = 0,
    int width = kWarpSize,
    unsigned shfl_sync_mask = kFullWarpMask) {
#if defined(USE_ROCM) || CUDA_VERSION < 9000
  return __shfl(val, srcLane, width);
#else
  return __shfl_sync(shfl_sync_mask, val, srcLane, width);
#endif
}

template <typename T>
DEVICE_INLINE T shfl_down_sync(
    const T val,
    unsigned delta,
    int width = kWarpSize,
    unsigned shfl_sync_mask = kFullWarpMask) {
#if defined(USE_ROCM) || CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(shfl_sync_mask, val, delta, width);
#endif
}

#if defined(USE_ROCM) || CUDA_VERSION < 9000
DEVICE_INLINE uint64_t ballot_sync(
#else
DEVICE_INLINE uint32_t ballot_sync(
#endif
    int predicate,
    unsigned shfl_sync_mask = kFullWarpMask) {
#if defined(USE_ROCM) || CUDA_VERSION < 9000
  return __ballot(predicate);
#else
  return __ballot_sync(shfl_sync_mask, predicate);
#endif
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T
warpReduceAllSum(T val, unsigned shfl_sync_mask = kFullWarpMask) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val += shfl_xor(val, mask, ReduceWidth, shfl_sync_mask);
  }
  return val;
}

DEVICE_INLINE void syncwarp() {
#ifdef USE_ROCM
  // Performance - replace a block level __syncthreads with per CU
  // __threadfence_block. It is a fine replacement for __syncwarp on AMD GPUs,
  // it is because a. memory fencing: __threadfence_block ops. at CU level,
  // same as __syncwarp at SM b. threads re-converge: wavefront run in
  // lockstep, no need __syncwarp re-converge
  __threadfence_block();
#else
  __syncwarp();
#endif
}

} // namespace fbgemm_gpu
