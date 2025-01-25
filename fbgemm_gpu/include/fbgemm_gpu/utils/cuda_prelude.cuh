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

#ifdef __HIP_PLATFORM_AMD__
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h> // @manual
#else
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#endif
#include <cassert>

namespace {

inline int get_device_sm_cnt_() {
#ifdef __HIP_PLATFORM_AMD__
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, c10::hip::current_device());
  return deviceProp.multiProcessorCount;
#else
  cudaDeviceProp* deviceProp =
      at::cuda::getDeviceProperties(c10::cuda::current_device());
  return deviceProp->multiProcessorCount;
#endif
}

} // namespace

namespace fbgemm_gpu {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 9000
#define FBGEMM_USE_SUBWARP_SHUFFLE
#endif

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

#define CUDA_DEVICE_GUARD(TENSOR)           \
  at::cuda::OptionalCUDAGuard device_guard; \
  device_guard.set_index(TENSOR.get_device())

#define FBGEMM_CUDA_CHECK(X)               \
  do {                                     \
    cudaError_t err = X;                   \
    assert(err == cudaError::cudaSuccess); \
  } while (0)

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
#if defined(USE_ROCM)
static constexpr uint64_t kFullWarpMask = 0xff'ff'ff'ff'ff'ff'ff'ff;
#else
static constexpr uint32_t kFullWarpMask = 0xff'ff'ff'ff;
#endif

static constexpr float kQParamEps = 1e-8f;

/* For rowwise int8 quantization, two quantization parameters (qparams)
will be stored at the end of each row in FP32 formats, appending a total of
8 bytes to each row.
*/
static constexpr float kINT8QparamsBytes = 8;

template <typename T>
DEVICE_INLINE T shfl_xor(
    const T val,
    int laneMask,
    int width = kWarpSize,
    unsigned shfl_sync_mask = static_cast<unsigned>(kFullWarpMask)) {
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
    unsigned shfl_sync_mask = static_cast<unsigned>(kFullWarpMask)) {
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
    unsigned shfl_sync_mask = static_cast<unsigned>(kFullWarpMask)) {
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
    unsigned shfl_sync_mask = static_cast<unsigned>(kFullWarpMask)) {
#if defined(USE_ROCM) || CUDA_VERSION < 9000
  return __ballot(predicate);
#else
  return __ballot_sync(shfl_sync_mask, predicate);
#endif
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warpReduceAllSum(
    T val,
    unsigned shfl_sync_mask = static_cast<unsigned>(kFullWarpMask)) {
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

// ROCm does not natively support __any_sync(). Using __ballot()
// (https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html)
// to implement __any_sync(). Note: the "warp-size" of AMD GPU is 64.
#ifdef USE_ROCM
__device__ int __any_sync(uint64_t mask, int predicate) {
  uint64_t predicate_bit_pattern = __ballot(predicate);
  return (predicate_bit_pattern & mask) > 0;
}
#endif

__host__ DEVICE_INLINE int32_t div_round_up(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}

static __host__ DEVICE_INLINE int32_t round_up(int32_t a, int32_t b) {
  return ((a + b - 1) / b) * b;
}

__host__ DEVICE_INLINE int32_t round_down(int32_t a, int32_t b) {
  return a / b * b;
}

// Return if the address is aligned to the type (mainly for Vec4T).
template <class T>
DEVICE_INLINE bool is_aligned(const void* ptr) {
  auto iptr = reinterpret_cast<uintptr_t>(ptr);
  return !(iptr % alignof(T));
}

} // namespace fbgemm_gpu
