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
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits>
#include <mutex>

#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"

#define SHFL_SYNC(val, srcLane) \
  shfl_sync(val, srcLane, kThreadGroupSize, shfl_sync_mask)

constexpr int32_t kCacheLocationMissing = -1;
constexpr size_t kForwardMaxThreads = 512;

namespace nbit {
// "Effective" number of elements in the row when we include the row-wise
// quantization parameters.
__device__ inline int32_t padded_D(
    const int32_t dim,
    const fbgemm_gpu::SparseType weight_ty) {
  switch (weight_ty) {
    case fbgemm_gpu::SparseType::FP32:
      return dim;
    case fbgemm_gpu::SparseType::FP16:
      return dim;
    case fbgemm_gpu::SparseType::FP8:
      return dim;
    case fbgemm_gpu::SparseType::INT8:
      return dim + 4;
    case fbgemm_gpu::SparseType::INT4:
      return dim + 8;
    case fbgemm_gpu::SparseType::INT2:
      return dim + 16;
    default:
      return 0;
  }
}

__device__ inline uint32_t pruned_hash_function(uint32_t h) {
  // MurmorHash3 32-bit mixing function.
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

// ---------------------- START cp.async helpers, copied from CUTLASS

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void* ptr) {
// We prefer to use the new CVTA intrinsics if they are available, otherwise we
// will fall back to the previous internal intrinsics if they are available.
#if (                                                \
    !defined(__clang__) && defined(__CUDA_ARCH__) && \
    __CUDACC_VER_MAJOR__ >= 11)

  //
  // This NVVM intrinsic converts an address in shared memory to a plain
  // unsigned integer. This is necessary to pass to shared memory instructions
  // in inline PTX.
  //
  // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only
  // available in 10.2].

  //
  //__device__ size_t __cvta_generic_to_shared(void* ptr);
  /// CUTLASS helper to get SMEM pointer
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
#elif (                                              \
    !defined(__clang__) && defined(__CUDA_ARCH__) && \
    __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)

  return __nvvm_get_smem_pointer(ptr);
#elif defined(__CUDA_ARCH__)
  uint32_t smem_ptr;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr)
      : "l"(ptr));

  return smem_ptr;
#else
  return 0;

#endif
}

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void const* ptr) {
  return cutlass_get_smem_pointer(const_cast<void*>(ptr));
}

__device__ __forceinline__ void cp_async_fence() {
#if __CUDA_ARCH__ >= 800

  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/// Partial specialization

/// Blocks until all but <N> previous cp.async.commit_group operations have
/// committed.

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800

  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

/// Blocks until all previous cp.async.commit_group operations have committed.
template <>
__device__ __forceinline__ void cp_async_wait<0>() {
#if __CUDA_ARCH__ >= 800

  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

/// Partial specialization
template <int SizeInBytes>
__device__ __forceinline__ void
cp_async_zfill_cg(void* smem_ptr, void const* global_ptr, bool pred_guard) {
#if __CUDA_ARCH__ >= 800
  static_assert(
      SizeInBytes == 16,
      "cp.async only supports CacheOperation::Global when access size is 16B.");

  unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
  int src_in_bytes = (pred_guard ? SizeInBytes : 0);
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
      "l"(global_ptr),
      "n"(SizeInBytes),
      "r"(src_in_bytes));

#else
  static_assert(SizeInBytes == 16, "");
  using AccessType = uint4;
  if (pred_guard) {
    *static_cast<AccessType*>(smem_ptr) =
        *static_cast<AccessType const*>(global_ptr);
  } else {
    AccessType zeros;
    zeros.x = 0;
    zeros.y = 0;
    zeros.z = 0;
    zeros.w = 0;
    *static_cast<AccessType*>(smem_ptr) = zeros;
  }

#endif
}

/// Copy with zero fill
template <int SizeInBytes>
__device__ __forceinline__ void
cp_async_zfill(void* smem_ptr, void const* global_ptr, bool pred_guard) {
#if __CUDA_ARCH__ >= 800
  // Make sure the size is supported.
  static_assert(
      (SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
      "Size is not supported");

  unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
  const int src_in_bytes = pred_guard ? SizeInBytes : 0;

  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
      "l"(global_ptr),
      "n"(SizeInBytes),
      "r"(src_in_bytes));

#else
  static_assert(SizeInBytes == 16, "");
  using AccessType = uint4;
  if (pred_guard) {
    *static_cast<AccessType*>(smem_ptr) =
        *static_cast<AccessType const*>(global_ptr);
  } else {
    AccessType zeros;
    zeros.x = 0;
    zeros.y = 0;
    zeros.z = 0;
    zeros.w = 0;
    *static_cast<AccessType*>(smem_ptr) = zeros;
  }

#endif
}

// ---------------------- END cp.async helpers, copied from CUTLASS
} // namespace nbit
