/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/macros/Macros.h>

#include <cstdint>

// Warp size definitions, split out of cuda_prelude.cuh so that host sources
// can use them. cuda_prelude.cuh is not includable from a .cpp: its device
// helpers call intrinsics the host compiler has no declaration for.
//
// On ROCm, at::cuda::warp_size() is declared by <c10/macros/Macros.h>, so
// this header needs no CUDA or HIP header of its own.

namespace fbgemm_gpu {

// Warp size -- device-pass constant
//
// Device code: per-arch constexpr. __GFX9__ is defined by HIP-Clang during
// device compilation for gfx9xx targets (warpSize 64); it is undefined for
// warpSize 32 targets. HIP-Clang runs the device-code backend once per
// --offload-arch, so the same source produces correct per-arch device code.
//
// Host code on ROCm: warpSize is only known at runtime. The 64 below is a
// stop-gap so __global__ kernel bodies parse on the host pass; never rely
// on it in host-side computation. Use kWarpSizeHost (below) instead.
#if !defined(USE_ROCM)
static constexpr int32_t kWarpSize = 32;
#elif defined(__GFX9__)
static constexpr int32_t kWarpSize = 64;
#elif defined(__HIP_DEVICE_COMPILE__)
static constexpr int32_t kWarpSize = 32;
#else
static constexpr int32_t kWarpSize = 64;
#endif

// Host-side warp size
//
// Use this in host code anywhere kWarpSize would be wrong on a ROCm
// multi-arch build (block-dim computations, grid sizing, etc.). It is:
//   * CUDA: a constexpr function returning 32. Usable as a template
//     argument and in static_assert.
//   * ROCm: an inline function returning at::cuda::warp_size() -- a runtime
//     query of the active device. NOT constexpr; do not use as a template
//     argument. Cheap (a cached device-properties array lookup).
//
// Always invoke with parentheses (kWarpSizeHost()) so the same call shape
// works under both back-ends and inside namespace-qualified call sites.
//
// Do not use in __device__ code: on ROCm the at::cuda::warp_size() call
// is not callable from device. Use kWarpSize there, which is per-arch
// correct via the device pass.
#if defined(USE_ROCM)
inline int32_t kWarpSizeHost() {
  return at::cuda::warp_size();
}
#else
inline constexpr int32_t kWarpSizeHost() {
  return 32;
}
#endif

} // namespace fbgemm_gpu
