/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <gtest/gtest.h>
#include <torch/types.h> // @manual=//caffe2:torch-cpp-cpu

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
// Compile-time contract checks for cuda_prelude.cuh
//
// These static_asserts pin the public header-only contract: kWarpSize and
// kFullWarpMask must have the right type and value for each platform.
// Regressions here would break warpSize 32/64 dual-build support on ROCm.
////////////////////////////////////////////////////////////////////////////////

// kWarpSize type: always int32_t.
static_assert(
    std::is_same_v<std::remove_cv_t<decltype(fbgemm_gpu::kWarpSize)>, int32_t>,
    "kWarpSize must be int32_t");

#if defined(USE_ROCM)
// kFullWarpMask on ROCm: uint64_t because HIP's __ballot_sync / __any_sync /
// __all_sync templates statically require a 64-bit mask type. Narrowing
// would break compilation on both warpSize 32 and warpSize 64 archs.
static_assert(
    std::is_same_v<
        std::remove_cv_t<decltype(fbgemm_gpu::kFullWarpMask)>,
        uint64_t>,
    "kFullWarpMask on ROCm must be uint64_t");
static_assert(
    fbgemm_gpu::kFullWarpMask == 0xffffffffffffffffull,
    "kFullWarpMask on ROCm must be 0xFFFFFFFFFFFFFFFF");
#else
// kFullWarpMask on CUDA: uint32_t (the NVIDIA ballot mask width).
static_assert(
    std::is_same_v<
        std::remove_cv_t<decltype(fbgemm_gpu::kFullWarpMask)>,
        uint32_t>,
    "kFullWarpMask on CUDA must be uint32_t");
static_assert(
    fbgemm_gpu::kFullWarpMask == 0xffffffffu,
    "kFullWarpMask on CUDA must be 0xFFFFFFFF");
#endif

////////////////////////////////////////////////////////////////////////////////
// Test Kernels
////////////////////////////////////////////////////////////////////////////////

// Writes the device-side value of fbgemm_gpu::kWarpSize from thread 0 into
// the output buffer. Used to verify that the device-side per-arch resolution
// of kWarpSize matches the host's at::cuda::warp_size() on the active
// device.
__global__ void write_device_kWarpSize(int32_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = fbgemm_gpu::kWarpSize;
  }
}

// For lane l in [0, warp_size), evaluates `(l < n)` as the predicate to
// fbgemm_gpu::ballot_sync, then thread 0 writes the returned bitmap into
// out[n-1]. With a single warp launch the expected bitmap is
// ((1ULL << n) - 1ULL) for n in [1, warp_size].
__global__ void ballot_roundtrip_kernel(uint64_t* out, int32_t n) {
  const auto lane = threadIdx.x;
  const bool pred = lane < static_cast<uint32_t>(n);
#if defined(USE_ROCM)
  const uint64_t bitmap = fbgemm_gpu::ballot_sync(pred);
#else
  const uint64_t bitmap = static_cast<uint64_t>(fbgemm_gpu::ballot_sync(pred));
#endif
  if (lane == 0) {
    out[0] = bitmap;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

TEST(WarpPrimitivesTest, device_kWarpSize_matches_at_cuda_warp_size) {
  // at::cuda::warp_size() is the host-side source of truth for the active
  // device's warpSize. The device-side kWarpSize is resolved per-arch at
  // compile time (__GFX9__ => 64, else 32 on ROCm; always 32 on CUDA).
  // For the arch the test runs on, they must agree.
  auto out = torch::full(
      {1},
      -1,
      torch::dtype(torch::kInt32)
          .device(torch::kCUDA, at::cuda::current_device()));
  FBGEMM_LAUNCH_KERNEL(
      write_device_kWarpSize,
      1,
      1,
      0,
      at::cuda::getCurrentCUDAStream(),
      out.data_ptr<int32_t>());

  const int32_t device_warp_size = out.cpu().item<int32_t>();
  const int32_t host_warp_size = at::cuda::warp_size();
  EXPECT_EQ(device_warp_size, host_warp_size);
  // And the runtime-known warpSize must be one of the two supported values.
  EXPECT_TRUE(host_warp_size == 32 || host_warp_size == 64)
      << "Unsupported warpSize: " << host_warp_size;
}

TEST(WarpPrimitivesTest, ballot_sync_roundtrip) {
  const int32_t warp_size = at::cuda::warp_size();
  auto out = torch::zeros(
      {1},
      torch::dtype(torch::kUInt64)
          .device(torch::kCUDA, at::cuda::current_device()));

  // For each n in [1, warp_size], check the returned bitmap has exactly the
  // low n bits set. This exercises both the CUDA (__ballot_sync) and HIP
  // (__ballot) wrappers and — on ROCm — transitively the __any_sync shim
  // defined in cuda_prelude.cuh (which is implemented via __ballot).
  for (int32_t n = 1; n <= warp_size; ++n) {
    FBGEMM_LAUNCH_KERNEL(
        ballot_roundtrip_kernel,
        1,
        warp_size,
        0,
        at::cuda::getCurrentCUDAStream(),
        out.data_ptr<uint64_t>(),
        n);

    const uint64_t expected =
        n == 64 ? 0xffffffffffffffffull : ((1ULL << n) - 1ULL);
    const auto out_cpu = out.cpu();
    EXPECT_EQ(out_cpu.data_ptr<uint64_t>()[0], expected) << "n=" << n;
  }
}

} // namespace fbgemm_gpu::utils
