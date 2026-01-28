/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fbgemm_gpu/utils/assert.h"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"

namespace fbgemm_gpu::utils {

using ::testing::HasSubstr;

// Kernel that conditionally fails based on thread index
__global__ void assert_conditional_fail_kernel(
    const int bad_block,
    const int bad_thread,
    TORCH_DSA_KERNEL_ARGS) {
  FBGEMM_KERNEL_ASSERT(!(blockIdx.x == bad_block && threadIdx.x == bad_thread));
}

////////////////////////////////////////////////////////////////////////////////
// FBGEMM_KERNEL_ASSERT DSA Failure Test
//
// These tests verify that FBGEMM_KERNEL_ASSERT correctly triggers Device-Side
// Assertion (DSA) failures when conditions are not met.
//
// IMPORTANT: DSA failures leave the CUDA device in a corrupted state that
// cannot be recovered without cudaDeviceReset(), which also destroys PyTorch's
// CUDA context. Therefore, each test that triggers a DSA failure should run
// in a separate process.
////////////////////////////////////////////////////////////////////////////////

// NOTE: Test is disabled for HIP because the device-side abort() (when
// TORCH_USE_CUDA_DSA is defined) causes the HIP runtime and host process itself
// to crash, which is not catchable by the test framework.
#ifndef __HIPCC__

TEST(FbgemmKernelAssertTest, assert_conditional_fail_single_thread) {
  EXPECT_NO_THROW({
    // Track the exception failure for the case where runtime DSA is disabled
    auto count = 0;

    // Test that only the specified thread triggers the assertion
    FBGEMM_LAUNCH_DSA_KERNEL(
        assert_conditional_fail_kernel,
        4, // 4 blocks
        128, // 128 threads per block
        0,
        at::cuda::getCurrentCUDAStream(),
        2, // bad_block = 2
        64 // bad_thread = 64
    );

    try {
      c10::cuda::device_synchronize();
      FAIL() << "Expected device synchronize to throw due to assertion failure";

    } catch (const c10::Error& err) {
      count++;

      if (isPytorchDsaEnabled()) {
        const auto err_str = std::string(err.what());
        EXPECT_THAT(
            err_str,
            HasSubstr("CUDA device-side assertion failures were found"));
      }
    }

    EXPECT_EQ(count, 1);
  });
}

#endif // __HIPCC__

} // namespace fbgemm_gpu::utils
