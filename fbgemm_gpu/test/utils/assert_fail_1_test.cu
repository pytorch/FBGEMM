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

// Kernel that always fails the assertion
__global__ void assert_always_fail_kernel(TORCH_DSA_KERNEL_ARGS) {
  FBGEMM_KERNEL_ASSERT(false && "This assertion should always fail");
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

TEST(FbgemmKernelAssertTest, assert_fail_throws_exception) {
  // Test that a failing assertion causes a device-side error
  FBGEMM_LAUNCH_DSA_KERNEL(
      assert_always_fail_kernel, 1, 1, 0, at::cuda::getCurrentCUDAStream());

  EXPECT_NO_THROW({
    // Track the exception failure for the case where runtime DSA is disabled
    auto count = 0;

    try {
      c10::cuda::device_synchronize();
      FAIL() << "Expected device synchronize to throw due to assertion failure";

    } catch (const c10::Error& err) {
      count++;

      if (isPytorchDsaEnabled()) {
        const auto err_str = std::string(err.what());

        // Verify that the error message contains DSA failure information
        EXPECT_THAT(
            err_str,
            HasSubstr("CUDA device-side assertion failures were found"));

        // Assertion failure captures the condition string
        EXPECT_THAT(err_str, HasSubstr("This assertion should always fail"));

        // Assertion failure should contain file information
        EXPECT_THAT(err_str, HasSubstr(__FILE__));
      }
    }

    EXPECT_EQ(count, 1);
  });
}

#endif // __HIPCC__

} // namespace fbgemm_gpu::utils
