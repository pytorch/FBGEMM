/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/types.h> // @manual=//caffe2:torch-cpp-cpu

#include "fbgemm_gpu/utils/assert.h"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"

namespace fbgemm_gpu::utils {

using ::testing::HasSubstr;

////////////////////////////////////////////////////////////////////////////////
// Test Kernels
////////////////////////////////////////////////////////////////////////////////

// Kernel that always passes the assertion
__global__ void assert_always_pass_kernel(TORCH_DSA_KERNEL_ARGS) {
  FBGEMM_KERNEL_ASSERT(true);
}

// Kernel that asserts on a condition
__global__ void assert_condition_kernel(
    const int* data,
    const int expected,
    TORCH_DSA_KERNEL_ARGS) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  FBGEMM_KERNEL_ASSERT(data[idx] == expected);
}

// Kernel that asserts index bounds
__global__ void
assert_bounds_kernel(const int index, const int size, TORCH_DSA_KERNEL_ARGS) {
  FBGEMM_KERNEL_ASSERT(index >= 0 && index < size);
}

////////////////////////////////////////////////////////////////////////////////
// FBGEMM_KERNEL_ASSERT Tests
////////////////////////////////////////////////////////////////////////////////

TEST(FbgemmKernelAssertTest, assert_pass_does_not_throw) {
  // Test that a passing assertion does not cause any errors
  const auto params = {std::tuple{1, 1}, std::tuple{128, 256}};
  for (const auto& [grid, block] : params) {
    EXPECT_NO_THROW({
      FBGEMM_LAUNCH_DSA_KERNEL(
          assert_always_pass_kernel,
          grid,
          block,
          0,
          at::cuda::getCurrentCUDAStream());
      c10::cuda::device_synchronize();
    });
  }
}

TEST(FbgemmKernelAssertTest, assert_condition_pass) {
  // Test assertion with a condition that passes
  const int size = 1024;
  const int expected_value = 42;

  auto tensor = torch::full(
      {size},
      expected_value,
      torch::dtype(torch::kInt32)
          .device(torch::kCUDA, at::cuda::current_device()));

  EXPECT_NO_THROW({
    FBGEMM_LAUNCH_DSA_KERNEL(
        assert_condition_kernel,
        (size + 255) / 256,
        256,
        0,
        at::cuda::getCurrentCUDAStream(),
        tensor.data_ptr<int>(),
        expected_value);
    c10::cuda::device_synchronize();
  });
}

TEST(FbgemmKernelAssertTest, assert_bounds_pass) {
  // Test bounds checking assertion that passes
  EXPECT_NO_THROW({
    FBGEMM_LAUNCH_DSA_KERNEL(
        assert_bounds_kernel, 1, 1, 0, at::cuda::getCurrentCUDAStream(), 5, 10);
    c10::cuda::device_synchronize();
  });
}

} // namespace fbgemm_gpu::utils
