/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDADeviceAssertion.h>
#include <cuda.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/types.h> // @manual=//caffe2:torch-cpp-cpu

#include "fbgemm_gpu/utils/device_properties.cuh"
#include "fbgemm_gpu/utils/host_device_buffer_pair.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

namespace fbgemm_gpu::utils {

#define U32(x) static_cast<uint32_t>(x)

using ::testing::HasSubstr;

////////////////////////////////////////////////////////////////////////////////
// Test Kernels
////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void array_sum_kernel(T* C, const T* A, const T* B, size_t size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] + B[idx];
  }
}

template <typename T>
__global__ void array_sum_dsa_kernel(
    T* C,
    const T* A,
    const T* B,
    size_t size,
    TORCH_DSA_KERNEL_ARGS) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] + B[idx];
  }
}

template <typename T>
__global__ void tensor_sum_kernel(
    pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> C,
    const pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> A,
    const pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> B,
    TORCH_DSA_KERNEL_ARGS) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < C.size(0)) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void always_fail_assertion_kernel(
    const int a,
    TORCH_DSA_KERNEL_ARGS) {
  CUDA_KERNEL_ASSERT2((a != a) && "This assertion should always fail");
}

auto sample_tensors(const long size) {
  auto A = torch::full(
      {size},
      2,
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device()));

  auto B = torch::full(
      {size},
      3,
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device()));

  auto C = torch::full(
      {size},
      -1,
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device()));

  return std::make_tuple(A, B, C);
}

////////////////////////////////////////////////////////////////////////////////
// Kernel Launcher Tests
////////////////////////////////////////////////////////////////////////////////

TEST(KernelLauncherTest, test_array_kernel_launch) {
  constexpr auto size = 1024;
  auto A = HostDeviceBufferPair<float>(size, 2);
  auto B = HostDeviceBufferPair<float>(size, 3);
  auto C = HostDeviceBufferPair<float>(size, -1);

  EXPECT_NO_THROW({
    FBGEMM_LAUNCH_KERNEL(
        array_sum_kernel<float>,
        8,
        1024,
        0,
        at::cuda::getCurrentCUDAStream(),
        C.device,
        A.device,
        B.device,
        size);

    C.syncToHost();

    for (const auto x : C.host) {
      EXPECT_EQ(x, 5.0f);
    }
  });
}

TEST(KernelLauncherTest, test_array_kernel_launch_dsa) {
  constexpr auto size = 1024;
  auto A = HostDeviceBufferPair<float>(size, 2);
  auto B = HostDeviceBufferPair<float>(size, 3);
  auto C = HostDeviceBufferPair<float>(size, -1);

  EXPECT_NO_THROW({
    FBGEMM_LAUNCH_DSA_KERNEL(
        array_sum_dsa_kernel<float>,
        8,
        1024,
        0,
        at::cuda::getCurrentCUDAStream(),
        C.device,
        A.device,
        B.device,
        size);

    C.syncToHost();

    for (const auto x : C.host) {
      EXPECT_EQ(x, 5.0f);
    }
  });
}

TEST(KernelLauncherTest, tensor_array_kernel_launch) {
  const auto size = 1024;
  // Not using structured bindings bc it fails on ROCm with:
  // `capturing a structured binding is not yet supported in OpenMP`
  at::Tensor A, B, C;
  std::tie(A, B, C) = sample_tensors(size);

  // Test normal kernel launch succeeds
  EXPECT_NO_THROW({
    FBGEMM_LAUNCH_DSA_KERNEL(
        tensor_sum_kernel<float>,
        8,
        1024,
        0,
        at::cuda::getCurrentCUDAStream(),
        PTA_B(C, float, 1, 64),
        PTA_B(A, float, 1, 64),
        PTA_B(B, float, 1, 64));
  });

  EXPECT_EQ(
      C.equal(torch::full(
          {size},
          5,
          torch::dtype(torch::kFloat32)
              .device(torch::kCUDA, at::cuda::current_device()))),
      true);
}

TEST(KernelLauncherTest, kernel_launch_checks) {
  const auto size = 1024;
  // Not using structured bindings bc it fails on ROCm with:
  // `capturing a structured binding is not yet supported in OpenMP`
  at::Tensor A, B, C;
  std::tie(A, B, C) = sample_tensors(size);

  const auto device = get_device_for_stream(at::cuda::getCurrentCUDAStream());
  const auto properties = get_device_properties(device);
  const auto grid_max = properties.maxGridSize;
  const auto block_max = properties.maxThreadsDim;

  // Test grid size bounds checking
  EXPECT_THROW(
      {
        FBGEMM_LAUNCH_DSA_KERNEL(
            tensor_sum_kernel<float>,
            // grid dims are too large
            grid_max[0] + 1,
            1024,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(C, float, 1, 64),
            PTA_B(A, float, 1, 64),
            PTA_B(B, float, 1, 64));
      },
      std::exception);

  // Test block size bounds checking
  EXPECT_THROW(
      {
        FBGEMM_LAUNCH_DSA_KERNEL(
            tensor_sum_kernel<float>,
            8,
            // block dims are too large
            block_max[0] + 1,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(C, float, 1, 64),
            PTA_B(A, float, 1, 64),
            PTA_B(B, float, 1, 64));
      },
      std::exception);

#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
  // Test max thread count
  EXPECT_THROW(
      {
        FBGEMM_LAUNCH_DSA_KERNEL(
            tensor_sum_kernel<float>,
            // Both grid and block dims conform, but the total number of threads
            // exceeds the max
            {U32(grid_max[0]), U32(grid_max[1]), U32(grid_max[2])},
            {U32(block_max[0]), U32(block_max[1]), U32(block_max[2])},
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(C, float, 1, 64),
            PTA_B(A, float, 1, 64),
            PTA_B(B, float, 1, 64));
      },
      std::exception);
#endif

  // Test shared memory size bounds checking
  EXPECT_THROW(
      {
        FBGEMM_LAUNCH_DSA_KERNEL(
            tensor_sum_kernel<float>,
            8,
            1024,
            // shared memory size is too large
            properties.sharedMemPerBlock + 1,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(C, float, 1, 64),
            PTA_B(A, float, 1, 64),
            PTA_B(B, float, 1, 64));
      },
      std::exception);
}

// NOTE: This test currently fails in fbcode CI for HIP with the following
// error:
//
// void fbgemm_gpu::utils::always_fail_assertion_kernel(const int,
// c10::hip::DeviceAssertionsData *const, uint32_t): Device-side assertion `(a
// != a) && "This assertion should always fail"' failed. :0:rocdevice.cpp :2984:
// 1311044151769 us: [pid:1082329 tid:0x7fc06c9ff640] Callback: Queue
// 0x7fc06b500000 aborting with error : HSA_STATUS_ERROR_EXCEPTION: An HSAIL
// operation resulted in a hardware exception. code: 0x1016
//
// Disabled for now until we can figure out why this is happening.
#ifndef __HIPCC__

TEST(KernelLauncherTest, throws_dsa_exception) {
  FBGEMM_LAUNCH_DSA_KERNEL(
      always_fail_assertion_kernel,
      1,
      1,
      0,
      at::cuda::getCurrentCUDAStream(),
      42);

  EXPECT_NO_THROW({
    try {
      c10::cuda::device_synchronize();
      throw std::runtime_error("Test didn't fail, but should have.");

    } catch (const c10::Error& err) {
      const auto err_str = std::string(err.what());

      ASSERT_THAT(
          err_str,
          HasSubstr(
              "CUDA device-side assertion failures were found on GPU #0!"));

      ASSERT_THAT(
          err_str, HasSubstr("File containing kernel launch = " __FILE__));

      ASSERT_THAT(
          err_str,
          HasSubstr(
              "Name of kernel launched that led to failure = always_fail_assertion_kernel"));
    }
  });
}

#endif // __HIPCC__

} // namespace fbgemm_gpu::utils
