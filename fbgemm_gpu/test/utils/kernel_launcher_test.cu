/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <cuda.h>
#include <gtest/gtest.h>
#include <torch/types.h> // @manual=//caffe2:torch-cpp-cpu

#include "fbgemm_gpu/utils/device_properties.cuh"
#include "fbgemm_gpu/utils/host_device_buffer_pair.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

namespace fbgemm_gpu::utils {

#define U32(x) static_cast<uint32_t>(x)

template <typename T>
__global__ void array_sum_kernel(T* C, const T* A, const T* B, size_t size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] + B[idx];
  }
}

template <typename T>
__global__ void tensor_sum_kernel(
    pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> C,
    const pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> A,
    const pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> B) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < C.size(0)) {
    C[idx] = A[idx] + B[idx];
  }
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

TEST(KernelLauncherTest, tensor_array_kernel_launch) {
  const auto size = 1024;
  // Not using structured bindings bc it fails on ROCm with:
  // `capturing a structured binding is not yet supported in OpenMP`
  at::Tensor A, B, C;
  std::tie(A, B, C) = sample_tensors(size);

  // Test normal kernel launch succeeds
  EXPECT_NO_THROW({
    FBGEMM_LAUNCH_KERNEL(
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
        FBGEMM_LAUNCH_KERNEL(
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
        FBGEMM_LAUNCH_KERNEL(
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
        FBGEMM_LAUNCH_KERNEL(
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
        FBGEMM_LAUNCH_KERNEL(
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

} // namespace fbgemm_gpu::utils
