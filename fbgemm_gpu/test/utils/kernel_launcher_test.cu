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

#include "fbgemm_gpu/utils/host_device_buffer_pair.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

namespace fbgemm_gpu::utils {

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

#ifdef USE_ROCM
  // Test grid size bounds checking for ROCm
  EXPECT_THROW(
      {
        FBGEMM_LAUNCH_KERNEL(
            array_sum_kernel<float>,
            // block size x grid size > 2**32
            1LL << 30, // 2**30
            1LL << 30, // 2**30
            0,
            at::cuda::getCurrentCUDAStream(),
            C.device,
            A.device,
            B.device,
            size);
      },
      std::exception);
#endif
}

TEST(KernelLauncherTest, tensor_array_kernel_launch) {
  const auto size = 1024;
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

#ifdef USE_ROCM
  // Test grid size bounds checking for ROCm
  EXPECT_THROW(
      {
        FBGEMM_LAUNCH_KERNEL(
            tensor_sum_kernel<float>,
            // block size x grid size > 2**32
            1LL << 30, // 2**30
            1LL << 30, // 2**30
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(C, float, 1, 64),
            PTA_B(A, float, 1, 64),
            PTA_B(B, float, 1, 64));
      },
      std::exception);
#endif
}

} // namespace fbgemm_gpu::utils
