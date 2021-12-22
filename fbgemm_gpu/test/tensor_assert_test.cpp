// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include "fbgemm_gpu/sparse_ops_utils.h"

using namespace ::testing;

TEST(tensor_assert_test, gpu_asserts) {
  at::Tensor on_cpu_empty;

  ASSERT_EQ(on_cpu_empty.numel(), 0);
  EXPECT_NO_THROW(TENSOR_ON_CPU(on_cpu_empty));
  ASSERT_TRUE(torch_tensor_empty_or_on_cuda_gpu_check(on_cpu_empty));
  EXPECT_NO_THROW(TENSOR_EMPTY_OR_ON_CUDA_GPU(on_cpu_empty));
  EXPECT_ANY_THROW(TENSOR_ON_CUDA_GPU(on_cpu_empty));

  auto on_cpu_non_empty = at::randint(10, 32);
  const auto on_cuda_non_empty = on_cpu_non_empty.to(at::device(at::kCUDA));

  ASSERT_NE(on_cpu_non_empty.numel(), 0);
  EXPECT_NO_THROW(TENSOR_ON_CPU(on_cpu_non_empty));
  EXPECT_ANY_THROW(TENSOR_ON_CPU(on_cuda_non_empty));
  EXPECT_NO_THROW(TENSOR_ON_CUDA_GPU(on_cuda_non_empty));
  EXPECT_NO_THROW(TENSOR_EMPTY_OR_ON_CUDA_GPU(on_cuda_non_empty));
}
