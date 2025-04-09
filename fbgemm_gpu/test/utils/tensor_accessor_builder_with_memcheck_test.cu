/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/types.h> // @manual=//caffe2:torch-cpp-cpu

// ENABLE compilation in FBGEMM_GPU_MEMCHECK mode as a test
#ifndef FBGEMM_GPU_MEMCHECK
#define FBGEMM_GPU_MEMCHECK
#endif

#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

template <typename T>
void test_ta_create_1(const at::Tensor& tensor) {
  TA_B(tensor, T, 1, 64).build("test_ta_create");
}

template <size_t N>
void test_ta_create_2(const at::Tensor& tensor) {
  TA_B(tensor, float, N, 64).build("test_ta_create");
}

void test_ta_create_3(const at::Tensor& tensor) {
  TA_B(tensor, float, 1, 64).build("test_ta_create");
}

TEST(TensorAccessorBuilderWithMemcheckTest, test_ta_create) {
  const auto tensor = torch::tensor(
      {1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f}, torch::kFloat32);
  // Test mismatched types
  EXPECT_THROW({ test_ta_create_1<int32_t>(tensor); }, std::exception);
  EXPECT_THROW({ test_ta_create_1<int64_t>(tensor); }, std::exception);
  EXPECT_THROW({ test_ta_create_1<double>(tensor); }, std::exception);

  // Test invalid dimensions
  EXPECT_THROW({ test_ta_create_2<2>(tensor); }, std::exception);
  EXPECT_THROW({ test_ta_create_2<3>(tensor); }, std::exception);
  EXPECT_THROW({ test_ta_create_2<4>(tensor); }, std::exception);

  // Test valid type and dimension
  EXPECT_NO_THROW({ test_ta_create_3(tensor); });
}

template <c10::ScalarType DType, typename T>
void test_ta_access() {
  const auto tensor = at::empty({0}, at::TensorOptions().dtype(DType));
  const auto accessor = TA_B(tensor, T, 1, 64).build("ta_access");

  EXPECT_DEATH({ accessor.at(10); }, "idx < numel_");
}

// NOTE: CUDA_KERNEL_ASSERT appears to be a no-op when HIPified
#ifndef __HIPCC__
TEST(TensorAccessorBuilderWithMemcheckTest, test_ta_access) {
  test_ta_access<torch::kInt32, int32_t>();
  test_ta_access<torch::kInt64, int64_t>();
}
#endif

template <typename T>
void test_pta_create_1(const at::Tensor& tensor) {
  PTA_B(tensor, T, 1, 64).build("test_pta_create");
}

template <size_t N>
void test_pta_create_2(const at::Tensor& tensor) {
  PTA_B(tensor, float, N, 64).build("test_pta_create");
}

void test_pta_create_3(const at::Tensor& tensor) {
  PTA_B(tensor, float, 1, 64).build("test_pta_create");
}

TEST(PackedTensorAccessorTest, test_pta_create) {
  const auto tensor = torch::tensor(
      {1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f}, torch::kFloat32);
  // Test mismatched types
  EXPECT_THROW({ test_pta_create_1<int32_t>(tensor); }, std::exception);
  EXPECT_THROW({ test_pta_create_1<int64_t>(tensor); }, std::exception);
  EXPECT_THROW({ test_pta_create_1<double>(tensor); }, std::exception);

  // Test invalid dimensions
  EXPECT_THROW({ test_pta_create_2<2>(tensor); }, std::exception);
  EXPECT_THROW({ test_pta_create_2<3>(tensor); }, std::exception);
  EXPECT_THROW({ test_pta_create_2<4>(tensor); }, std::exception);

  // Test valid type and dimension
  EXPECT_NO_THROW({ test_pta_create_3(tensor); });
}

template <c10::ScalarType DType, typename T>
void test_pta_access() {
  const auto tensor = at::empty({0}, at::TensorOptions().dtype(DType));
  const auto accessor = PTA_B(tensor, T, 1, 64).build("test_pta_access");

  EXPECT_DEATH({ accessor.at(10); }, "idx < numel_");
}

#ifndef __HIPCC__
TEST(PackedTensorAccessorWithMemcheckTest, test_pta_access) {
  test_pta_access<torch::kInt32, int32_t>();
  test_pta_access<torch::kInt64, int64_t>();
}
#endif

#undef FBGEMM_GPU_MEMCHECK
