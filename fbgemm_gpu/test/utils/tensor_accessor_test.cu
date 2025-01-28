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

// DISABLE compilation in FBGEMM_GPU_MEMCHECK mode as a test
#ifdef FBGEMM_GPU_MEMCHECK
#undef FBGEMM_GPU_MEMCHECK
#endif

#include "fbgemm_gpu/utils/tensor_accessor.h"

template <typename T>
void test_ta_create_1(const at::Tensor& tensor) {
  [[maybe_unused]] const auto func_name = "test_ta_create";
  [[maybe_unused]] const auto accessor =
      MAKE_TA_WITH_NAME(func_name, tensor, T, 1);
}

template <size_t N>
void test_ta_create_2(const at::Tensor& tensor) {
  [[maybe_unused]] const auto func_name = "test_ta_create";
  [[maybe_unused]] const auto accessor =
      MAKE_TA_WITH_NAME(func_name, tensor, float, N);
}

void test_ta_create_3(const at::Tensor& tensor) {
  [[maybe_unused]] const auto func_name = "test_ta_create";
  [[maybe_unused]] const auto accessor =
      MAKE_TA_WITH_NAME(func_name, tensor, float, 1);
}

TEST(TensorAccessorTest, test_ta_create) {
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

template <typename T>
void test_pta_create_1(const at::Tensor& tensor) {
  [[maybe_unused]] const auto func_name = "test_pta_create";
  [[maybe_unused]] const auto accessor =
      MAKE_PTA_WITH_NAME(func_name, tensor, T, 1, 64);
}

template <size_t N>
void test_pta_create_2(const at::Tensor& tensor) {
  [[maybe_unused]] const auto func_name = "test_pta_create";
  [[maybe_unused]] const auto accessor =
      MAKE_PTA_WITH_NAME(func_name, tensor, float, N, 64);
}

void test_pta_create_3(const at::Tensor& tensor) {
  [[maybe_unused]] const auto func_name = "test_pta_create";
  [[maybe_unused]] const auto accessor =
      MAKE_PTA_WITH_NAME(func_name, tensor, float, 1, 64);
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
