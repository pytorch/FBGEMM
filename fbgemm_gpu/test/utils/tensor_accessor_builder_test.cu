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

#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

namespace fbgemm_gpu::utils {

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

TEST(TensorAccessorBuilderTest, test_ta_create) {
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
  PTA_B(tensor, T, 1, 64).build("test_pta_create");
}

template <size_t N>
void test_pta_create_2(const at::Tensor& tensor) {
  PTA_B(tensor, float, N, 64).build("test_pta_create");
}

void test_pta_create_3(const at::Tensor& tensor) {
  PTA_B(tensor, float, 1, 64).build("test_pta_create");
}

TEST(TensorAccessorBuilderTest, test_pta_create) {
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

template <typename T>
std::array<T, 3> special_values() {
  static_assert(
      std::is_floating_point_v<T>, "Only floating point types supported");

  return {
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::infinity(),
      -std::numeric_limits<T>::infinity()};
}

template <typename T>
void test_check_values() {
  const auto invalids = special_values<T>();

  for (const auto i : invalids) {
    const auto tensor = torch::tensor(
        {1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f}, torch::kFloat32);
    tensor[rand() % tensor.numel()] = i;

    const auto builder = PTA_B(tensor, float, 1, 64);
    EXPECT_THROW({ builder.checkValues("test_check_values"); }, std::exception);
  }
}

TEST(TensorAccessorBuilderTest, test_check_values) {
  test_check_values<float>();
  test_check_values<double>();
}

} // namespace fbgemm_gpu::utils
