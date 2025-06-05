/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Enable kernel asserts on ROCm as it is disabled by default
// https://github.com/pytorch/pytorch/blob/main/c10/macros/Macros.h#L407
#define C10_USE_ROCM_KERNEL_ASSERT

#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/types.h> // @manual=//caffe2:torch-cpp-cpu

// ENABLE compilation in FBGEMM_GPU_MEMCHECK mode as a test
#ifndef FBGEMM_GPU_MEMCHECK
#define FBGEMM_GPU_MEMCHECK
#endif

#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

namespace fbgemm_gpu::utils {

template <typename T>
void test_ta_access() {
  constexpr auto DType = c10::CppTypeToScalarType<T>::value;

  const auto tensor = at::empty({0}, at::TensorOptions().dtype(DType));
  const auto accessor = TA_B(tensor, T, 1, 64).build("test_ta_access");

  EXPECT_DEATH({ accessor.at(10); }, "idx < numel_");
}

TEST(TensorAccessorWithMemcheckTest, test_ta_access) {
  test_ta_access<int32_t>();
  test_ta_access<int64_t>();
}

template <typename T>
void test_pta_access() {
  constexpr auto DType = c10::CppTypeToScalarType<T>::value;

  const auto tensor = at::empty({0}, at::TensorOptions().dtype(DType));
  const auto accessor = PTA_B(tensor, T, 1, 64).build("test_pta_access");

  EXPECT_DEATH({ accessor.at(10); }, "idx < numel_");
}

TEST(PackedTensorAccessorWithMemcheckTest, test_pta_access) {
  test_pta_access<int32_t>();
  test_pta_access<int64_t>();
}

} // namespace fbgemm_gpu::utils

#undef FBGEMM_GPU_MEMCHECK
