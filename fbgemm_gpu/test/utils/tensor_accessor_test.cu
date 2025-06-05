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

#include "fbgemm_gpu/utils/tensor_accessor.h"

namespace fbgemm_gpu::utils {

TEST(TensorAccessorTest, tensor_access) {
  const auto tensor1 = torch::tensor(
      {{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f},
       {2.0f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f}},
      torch::kFloat32);

  const auto tensor2 = torch::tensor(
      {{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f},
       {2.0f, 2.1f, 2.2f, 2.3f, 42.0f, 2.5f, 2.6f, 2.7f}},
      torch::kFloat32);

  auto accessor = TensorAccessor<float, 2, DefaultPtrTraits, int64_t>(
      static_cast<typename DefaultPtrTraits<float>::PtrType>(
          tensor1.data_ptr<float>()),
      tensor1.sizes().data(),
      tensor1.strides().data(),
      "tensor",
      "context");

  EXPECT_NO_THROW({
    // Value update through accessor should work as expected
    accessor[1][4] = 42.0f;

    EXPECT_TRUE(torch::equal(tensor1, tensor2))
        << "tensor1 is not equal to tensor2";
  });

  EXPECT_DEATH({ accessor[10][20] = 3.14f; }, "idx < numel_");
}

TEST(PackedTensorAccessorTest, tensor_access) {
  const auto tensor1 = torch::tensor(
      {{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f},
       {2.0f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f}},
      torch::kFloat32);

  const auto tensor2 = torch::tensor(
      {{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f},
       {2.0f, 2.1f, 2.2f, 2.3f, 42.0f, 2.5f, 2.6f, 2.7f}},
      torch::kFloat32);

  auto accessor = PackedTensorAccessor<float, 2, RestrictPtrTraits, int64_t>(
      static_cast<typename RestrictPtrTraits<float>::PtrType>(
          tensor1.data_ptr<float>()),
      tensor1.sizes().data(),
      tensor1.strides().data(),
      "tensor",
      "context");

  EXPECT_NO_THROW({
    // Value update through accessor should work as expected
    accessor[1][4] = 42.0f;

    EXPECT_TRUE(torch::equal(tensor1, tensor2))
        << "tensor1 is not equal to tensor2";

    const auto transposed1 = accessor.transpose(0, 1);
    const auto transposed2 = transposed1.transpose(0, 1);

    for (auto i = 0; i < accessor.size(0); ++i) {
      for (auto j = 0; j < accessor.size(1); ++j) {
        // Transpose should work as expected
        EXPECT_EQ(transposed1[j][i], accessor[i][j]);
        // Twice-transpose should return the original tensor
        EXPECT_EQ(transposed2[i][j], accessor[i][j]);
      }
    }
  });

  EXPECT_DEATH({ accessor[10][20] = 3.14f; }, "idx < numel_");
}

} // namespace fbgemm_gpu::utils
