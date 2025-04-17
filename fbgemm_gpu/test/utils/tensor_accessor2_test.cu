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

#include "fbgemm_gpu/utils/tensor_accessor2.h"

namespace fbgemm_gpu::utils {

TEST(TensorAccessorTest, tensor_access) {
  const auto tensor1 = torch::tensor(
      {{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f},
       {1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f}},
      torch::kFloat32);

  const auto tensor2 = torch::tensor(
      {{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f},
       {1.0f, 1.1f, 1.2f, 1.3f, 42.0f, 1.5f, 1.6f, 1.7f}},
      torch::kFloat32);

  auto accessor = TensorAccessor<float, 2, DefaultPtrTraits, int64_t>(
      static_cast<typename DefaultPtrTraits<float>::PtrType>(
          tensor1.data_ptr<float>()),
      tensor1.sizes().data(),
      tensor1.strides().data(),
      "tensor",
      "context");

  // Accessor should work as expected
  accessor[1][4] = 42.0f;

  EXPECT_TRUE(torch::equal(tensor1, tensor1))
      << "tensor1 is not equal to tensor2";

#ifndef __HIPCC__
  EXPECT_DEATH({ accessor[10][20] = 3.14f; }, "idx < numel_");
#endif
}

} // namespace fbgemm_gpu::utils
