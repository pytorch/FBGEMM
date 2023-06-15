/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "deeplearning/fbgemm/fbgemm_gpu/include/fbgemm_gpu/sparse_ops_utils.h"

using namespace testing;

at::Tensor get_valid_cpu_tensor() {
  std::vector<int32_t> test_data = {1};
  return torch::from_blob(
      test_data.data(), {static_cast<long>(test_data.size())}, torch::kInt);
}

TEST(sparse_ops_utils_test, undefined_tensors_do_not_trigger) {
  const auto ten1 = at::Tensor();
  const auto ten2 = at::Tensor();
  const auto func = [&]() {
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(ten1, ten2);
  };
  EXPECT_NO_THROW(func());
}

TEST(sparse_ops_utils_test, cpu_tensors_fail) {
  const auto ten1 = get_valid_cpu_tensor();
  const auto ten2 = get_valid_cpu_tensor();
  const auto func = [&]() {
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(ten1, ten2);
  };

  EXPECT_THAT(
      func,
      Throws<c10::Error>(Property(
          &c10::Error::what,
          HasSubstr(
              "Not all tensors were on the same GPU: ten1(CPU:-1),  ten2(CPU:-1)"))));
}

TEST(sparse_ops_utils_test, gpu_tensors_pass) {
  const auto ten1 = get_valid_cpu_tensor().cuda();
  const auto ten2 = get_valid_cpu_tensor().cuda();
  const auto func = [&]() {
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(ten1, ten2);
  };
  EXPECT_NO_THROW(func());
}

TEST(sparse_ops_utils_test, optional_tensor_passes) {
  const auto ten1 = get_valid_cpu_tensor().cuda();
  const c10::optional<at::Tensor> ten2;
  const auto func = [&]() {
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(ten1, ten2);
  };
  EXPECT_NO_THROW(func());
}
