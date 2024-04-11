/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/embedding_forward_split_cpu.h"
#include "torch/types.h" // @manual=//caffe2:torch-cpp-cpu

TEST(CpuKernelTest, csr2csc_test) {
  internal::HyperCompressedSparseColumn csc;
  int B = 2;
  at::Tensor offsets = torch::tensor({0, 4, 8});
  at::Tensor indices = torch::tensor({1, 2, 4, 5, 4, 3, 2, 9});
  int64_t pooling_mode = (int64_t)fbgemm_gpu::PoolingMode::SUM;
  int table_to_feature_offset[2] = {0, 1};
  int num_embeddings = 10;

  ::internal::csr2csc(
      csc,
      B,
      offsets.accessor<int64_t, 1>(),
      indices.accessor<int64_t, 1>(),
      at::TensorAccessor<at::acc_type<float, true>, 1>(
          nullptr, nullptr, nullptr), // no weights
      pooling_mode,
      table_to_feature_offset,
      num_embeddings);

  // sorted list of unique elements in indices
  std::array<int, 6> expect_cs_indices = {1, 2, 3, 4, 5, 9};
  for (int i = 0; i < expect_cs_indices.size(); ++i) {
    EXPECT_EQ(expect_cs_indices[i], csc.column_segment_indices[i]);
  }

  // column_segment_ptr[i+1]-column_segment_ptr[i] gives the count of
  // column_segment_indices[i] in indices
  std::array<int, 7> expect_cs_ptr = {0, 1, 3, 4, 6, 7, 8};
  for (int i = 0; i < expect_cs_ptr.size(); ++i) {
    EXPECT_EQ(expect_cs_ptr[i], csc.column_segment_ptr[i]);
  }

  // gives the bag of the ith lowest value in indices, where the bag is
  // determined according to offsets
  std::array<int, 8> expect_row_indices = {0, 0, 1, 1, 0, 1, 0, 1};
  for (int i = 0; i < expect_row_indices.size(); ++i) {
    EXPECT_EQ(expect_row_indices[i], csc.row_indices[i]);
  }

  internal::HyperCompressedSparseColumn csc_weighted;
  at::Tensor indice_weights = torch::tensor(
      {1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f}, torch::kFloat32);
  ::internal::csr2csc(
      csc_weighted,
      B,
      offsets.accessor<int64_t, 1>(),
      indices.accessor<int64_t, 1>(),
      indice_weights.accessor<at::acc_type<float, true>, 1>(),
      pooling_mode,
      table_to_feature_offset,
      num_embeddings);

  for (int i = 0; i < expect_cs_indices.size(); ++i) {
    EXPECT_EQ(expect_cs_indices[i], csc_weighted.column_segment_indices[i]);
  }

  for (int i = 0; i < expect_cs_ptr.size(); ++i) {
    EXPECT_EQ(expect_cs_ptr[i], csc_weighted.column_segment_ptr[i]);
  }

  for (int i = 0; i < expect_row_indices.size(); ++i) {
    EXPECT_EQ(expect_row_indices[i], csc_weighted.row_indices[i]);
  }

  // sorting should be exact, no arithmetic needed. check for strict equality
  // of floats, not relative error
  std::array<float, 8> expect_weights = {
      1.0f, 1.1f, 1.6f, 1.5f, 1.2f, 1.4f, 1.3f, 1.7f};
  for (int i = 0; i < expect_weights.size(); ++i) {
    EXPECT_EQ(expect_weights[i], csc_weighted.weights[i]);
  }
}
