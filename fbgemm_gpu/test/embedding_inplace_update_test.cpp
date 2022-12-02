/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <folly/Random.h>
#include <gtest/gtest.h>
#include "deeplearning/fbgemm/fbgemm_gpu/src/embedding_inplace_update.h"

using namespace ::testing;
using namespace fbgemm_gpu;

TEST(embedding_inplace_update_test, random_update) {
  int T = folly::Random::rand32() % 3 + 2; // total number of tables
  std::vector<SparseType> weight_ty_list = {
      SparseType::FP32,
      SparseType::FP16,
      SparseType::INT8,
      SparseType::INT4,
      SparseType::INT2};
  std::vector<int32_t> D_offsets = {0};
  std::vector<int32_t> weights_placements = {};
  std::vector<int64_t> weights_offsets = {};
  std::vector<uint8_t> weights_tys;
  int64_t update_size = 0;
  std::vector<int32_t> update_tables;
  std::vector<int32_t> update_rows;
  int64_t dev_weights_offset = 0;
  int64_t uvm_weights_offset = 0;
  for (int i = 0; i < T; i++) {
    SparseType weight_ty =
        weight_ty_list[folly::Random::rand32() % weight_ty_list.size()];
    weights_tys.push_back(uint8_t(weight_ty));
    int D = 1 << (1 + folly::Random::rand32() % 5); // table dimension
    int32_t D_bytes = nbit::padded_row_size_in_bytes(D, weight_ty, 16);
    int total_rows = 10 + folly::Random::rand32() % 50;
    D_offsets.push_back(D_offsets.back() + D);

    int32_t weights_placement = folly::Random::rand32() % 2;
    weights_placements.push_back(weights_placement);
    if (weights_placement == 0) {
      weights_offsets.push_back(dev_weights_offset);
      dev_weights_offset += D_bytes * total_rows;
    } else {
      weights_offsets.push_back(uvm_weights_offset);
      uvm_weights_offset += D_bytes * total_rows;
    }
    int n = folly::Random::rand32() % 10 + 5;
    std::set<int32_t> rows;
    for (int j = 0; j < n; j++) {
      rows.insert(folly::Random::rand32() % total_rows);
    }
    std::string update_rows_str = "";
    for (int32_t r : rows) {
      update_tables.push_back(i);
      update_rows.push_back(r);
      update_size += D_bytes;
      update_rows_str += std::to_string(r) + ",";
    }
    LOG(INFO) << "table idx: " << i << ", D: " << D
              << ", weight type: " << int(weight_ty) << ", D bytes: " << D_bytes
              << ", total rows: " << total_rows
              << ", weight placement: " << weights_placement
              << ", weight offset: " << weights_offsets.back()
              << ", update rows: " << update_rows_str;
  }

  auto dev_weight = at::randint(
      0, 255, {dev_weights_offset}, at::device(at::kCUDA).dtype(at::kByte));
  auto uvm_weight = at::randint(
      0, 255, {uvm_weights_offset}, at::device(at::kCUDA).dtype(at::kByte));
  auto update_weight = at::randint(
      0, 255, {update_size}, at::device(at::kCUDA).dtype(at::kByte));

  fbgemm_gpu::embedding_inplace_update_host_weight_cuda(
      dev_weight,
      uvm_weight,
      at::tensor(weights_placements, at::device(at::kCUDA).dtype(at::kInt)),
      at::tensor(weights_offsets, at::device(at::kCUDA).dtype(at::kLong)),
      at::tensor(weights_tys, at::device(at::kCUDA).dtype(at::kByte)),
      at::tensor(D_offsets, at::device(at::kCUDA).dtype(at::kInt)),
      update_weight,
      update_tables,
      update_rows,
      16L /* row_alignment */);

  int offset = 0;
  for (int i = 0; i < update_tables.size(); i++) {
    auto table_idx = update_tables[i];
    auto row_idx = update_rows[i];
    auto weight_offset = weights_offsets[table_idx];
    auto weight_placement = weights_placements[table_idx];
    auto D = D_offsets[table_idx + 1] - D_offsets[table_idx];
    SparseType ty = static_cast<SparseType>(weights_tys[table_idx]);
    int32_t D_bytes = nbit::padded_row_size_in_bytes(D, ty, 16);
    if (weight_placement == 0) {
      for (int j = 0; j < D_bytes; j++) {
        ASSERT_EQ(
            dev_weight[weight_offset + D_bytes * row_idx + j].item<uint8_t>(),
            update_weight[offset + j].item<uint8_t>());
      }
    } else {
      for (int j = 0; j < D_bytes; j++) {
        ASSERT_EQ(
            uvm_weight[weight_offset + D_bytes * row_idx + j].item<uint8_t>(),
            update_weight[offset + j].item<uint8_t>());
      }
    }
    offset += D_bytes;
  }
}
