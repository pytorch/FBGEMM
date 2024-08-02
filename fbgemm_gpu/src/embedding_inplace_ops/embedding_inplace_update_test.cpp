/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <folly/Random.h>
#include <gtest/gtest.h>
#include "fbgemm_gpu/embedding_inplace_update.h"

using namespace ::testing;
using namespace fbgemm_gpu;

int32_t get_D_bytes(
    Tensor D_offsets,
    Tensor weights_tys,
    const int32_t table_idx,
    const int64_t row_alignment) {
  const int32_t D_start = D_offsets[table_idx].item<int32_t>();
  const int32_t D_end = D_offsets[table_idx + 1].item<int32_t>();
  const int32_t D = D_end - D_start;
  SparseType weight_ty =
      static_cast<SparseType>(weights_tys[table_idx].item<uint8_t>());
  return nbit::padded_row_size_in_bytes(D, weight_ty, row_alignment);
}

template <typename index_t>
void test_embedding_inplace_update() {
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
  std::vector<int32_t> update_table_idx;
  std::vector<index_t> update_row_idx;
  int64_t dev_weights_offset = 0;
  int64_t uvm_weights_offset = 0;
  for (const auto i : c10::irange(T)) {
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
    for (const auto j : c10::irange(n)) {
      rows.insert(folly::Random::rand32() % total_rows);
    }
    std::string update_row_idx_str = "";
    for (int32_t r : rows) {
      update_table_idx.push_back(i);
      update_row_idx.push_back(r);
      update_size += D_bytes;
      update_row_idx_str += std::to_string(r) + ",";
    }
    LOG(INFO) << "table idx: " << i << ", D: " << D
              << ", weight type: " << int(weight_ty) << ", D bytes: " << D_bytes
              << ", total rows: " << total_rows
              << ", weight placement: " << weights_placement
              << ", weight offset: " << weights_offsets.back()
              << ", update rows: " << update_row_idx_str;
  }

  bool use_cpu = folly::Random::rand32() % 2;
  auto device = use_cpu ? at::kCPU : at::kCUDA;
  int64_t row_alignment = use_cpu ? 1L : 16L;

  auto dev_weight = at::randint(
      0, 255, {dev_weights_offset}, at::device(device).dtype(at::kByte));
  auto uvm_weight = at::randint(
      0, 255, {uvm_weights_offset}, at::device(device).dtype(at::kByte));
  auto update_weight =
      at::randint(0, 255, {update_size}, at::device(device).dtype(at::kByte));

  auto D_offsets_tensor =
      at::tensor(D_offsets, at::device(device).dtype(at::kInt));
  auto weights_tys_tensor =
      at::tensor(weights_tys, at::device(device).dtype(at::kByte));

  std::vector<int64_t> update_offsets;
  int64_t update_offset = 0;
  update_offsets.push_back(0);
  for (int i = 0; i < update_table_idx.size(); ++i) {
    int32_t idx = update_table_idx[i];
    update_offset +=
        get_D_bytes(D_offsets_tensor, weights_tys_tensor, idx, row_alignment);
    update_offsets.push_back(update_offset);
  }

  auto update_offsets_tensor =
      at::tensor(update_offsets, at::device(device).dtype(at::kLong));
  auto table_idx_tensor =
      at::tensor(update_table_idx, at::device(device).dtype(at::kInt));
  auto row_idx_tensor =
      at::tensor(update_row_idx, at::device(device).dtype(at::kLong));

  if (use_cpu) {
    fbgemm_gpu::embedding_inplace_update_cpu(
        dev_weight,
        uvm_weight,
        at::tensor(weights_placements, at::device(device).dtype(at::kInt)),
        at::tensor(weights_offsets, at::device(device).dtype(at::kLong)),
        weights_tys_tensor,
        D_offsets_tensor,
        update_weight,
        table_idx_tensor,
        row_idx_tensor,
        update_offsets_tensor,
        row_alignment);

  } else {
    fbgemm_gpu::embedding_inplace_update_cuda(
        dev_weight,
        uvm_weight,
        at::tensor(weights_placements, at::device(device).dtype(at::kInt)),
        at::tensor(weights_offsets, at::device(device).dtype(at::kLong)),
        weights_tys_tensor,
        D_offsets_tensor,
        update_weight,
        table_idx_tensor,
        row_idx_tensor,
        update_offsets_tensor,
        row_alignment);
  }

  // Validation
  auto dev_weight_cpu = dev_weight.cpu();
  auto uvm_weight_cpu = uvm_weight.cpu();
  auto update_weight_cpu = update_weight.cpu();
  int offset = 0;
  for (int i = 0; i < update_table_idx.size(); i++) {
    auto table_idx = update_table_idx[i];
    auto row_idx = update_row_idx[i];
    auto weight_offset = weights_offsets[table_idx];
    auto weight_placement = weights_placements[table_idx];
    auto D = D_offsets[table_idx + 1] - D_offsets[table_idx];
    SparseType ty = static_cast<SparseType>(weights_tys[table_idx]);
    int32_t D_bytes = nbit::padded_row_size_in_bytes(D, ty, 16);
    auto dev_weight_acc = dev_weight_cpu.data_ptr<uint8_t>();
    auto uvm_weight_acc = uvm_weight_cpu.data_ptr<uint8_t>();
    auto update_weight_acc = update_weight_cpu.data_ptr<uint8_t>();
    if (weight_placement == 0) {
      for (const auto j : c10::irange(D_bytes)) {
        ASSERT_EQ(
            dev_weight_acc[weight_offset + D_bytes * row_idx + j],
            update_weight_acc[offset + j]);
      }
    } else {
      for (const auto j : c10::irange(D_bytes)) {
        ASSERT_EQ(
            uvm_weight_acc[weight_offset + D_bytes * row_idx + j],
            update_weight_acc[offset + j]);
      }
    }
    offset += D_bytes;
  }
}

TEST(EmbeddingInplaceUpdateTest, random_update) {
  // TODO: Skipping test_embedding_inplace_update<int32_t> because it is
  // unreliable and crashes occasionally.  This should be fixed and re-enabled.
  //
  // test_embedding_inplace_update<int32_t>();
  test_embedding_inplace_update<int64_t>();
}
