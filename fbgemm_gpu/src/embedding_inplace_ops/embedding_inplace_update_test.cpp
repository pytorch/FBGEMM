/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <c10/util/irange.h>
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
void test_embedding_inplace_update(bool use_cpu) {
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
  // dev_weights vs uvm_weights routing differs by kernel: the CUDA kernel
  // treats PlacementType::DEVICE as dev_weights, while the CPU kernel treats
  // PlacementType::HOST as dev_weights. Pick the matching enum per device so
  // the harness drives both buffers correctly on either path (the old
  // hard-coded 0/1 convention only matched the CUDA kernel).
  const int32_t devPlacement = use_cpu
      ? static_cast<int32_t>(PlacementType::HOST)
      : static_cast<int32_t>(PlacementType::DEVICE);
  const int32_t uvmPlacement = static_cast<int32_t>(PlacementType::MANAGED);
  // Row alignment differs by kernel (16B on device, 1B on CPU). D_bytes below
  // MUST use the same alignment the kernel will use, otherwise the harness lays
  // out buffers/offsets differently than the kernel scatters (the old
  // hard-coded 16 only matched the device path).
  const int64_t row_alignment = use_cpu ? 1L : 16L;
  for (const auto i : c10::irange(T)) {
    SparseType weight_ty =
        weight_ty_list[folly::Random::rand32() % weight_ty_list.size()];
    weights_tys.push_back(uint8_t(weight_ty));
    int D = 1 << (1 + folly::Random::rand32() % 5); // table dimension
    int32_t D_bytes =
        nbit::padded_row_size_in_bytes(D, weight_ty, row_alignment);
    int total_rows = 10 + folly::Random::rand32() % 50;
    D_offsets.push_back(D_offsets.back() + D);

    int32_t weights_placement =
        (folly::Random::rand32() % 2 == 0) ? devPlacement : uvmPlacement;
    weights_placements.push_back(weights_placement);
    if (weights_placement == devPlacement) {
      weights_offsets.push_back(dev_weights_offset);
      dev_weights_offset += D_bytes * total_rows;
    } else {
      weights_offsets.push_back(uvm_weights_offset);
      uvm_weights_offset += D_bytes * total_rows;
    }
    int n = folly::Random::rand32() % 10 + 5;
    std::set<int32_t> rows;
    for ([[maybe_unused]] const auto j : c10::irange(n)) {
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

  auto device = use_cpu ? at::kCPU : at::kCUDA;

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
  for (const auto i : c10::irange(update_table_idx.size())) {
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
  for (const auto i : c10::irange(update_table_idx.size())) {
    auto table_idx = update_table_idx[i];
    auto row_idx = update_row_idx[i];
    auto weight_offset = weights_offsets[table_idx];
    auto weight_placement = weights_placements[table_idx];
    auto D = D_offsets[table_idx + 1] - D_offsets[table_idx];
    SparseType ty = static_cast<SparseType>(weights_tys[table_idx]);
    int32_t D_bytes = nbit::padded_row_size_in_bytes(D, ty, row_alignment);
    auto dev_weight_acc = dev_weight_cpu.const_data_ptr<uint8_t>();
    auto uvm_weight_acc = uvm_weight_cpu.const_data_ptr<uint8_t>();
    auto update_weight_acc = update_weight_cpu.const_data_ptr<uint8_t>();
    if (weight_placement == devPlacement) {
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

// Positive coverage via the shared harness: a valid (in-range) random update
// must scatter correctly and must NOT trip the new bounds asserts. The harness
// randomly assigns per-table DEVICE/UVM placement, so this also exercises the
// mixed-placement path that the allocation-size row bound must not false-abort.
TEST(EmbeddingInplaceUpdateTest, CpuRandomValidUpdate) {
  test_embedding_inplace_update<int32_t>(/*use_cpu=*/true);
  test_embedding_inplace_update<int64_t>(/*use_cpu=*/true);
}

// Real device-kernel (.cu) coverage: proves the CUDA_KERNEL_ASSERT guards do
// not false-abort valid callers. Skipped when no GPU is present.
//
// There is intentionally NO device NEGATIVE (assert-fires) test: on ROCm
// CUDA_KERNEL_ASSERT compiles to a bare abort(), which SIGABRTs the whole
// process and poisons the GPU/HIP context; gtest death tests fork and CUDA/HIP
// is not fork-safe, so an out-of-range device index cannot be reliably caught
// in-process (it would need the DSA framework / TORCH_USE_CUDA_DSA). The
// out-of-range negative path is instead covered at the CPU-kernel layer
// (Cpu*OutOfRange* below, TORCH_CHECK -> catchable) and the host layer
// (parent diff D112649885).
TEST(EmbeddingInplaceUpdateTest, GpuRandomValidUpdate) {
  if (!at::hasCUDA()) {
    GTEST_SKIP() << "No CUDA/HIP device available";
  }
  test_embedding_inplace_update<int32_t>(/*use_cpu=*/false);
  test_embedding_inplace_update<int64_t>(/*use_cpu=*/false);
}

// Tests for the defense-in-depth bounds asserts added to the inplace-update
// kernels. Only the CPU kernel is exercised here: it uses TORCH_CHECK, which
// throws a catchable c10::Error. The CUDA kernel uses CUDA_KERNEL_ASSERT, which
// on ROCm compiles to a bare abort() that crashes the whole process, so the
// device out-of-range case cannot be a catchable/death gtest in-process; it is
// compile-verified only.

namespace {

// Apply one delta update to (tableIdx, rowIdx) of a single HOST-placement FP32
// table with `totalRows` rows of dimension `D`, via the CPU kernel. Throws (via
// TORCH_CHECK) if a bounds assert fires.
void runSingleTableCpuUpdate(
    int64_t totalRows,
    int32_t D,
    int32_t tableIdx,
    int64_t rowIdx) {
  const SparseType weight_ty = SparseType::FP32;
  const int64_t row_alignment = 1;
  const int32_t D_bytes =
      nbit::padded_row_size_in_bytes(D, weight_ty, row_alignment);

  auto dev_weights =
      at::zeros({totalRows * D_bytes}, at::device(at::kCPU).dtype(at::kByte));
  auto uvm_weights = at::zeros({0}, at::device(at::kCPU).dtype(at::kByte));
  // CPU kernel routes PlacementType::HOST to dev_weights.
  auto weights_placements = at::tensor(
      {static_cast<int32_t>(PlacementType::HOST)}, at::dtype(at::kInt));
  auto weights_offsets = at::tensor({int64_t(0)}, at::dtype(at::kLong));
  auto weights_tys =
      at::tensor({static_cast<uint8_t>(weight_ty)}, at::dtype(at::kByte));
  auto D_offsets = at::tensor({0, D}, at::dtype(at::kInt));
  auto update_weights =
      at::zeros({D_bytes}, at::device(at::kCPU).dtype(at::kByte));
  auto update_table_idx = at::tensor({tableIdx}, at::dtype(at::kInt));
  auto update_row_idx = at::tensor({rowIdx}, at::dtype(at::kLong));
  auto update_offsets =
      at::tensor({int64_t(0), int64_t(D_bytes)}, at::dtype(at::kLong));

  embedding_inplace_update_cpu(
      dev_weights,
      uvm_weights,
      weights_placements,
      weights_offsets,
      weights_tys,
      D_offsets,
      update_weights,
      update_table_idx,
      update_row_idx,
      update_offsets,
      row_alignment);
}

} // namespace

TEST(EmbeddingInplaceUpdateTest, CpuRejectsOutOfRangeTableIdx) {
  // Single table (num_tables == 1); table index 5 is out of range and would
  // index per-table metadata out of bounds. Must throw, not read wild memory.
  EXPECT_ANY_THROW(runSingleTableCpuUpdate(
      /*totalRows=*/8, /*D=*/4, /*tableIdx=*/5, /*rowIdx=*/0));
}

TEST(EmbeddingInplaceUpdateTest, CpuRejectsOutOfRangeRowIdx) {
  // Row 100 in an 8-row table scatters past dev_weights. Must throw.
  EXPECT_ANY_THROW(runSingleTableCpuUpdate(
      /*totalRows=*/8, /*D=*/4, /*tableIdx=*/0, /*rowIdx=*/100));
}

TEST(EmbeddingInplaceUpdateTest, CpuValidUpdateSucceeds) {
  // In-range update must not false-abort.
  EXPECT_NO_THROW(runSingleTableCpuUpdate(
      /*totalRows=*/8, /*D=*/4, /*tableIdx=*/0, /*rowIdx=*/5));
}

TEST(EmbeddingInplaceUpdateTest, CpuMixedPlacementValidUpdateSucceeds) {
  // Two tables under MIXED placement: table 0 HOST (dev_weights), table 1
  // non-HOST (uvm_weights). weights_offsets is per-placement-buffer, so both
  // start at offset 0 in their own buffer. A valid in-range update to each
  // table must NOT abort. This locks in the allocation-size row bound: a
  // per-table weights_offsets[table_idx + 1] bound would treat table 1's
  // offset (0, in the uvm buffer) as table 0's end and false-abort this
  // valid caller.
  const SparseType weight_ty = SparseType::FP32;
  const int64_t row_alignment = 1;
  const int32_t D = 4;
  const int32_t D_bytes =
      nbit::padded_row_size_in_bytes(D, weight_ty, row_alignment);
  const int64_t totalRows = 8;

  auto dev_weights =
      at::zeros({totalRows * D_bytes}, at::device(at::kCPU).dtype(at::kByte));
  auto uvm_weights =
      at::zeros({totalRows * D_bytes}, at::device(at::kCPU).dtype(at::kByte));
  // table 0 -> dev_weights (HOST), table 1 -> uvm_weights (non-HOST).
  auto weights_placements = at::tensor(
      {static_cast<int32_t>(PlacementType::HOST),
       static_cast<int32_t>(PlacementType::DEVICE)},
      at::dtype(at::kInt));
  auto weights_offsets =
      at::tensor({int64_t(0), int64_t(0)}, at::dtype(at::kLong));
  auto weights_tys = at::tensor(
      {static_cast<uint8_t>(weight_ty), static_cast<uint8_t>(weight_ty)},
      at::dtype(at::kByte));
  auto D_offsets = at::tensor({0, D, 2 * D}, at::dtype(at::kInt));
  auto update_weights =
      at::zeros({2 * D_bytes}, at::device(at::kCPU).dtype(at::kByte));
  auto update_table_idx = at::tensor({0, 1}, at::dtype(at::kInt));
  auto update_row_idx =
      at::tensor({int64_t(5), int64_t(6)}, at::dtype(at::kLong));
  auto update_offsets = at::tensor(
      {int64_t(0), int64_t(D_bytes), int64_t(2 * D_bytes)},
      at::dtype(at::kLong));

  EXPECT_NO_THROW(embedding_inplace_update_cpu(
      dev_weights,
      uvm_weights,
      weights_placements,
      weights_offsets,
      weights_tys,
      D_offsets,
      update_weights,
      update_table_idx,
      update_row_idx,
      update_offsets,
      row_alignment));
}
