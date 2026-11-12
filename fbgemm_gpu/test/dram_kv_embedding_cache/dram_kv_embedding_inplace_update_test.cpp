/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fbgemm_gpu/embedding_common.h>
#include <fbgemm_gpu/embedding_inplace_update.h>
#include <gtest/gtest.h>
#include <torch/csrc/jit/api/module.h>

namespace fbgemm_gpu {
namespace {

// Stands in for KVEmbeddingInference and keeps the table ids, row ids and rows
// embedding_inplace_update_per_table receives.
torch::jit::Module makeKvModuleKeepingWrittenRows() {
  torch::jit::Module kvModule("KvModuleKeepingWrittenRows");
  kvModule.register_attribute(
      "written_table_ids",
      c10::ListType::create(c10::IntType::get()),
      c10::List<int64_t>());
  kvModule.register_attribute(
      "written_row_ids",
      c10::ListType::create(c10::TensorType::get()),
      c10::List<at::Tensor>());
  kvModule.register_attribute(
      "written_rows",
      c10::ListType::create(c10::TensorType::get()),
      c10::List<at::Tensor>());
  kvModule.define(R"(
def embedding_inplace_update_per_table(self, table_id: int, row_ids: Tensor, rows: Tensor) -> None:
    self.written_table_ids.append(table_id)
    self.written_row_ids.append(row_ids.clone())
    self.written_rows.append(rows.clone())
)");
  return kvModule;
}

// Writes rows of two INT4 tables (dims 64 and 128) through the module.
void writeKvRows(
    torch::jit::Module& kvModule,
    const at::Tensor& updateWeights,
    const at::Tensor& updateTableIdx,
    const at::Tensor& updateRowIdx,
    const at::Tensor& updateOffsets) {
  const auto int4 = static_cast<uint8_t>(SparseType::INT4);
  dram_kv_embedding_inplace_update_cpu(
      &kvModule,
      "embedding_inplace_update_per_table",
      /*weights_placements=*/at::zeros({2}, at::kInt),
      /*weights_offsets=*/at::zeros({2}, at::kLong),
      /*weights_tys=*/at::tensor({int4, int4}, at::kByte),
      /*D_offsets=*/at::tensor({0, 64, 192}, at::kInt),
      updateWeights,
      updateTableIdx,
      updateRowIdx,
      updateOffsets,
      /*row_alignment=*/1);
}

} // namespace

// Each table's rows are cut at the width in update_offsets: 2 rows of a dim-64
// INT4 table (36 B, its rowwise width), then 1 row of a dim-128 INT4 table
// stored in an 80 B layout (wider than its 68 B rowwise width).
TEST(DramKvEmbeddingInplaceUpdateTest, CutsEachTableAtItsUpdateOffsetsWidth) {
  auto kvModule = makeKvModuleKeepingWrittenRows();
  const auto updateWeights =
      at::arange(2 * 36 + 80, at::kInt).remainder(251).to(at::kByte);

  writeKvRows(
      kvModule,
      updateWeights,
      /*updateTableIdx=*/at::tensor({0, 0, 1}, at::kInt),
      /*updateRowIdx=*/at::tensor({5L, 9L, 42L}, at::kLong),
      /*updateOffsets=*/at::tensor({0L, 36L, 72L, 152L}, at::kLong));

  const auto writtenRows = kvModule.attr("written_rows").toTensorVector();
  ASSERT_EQ(writtenRows.size(), 2);
  EXPECT_TRUE(
      at::equal(writtenRows[0], updateWeights.slice(0, 0, 72).view({2, 36})));
  EXPECT_TRUE(
      at::equal(writtenRows[1], updateWeights.slice(0, 72, 152).view({1, 80})));
  EXPECT_EQ(
      kvModule.attr("written_table_ids").toIntVector(),
      (std::vector<int64_t>{0, 1}));
  const auto writtenRowIds = kvModule.attr("written_row_ids").toTensorVector();
  ASSERT_EQ(writtenRowIds.size(), 2);
  EXPECT_TRUE(at::equal(writtenRowIds[0], at::tensor({5L, 9L}, at::kLong)));
  EXPECT_TRUE(at::equal(writtenRowIds[1], at::tensor({42L}, at::kLong)));
}

// 3 rows need 4 offsets: a start for each row and the end of the last one.
TEST(DramKvEmbeddingInplaceUpdateTest, RejectsUpdateOffsetsWithoutAnEndOffset) {
  auto kvModule = makeKvModuleKeepingWrittenRows();

  try {
    writeKvRows(
        kvModule,
        at::zeros({152}, at::kByte),
        /*updateTableIdx=*/at::tensor({0, 0, 1}, at::kInt),
        /*updateRowIdx=*/at::tensor({5L, 9L, 42L}, at::kLong),
        /*updateOffsets=*/at::tensor({0L, 36L, 72L}, at::kLong));
    FAIL() << "3 offsets for 3 rows were accepted";
  } catch (const c10::Error& e) {
    EXPECT_NE(
        e.msg().find("update_offsets needs 4 entries, got 3"),
        std::string::npos)
        << e.msg();
  }
  EXPECT_TRUE(kvModule.attr("written_rows").toTensorVector().empty());
}

// 73 bytes of table 0 cannot be cut into 2 equal rows.
TEST(
    DramKvEmbeddingInplaceUpdateTest,
    RejectsTableBytesThatDoNotSplitIntoEqualRows) {
  auto kvModule = makeKvModuleKeepingWrittenRows();

  try {
    writeKvRows(
        kvModule,
        at::zeros({73}, at::kByte),
        /*updateTableIdx=*/at::tensor({0, 0}, at::kInt),
        /*updateRowIdx=*/at::tensor({5L, 9L}, at::kLong),
        /*updateOffsets=*/at::tensor({0L, 36L, 73L}, at::kLong));
    FAIL() << "73 bytes were cut into 2 rows";
  } catch (const c10::Error& e) {
    EXPECT_NE(
        e.msg().find("update_offsets give 73 bytes for 2 rows of table 0"),
        std::string::npos)
        << e.msg();
  }
  EXPECT_TRUE(kvModule.attr("written_rows").toTensorVector().empty());
}

} // namespace fbgemm_gpu
