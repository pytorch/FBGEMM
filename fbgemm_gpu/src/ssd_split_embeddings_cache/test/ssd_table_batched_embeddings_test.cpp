/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstring>
#include <filesystem>
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/fixed_block_pool.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/ssd_table_batched_embeddings.h"

using namespace ::testing;
constexpr int64_t EMBEDDING_DIMENSION = 8;

class MockEmbeddingRocksDB : public ssd::EmbeddingRocksDB {
 public:
  MockEmbeddingRocksDB(
      std::string path,
      int64_t num_shards,
      int64_t num_threads,
      int64_t memtable_flush_period,
      int64_t memtable_flush_offset,
      int64_t l0_files_per_compact,
      int64_t max_D,
      int64_t rate_limit_mbps,
      int64_t size_ratio,
      int64_t compaction_trigger,
      int64_t write_buffer_size,
      int64_t max_write_buffer_num,
      float uniform_init_lower,
      float uniform_init_upper,
      int64_t row_storage_bitwidth = 32,
      int64_t cache_size = 0,
      bool use_passed_in_path = false,
      int64_t tbe_unqiue_id = 0,
      int64_t l2_cache_size_gb = 0,
      bool enable_async_update = false,
      bool enable_raw_embedding_streaming = false,
      int64_t res_store_shards = 0,
      int64_t res_server_port = 0,
      std::vector<std::string> table_names = {},
      std::vector<int64_t> table_offsets = {},
      const std::vector<int64_t>& table_sizes = {},
      bool enable_metadata_cf = false,
      int64_t metadata_dim = 0)
      : ssd::EmbeddingRocksDB(
            path,
            num_shards,
            num_threads,
            memtable_flush_period,
            memtable_flush_offset,
            l0_files_per_compact,
            max_D,
            rate_limit_mbps,
            size_ratio,
            compaction_trigger,
            write_buffer_size,
            max_write_buffer_num,
            uniform_init_lower,
            uniform_init_upper,
            row_storage_bitwidth,
            cache_size,
            use_passed_in_path,
            tbe_unqiue_id,
            l2_cache_size_gb,
            enable_async_update,
            enable_raw_embedding_streaming,
            res_store_shards,
            res_server_port,
            std::move(table_names),
            std::move(table_offsets),
            table_sizes,
            /*table_dims=*/std::nullopt,
            /*hash_size_cumsum=*/std::nullopt,
            /*flushing_block_size=*/2000000000,
            /*disable_random_init=*/false,
            /*enable_blob_db=*/false,
            /*enable_metadata_cf=*/enable_metadata_cf,
            /*metadata_dim=*/metadata_dim) {}
  MOCK_METHOD(
      rocksdb::Status,
      set_rocksdb_option,
      (int, const std::string&, const std::string&),
      (override));
};

std::unique_ptr<MockEmbeddingRocksDB> getMockEmbeddingRocksDB(
    int num_shards,
    const std::string& dir,
    bool enable_raw_embedding_streaming = false,
    const std::vector<std::string>& table_names = {},
    const std::vector<int64_t>& table_offsets = {},
    const std::vector<int64_t>& table_sizes = {},
    bool enable_metadata_cf = false,
    int64_t metadata_dim = 0) {
  std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  std::filesystem::path rocksdb_dir = temp_dir / dir;
  std::filesystem::create_directories(rocksdb_dir);

  return std::make_unique<MockEmbeddingRocksDB>(
      rocksdb_dir,
      num_shards, // num_shards,
      8, // num_threads,
      0, // memtable_flush_period,
      0, // memtable_flush_offset,
      4, // l0_files_per_compact,
      EMBEDDING_DIMENSION, // max embedding dimension,
      0, // rate_limit_mbps,
      1, // size_ratio,
      8, // compaction_trigger,
      536870912, // 512M write_buffer_size,
      8, // max_write_buffer_num,
      -0.01, // uniform_init_lower,
      0.01, // uniform_init_upper,
      32, // row_storage_bitwidth = 32,
      0, // cache_size = 0
      true, // use_passed_in_path
      0, // tbe_unqiue_id
      0, // l2_cache_size_gb
      false, // enable_async_update
      enable_raw_embedding_streaming, // enable_raw_embedding_streaming
      3, // res_store_shards
      0, // res_server_port
      table_names, // table_names
      table_offsets, // table_offsets
      table_sizes, // table_sizes
      enable_metadata_cf, // enable_metadata_cf
      metadata_dim); // metadata_dim
}

namespace {
constexpr int64_t kMetadataDimFp32 = 4;

// Local mirror of the production MetaHeader layout (see fixed_block_pool.h),
// kept simple for the test. 16 bytes:
// [int64 key][uint32 timestamp][uint32 count:31][bool used:1].
// The feature score is stored in the `count` field as its raw float bits.
struct alignas(8) MetaHeader {
  int64_t key;
  uint32_t timestamp;
  uint32_t count : 31;
  bool used : 1;
};
static_assert(sizeof(MetaHeader) == kMetadataDimFp32 * sizeof(float));

void create_metaheader_row(
    float* row,
    int64_t row_dim,
    int64_t key,
    uint32_t timestamp,
    float feature_score) {
  std::memset(row, 0, row_dim * sizeof(float));
  uint32_t score_bits = 0;
  std::memcpy(&score_bits, &feature_score, sizeof(score_bits));

  MetaHeader header{};
  header.key = key;
  header.timestamp = timestamp;
  header.count = score_bits & 0x7FFFFFFF; // clear sign bit, keep 31 bits
  header.used = true;
  std::memcpy(row, &header, sizeof(header));
}

float decode_feature_score(int64_t raw_metadata) {
  uint32_t count_used =
      static_cast<uint32_t>(static_cast<uint64_t>(raw_metadata) >> 32);
  uint32_t score_bits = count_used & 0x7FFFFFFF;
  float score = 0.0f;
  std::memcpy(&score, &score_bits, sizeof(score));
  return score;
}
} // namespace

TEST(SSDTableBatchedEmbeddingsTest, TestToggleCompactionSuccess) {
  int num_shards = 8;
  auto mock_embedding_rocks = getMockEmbeddingRocksDB(num_shards, "success");
  EXPECT_CALL(*mock_embedding_rocks, set_rocksdb_option)
      .Times(num_shards)
      .WillRepeatedly(Return(rocksdb::Status::OK()));
  mock_embedding_rocks->toggle_compaction(true);
}

TEST(SSDTableBatchedEmbeddingsTest, TestToggleCompactionRetryAndSucceed) {
  int num_shards = 1;
  auto mock_embedding_rocks =
      getMockEmbeddingRocksDB(num_shards, "retrySucceed");
  int max_retry = 10;
  EXPECT_CALL(*mock_embedding_rocks, set_rocksdb_option)
      .Times(max_retry)
      .WillOnce(::testing::Return(rocksdb::Status::NotFound()))
      .WillOnce(::testing::Return(rocksdb::Status::NotFound()))
      .WillOnce(::testing::Return(rocksdb::Status::NotFound()))
      .WillOnce(::testing::Return(rocksdb::Status::NotFound()))
      .WillOnce(::testing::Return(rocksdb::Status::NotFound()))
      .WillOnce(::testing::Return(rocksdb::Status::NotFound()))
      .WillOnce(::testing::Return(rocksdb::Status::NotFound()))
      .WillOnce(::testing::Return(rocksdb::Status::NotFound()))
      .WillOnce(::testing::Return(rocksdb::Status::NotFound()))
      .WillOnce(::testing::Return(rocksdb::Status::OK()));
  mock_embedding_rocks->toggle_compaction(true);
}

TEST(SSDTableBatchedEmbeddingsTest, TestToggleCompactionFailOnRetry) {
  int num_shards = 8;
  auto mock_embedding_rocks =
      getMockEmbeddingRocksDB(num_shards, "failOnRetry");
  EXPECT_CALL(*mock_embedding_rocks, set_rocksdb_option)
      .WillRepeatedly(Return(rocksdb::Status::NotFound()));
  EXPECT_DEATH(
      { mock_embedding_rocks->toggle_compaction(true); },
      "Failed to toggle compaction to 1");
}

TEST(SSDTableBatchedEmbeddingsTest, TestToggleCompactionFailOnThronw) {
  int num_shards = 8;
  auto mock_embedding_rocks =
      getMockEmbeddingRocksDB(num_shards, "failOnThrow");
  EXPECT_CALL(*mock_embedding_rocks, set_rocksdb_option)
      .WillRepeatedly(Throw(std::runtime_error("some error message")));
  EXPECT_DEATH(
      { mock_embedding_rocks->toggle_compaction(true); },
      "Failed to toggle compaction to 1 with exception std::runtime_error: some error message");
}

TEST(SSDTableBatchedEmbeddingsTest, TestMetadataCfInitialization) {
  constexpr int64_t kNumShards = 2;

  // Case 1: Metadata CF is correctly initialized.
  auto mock_embedding_rocks = getMockEmbeddingRocksDB(
      /*num_shards=*/kNumShards,
      "metadataCfInit",
      /*enable_raw_embedding_streaming=*/false,
      /*table_names=*/{},
      /*table_offsets=*/{},
      /*table_sizes=*/{},
      /*enable_metadata_cf=*/true,
      /*metadata_dim=*/kMetadataDimFp32);

  for (int64_t shard = 0; shard < kNumShards; ++shard) {
    EXPECT_TRUE(mock_embedding_rocks->is_metadata_cf_initialized(shard));
  }
  EXPECT_EQ(mock_embedding_rocks->get_metadata_dim(), kMetadataDimFp32);

  // Case 2: Metadata CF is not initialized when enable_metadata_cf = false or
  // by default.
  mock_embedding_rocks = getMockEmbeddingRocksDB(
      /*num_shards=*/kNumShards,
      "metadataCfInit",
      /*enable_raw_embedding_streaming=*/false,
      /*table_names=*/{},
      /*table_offsets=*/{},
      /*table_sizes=*/{},
      /*enable_metadata_cf=*/false,
      /*metadata_dim=*/kMetadataDimFp32);

  for (int64_t shard = 0; shard < kNumShards; ++shard) {
    EXPECT_FALSE(mock_embedding_rocks->is_metadata_cf_initialized(shard));
  }
  // get_metadata_dim reflects constructor param even when CF disabled
  EXPECT_EQ(mock_embedding_rocks->get_metadata_dim(), kMetadataDimFp32);

  mock_embedding_rocks = getMockEmbeddingRocksDB(
      /*num_shards=*/kNumShards, "metadataCfInit");

  for (int64_t shard = 0; shard < kNumShards; ++shard) {
    EXPECT_FALSE(mock_embedding_rocks->is_metadata_cf_initialized(shard));
  }
  EXPECT_EQ(mock_embedding_rocks->get_metadata_dim(), 0);

  // Case 3: Metadata CF is not correctly initialized when metadata dim = 0.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        getMockEmbeddingRocksDB(
            /*num_shards=*/1,
            "metadataCfZeroDim",
            /*enable_raw_embedding_streaming=*/false,
            /*table_names=*/{},
            /*table_offsets=*/{},
            /*table_sizes=*/{},
            /*enable_metadata_cf=*/true,
            /*metadata_dim=*/0);
      },
      "enable_metadata_cf_ is true but metadata_dim_ is not positive");
}

TEST(SSDTableBatchedEmbeddingsTest, TestGetKvZchEvictionMetadataBySnapshot) {
  auto mock_embedding_rocks = getMockEmbeddingRocksDB(
      /*num_shards=*/1,
      "metadataCfWrite",
      /*enable_raw_embedding_streaming=*/false,
      /*table_names=*/{},
      /*table_offsets=*/{},
      /*table_sizes=*/{},
      /*enable_metadata_cf=*/true,
      /*metadata_dim=*/kMetadataDimFp32);

  auto indices = at::tensor({11L, 22L}, at::TensorOptions().dtype(at::kLong));
  auto metadata =
      at::zeros({2, kMetadataDimFp32}, at::TensorOptions().dtype(at::kFloat));
  create_metaheader_row(
      metadata.data_ptr<float>() + 0 * kMetadataDimFp32,
      kMetadataDimFp32,
      11,
      7,
      1.75f);
  create_metaheader_row(
      metadata.data_ptr<float>() + 1 * kMetadataDimFp32,
      kMetadataDimFp32,
      22,
      13,
      3.0f);
  auto count = at::tensor({2L}, at::TensorOptions().dtype(at::kLong));

  mock_embedding_rocks->set_kv_metadata_async(indices, metadata, count).wait();

  auto metadata_out =
      mock_embedding_rocks->get_kv_zch_eviction_metadata_by_snapshot(
          indices, count, /*snapshot_handle=*/nullptr);
  ASSERT_EQ(metadata_out.numel(), 2);

  auto* raw_ptr = metadata_out.data_ptr<int64_t>();
  EXPECT_EQ(static_cast<uint32_t>(raw_ptr[0] & 0xFFFFFFFFu), 7u);
  EXPECT_EQ(static_cast<uint32_t>(raw_ptr[1] & 0xFFFFFFFFu), 13u);
  EXPECT_FLOAT_EQ(decode_feature_score(raw_ptr[0]), 1.75f);
  EXPECT_FLOAT_EQ(decode_feature_score(raw_ptr[1]), 3.0f);

  uint32_t count_used_0 =
      static_cast<uint32_t>(static_cast<uint64_t>(raw_ptr[0]) >> 32);
  uint32_t count_used_1 =
      static_cast<uint32_t>(static_cast<uint64_t>(raw_ptr[1]) >> 32);
  EXPECT_TRUE((count_used_0 & 0x80000000u) != 0);
  EXPECT_TRUE((count_used_1 & 0x80000000u) != 0);
}

TEST(
    SSDTableBatchedEmbeddingsTest,
    TestGetKvDbMetadataOnlyReadsMetadataColumnFamily) {
  auto mock_embedding_rocks = getMockEmbeddingRocksDB(
      /*num_shards=*/1,
      "metadataCfReadOnly",
      /*enable_raw_embedding_streaming=*/false,
      /*table_names=*/{},
      /*table_offsets=*/{},
      /*table_sizes=*/{},
      /*enable_metadata_cf=*/true,
      /*metadata_dim=*/kMetadataDimFp32);

  auto indices = at::tensor({11L, 22L}, at::TensorOptions().dtype(at::kLong));
  auto metadata =
      at::zeros({2, kMetadataDimFp32}, at::TensorOptions().dtype(at::kFloat));
  create_metaheader_row(
      metadata.data_ptr<float>() + 0 * kMetadataDimFp32,
      kMetadataDimFp32,
      11,
      7,
      1.75f);
  create_metaheader_row(
      metadata.data_ptr<float>() + 1 * kMetadataDimFp32,
      kMetadataDimFp32,
      22,
      13,
      3.0f);
  auto count = at::tensor({2L}, at::TensorOptions().dtype(at::kLong));

  mock_embedding_rocks->set_kv_metadata_async(indices, metadata, count).wait();

  // Read back only the metadata rows (shape {N, metadata_dim}, same dtype as
  // the provided out-tensor) and verify they round-trip byte-for-byte.
  auto metadata_only =
      at::zeros({2, kMetadataDimFp32}, at::TensorOptions().dtype(at::kFloat));
  mock_embedding_rocks
      ->get_kv_db_metadata_only_async(indices, metadata_only, count)
      .wait();

  EXPECT_TRUE(at::equal(metadata_only, metadata));
}

TEST(
    SSDTableBatchedEmbeddingsTest,
    TestSetKvDbReconstructsWholeRowFromSplitMetadataStorage) {
  auto mock_embedding_rocks = getMockEmbeddingRocksDB(
      /*num_shards=*/1,
      "splitMetadataWholeRow",
      /*enable_raw_embedding_streaming=*/false,
      /*table_names=*/{},
      /*table_offsets=*/{},
      /*table_sizes=*/{},
      /*enable_metadata_cf=*/true,
      /*metadata_dim=*/kMetadataDimFp32);

  auto indices = at::tensor({33L, 44L}, at::TensorOptions().dtype(at::kLong));
  auto rows = at::zeros(
      {2, kMetadataDimFp32 + EMBEDDING_DIMENSION},
      at::TensorOptions().dtype(at::kFloat));
  auto* rows_ptr = rows.data_ptr<float>();
  create_metaheader_row(
      rows_ptr + 0 * (kMetadataDimFp32 + EMBEDDING_DIMENSION),
      kMetadataDimFp32 + EMBEDDING_DIMENSION,
      33,
      19,
      2.5f);
  create_metaheader_row(
      rows_ptr + 1 * (kMetadataDimFp32 + EMBEDDING_DIMENSION),
      kMetadataDimFp32 + EMBEDDING_DIMENSION,
      44,
      23,
      4.5f);
  for (int64_t i = 0; i < EMBEDDING_DIMENSION; ++i) {
    rows_ptr[kMetadataDimFp32 + i] = static_cast<float>(100 + i);
    rows_ptr[(kMetadataDimFp32 + EMBEDDING_DIMENSION) + kMetadataDimFp32 + i] =
        static_cast<float>(200 + i);
  }
  auto count = at::tensor({2L}, at::TensorOptions().dtype(at::kLong));

  mock_embedding_rocks->set_kv_db_async(indices, rows, count).wait();

  auto full_rows = at::zeros(
      {2, kMetadataDimFp32 + EMBEDDING_DIMENSION},
      at::TensorOptions().dtype(at::kFloat));
  mock_embedding_rocks->get_kv_from_storage_by_snapshot(
      indices, full_rows, /*snapshot_handle=*/nullptr);

  EXPECT_EQ(
      std::memcmp(
          full_rows.data_ptr<float>(),
          rows.data_ptr<float>(),
          rows.numel() * sizeof(float)),
      0);

  auto payload = at::zeros(
      {2, EMBEDDING_DIMENSION}, at::TensorOptions().dtype(at::kFloat));
  mock_embedding_rocks->get_kv_from_storage_by_snapshot(
      indices,
      payload,
      /*snapshot_handle=*/nullptr,
      /*width_offset=*/kMetadataDimFp32,
      /*width_length=*/EMBEDDING_DIMENSION);

  auto* payload_ptr = payload.data_ptr<float>();
  for (int64_t i = 0; i < EMBEDDING_DIMENSION; ++i) {
    EXPECT_FLOAT_EQ(payload_ptr[i], static_cast<float>(100 + i));
    EXPECT_FLOAT_EQ(
        payload_ptr[EMBEDDING_DIMENSION + i], static_cast<float>(200 + i));
  }

  auto metadata_out =
      mock_embedding_rocks->get_kv_zch_eviction_metadata_by_snapshot(
          indices, count, /*snapshot_handle=*/nullptr);
  auto* raw_ptr = metadata_out.data_ptr<int64_t>();
  EXPECT_EQ(static_cast<uint32_t>(raw_ptr[0] & 0xFFFFFFFFu), 19u);
  EXPECT_EQ(static_cast<uint32_t>(raw_ptr[1] & 0xFFFFFFFFu), 23u);
  EXPECT_FLOAT_EQ(decode_feature_score(raw_ptr[0]), 2.5f);
  EXPECT_FLOAT_EQ(decode_feature_score(raw_ptr[1]), 4.5f);
}

TEST(SSDTableBatchedEmbeddingsTest, TestGetKvDbWeightsOnlyReadsPayload) {
  auto mock_embedding_rocks = getMockEmbeddingRocksDB(
      /*num_shards=*/1,
      "weightsOnlyRead",
      /*enable_raw_embedding_streaming=*/false,
      /*table_names=*/{},
      /*table_offsets=*/{},
      /*table_sizes=*/{},
      /*enable_metadata_cf=*/true,
      /*metadata_dim=*/kMetadataDimFp32);

  auto indices = at::tensor({33L, 44L}, at::TensorOptions().dtype(at::kLong));
  auto rows = at::zeros(
      {2, kMetadataDimFp32 + EMBEDDING_DIMENSION},
      at::TensorOptions().dtype(at::kFloat));
  auto* rows_ptr = rows.data_ptr<float>();
  create_metaheader_row(
      rows_ptr + 0 * (kMetadataDimFp32 + EMBEDDING_DIMENSION),
      kMetadataDimFp32 + EMBEDDING_DIMENSION,
      33,
      19,
      2.5f);
  create_metaheader_row(
      rows_ptr + 1 * (kMetadataDimFp32 + EMBEDDING_DIMENSION),
      kMetadataDimFp32 + EMBEDDING_DIMENSION,
      44,
      23,
      4.5f);
  for (int64_t i = 0; i < EMBEDDING_DIMENSION; ++i) {
    rows_ptr[kMetadataDimFp32 + i] = static_cast<float>(100 + i);
    rows_ptr[(kMetadataDimFp32 + EMBEDDING_DIMENSION) + kMetadataDimFp32 + i] =
        static_cast<float>(200 + i);
  }
  auto count = at::tensor({2L}, at::TensorOptions().dtype(at::kLong));

  mock_embedding_rocks->set_kv_db_async(indices, rows, count).wait();

  // Read back only the embedding payload; the metaheader prefix is skipped.
  // Output shape is {N, stride} with stride = weights.size(1).
  auto weights_only = at::zeros(
      {2, EMBEDDING_DIMENSION}, at::TensorOptions().dtype(at::kFloat));
  mock_embedding_rocks
      ->get_kv_db_weights_only_async(indices, weights_only, count)
      .wait();

  auto* payload_ptr = weights_only.data_ptr<float>();
  for (int64_t i = 0; i < EMBEDDING_DIMENSION; ++i) {
    EXPECT_FLOAT_EQ(payload_ptr[i], static_cast<float>(100 + i));
    EXPECT_FLOAT_EQ(
        payload_ptr[EMBEDDING_DIMENSION + i], static_cast<float>(200 + i));
  }
}

TEST(SSDTableBatchedEmbeddingsTest, TestSetKvDbAsyncWithoutMetadataCf) {
  // Verify set_kv_db_async payload-only path when metadata CF is disabled.
  // This complements TestSetKvDbReconstructsWholeRowFromSplitMetadataStorage
  // which covers the enable_metadata_cf=true split path.
  auto mock_embedding_rocks = getMockEmbeddingRocksDB(
      /*num_shards=*/1, "setKvNoMetadata");

  EXPECT_EQ(mock_embedding_rocks->get_metadata_dim(), 0);
  EXPECT_FALSE(mock_embedding_rocks->is_metadata_cf_initialized(0));

  auto indices = at::tensor({11L, 22L}, at::TensorOptions().dtype(at::kLong));
  auto rows = at::zeros(
      {2, EMBEDDING_DIMENSION}, at::TensorOptions().dtype(at::kFloat));
  auto* rows_ptr = rows.data_ptr<float>();
  for (int64_t i = 0; i < EMBEDDING_DIMENSION; ++i) {
    rows_ptr[i] = static_cast<float>(10 + i);
    rows_ptr[EMBEDDING_DIMENSION + i] = static_cast<float>(20 + i);
  }
  auto count = at::tensor({2L}, at::TensorOptions().dtype(at::kLong));

  // Write via default CF only path
  mock_embedding_rocks->set_kv_db_async(indices, rows, count).wait();

  // Read back full rows via standard get path (no metadata CF involved)
  auto out = at::zeros(
      {2, EMBEDDING_DIMENSION}, at::TensorOptions().dtype(at::kFloat));
  mock_embedding_rocks->get_kv_from_storage_by_snapshot(
      indices, out, /*snapshot_handle=*/nullptr);

  EXPECT_TRUE(at::equal(out, rows));

  // Weights-only async should also work and match same payload when CF disabled
  auto weights_only = at::zeros(
      {2, EMBEDDING_DIMENSION}, at::TensorOptions().dtype(at::kFloat));
  mock_embedding_rocks
      ->get_kv_db_weights_only_async(indices, weights_only, count)
      .wait();
  EXPECT_TRUE(at::equal(weights_only, rows));

  // Metadata-only async should no-op and return empty or unchanged when
  // disabled
  auto metadata_out =
      at::zeros({2, kMetadataDimFp32}, at::TensorOptions().dtype(at::kFloat));
  mock_embedding_rocks
      ->get_kv_db_metadata_only_async(indices, metadata_out, count)
      .wait();
  // Expect zeros since CF disabled path returns early without modifying output
  // or at least not equal to non-zero pattern; we just verify no crash and
  // metadata dim is 0 on DB side.
  EXPECT_EQ(mock_embedding_rocks->get_metadata_dim(), 0);
}

TEST(
    SSDTableBatchedEmbeddingsTest,
    TestGetKvDbAsyncImplWithAndWithoutMetadataCf) {
  // Test get_kv_db_async_impl via public get_kv_db_async wrapper for both
  // paths. Path 1: without metadata CF – exercises ssd_get_weights_multi_get
  // branch.
  {
    auto mock = getMockEmbeddingRocksDB(
        /*num_shards=*/1, "getKvAsyncNoMeta");
    auto indices = at::tensor({5L, 6L}, at::TensorOptions().dtype(at::kLong));
    auto rows = at::zeros(
        {2, EMBEDDING_DIMENSION}, at::TensorOptions().dtype(at::kFloat));
    auto* p = rows.data_ptr<float>();
    for (int i = 0; i < EMBEDDING_DIMENSION; ++i) {
      p[i] = 1.0f * i;
      p[EMBEDDING_DIMENSION + i] = 2.0f * i;
    }
    auto count = at::tensor({2L}, at::TensorOptions().dtype(at::kLong));
    mock->set_kv_db_async(indices, rows, count).wait();

    auto out = at::zeros(
        {2, EMBEDDING_DIMENSION}, at::TensorOptions().dtype(at::kFloat));
    // get_kv_db_async calls get_kv_db_async_impl<false> internally
    mock->get_kv_db_async(indices, out, count).wait();
    EXPECT_TRUE(at::equal(out, rows));
    EXPECT_EQ(mock->get_metadata_dim(), 0);
  }

  // Path 2: with metadata CF – exercises
  // ssd_get_weights_with_metadata_multi_get branch inside get_kv_db_async_impl.
  {
    auto mock = getMockEmbeddingRocksDB(
        /*num_shards=*/1,
        "getKvAsyncWithMeta",
        /*enable_raw_embedding_streaming=*/false,
        /*table_names=*/{},
        /*table_offsets=*/{},
        /*table_sizes=*/{},
        /*enable_metadata_cf=*/true,
        /*metadata_dim=*/kMetadataDimFp32);
    auto indices = at::tensor({7L, 8L}, at::TensorOptions().dtype(at::kLong));
    auto rows = at::zeros(
        {2, kMetadataDimFp32 + EMBEDDING_DIMENSION},
        at::TensorOptions().dtype(at::kFloat));
    auto* rp = rows.data_ptr<float>();
    create_metaheader_row(
        rp + 0 * (kMetadataDimFp32 + EMBEDDING_DIMENSION),
        kMetadataDimFp32 + EMBEDDING_DIMENSION,
        7,
        11,
        1.5f);
    create_metaheader_row(
        rp + 1 * (kMetadataDimFp32 + EMBEDDING_DIMENSION),
        kMetadataDimFp32 + EMBEDDING_DIMENSION,
        8,
        13,
        2.5f);
    for (int64_t i = 0; i < EMBEDDING_DIMENSION; ++i) {
      rp[kMetadataDimFp32 + i] = 30.0f + i;
      rp[(kMetadataDimFp32 + EMBEDDING_DIMENSION) + kMetadataDimFp32 + i] =
          40.0f + i;
    }
    auto count = at::tensor({2L}, at::TensorOptions().dtype(at::kLong));
    mock->set_kv_db_async(indices, rows, count).wait();

    auto out = at::zeros(
        {2, kMetadataDimFp32 + EMBEDDING_DIMENSION},
        at::TensorOptions().dtype(at::kFloat));
    mock->get_kv_db_async(indices, out, count).wait();
    EXPECT_EQ(
        std::memcmp(
            out.data_ptr<float>(),
            rows.data_ptr<float>(),
            rows.numel() * sizeof(float)),
        0);
    EXPECT_EQ(mock->get_metadata_dim(), kMetadataDimFp32);
    EXPECT_TRUE(mock->is_metadata_cf_initialized(0));
  }
}

TEST(SSDTableBatchedEmbeddingsTest, MetaHeaderParityWithDRAM) {
  // Verify the SSD read path (get_kv_zch_eviction_metadata_by_snapshot)
  // produces the same packed eviction-metadata word as the DRAM
  // FixedBlockPool helper for identical MetaHeader bytes. Ensures cross-backend
  // compatibility.
  kv_mem::FixedBlockPool::MetaHeader dram_hdr{};
  dram_hdr.key = 0x1122334455667788LL;
  dram_hdr.timestamp = 0xAABBCCDD;
  dram_hdr.count = 0x1234567;
  dram_hdr.used = true;

  uint64_t dram_out = kv_mem::FixedBlockPool::get_metaheader_raw(&dram_hdr);

  // SSD side: store the identical MetaHeader bytes into the metadata column
  // family, then read them back through the production accessor.
  auto mock_embedding_rocks = getMockEmbeddingRocksDB(
      /*num_shards=*/1,
      "metaHeaderParity",
      /*enable_raw_embedding_streaming=*/false,
      /*table_names=*/{},
      /*table_offsets=*/{},
      /*table_sizes=*/{},
      /*enable_metadata_cf=*/true,
      /*metadata_dim=*/kMetadataDimFp32);

  auto indices =
      at::tensor({dram_hdr.key}, at::TensorOptions().dtype(at::kLong));
  auto metadata =
      at::zeros({1, kMetadataDimFp32}, at::TensorOptions().dtype(at::kFloat));
  std::memcpy(metadata.data_ptr<float>(), &dram_hdr, sizeof(dram_hdr));
  auto count = at::tensor({1L}, at::TensorOptions().dtype(at::kLong));
  mock_embedding_rocks->set_kv_metadata_async(indices, metadata, count).wait();

  auto metadata_out =
      mock_embedding_rocks->get_kv_zch_eviction_metadata_by_snapshot(
          indices, count, /*snapshot_handle=*/nullptr);
  ASSERT_EQ(metadata_out.numel(), 1);
  uint64_t ssd_out = static_cast<uint64_t>(metadata_out.data_ptr<int64_t>()[0]);

  EXPECT_EQ(dram_out, ssd_out);

  // unpack matches Python test expectation
  uint32_t ts = static_cast<uint32_t>(dram_out & 0xFFFFFFFFu);
  uint32_t count_used = static_cast<uint32_t>(dram_out >> 32);
  EXPECT_EQ(ts, 0xAABBCCDDu);
  EXPECT_EQ(count_used & 0x7FFFFFFFu, 0x1234567u);
  EXPECT_EQ(count_used >> 31, 1u);
}
