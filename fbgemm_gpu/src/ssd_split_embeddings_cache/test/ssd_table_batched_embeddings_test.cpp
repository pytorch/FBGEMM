/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <folly/container/F14Map.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <future>
#include <mutex>
#include <thread>
#include <vector>
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/fixed_block_pool.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/ssd_table_batched_embeddings.h"

using namespace ::testing;
constexpr int64_t EMBEDDING_DIMENSION = 8;

/// Holds a lookup at the storage fetch -- the window between the L2 hit/miss
/// split and the fetch filling those misses, which a snapshot switch must not
/// slip into.
///
/// arm_at_fetch() stops at whichever fetch entry point the lookup takes,
/// including one that has not yet taken a guard, so a switch can land inside
/// the window. arm_at_registered_fetch() waits for a fetch that was handed a
/// guard, so the read is already counted. Both are one-shot per arming: the
/// unguarded path re-enters through the guarded one.
class ReadPark {
 public:
  void arm_at_fetch() {
    armed_at_fetch() = true;
    armed_at_registered_fetch() = true;
  }

  void arm_at_registered_fetch() {
    armed_at_registered_fetch() = true;
  }

  void park_at_fetch() {
    if (armed_at_fetch()) {
      park();
    }
  }

  void park_at_registered_fetch() {
    if (armed_at_registered_fetch()) {
      park();
    }
  }

  void wait_until_parked(int64_t reads) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return parked_ >= reads; });
  }

  void release() {
    {
      const std::scoped_lock lock(mutex_);
      released_ = true;
    }
    cv_.notify_all();
  }

 private:
  void park() {
    armed_at_fetch() = false;
    armed_at_registered_fetch() = false;
    std::unique_lock<std::mutex> lock(mutex_);
    ++parked_;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
  }

  static bool& armed_at_fetch() {
    static thread_local bool armed = false;
    return armed;
  }

  static bool& armed_at_registered_fetch() {
    static thread_local bool armed = false;
    return armed;
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  int64_t parked_{0};
  bool released_{false};
};

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

  // Null unless a test installs one, in which case both entry points the
  // lookup can reach the storage tier through are routed through it.
  void set_read_park(ReadPark* read_park) {
    read_park_ = read_park;
  }

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    if (read_park_ != nullptr) {
      read_park_->park_at_fetch();
    }
    return ssd::EmbeddingRocksDB::get_kv_db_async(indices, weights, count);
  }

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async_with_guard(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const std::shared_ptr<kv_db::ReadGuard>& read_guard) override {
    if (read_park_ != nullptr) {
      read_park_->park_at_registered_fetch();
    }
    return ssd::EmbeddingRocksDB::get_kv_db_async_with_guard(
        indices, weights, count, read_guard);
  }

 private:
  ReadPark* read_park_{nullptr};
};

std::unique_ptr<MockEmbeddingRocksDB> getMockEmbeddingRocksDB(
    int num_shards,
    const std::string& dir,
    bool enable_raw_embedding_streaming = false,
    const std::vector<std::string>& table_names = {},
    const std::vector<int64_t>& table_offsets = {},
    const std::vector<int64_t>& table_sizes = {},
    bool enable_metadata_cf = false,
    int64_t metadata_dim = 0,
    int64_t l2_cache_size_gb = 0) {
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
      l2_cache_size_gb, // l2_cache_size_gb
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

namespace {

// Write <value> into every element of the row for each id, one row per id.
void write_rows(
    kv_db::EmbeddingKVDB& db,
    const std::vector<int64_t>& ids,
    float value) {
  auto indices = at::tensor(ids, at::TensorOptions().dtype(at::kLong));
  auto rows = at::full(
      {static_cast<int64_t>(ids.size()), EMBEDDING_DIMENSION},
      value,
      at::TensorOptions().dtype(at::kFloat));
  auto count = at::tensor(
      {static_cast<int64_t>(ids.size())}, at::TensorOptions().dtype(at::kLong));
  db.set_kv_db_async(indices, rows, count).wait();
}

// Read the given ids through the embedding lookup path, the one that eval
// uses and the one that honours the active snapshot.
at::Tensor read_rows(
    ssd::EmbeddingRocksDB& db,
    const std::vector<int64_t>& ids) {
  auto indices = at::tensor(ids, at::TensorOptions().dtype(at::kLong));
  auto out = at::zeros(
      {static_cast<int64_t>(ids.size()), EMBEDDING_DIMENSION},
      at::TensorOptions().dtype(at::kFloat));
  auto count = at::tensor(
      {static_cast<int64_t>(ids.size())}, at::TensorOptions().dtype(at::kLong));
  db.get_kv_db_async(indices, out, count).wait();
  return out;
}

// Read the given ids through EmbeddingKVDB::get(), the cache-aware entry
// point eval actually calls. Unlike read_rows() this goes through the L2
// hit/miss split, so part of the answer can come from the cache and part from
// the storage tier. A fresh indices tensor every call because get() marks L2
// hits in place with a sentinel.
at::Tensor get_rows(kv_db::EmbeddingKVDB& db, const std::vector<int64_t>& ids) {
  auto indices = at::tensor(ids, at::TensorOptions().dtype(at::kLong));
  auto out = at::zeros(
      {static_cast<int64_t>(ids.size()), EMBEDDING_DIMENSION},
      at::TensorOptions().dtype(at::kFloat));
  auto count = at::tensor(
      {static_cast<int64_t>(ids.size())}, at::TensorOptions().dtype(at::kLong));
  db.get(indices, out, count, /*sleep_ms=*/0);
  return out;
}

// Write <value> into every element of the row for each id, through the
// cache-aware entry point so the L2 mutex is actually taken.
void set_rows(
    kv_db::EmbeddingKVDB& db,
    const std::vector<int64_t>& ids,
    float value) {
  auto indices = at::tensor(ids, at::TensorOptions().dtype(at::kLong));
  auto rows = at::full(
      {static_cast<int64_t>(ids.size()), EMBEDDING_DIMENSION},
      value,
      at::TensorOptions().dtype(at::kFloat));
  auto count = at::tensor(
      {static_cast<int64_t>(ids.size())}, at::TensorOptions().dtype(at::kLong));
  db.set(indices, rows, count);
}

std::vector<int64_t> contiguous_ids(int64_t begin, int64_t end) {
  std::vector<int64_t> ids;
  ids.reserve(end - begin);
  for (int64_t id = begin; id < end; ++id) {
    ids.push_back(id);
  }
  return ids;
}

at::Tensor uniform_rows(int64_t num_rows, float value) {
  return at::full(
      {num_rows, EMBEDDING_DIMENSION},
      value,
      at::TensorOptions().dtype(at::kFloat));
}

int64_t count_rows_with_value(const at::Tensor& rows, float value) {
  return at::all(rows == value, /*dim=*/1).to(at::kLong).sum().item<int64_t>();
}

// A read's registration is dropped when the last per-shard closure holding
// its guard is destroyed, which can trail the read returning by a hair.
// Waiting on the count instead of sampling it keeps that from being a flake,
// while a release that never happens still fails, just slowly.
void wait_for_no_reads_in_flight(ssd::EmbeddingRocksDB& db) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (db.get_active_snapshot_read_count() != 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_EQ(db.get_active_snapshot_read_count(), 0)
      << "a read that has already returned is still registered in flight";
}

} // namespace

TEST(SSDTableBatchedEmbeddingsTest, TestNoActiveSnapshotReadsLiveDb) {
  // The default has to stay exactly as it was: no snapshot installed means
  // the lookup path sees the newest write. This is what training relies on.
  auto db = getMockEmbeddingRocksDB(/*num_shards=*/2, "activeSnapshotDefault");
  const std::vector<int64_t> ids{1L, 2L, 3L};

  EXPECT_FALSE(db->has_active_snapshot());
  EXPECT_EQ(db->get_active_snapshot_read_count(), 0);

  write_rows(*db, ids, 1.0f);
  write_rows(*db, ids, 2.0f);

  EXPECT_TRUE(
      at::equal(
          read_rows(*db, ids),
          at::full(
              {3, EMBEDDING_DIMENSION},
              2.0f,
              at::TensorOptions().dtype(at::kFloat))));
}

TEST(SSDTableBatchedEmbeddingsTest, TestReinstallingTheActiveSnapshotIsANoOp) {
  // Re-installing the handle that is already active must return immediately.
  // If it fell through it would retire the same handle it installs, and
  // end_read_snapshot() resolves against the active generation first, so the
  // retiring generation's count would never reach zero and the drain would
  // hang. Bounded on another thread so a regression fails instead of hanging
  // the suite.
  auto db =
      getMockEmbeddingRocksDB(/*num_shards=*/2, "reinstallActiveSnapshot");
  const std::vector<int64_t> ids{1L, 2L, 3L};
  write_rows(*db, ids, 1.0f);

  const auto* snapshot = db->create_snapshot();
  db->set_active_snapshot(snapshot);
  ASSERT_TRUE(db->has_active_snapshot());

  std::promise<void> done;
  auto finished = done.get_future();
  std::thread reinstall([&] {
    db->set_active_snapshot(snapshot);
    done.set_value();
  });
  EXPECT_EQ(
      finished.wait_for(std::chrono::seconds(30)), std::future_status::ready)
      << "re-installing the active snapshot did not return; the drain is "
         "waiting on a generation nothing will decrement";
  reinstall.join();

  // Still installed, and still serving the snapshot rather than the live db.
  EXPECT_TRUE(db->has_active_snapshot());
  write_rows(*db, ids, 2.0f);
  EXPECT_TRUE(
      at::equal(
          read_rows(*db, ids),
          at::full(
              {3, EMBEDDING_DIMENSION},
              1.0f,
              at::TensorOptions().dtype(at::kFloat))));

  db->clear_active_snapshot();
  db->release_snapshot(snapshot);
}

TEST(SSDTableBatchedEmbeddingsTest, TestActiveSnapshotIsolatesLookupReads) {
  // The scenario async checkpoint load needs: a snapshot is installed before
  // a "checkpoint" is streamed into the live db, and lookups keep returning
  // the pre-load view for the whole write.
  auto db = getMockEmbeddingRocksDB(/*num_shards=*/2, "activeSnapshotIsolate");
  const std::vector<int64_t> ids{10L, 11L, 12L};
  const auto pre_load = at::full(
      {3, EMBEDDING_DIMENSION}, 1.0f, at::TensorOptions().dtype(at::kFloat));
  const auto post_load = at::full(
      {3, EMBEDDING_DIMENSION}, 9.0f, at::TensorOptions().dtype(at::kFloat));

  write_rows(*db, ids, 1.0f);

  const auto* snapshot_a = db->create_snapshot();
  db->set_active_snapshot(snapshot_a);
  EXPECT_TRUE(db->has_active_snapshot());

  // Streaming the new checkpoint into the live db.
  write_rows(*db, ids, 9.0f);

  EXPECT_TRUE(at::equal(read_rows(*db, ids), pre_load));

  // Commit: publish the post-load snapshot and switch onto it.
  const auto* snapshot_b = db->create_snapshot();
  db->set_active_snapshot(snapshot_b);
  EXPECT_TRUE(at::equal(read_rows(*db, ids), post_load));

  // Releasing the old snapshot after the switch must not disturb reads.
  db->release_snapshot(snapshot_a);
  EXPECT_TRUE(at::equal(read_rows(*db, ids), post_load));

  // Going back to the live db is the way to restore the training default.
  db->clear_active_snapshot();
  db->release_snapshot(snapshot_b);
  EXPECT_FALSE(db->has_active_snapshot());
  EXPECT_EQ(db->get_snapshot_count(), 0);
  EXPECT_TRUE(at::equal(read_rows(*db, ids), post_load));
}

TEST(
    SSDTableBatchedEmbeddingsTest,
    TestReleasingActiveSnapshotRevertsToLiveDb) {
  // Releasing the snapshot that is still installed must not leave the read
  // path pointing at a handle that is no longer registered. It falls back to
  // the live db, which is the safe default, and it must not crash.
  auto db = getMockEmbeddingRocksDB(/*num_shards=*/2, "activeSnapshotRelease");
  const std::vector<int64_t> ids{20L, 21L};

  write_rows(*db, ids, 3.0f);
  const auto* snapshot = db->create_snapshot();
  db->set_active_snapshot(snapshot);
  write_rows(*db, ids, 4.0f);

  EXPECT_TRUE(
      at::equal(
          read_rows(*db, ids),
          at::full(
              {2, EMBEDDING_DIMENSION},
              3.0f,
              at::TensorOptions().dtype(at::kFloat))));

  db->release_snapshot(snapshot);

  EXPECT_FALSE(db->has_active_snapshot());
  EXPECT_EQ(db->get_snapshot_count(), 0);
  EXPECT_TRUE(
      at::equal(
          read_rows(*db, ids),
          at::full(
              {2, EMBEDDING_DIMENSION},
              4.0f,
              at::TensorOptions().dtype(at::kFloat))));
}

TEST(SSDTableBatchedEmbeddingsTest, TestSwitchDrainsConcurrentReaders) {
  // The lifetime guarantee: a switch installs the new handle and then blocks
  // until every read issued against the old one has finished, so the old
  // snapshot is never freed under a running reader. Reads issued during the
  // drain join the new generation, so the drain terminates even though
  // lookups keep arriving throughout, which is the online eval case.
  auto db = getMockEmbeddingRocksDB(/*num_shards=*/4, "activeSnapshotLifetime");
  std::vector<int64_t> ids;
  ids.reserve(512);
  for (int64_t i = 0; i < 512; ++i) {
    ids.push_back(i);
  }
  const auto pre_load = at::full(
      {512, EMBEDDING_DIMENSION}, 5.0f, at::TensorOptions().dtype(at::kFloat));

  write_rows(*db, ids, 5.0f);
  const auto* snapshot = db->create_snapshot();
  db->set_active_snapshot(snapshot);
  write_rows(*db, ids, 6.0f);

  // Nothing registered while idle.
  EXPECT_EQ(db->get_active_snapshot_read_count(), 0);

  const auto post_load = at::full(
      {512, EMBEDDING_DIMENSION}, 6.0f, at::TensorOptions().dtype(at::kFloat));

  std::atomic<bool> stop{false};
  std::atomic<int64_t> reads{0};
  std::atomic<int64_t> mixed_reads{0};
  constexpr int kNumReaders = 4;
  std::vector<std::thread> readers;
  readers.reserve(kNumReaders);
  for (int t = 0; t < kNumReaders; ++t) {
    readers.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        // Every read either observes the snapshot or, after the switch, the
        // live db. It must never observe freed memory, and it must not come
        // back as a mixture of the two.
        auto out = read_rows(*db, ids);
        if (!at::equal(out, pre_load) && !at::equal(out, post_load)) {
          mixed_reads.fetch_add(1, std::memory_order_relaxed);
        }
        reads.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Let the readers get going, then switch out from under them. The readers
  // keep issuing lookups for the whole duration of this call.
  while (reads.load(std::memory_order_relaxed) < 8) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  db->clear_active_snapshot();
  // Returning means the old generation drained, so releasing is safe now.
  EXPECT_EQ(db->get_active_snapshot_read_count(), 0);
  db->release_snapshot(snapshot);

  stop.store(true, std::memory_order_relaxed);
  for (auto& reader : readers) {
    reader.join();
  }

  EXPECT_EQ(mixed_reads.load(), 0)
      << "a concurrent read came back as neither the snapshot view nor the "
         "live one";
  EXPECT_EQ(db->get_snapshot_count(), 0);
  EXPECT_FALSE(db->has_active_snapshot());
  EXPECT_TRUE(at::equal(read_rows(*db, ids), post_load));
  // Sanity: the pre-load view really was different, so the isolation above
  // was not comparing a value to itself.
  EXPECT_FALSE(at::equal(read_rows(*db, ids), pre_load));
}

TEST(SSDTableBatchedEmbeddingsTest, TestReleasedSnapshotIdStaysInvalid) {
  // Ids identify "my snapshot" after the handle is gone, so a released one
  // must never read as valid again -- not even when the next snapshot reuses
  // the released handle's memory, which is why ids are not addresses.
  auto db = getMockEmbeddingRocksDB(/*num_shards=*/1, "snapshotIdReuse");
  const auto* first = db->create_snapshot();
  const int64_t first_id = first->id();
  db->release_snapshot(first);

  const auto* second = db->create_snapshot();

  EXPECT_NE(second->id(), first_id);
  EXPECT_FALSE(db->is_valid_snapshot_id(first_id));
  EXPECT_TRUE(db->is_valid_snapshot_id(second->id()));
  db->release_snapshot(second);
}

TEST(
    SSDTableBatchedEmbeddingsTest,
    TestSnapshotSwitchDuringLookupDoesNotTearBatch) {
  // The torn read the guard prevents: half the batch answered from L2 warmed
  // on the pre-load view, half fetched from storage. A switch between the two
  // must not return a batch that mixes them. The lookup is parked at the
  // fetch and the switch driven from another thread, so the interleaving is
  // fixed rather than raced for.
  auto db = getMockEmbeddingRocksDB(
      /*num_shards=*/2,
      "switchMidLookup",
      /*enable_raw_embedding_streaming=*/false,
      /*table_names=*/{},
      /*table_offsets=*/{},
      /*table_sizes=*/{},
      /*enable_metadata_cf=*/false,
      /*metadata_dim=*/0,
      /*l2_cache_size_gb=*/1);

  constexpr int64_t kNumRows = 64;
  constexpr float kPreLoad = 1.0f;
  constexpr float kPostLoad = 9.0f;
  const auto ids = contiguous_ids(0, kNumRows);
  const auto cached_ids = contiguous_ids(0, kNumRows / 2);

  write_rows(*db, ids, kPreLoad);
  const auto* snapshot = db->create_snapshot();
  db->set_active_snapshot(snapshot);

  // Warms L2 with the pre-load view; these become the cache hits below.
  EXPECT_TRUE(
      at::equal(
          get_rows(*db, cached_ids), uniform_rows(kNumRows / 2, kPreLoad)));

  // The checkpoint load streaming into the live db behind the snapshot.
  write_rows(*db, ids, kPostLoad);

  ReadPark park;
  db->set_read_park(&park);

  at::Tensor out;
  std::thread reader([&] {
    park.arm_at_fetch();
    out = get_rows(*db, ids);
  });
  park.wait_until_parked(1);

  // The switch blocks until the parked read drains, so it needs its own
  // thread. has_active_snapshot() flips when the new view is installed, which
  // is before the drain, and is therefore the signal that the switch has
  // taken effect for anyone starting a read from now on.
  std::thread switcher([&] { db->clear_active_snapshot(); });
  while (db->has_active_snapshot()) {
    std::this_thread::yield();
  }
  park.release();
  reader.join();
  switcher.join();
  db->set_read_park(nullptr);
  db->release_snapshot(snapshot);

  // The read registered before the switch, so every row has to be pre-load.
  EXPECT_EQ(count_rows_with_value(out, kPreLoad), kNumRows)
      << "batch was stitched from both views: "
      << count_rows_with_value(out, kPreLoad) << " pre-load rows and "
      << count_rows_with_value(out, kPostLoad) << " post-load rows";
}

TEST(SSDTableBatchedEmbeddingsTest, TestInFlightReadCountMatchesLiveReads) {
  // The in-flight count is about to be asserted on by the drain path, so it
  // has to be right for reasons beyond "a read is running". A counter stuck
  // at one would satisfy the single-reader case, and a release that fails to
  // decrement would too, so all three cases are checked.
  auto db = getMockEmbeddingRocksDB(/*num_shards=*/2, "inFlightReadCount");
  const std::vector<int64_t> ids{1L, 2L, 3L, 4L};

  write_rows(*db, ids, 2.0f);
  const auto* snapshot = db->create_snapshot();
  db->set_active_snapshot(snapshot);
  EXPECT_EQ(db->get_active_snapshot_read_count(), 0);

  // One read parked inside get(), nothing else started.
  {
    ReadPark park;
    db->set_read_park(&park);
    std::thread reader([&] {
      park.arm_at_registered_fetch();
      get_rows(*db, ids);
    });
    park.wait_until_parked(1);
    EXPECT_EQ(db->get_active_snapshot_read_count(), 1);
    park.release();
    reader.join();
    db->set_read_park(nullptr);
  }

  // Same, but with a read that has already run to completion behind it. The
  // release for that one has to have decremented, which is the failure that
  // bit us before: a reference outliving the logical read leaves the count
  // permanently high and every drain after it waits on a phantom.
  {
    read_rows(*db, ids);
    ASSERT_NO_FATAL_FAILURE(wait_for_no_reads_in_flight(*db));

    ReadPark park;
    db->set_read_park(&park);
    std::thread reader([&] {
      park.arm_at_registered_fetch();
      get_rows(*db, ids);
    });
    park.wait_until_parked(1);
    EXPECT_EQ(db->get_active_snapshot_read_count(), 1);
    park.release();
    reader.join();
    db->set_read_park(nullptr);
  }

  // Two reads registered at once, so the set above cannot be satisfied by a
  // counter stuck at one. The second read goes to the storage tier directly
  // rather than through get(), because get() holds l2_cache_mtx_ for its
  // whole body and two lookups can never be inside it at the same time.
  {
    ASSERT_NO_FATAL_FAILURE(wait_for_no_reads_in_flight(*db));

    ReadPark park;
    db->set_read_park(&park);
    std::thread lookup([&] {
      park.arm_at_registered_fetch();
      get_rows(*db, ids);
    });
    std::thread fetch([&] {
      park.arm_at_registered_fetch();
      read_rows(*db, ids);
    });
    park.wait_until_parked(2);
    EXPECT_EQ(db->get_active_snapshot_read_count(), 2);
    park.release();
    lookup.join();
    fetch.join();
    db->set_read_park(nullptr);
  }
  ASSERT_NO_FATAL_FAILURE(wait_for_no_reads_in_flight(*db));

  db->clear_active_snapshot();
  db->release_snapshot(snapshot);
}

TEST(
    SSDTableBatchedEmbeddingsTest,
    TestConcurrentLookupsAndSwitchesDoNotDeadlock) {
  // A lookup takes the read guard and then l2_cache_mtx_; a switch holds
  // active_snapshot_mutex_ and waits for guards to drop. That is acyclic only
  // because no lookup ever touches the snapshot state while holding
  // l2_cache_mtx_ -- the guard is declared before the L2 lock and so released
  // after it. This is the test of that reasoning.
  auto db = getMockEmbeddingRocksDB(
      /*num_shards=*/4,
      "lookupSwitchLockOrder",
      /*enable_raw_embedding_streaming=*/false,
      /*table_names=*/{},
      /*table_offsets=*/{},
      /*table_sizes=*/{},
      /*enable_metadata_cf=*/false,
      /*metadata_dim=*/0,
      /*l2_cache_size_gb=*/1);

  const auto ids = contiguous_ids(0, 128);
  write_rows(*db, ids, 4.0f);
  // Two snapshots, the first installed before any lookup: moving off the live
  // DB is only allowed then. Switching between them afterwards is what
  // exercises the drain under load.
  const auto* snapshot_a = db->create_snapshot();
  const auto* snapshot_b = db->create_snapshot();
  db->set_active_snapshot(snapshot_a);

  std::atomic<bool> stop{false};
  std::atomic<int64_t> lookups{0};
  std::atomic<int64_t> switches{0};
  std::atomic<int64_t> writes{0};

  std::vector<std::thread> threads;
  constexpr int kNumReaders = 4;
  threads.reserve(kNumReaders + 1); // readers, plus the single switcher below
  for (int t = 0; t < kNumReaders; ++t) {
    threads.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        get_rows(*db, ids);
        lookups.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  // Exactly one switcher: concurrent switches CHECK-fail by design.
  threads.emplace_back([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      db->set_active_snapshot(snapshot_b);
      db->set_active_snapshot(snapshot_a);
      switches.fetch_add(1, std::memory_order_relaxed);
    }
  });
  // The other holder of l2_cache_mtx_, so the lock is genuinely contended
  // from both sides while snapshots are being switched.
  threads.emplace_back([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      set_rows(*db, ids, 4.0f);
      writes.fetch_add(1, std::memory_order_relaxed);
    }
  });

  constexpr int64_t kMinLookups = 200;
  constexpr int64_t kMinSwitches = 50;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while ((lookups.load() < kMinLookups || switches.load() < kMinSwitches) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  stop.store(true, std::memory_order_relaxed);

  auto joined = std::async(std::launch::async, [&] {
    for (auto& thread : threads) {
      thread.join();
    }
  });
  ASSERT_EQ(
      joined.wait_for(std::chrono::seconds(60)), std::future_status::ready)
      << "lookups, switches and writes did not all finish; the lookup and "
         "switch lock orders have formed a cycle";

  EXPECT_GE(lookups.load(), kMinLookups);
  EXPECT_GE(switches.load(), kMinSwitches);
  EXPECT_GT(writes.load(), 0);

  db->clear_active_snapshot();
  db->release_snapshot(snapshot_a);
  db->release_snapshot(snapshot_b);
  EXPECT_TRUE(at::equal(get_rows(*db, ids), uniform_rows(128, 4.0f)));
}

namespace {

// A backend with a single, unswitchable read view: it does not override
// acquire_read_guard(), which is what DRAM and PS do. Reading through it
// exercises the null-guard default on the base class end to end.
class SingleViewKVDB : public kv_db::EmbeddingKVDB {
 public:
  SingleViewKVDB(int64_t num_shards, int64_t l2_cache_size_gb)
      : kv_db::EmbeddingKVDB(
            num_shards,
            EMBEDDING_DIMENSION,
            l2_cache_size_gb,
            /*unique_id=*/7,
            /*ele_size_bytes=*/static_cast<int64_t>(sizeof(float))) {}

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    fetches.fetch_add(1, std::memory_order_relaxed);
    const auto num_rows = count.item<int64_t>();
    const auto* ids = indices.const_data_ptr<int64_t>();
    auto* out = weights.mutable_data_ptr<float>();
    const std::scoped_lock lock(mutex_);
    for (int64_t row = 0; row < num_rows; ++row) {
      // Negative ids are the sentinel get() writes for rows L2 has answered.
      if (ids[row] < 0) {
        continue;
      }
      const auto found = rows_.find(ids[row]);
      std::fill_n(
          out + row * EMBEDDING_DIMENSION,
          EMBEDDING_DIMENSION,
          found == rows_.end() ? 0.0f : found->second);
    }
    return std::vector<folly::Unit>(1);
  }

  folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const kv_db::RocksdbWriteMode /*w_mode*/ =
          kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ) override {
    const auto num_rows = count.item<int64_t>();
    const auto* ids = indices.const_data_ptr<int64_t>();
    const auto* in = weights.const_data_ptr<float>();
    const std::scoped_lock lock(mutex_);
    for (int64_t row = 0; row < num_rows; ++row) {
      if (ids[row] < 0) {
        continue;
      }
      rows_[ids[row]] = in[row * EMBEDDING_DIMENSION];
    }
    return std::vector<folly::Unit>(1);
  }

  folly::SemiFuture<std::vector<folly::Unit>>
  set_kv_zch_eviction_metadata_async(
      at::Tensor /*indices*/,
      at::Tensor /*count*/,
      at::Tensor /*engage_show_count*/) override {
    return std::vector<folly::Unit>(1);
  }

  void set_embedding_cache_enrich_query_id_async(
      at::Tensor /*hashed_indices*/,
      at::Tensor /*unhashed_indices*/,
      at::Tensor /*count*/) override {}

  void compact() override {}

  void maybe_evict() override {}

  std::atomic<int64_t> fetches{0};

 private:
  void flush_or_compact(const int64_t /*timestep*/) override {}

  std::mutex mutex_;
  folly::F14FastMap<int64_t, float> rows_;
};

} // namespace

TEST(
    SSDTableBatchedEmbeddingsTest,
    TestBackendWithoutReadGuardServesLookupsThroughDefault) {
  // acquire_read_guard() was added to the shared base and overridden in one
  // backend out of four. The other three keep the null default, so get() has
  // to hand them a null guard and the default get_kv_db_async_with_guard()
  // has to fall through to the plain fetch. Both the L2-enabled and the
  // L2-disabled shapes of get() go through that default.
  for (const int64_t l2_cache_size_gb : {int64_t{0}, int64_t{1}}) {
    SingleViewKVDB db(/*num_shards=*/2, l2_cache_size_gb);
    EXPECT_EQ(db.acquire_read_guard(), nullptr);

    const std::vector<int64_t> ids{50L, 51L, 52L};
    const auto expected = uniform_rows(3, 5.0f);
    write_rows(db, ids, 5.0f);

    EXPECT_TRUE(at::equal(get_rows(db, ids), expected));
    EXPECT_GT(db.fetches.load(), 0);
    // Again, so the L2-enabled case also covers the all-hits branch.
    EXPECT_TRUE(at::equal(get_rows(db, ids), expected));
  }
}

TEST(SSDTableBatchedEmbeddingsTest, TestLeavingTheLiveDbAfterALookupIsRefused) {
  // A lookup against the live DB takes no registration, so a switch off the
  // live DB cannot drain it. Once one has run, the switch has to refuse
  // rather than let a load start under a lookup it cannot see.
  auto db = getMockEmbeddingRocksDB(/*num_shards=*/2, "leaveLiveAfterLookup");
  const std::vector<int64_t> ids{30L, 31L};
  write_rows(*db, ids, 1.0f);
  get_rows(*db, ids);
  const auto* snapshot = db->create_snapshot();

  EXPECT_THROW(db->set_active_snapshot(snapshot), c10::Error);

  // Refused, not half-installed: reads still go to the live DB.
  EXPECT_FALSE(db->has_active_snapshot());
  write_rows(*db, ids, 2.0f);
  EXPECT_TRUE(at::equal(get_rows(*db, ids), uniform_rows(2, 2.0f)));
  db->release_snapshot(snapshot);
}

TEST(
    SSDTableBatchedEmbeddingsTest,
    TestFirstSnapshotBeforeAnyLookupThenSwitchesBetweenSnapshots) {
  // The supported order: install a snapshot before serving starts, then only
  // ever switch between snapshots.
  auto db = getMockEmbeddingRocksDB(/*num_shards=*/2, "firstSnapshotFirst");
  const std::vector<int64_t> ids{40L, 41L};
  write_rows(*db, ids, 1.0f);
  const auto* before_load = db->create_snapshot();

  db->set_active_snapshot(before_load);
  EXPECT_TRUE(at::equal(get_rows(*db, ids), uniform_rows(2, 1.0f)));
  write_rows(*db, ids, 2.0f);
  const auto* after_load = db->create_snapshot();
  db->set_active_snapshot(after_load);

  EXPECT_TRUE(at::equal(read_rows(*db, ids), uniform_rows(2, 2.0f)));
  db->release_snapshot(before_load);
  db->clear_active_snapshot();
  db->release_snapshot(after_load);
}

TEST(
    SSDTableBatchedEmbeddingsTest,
    TestReturningFromTheLiveDbToASnapshotIsRefused) {
  // The rule is about leaving the live DB, not about the first install only:
  // after a revert to live, lookups are untracked again, so re-installing a
  // snapshot has the same gap.
  auto db = getMockEmbeddingRocksDB(/*num_shards=*/2, "returnFromLive");
  const std::vector<int64_t> ids{50L, 51L};
  write_rows(*db, ids, 1.0f);
  const auto* snapshot = db->create_snapshot();
  db->set_active_snapshot(snapshot);
  get_rows(*db, ids);
  db->clear_active_snapshot();

  EXPECT_THROW(db->set_active_snapshot(snapshot), c10::Error);

  EXPECT_FALSE(db->has_active_snapshot());
  db->release_snapshot(snapshot);
}
