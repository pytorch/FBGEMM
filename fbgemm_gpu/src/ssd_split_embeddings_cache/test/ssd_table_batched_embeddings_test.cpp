/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <filesystem>
#include "deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/ssd_table_batched_embeddings.h"

using namespace ::testing;

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
      bool enable_async_update = false)
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
            enable_async_update){};
  MOCK_METHOD(
      rocksdb::Status,
      set_rocksdb_option,
      (int, const std::string&, const std::string&),
      (override));
};

std::unique_ptr<MockEmbeddingRocksDB> getMockEmbeddingRocksDB(
    int num_shards,
    std::string dir) {
  std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  std::filesystem::path rocksdb_dir = temp_dir / dir;
  std::filesystem::create_directories(rocksdb_dir);
  auto EMBEDDING_DIMENSION = 8;

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
      false,
      0,
      0,
      false);
}
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
