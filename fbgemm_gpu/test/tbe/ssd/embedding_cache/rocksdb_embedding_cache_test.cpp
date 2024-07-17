/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <ATen/ATen.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>
#include <filesystem>
#include "deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/rocksdb_embedding_cache.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/ssd_table_batched_embeddings.h"

namespace embedding_cache {

TEST(RocksDbEmbeddingCacheTest, TestPutAndGet) {
  std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
  std::filesystem::path rocksdb_dir = temp_dir / "rocksdb";
  std::filesystem::create_directories(rocksdb_dir);
  auto rocks_db = std::make_unique<ssd::EmbeddingRocksDB>(
      rocksdb_dir,
      8, // num_shards,
      8, // num_threads,
      0, // memtable_flush_period,
      0, // memtable_flush_offset,
      4, // l0_files_per_compact,
      128, // max embedding dimension,
      0, // rate_limit_mbps,
      1, // size_ratio,
      8, // compaction_trigger,
      536870912, // 512M write_buffer_size,
      8, // max_write_buffer_num,
      -0.01, // uniform_init_lower,
      0.01, // uniform_init_upper,
      32, // row_storage_bitwidth = 32,
      0 // cache_size = 0
  );

  auto cache = std::make_unique<RocksDBEmbeddingCache>(std::move(rocks_db));
  auto write_indices =
      at::tensor({10, 2, 1}, at::TensorOptions().dtype(at::kLong));

  auto EMBEDDING_DIMENSION = 3;
  auto write_buffer = at::randn(
      {write_indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().dtype(at::kFloat));
  XLOG(INFO) << "weights to write:\n" << write_buffer;
  auto write_count = at::tensor({3}, at::TensorOptions().dtype(at::kLong));
  cache->set(write_indices, write_buffer, write_count);

  auto read_indices = at::tensor({1, 2}, at::TensorOptions().dtype(at::kLong));
  auto read_buffer = at::empty(
      {read_indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().dtype(at::kFloat));
  XLOG(INFO) << "read_indices:\n" << read_indices;
  auto read_count = at::tensor({2}, at::TensorOptions().dtype(at::kLong));
  cache->get(read_indices, read_buffer, read_count);
  XLOG(INFO) << "weights loaded for index 1:\n" << read_buffer;

  EXPECT_EQ(
      write_buffer.index({2, 0}).item<float>(),
      read_buffer.index({0, 0}).item<float>());

  std::filesystem::remove_all(rocksdb_dir);
}
} // namespace embedding_cache
