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
#include "fbgemm_gpu/split_embeddings_cache/cachelib_cache.h" // @manual=//deeplearning/fbgemm/fbgemm_gpu/src/split_embeddings_cache:cachelib_cache

namespace l2_cache {

/**
 * @brief Tests basic put and get operations with regular allocator mode.
 */
TEST(CacheLibCacheTest, TestPutAndGetRegularMode) {
  const int64_t EMBEDDING_DIM = 8;
  const int64_t NUM_SHARDS = 4;
  const int64_t CACHE_SIZE = 100 * 1024 * 1024; // 100MB

  CacheLibCache::CacheConfig config;
  config.cache_size_bytes = CACHE_SIZE;
  config.item_size_bytes = EMBEDDING_DIM * sizeof(float);
  config.num_shards = NUM_SHARDS;
  config.max_D_ = EMBEDDING_DIM;
  config.use_object_cache = false; // Regular allocator mode

  auto cache = std::make_unique<CacheLibCache>(config, 0 /* unique_tbe_id */);

  // Create test data
  auto key1 = at::tensor({100}, at::TensorOptions().dtype(at::kLong));
  auto key2 = at::tensor({200}, at::TensorOptions().dtype(at::kLong));
  auto key3 = at::tensor({300}, at::TensorOptions().dtype(at::kLong));

  auto data1 = at::ones({EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  auto data2 =
      at::ones({EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat)) * 2.0;
  auto data3 =
      at::ones({EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat)) * 3.0;

  // Test put operations
  EXPECT_TRUE(cache->put(key1, data1));
  EXPECT_TRUE(cache->put(key2, data2));
  EXPECT_TRUE(cache->put(key3, data3));

  // Test get operations
  auto result1 = cache->get(key1);
  ASSERT_TRUE(result1.has_value());
  auto result1_tensor = at::from_blob(
      result1.value(), {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_TRUE(at::allclose(result1_tensor, data1));

  auto result2 = cache->get(key2);
  ASSERT_TRUE(result2.has_value());
  auto result2_tensor = at::from_blob(
      result2.value(), {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_TRUE(at::allclose(result2_tensor, data2));

  auto result3 = cache->get(key3);
  ASSERT_TRUE(result3.has_value());
  auto result3_tensor = at::from_blob(
      result3.value(), {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_TRUE(at::allclose(result3_tensor, data3));

  // Test cache miss
  auto key_miss = at::tensor({999}, at::TensorOptions().dtype(at::kLong));
  auto result_miss = cache->get(key_miss);
  EXPECT_FALSE(result_miss.has_value());
}

/**
 * @brief Tests basic put and get operations with ObjectCache mode.
 */
TEST(CacheLibCacheTest, TestPutAndGetObjectCacheMode) {
  const int64_t EMBEDDING_DIM = 8;
  const int64_t NUM_SHARDS = 4;
  const int64_t CACHE_SIZE = 100 * 1024 * 1024; // 100MB

  CacheLibCache::CacheConfig config;
  config.cache_size_bytes = CACHE_SIZE;
  config.item_size_bytes = EMBEDDING_DIM * sizeof(float);
  config.num_shards = NUM_SHARDS;
  config.max_D_ = EMBEDDING_DIM;
  config.use_object_cache = true; // ObjectCache mode

  auto cache = std::make_unique<CacheLibCache>(config, 1 /* unique_tbe_id */);

  // Create test data
  auto key1 = at::tensor({100}, at::TensorOptions().dtype(at::kLong));
  auto key2 = at::tensor({200}, at::TensorOptions().dtype(at::kLong));
  auto key3 = at::tensor({300}, at::TensorOptions().dtype(at::kLong));

  auto data1 = at::ones({EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  auto data2 =
      at::ones({EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat)) * 2.0;
  auto data3 =
      at::ones({EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat)) * 3.0;

  // Test put operations
  EXPECT_TRUE(cache->put(key1, data1));
  EXPECT_TRUE(cache->put(key2, data2));
  EXPECT_TRUE(cache->put(key3, data3));

  // Test get operations
  auto result1 = cache->get(key1);
  ASSERT_TRUE(result1.has_value());
  auto result1_tensor = at::from_blob(
      result1.value(), {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_TRUE(at::allclose(result1_tensor, data1));

  auto result2 = cache->get(key2);
  ASSERT_TRUE(result2.has_value());
  auto result2_tensor = at::from_blob(
      result2.value(), {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_TRUE(at::allclose(result2_tensor, data2));

  auto result3 = cache->get(key3);
  ASSERT_TRUE(result3.has_value());
  auto result3_tensor = at::from_blob(
      result3.value(), {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_TRUE(at::allclose(result3_tensor, data3));

  // Test cache miss
  auto key_miss = at::tensor({999}, at::TensorOptions().dtype(at::kLong));
  auto result_miss = cache->get(key_miss);
  EXPECT_FALSE(result_miss.has_value());
}

/**
 * @brief Tests cache update operations (put with existing key).
 */
TEST(CacheLibCacheTest, TestCacheUpdate) {
  const int64_t EMBEDDING_DIM = 8;
  const int64_t NUM_SHARDS = 4;
  const int64_t CACHE_SIZE = 100 * 1024 * 1024; // 100MB

  CacheLibCache::CacheConfig config;
  config.cache_size_bytes = CACHE_SIZE;
  config.item_size_bytes = EMBEDDING_DIM * sizeof(float);
  config.num_shards = NUM_SHARDS;
  config.max_D_ = EMBEDDING_DIM;
  config.use_object_cache = true; // ObjectCache mode

  auto cache = std::make_unique<CacheLibCache>(config, 2 /* unique_tbe_id */);

  auto key = at::tensor({100}, at::TensorOptions().dtype(at::kLong));
  auto data1 = at::ones({EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  auto data2 =
      at::ones({EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat)) * 5.0;

  // Insert initial value
  EXPECT_TRUE(cache->put(key, data1));

  // Verify initial value
  auto result1 = cache->get(key);
  ASSERT_TRUE(result1.has_value());
  auto result1_tensor = at::from_blob(
      result1.value(), {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_TRUE(at::allclose(result1_tensor, data1));

  // Update with new value
  EXPECT_TRUE(cache->put(key, data2));

  // Verify updated value
  auto result2 = cache->get(key);
  ASSERT_TRUE(result2.has_value());
  auto result2_tensor = at::from_blob(
      result2.value(), {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_TRUE(at::allclose(result2_tensor, data2));
}

/**
 * @brief Tests cache usage statistics.
 */
TEST(CacheLibCacheTest, TestCacheUsageStats) {
  const int64_t EMBEDDING_DIM = 8;
  const int64_t NUM_SHARDS = 4;
  const int64_t CACHE_SIZE = 100 * 1024 * 1024; // 100MB

  CacheLibCache::CacheConfig config;
  config.cache_size_bytes = CACHE_SIZE;
  config.item_size_bytes = EMBEDDING_DIM * sizeof(float);
  config.num_shards = NUM_SHARDS;
  config.max_D_ = EMBEDDING_DIM;
  config.use_object_cache = false; // Regular mode

  auto cache = std::make_unique<CacheLibCache>(config, 3 /* unique_tbe_id */);

  auto stats = cache->get_cache_usage();
  EXPECT_EQ(stats.size(), 2); // [freeBytes, capacity]
  EXPECT_EQ(stats[1], CACHE_SIZE); // capacity should match config
  EXPECT_GT(stats[0], 0); // should have some free bytes
}

/**
 * @brief Tests cache usage statistics with ObjectCache mode.
 */
TEST(CacheLibCacheTest, TestCacheUsageStatsObjectCache) {
  const int64_t EMBEDDING_DIM = 8;
  const int64_t NUM_SHARDS = 4;
  const int64_t CACHE_SIZE = 100 * 1024 * 1024; // 100MB

  CacheLibCache::CacheConfig config;
  config.cache_size_bytes = CACHE_SIZE;
  config.item_size_bytes = EMBEDDING_DIM * sizeof(float);
  config.num_shards = NUM_SHARDS;
  config.max_D_ = EMBEDDING_DIM;
  config.use_object_cache = true; // ObjectCache mode

  auto cache = std::make_unique<CacheLibCache>(config, 5 /* unique_tbe_id */);

  // Get stats before inserting any data
  auto stats_before = cache->get_cache_usage();
  EXPECT_EQ(stats_before.size(), 2); // [freeBytes, capacity]
  EXPECT_EQ(stats_before[1], CACHE_SIZE); // capacity should match config
  // With object size tracking enabled, should report full cache as free
  EXPECT_EQ(stats_before[0], CACHE_SIZE); // all bytes should be free initially

  // Insert some data
  auto key = at::tensor({100}, at::TensorOptions().dtype(at::kLong));
  auto data = at::ones({EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_TRUE(cache->put(key, data));

  // Get stats after inserting data
  auto stats_after = cache->get_cache_usage();
  EXPECT_EQ(stats_after.size(), 2);
  EXPECT_EQ(stats_after[1], CACHE_SIZE); // capacity unchanged
  // Free bytes should be less than before since we inserted data
  EXPECT_LT(stats_after[0], stats_before[0]);
  // Used memory should equal the data size we inserted
  int64_t used_memory = CACHE_SIZE - stats_after[0];
  int64_t expected_used = EMBEDDING_DIM * sizeof(float);
  EXPECT_EQ(used_memory, expected_used);
}

/**
 * @brief Tests cache with different data types.
 */
TEST(CacheLibCacheTest, TestDifferentDataTypes) {
  const int64_t EMBEDDING_DIM = 16;
  const int64_t NUM_SHARDS = 4;
  const int64_t CACHE_SIZE = 100 * 1024 * 1024; // 100MB

  CacheLibCache::CacheConfig config;
  config.cache_size_bytes = CACHE_SIZE;
  config.item_size_bytes = EMBEDDING_DIM * sizeof(uint8_t);
  config.num_shards = NUM_SHARDS;
  config.max_D_ = EMBEDDING_DIM;
  config.use_object_cache = true; // ObjectCache mode

  auto cache = std::make_unique<CacheLibCache>(config, 6 /* unique_tbe_id */);

  // Test with int64 keys and uint8 data
  auto key = at::tensor({100}, at::TensorOptions().dtype(at::kLong));
  auto data = at::randint(
      0, 255, {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kByte));

  EXPECT_TRUE(cache->put(key, data));

  auto result = cache->get(key);
  ASSERT_TRUE(result.has_value());
  auto result_tensor = at::from_blob(
      result.value(), {EMBEDDING_DIM}, at::TensorOptions().dtype(at::kByte));
  EXPECT_TRUE(at::equal(result_tensor, data));
}

} // namespace l2_cache
