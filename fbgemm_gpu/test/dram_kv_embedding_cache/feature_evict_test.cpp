/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/feature_evict.h"

#include <cstdio>
#include <iostream>
#include <limits>

#include <fmt/format.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>
#include <array>
#include <cstdint>

// #include
// "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/SynchronizedShardedMap.h"

namespace kv_mem {
static constexpr int DIMENSION = 128;
size_t BLOCK_SIZE = FixedBlockPool::calculate_block_size<float>(DIMENSION);
size_t BLOCK_ALIGNMENT = FixedBlockPool::calculate_block_alignment<float>();

TEST(FeatureEvictTest, CounterBasedEviction) {
  static constexpr int NUM_SHARDS = 8;
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(
      NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);

  std::vector<int64_t> sub_table_hash_cumsum = {1000, 2000};
  // Insert test data table 1
  for (int i = 0; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, i < 400 ? 1 : 2); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }
  // Insert test data table 2
  for (int i = 1000; i < 2000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, i < 1500 ? 10 : 15); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  c10::intrusive_ptr<FeatureEvictConfig> feature_evict_config =
      c10::make_intrusive<FeatureEvictConfig>(
          1, // evict_trigger_mode, not needed since no scheduler in this UT
          1, // evict_trigger_strategy, not needed since no scheduler in this UT
          2, // trigger_step_interval, not needed since no scheduler in this UT
          std::nullopt, // mem_util_threshold_in_GB, not needed since no
                        // scheduler in this UT
          std::nullopt, // ttls_in_mins
          std::vector<int64_t>({1, 0}), // counter_thresholds
          std::vector<double>({0.5, 0.6}), // counter_decay_rates
          std::nullopt, // l2_weight_thresholds
          std::nullopt,
          0,
          0); // embedding_dims

  auto feature_evict = create_feature_evict(
      feature_evict_config,
      *kv_store_.get(),
      sub_table_hash_cumsum,
      true, // is training
      TestMode::NORMAL);

  // Initial validation
  size_t total_blocks = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    total_blocks += rlock->size();
  }
  ASSERT_EQ(total_blocks, 2000);

  // Perform eviction
  feature_evict->trigger_evict();

  // Validate eviction process
  while (feature_evict->is_evicting()) {
    feature_evict->resume();
    std::this_thread::sleep_for(std::chrono::microseconds(5));
    feature_evict->pause();
  }

  // Validate results
  size_t remaining = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    remaining += rlock->size();
  }
  LOG(INFO) << "remaining: " << remaining;
  ASSERT_EQ(remaining, 1600);
}

TEST(FeatureEvictTest, TimeBasedEviction) {
  static constexpr int NUM_SHARDS = 8;
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(
      NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);
  uint32_t current_time = FixedBlockPool::current_timestamp();
  std::vector<int64_t> sub_table_hash_cumsum = {1000, 2000};
  // Insert test data
  for (int i = 0; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_timestamp(
        block, i < 400 ? current_time - 7200 : current_time); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  for (int i = 1000; i < 2000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_timestamp(block, current_time); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  std::vector<int64_t> ttls = {1, std::numeric_limits<int64_t>::max() / 1000};
  // feature evict config
  c10::intrusive_ptr<FeatureEvictConfig> feature_evict_config =
      c10::make_intrusive<FeatureEvictConfig>(
          1, // evict_trigger_mode, not needed since no scheduler in this UT
          0, // evict_trigger_strategy, not needed since no scheduler in this UT
          2, // trigger_step_interval, not needed since no scheduler
             // in this UT
          std::nullopt, // mem_util_threshold_in_GB, not needed since no
                        // scheduler in this UT
          ttls, // ttls_in_mins
          std::nullopt, // counter_thresholds
          std::nullopt, // counter_decay_rates
          std::nullopt, // l2_weight_thresholds
          std::nullopt,
          0,
          0); // embedding_dims

  auto feature_evict = create_feature_evict(
      feature_evict_config,
      *kv_store_.get(),
      sub_table_hash_cumsum,
      true // is training
  );

  // Initial validation
  size_t total_blocks = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    total_blocks += rlock->size();
  }
  ASSERT_EQ(total_blocks, 2000);

  // Perform eviction
  feature_evict->trigger_evict();

  // Validate eviction process
  while (feature_evict->is_evicting()) {
    feature_evict->resume();
    std::this_thread::sleep_for(std::chrono::microseconds(5));
    feature_evict->pause();
  }

  // Validate results
  size_t remaining = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    remaining += rlock->size();
  }
  LOG(INFO) << "remaining: " << remaining;
  ASSERT_EQ(remaining, 1600);
}

TEST(FeatureEvictTest, TimeCounterBasedEviction) {
  static constexpr int NUM_SHARDS = 8;
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(
      NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);

  uint32_t current_time = FixedBlockPool::current_timestamp();
  std::vector<int64_t> sub_table_hash_cumsum = {2000};
  // Insert test data
  for (int i = 0; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_timestamp(
        block, i < 500 ? current_time - 7200 : current_time); // Initial score
    FixedBlockPool::set_count(block, 1);
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  for (int i = 1000; i < 2000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_timestamp(block, current_time); // Initial score
    FixedBlockPool::set_count(block, 2);
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  std::vector<int64_t> ttls = {1};
  std::vector<int64_t> counter_thresholds = {1};
  std::vector<double> counter_decay_rates = {0.5};

  // feature evict config
  c10::intrusive_ptr<FeatureEvictConfig> feature_evict_config =
      c10::make_intrusive<FeatureEvictConfig>(
          2, // evict_trigger_mode, not needed since no scheduler in this UT
          2, // evict_trigger_strategy, not needed since no scheduler in this UT
          std::nullopt, // trigger_step_interval, not needed since no scheduler
                        // in this UT
          2, // mem_util_threshold_in_GB, not needed since no
             // scheduler in this UT
          ttls, // ttls_in_mins
          counter_thresholds, // counter_thresholds
          counter_decay_rates, // counter_decay_rates
          std::nullopt, // l2_weight_thresholds
          std::nullopt,
          0,
          0); // embedding_dims

  auto feature_evict = create_feature_evict(
      feature_evict_config,
      *kv_store_.get(),
      sub_table_hash_cumsum,
      true // is training
  );

  // Initial validation
  size_t total_blocks = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    total_blocks += rlock->size();
  }
  ASSERT_EQ(total_blocks, 2000);

  // Perform eviction
  feature_evict->trigger_evict();

  // Validate eviction process
  while (feature_evict->is_evicting()) {
    feature_evict->resume();
    std::this_thread::sleep_for(std::chrono::microseconds(5));
    feature_evict->pause();
  }

  // Validate results
  size_t remaining = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    remaining += rlock->size();
  }
  LOG(INFO) << "remaining: " << remaining;
  ASSERT_EQ(remaining, 1500);
}

TEST(FeatureEvictTest, L2WeightBasedEviction) {
  static constexpr int NUM_SHARDS = 8;
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(
      NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);
  int dim = 4;
  std::vector<int64_t> sub_table_hash_cumsum = {2000};
  std::vector<float> weight1(dim, 1.0);
  // Insert test data
  for (int i = 0; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    auto* data_ptr = FixedBlockPool::data_ptr<float>(block);
    FixedBlockPool::set_key(block, i);
    std::copy(weight1.begin(), weight1.end(), data_ptr);
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }
  std::vector<float> weight2(dim, 2.0);
  for (int i = 1000; i < 2000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    auto* data_ptr = FixedBlockPool::data_ptr<float>(block);
    FixedBlockPool::set_key(block, i);
    std::copy(weight2.begin(), weight2.end(), data_ptr);
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  std::vector<double> l2_weight_thresholds = {3.0};
  // feature evict config
  c10::intrusive_ptr<FeatureEvictConfig> feature_evict_config =
      c10::make_intrusive<FeatureEvictConfig>(
          2, // evict_trigger_mode, not needed since no scheduler in this UT
          3, // evict_trigger_strategy, not needed since no scheduler in this UT
          std::nullopt, // trigger_step_interval, not needed since no scheduler
                        // in this UT
          2, // mem_util_threshold_in_GB, not needed since no
             // scheduler in this UT
          std::nullopt, // ttls_in_mins
          std::nullopt, // counter_thresholds
          std::nullopt, // counter_decay_rates
          l2_weight_thresholds, // l2_weight_thresholds
          std::vector<int64_t>({dim}),
          0,
          0); // embedding_dims

  auto feature_evict = create_feature_evict(
      feature_evict_config,
      *kv_store_.get(),
      sub_table_hash_cumsum,
      true // is training
  );

  // Initial validation
  size_t total_blocks = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    total_blocks += rlock->size();
  }
  ASSERT_EQ(total_blocks, 2000);

  // Perform eviction
  feature_evict->trigger_evict();

  // Validate eviction process
  while (feature_evict->is_evicting()) {
    feature_evict->resume();
    std::this_thread::sleep_for(std::chrono::microseconds(5));
    feature_evict->pause();
  }

  // Validate results
  size_t remaining = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    remaining += rlock->size();
  }
  LOG(INFO) << "remaining: " << remaining;
  ASSERT_EQ(remaining, 1000);
}

TEST(FeatureEvictTest, PerformanceTest) {
  static constexpr int NUM_SHARDS = 1;
  // Test configurations
  const std::vector<int> test_sizes = {
      100'000, 500'000, 1'000'000, 5'000'000, 10'000'000};

  fmt::print("\nPerformance Test Results:\n");
  fmt::print("{:<15} {:<15} {:<15}\n", "Size", "Time(ms)", "Items/ms");
  fmt::print("{:-<45}\n", ""); // 分隔线

  for (const auto& size : test_sizes) {
    auto kv_store = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(
        NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT, 1000);

    // Insert test data with different initial scores
    for (int i = 0; i < size; ++i) {
      int shard_id = i % NUM_SHARDS;
      auto wlock = kv_store->by(shard_id).wlock();
      auto* pool = kv_store->pool_by(shard_id);
      auto* block = pool->allocate_t<float>();
      FixedBlockPool::set_key(block, i);
      FixedBlockPool::set_count(
          block,
          (i % 2) ? 1 : 2); // Alternate between scores
      FixedBlockPool::set_used(block, true);
      wlock->insert({i, block});
    }

    // Measure eviction time
    std::vector<double> execution_times;
    std::vector<int64_t> counter_thresholds = {1};
    std::vector<double> counter_decay_rates = {0.5};
    std::vector<int64_t> sub_table_hash_cumsum = {size};
    CounterBasedEvict evictor(
        *kv_store.get(),
        sub_table_hash_cumsum,
        counter_decay_rates,
        counter_thresholds,
        0,
        0,
        true, // is training
        TestMode::NORMAL);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform eviction
    evictor.trigger_evict();
    evictor.resume();
    while (evictor.is_evicting()) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    std::size_t current_size = 0;
    for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
      auto wlock = kv_store->by(shard_id).wlock();
      current_size += wlock->size();
    }
    double eviction_rate =
        static_cast<double>(size - current_size) / static_cast<double>(size);

    // Print results
    fmt::print("{:<15d} {:<15d} {:<15.2f}\n", size, duration, eviction_rate);
  }
}

TEST(FeatureEvictTest, DupAPINoOpCheck) {
  static constexpr int NUM_SHARDS = 8;
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(
      NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);

  std::vector<int64_t> sub_table_hash_cumsum = {1000, 2000};
  // Insert test data table 1
  for (int i = 0; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, i < 400 ? 1 : 2); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }
  // Insert test data table 2
  for (int i = 1000; i < 2000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, i < 1500 ? 10 : 15); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  std::unique_ptr<FeatureEvict<float>> feature_evict;
  std::vector<int64_t> counter_thresholds = {1, 0};
  std::vector<double> counter_decay_rates = {0.5, 0.6};
  // feature evict config
  c10::intrusive_ptr<FeatureEvictConfig> feature_evict_config =
      c10::make_intrusive<FeatureEvictConfig>(
          2, // evict_trigger_mode, not needed since no scheduler in this UT
          1, // evict_trigger_strategy, not needed since no scheduler in this UT
          std::nullopt, // trigger_step_interval, not needed since no scheduler
                        // in this UT
          2, // mem_util_threshold_in_GB, not needed since no
             // scheduler in this UT
          std::nullopt, // ttls_in_mins
          counter_thresholds, // counter_thresholds
          counter_decay_rates, // counter_decay_rates
          std::nullopt, // l2_weight_thresholds
          std::nullopt,
          0,
          0); // embedding_dims

  feature_evict = create_feature_evict(
      feature_evict_config,
      *kv_store_.get(),
      sub_table_hash_cumsum,
      true, // is training
      TestMode::NORMAL);

  // Initial validation
  size_t total_blocks = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    total_blocks += rlock->size();
  }
  ASSERT_EQ(total_blocks, 2000);
  // Perform eviction 2 rounds
  for (int i = 0; i < 2; ++i) {
    feature_evict->trigger_evict();
    EXPECT_EQ(feature_evict->metrics_.full_duration_ms, 0);
    EXPECT_EQ(feature_evict->evict_interrupt_.load(), true);
    EXPECT_EQ(feature_evict->futures_.size(), NUM_SHARDS);
    for (int j = 0; j < NUM_SHARDS; ++j) {
      // one chunk has 8192 blocks, it is enough to holde the allocations in
      // this UT
      EXPECT_EQ(feature_evict->block_nums_snapshot_[j], 8192);
    }

    // verify trigger evict is no op
    auto marked_ts = feature_evict->metrics_.start_time_ms;
    feature_evict->trigger_evict();
    EXPECT_EQ(feature_evict->metrics_.start_time_ms, marked_ts);

    auto get_num_paused_evicts = [&]() -> int {
      std::unique_lock<std::mutex> lock(feature_evict->mutex_);
      return feature_evict->num_waiting_evicts_.load();
    };

    // Validate eviction process
    while (feature_evict->is_evicting()) {
      feature_evict->resume();

      // verify resume is no op
      auto num_paused_evicts_mark = get_num_paused_evicts();
      feature_evict->resume();
      EXPECT_LE(get_num_paused_evicts(), num_paused_evicts_mark);

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      feature_evict->pause();
      feature_evict->pause();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      EXPECT_EQ(feature_evict->has_running_evict(), false);
    }
    EXPECT_GT(feature_evict->metrics_.full_duration_ms, 0);
    EXPECT_EQ(feature_evict->has_running_evict(), false);
    EXPECT_EQ(feature_evict->evict_flag_.load(), false);
    for (int j = 0; j < NUM_SHARDS; ++j) {
      // one chunk has 8192 blocks, it is enough to holde the allocations in
      // this UT
      EXPECT_EQ(feature_evict->block_nums_snapshot_[j], 8192);
    }
    EXPECT_EQ(feature_evict->finished_evictions_.load(), NUM_SHARDS);

    // Validate results
    size_t remaining = 0;
    for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
      auto rlock = kv_store_->by(shard_id).rlock();
      remaining += rlock->size();
    }
    LOG(INFO) << "remaining: " << remaining;
    if (i == 0) {
      ASSERT_EQ(remaining, 1600);
    } else {
      ASSERT_EQ(remaining, 1000);
    }
  }
  // for executors to finish
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

TEST(FeatureEvictTest, EdgeCase_NoPause) {
  // test 2 eviction rounds
  static constexpr int NUM_SHARDS = 8;
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(
      NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);

  std::vector<int64_t> sub_table_hash_cumsum = {1000, 2000};
  // Insert test data table 1
  for (int i = 0; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, i < 400 ? 1 : 2); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }
  // Insert test data table 2
  for (int i = 1000; i < 2000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, i < 1500 ? 10 : 15); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  std::unique_ptr<FeatureEvict<float>> feature_evict;
  std::vector<int64_t> counter_thresholds = {1, 0};
  std::vector<double> counter_decay_rates = {0.5, 0.6};
  // feature evict config
  c10::intrusive_ptr<FeatureEvictConfig> feature_evict_config =
      c10::make_intrusive<FeatureEvictConfig>(
          2, // evict_trigger_mode, not needed since no scheduler in this UT
          1, // evict_trigger_strategy, not needed since no scheduler in this UT
          std::nullopt, // trigger_step_interval, not needed since no scheduler
                        // in this UT
          2, // mem_util_threshold_in_GB, not needed since no
             // scheduler in this UT
          std::nullopt, // ttls_in_mins
          counter_thresholds, // counter_thresholds
          counter_decay_rates, // counter_decay_rates
          std::nullopt, // l2_weight_thresholds
          std::nullopt,
          0,
          0); // embedding_dims

  feature_evict = create_feature_evict(
      feature_evict_config,
      *kv_store_.get(),
      sub_table_hash_cumsum,
      true, // is training
      TestMode::NORMAL);

  // Initial validation
  size_t total_blocks = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    total_blocks += rlock->size();
  }
  ASSERT_EQ(total_blocks, 2000);

  // Perform eviction 2 rounds
  for (int i = 0; i < 2; ++i) {
    feature_evict->trigger_evict();
    EXPECT_EQ(feature_evict->metrics_.full_duration_ms, 0);
    EXPECT_EQ(feature_evict->evict_interrupt_.load(), true);
    EXPECT_EQ(feature_evict->futures_.size(), NUM_SHARDS);
    for (int j = 0; j < NUM_SHARDS; ++j) {
      // one chunk has 8192 blocks, it is enough to holde the allocations in
      // this UT
      EXPECT_EQ(feature_evict->block_nums_snapshot_[j], 8192);
    }
    EXPECT_EQ(feature_evict->finished_evictions_.load(), 0);
    feature_evict->resume();
    // Validate eviction process
    if (i == 0) {
      feature_evict->wait_until_eviction_done();
    } else {
      while (feature_evict->is_evicting()) {
        std::this_thread::sleep_for(std::chrono::microseconds(5));
      }
    }

    EXPECT_GT(feature_evict->metrics_.full_duration_ms, 0);
    EXPECT_EQ(feature_evict->has_running_evict(), false);
    EXPECT_EQ(feature_evict->evict_flag_.load(), false);
    for (int j = 0; j < NUM_SHARDS; ++j) {
      // one chunk has 8192 blocks, it is enough to holde the allocations in
      // this UT
      EXPECT_EQ(feature_evict->block_nums_snapshot_[j], 8192);
    }
    EXPECT_EQ(feature_evict->finished_evictions_.load(), NUM_SHARDS);

    // Validate results
    size_t remaining = 0;
    for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
      auto rlock = kv_store_->by(shard_id).rlock();
      remaining += rlock->size();
    }
    LOG(INFO) << "remaining: " << remaining;

    if (i == 0) {
      ASSERT_EQ(remaining, 1600);
    } else {
      ASSERT_EQ(remaining, 1000);
    }
  }
  // for executors to finish
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

TEST(FeatureEvictTest, EdgeCase_PauseOnLastIter) {
  // test 2 eviction rounds
  static constexpr int NUM_SHARDS = 8;
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(
      NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);

  std::vector<int64_t> sub_table_hash_cumsum = {1000, 2000};
  // Insert test data table 1
  for (int i = 0; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, i < 400 ? 1 : 2); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }
  // Insert test data table 2
  for (int i = 1000; i < 2000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, i < 1500 ? 10 : 15); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  std::unique_ptr<FeatureEvict<float>> feature_evict;
  std::vector<int64_t> counter_thresholds = {1, 0};
  std::vector<double> counter_decay_rates = {0.5, 0.6};
  // feature evict config
  c10::intrusive_ptr<FeatureEvictConfig> feature_evict_config =
      c10::make_intrusive<FeatureEvictConfig>(
          2, // evict_trigger_mode, not needed since no scheduler in this UT
          1, // evict_trigger_strategy, not needed since no scheduler in this UT
          std::nullopt, // trigger_step_interval, not needed since no scheduler
                        // in this UT
          2, // mem_util_threshold_in_GB, not needed since no
             // scheduler in this UT
          std::nullopt, // ttls_in_mins
          counter_thresholds, // counter_thresholds
          counter_decay_rates, // counter_decay_rates
          std::nullopt, // l2_weight_thresholds
          std::nullopt,
          0,
          0); // embedding_dims

  feature_evict = create_feature_evict(
      feature_evict_config,
      *kv_store_.get(),
      sub_table_hash_cumsum,
      true, // is training
      TestMode::PAUSE_ON_LAST_ITERATION);

  // Initial validation
  size_t total_blocks = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    total_blocks += rlock->size();
  }
  ASSERT_EQ(total_blocks, 2000);

  // Perform eviction 2 rounds
  for (int i = 0; i < 2; ++i) {
    feature_evict->trigger_evict();
    EXPECT_EQ(feature_evict->metrics_.full_duration_ms, 0);
    EXPECT_EQ(feature_evict->evict_interrupt_.load(), true);
    EXPECT_EQ(feature_evict->futures_.size(), NUM_SHARDS);
    for (int j = 0; j < NUM_SHARDS; ++j) {
      // one chunk has 8192 blocks, it is enough to holde the allocations in
      // this UT
      EXPECT_EQ(feature_evict->block_nums_snapshot_[j], 8192);
    }
    EXPECT_EQ(feature_evict->finished_evictions_.load(), 0);
    feature_evict->resume();
    // Validate eviction process
    while (!feature_evict->should_call_) {
      std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
    std::vector<int> shard_ids;
    for (int j = 0; j < feature_evict->last_iter_shards_.size(); ++j) {
      if (!feature_evict->last_iter_shards_[j]->load()) {
        continue;
      }
      auto shard_id = j;
      shard_ids.push_back(shard_id);
      EXPECT_EQ(
          feature_evict->block_cursors_[shard_id],
          feature_evict->block_nums_snapshot_[shard_id]);
    }
    EXPECT_EQ(feature_evict->finished_evictions_.load(), 0);
    feature_evict->pause();

    feature_evict->resume();
    feature_evict->wait_until_eviction_done();
    EXPECT_GT(feature_evict->metrics_.full_duration_ms, 0);
    EXPECT_EQ(feature_evict->has_running_evict(), false);
    EXPECT_EQ(feature_evict->evict_flag_.load(), false);
    for (int j = 0; j < NUM_SHARDS; ++j) {
      // one chunk has 8192 blocks, it is enough to holde the allocations in
      // this UT
      EXPECT_EQ(feature_evict->block_nums_snapshot_[j], 8192);
    }
    EXPECT_EQ(feature_evict->finished_evictions_.load(), NUM_SHARDS);

    // Validate results
    size_t remaining = 0;
    for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
      auto rlock = kv_store_->by(shard_id).rlock();
      remaining += rlock->size();
    }
    LOG(INFO) << "remaining: " << remaining;

    if (i == 0) {
      ASSERT_EQ(remaining, 1600);
    } else {
      ASSERT_EQ(remaining, 1000);
    }
    feature_evict->reset_ut_specific();
  }
  // for executors to finish
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
}
} // namespace kv_mem
