#include "fbgemm_gpu/src/dram_kv_embedding_cache/feature_evict.h"

#include <cstdio>
#include <iostream>

#include <array>
#include <fmt/format.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include "fbgemm_gpu/src/dram_kv_embedding_cache/SynchronizedShardedMap.h"

namespace kv_mem {
static constexpr int DIMENSION = 128;
size_t BLOCK_SIZE = FixedBlockPool::calculate_block_size<float>(DIMENSION);
size_t BLOCK_ALIGNMENT = FixedBlockPool::calculate_block_alignment<float>();

TEST(FeatureEvictTest, CounterBasedEviction) {
  static constexpr int NUM_SHARDS = 8;
  auto executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(4);
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);

  // Insert test data
  for (int i = 0; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, 1);  // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  for (int i = 1000; i < 2000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, 2);  // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  std::unique_ptr<FeatureEvict<float>> feature_evict;
  int evict_trigger_mode = 2;
  int evict_trigger_strategy = 1;
  uint32_t count_threshold = 1;
  float count_decay_rate = 0.5;
  // feature evict config
  FeatureEvictConfig feature_evict_config;
  feature_evict_config.trigger_mode = static_cast<EvictTriggerMode>(evict_trigger_mode);
  feature_evict_config.trigger_strategy = static_cast<EvictTriggerStrategy>(evict_trigger_strategy);
  feature_evict_config.count_threshold = count_threshold;
  feature_evict_config.count_decay_rate = count_decay_rate;

  if (feature_evict_config.trigger_mode != EvictTriggerMode::DISABLED) {
    feature_evict = create_feature_evict(feature_evict_config, executor_.get(),*kv_store_.get(), 4);
  }

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
    // Validate score decay
    for (const auto& [key, block] : *rlock) {
      ASSERT_EQ(FixedBlockPool::get_count(block), 1);
    }
  }
  std::cout << "remaining: " << remaining << std::endl;
  ASSERT_EQ(remaining, 1000);
}

TEST(FeatureEvictTest, TimeBasedEviction) {
  static constexpr int NUM_SHARDS = 8;
  auto executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(4);
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);

  // Insert test data
  for (int i = 0; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::update_timestamp(block);  // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }
  std::this_thread::sleep_for(std::chrono::seconds(5));

  for (int i = 1000; i < 2000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::update_timestamp(block); // Initial score
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  std::unique_ptr<FeatureEvict<float>> feature_evict;
  int evict_trigger_mode = 2;
  int evict_trigger_strategy = 0;
  uint32_t ttl = 4;
  // feature evict config
  FeatureEvictConfig feature_evict_config;
  feature_evict_config.trigger_mode = static_cast<EvictTriggerMode>(evict_trigger_mode);
  feature_evict_config.trigger_strategy = static_cast<EvictTriggerStrategy>(evict_trigger_strategy);
  feature_evict_config.ttl = ttl;

  if (feature_evict_config.trigger_mode != EvictTriggerMode::DISABLED) {
    feature_evict = create_feature_evict(feature_evict_config, executor_.get(),*kv_store_.get(), 4);
  }

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
  std::cout << "remaining: " << remaining << std::endl;
  ASSERT_EQ(remaining, 1000);
}

TEST(FeatureEvictTest, TimeCounterBasedEviction) {
  static constexpr int NUM_SHARDS = 8;
  auto executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(4);
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);

  // Insert test data
  for (int i = 0; i < 500; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::update_timestamp(block);  // Initial score
    FixedBlockPool::set_count(block, 1);
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }
  std::this_thread::sleep_for(std::chrono::seconds(5));
  for (int i = 500; i < 1000; ++i) {
    int shard_id = i % NUM_SHARDS;
    auto wlock = kv_store_->by(shard_id).wlock();
    auto* pool = kv_store_->pool_by(shard_id);
    auto* block = pool->allocate_t<float>();
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::update_timestamp(block);  // Initial score
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
    FixedBlockPool::update_timestamp(block); // Initial score
    FixedBlockPool::set_count(block, 2);
    FixedBlockPool::set_used(block, true);
    wlock->insert({i, block});
  }

  std::unique_ptr<FeatureEvict<float>> feature_evict;
  int evict_trigger_mode = 2;
  int evict_trigger_strategy = 2;
  uint32_t ttl = 4;
  uint32_t count_threshold = 1;
  float count_decay_rate = 0.5;

  // feature evict config
  FeatureEvictConfig feature_evict_config;
  feature_evict_config.trigger_mode = static_cast<EvictTriggerMode>(evict_trigger_mode);
  feature_evict_config.trigger_strategy = static_cast<EvictTriggerStrategy>(evict_trigger_strategy);
  feature_evict_config.ttl = ttl;
  feature_evict_config.count_threshold = count_threshold;
  feature_evict_config.count_decay_rate = count_decay_rate;

  if (feature_evict_config.trigger_mode != EvictTriggerMode::DISABLED) {
    feature_evict = create_feature_evict(feature_evict_config, executor_.get(),*kv_store_.get(), 4);
  }

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
  std::cout << "remaining: " << remaining << std::endl;
  ASSERT_EQ(remaining, 1500);
}

TEST(FeatureEvictTest, L2WeightBasedEviction) {
  static constexpr int NUM_SHARDS = 8;
  auto executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(4);
  auto kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);
  int dim = 4;
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

  std::unique_ptr<FeatureEvict<float>> feature_evict;
  int evict_trigger_mode = 2;
  int evict_trigger_strategy = 3;
  double l2_weight_threshold = 3.0;
  // feature evict config
  FeatureEvictConfig feature_evict_config;
  feature_evict_config.trigger_mode = static_cast<EvictTriggerMode>(evict_trigger_mode);
  feature_evict_config.trigger_strategy = static_cast<EvictTriggerStrategy>(evict_trigger_strategy);
  feature_evict_config.l2_weight_threshold = l2_weight_threshold;

  if (feature_evict_config.trigger_mode != EvictTriggerMode::DISABLED) {
    feature_evict = create_feature_evict(feature_evict_config, executor_.get(),*kv_store_.get(), dim);
  }

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
  std::cout << "remaining: " << remaining << std::endl;
  ASSERT_EQ(remaining, 1000);
}

TEST(FeatureEvictTest, PerformanceTest) {
  static constexpr int NUM_SHARDS = 1;
  // Test configurations
  const std::vector<int> test_sizes = {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000};

  fmt::print("\nPerformance Test Results:\n");
  fmt::print("{:<15} {:<15} {:<15}\n", "Size", "Time(ms)", "Items/ms");
  fmt::print("{:-<45}\n", "");  // 分隔线

  for (const auto& size : test_sizes) {
    // Create executor and store for each test size
    auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(8);
    auto kv_store =
        std::make_unique<SynchronizedShardedMap<int64_t, float*>>(NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT, 1000);

    // Insert test data with different initial scores
    for (int i = 0; i < size; ++i) {
      int shard_id = i % NUM_SHARDS;
      auto wlock = kv_store->by(shard_id).wlock();
      auto* pool = kv_store->pool_by(shard_id);
      auto* block = pool->allocate_t<float>();
      FixedBlockPool::set_key(block, i);
      FixedBlockPool::set_count(block, (i % 2) ? 1 : 2);  // Alternate between scores
      FixedBlockPool::set_used(block, true);
      wlock->insert({i, block});
    }

    // Measure eviction time
    std::vector<double> execution_times;
    CounterBasedEvict evictor(executor.get(), *kv_store.get(), 0.5f, 1);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform eviction
    evictor.trigger_evict();
    evictor.resume();
    while (evictor.is_evicting()) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::size_t current_size = 0;
    for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
      auto wlock = kv_store->by(shard_id).wlock();
      current_size += wlock->size();
    }
    double eviction_rate = static_cast<double>(size - current_size) / static_cast<double>(size);

    // Print results
    fmt::print("{:<15d} {:<15d} {:<15.2f}\n", size, duration, eviction_rate);
  }
}
}  // namespace kv_mem