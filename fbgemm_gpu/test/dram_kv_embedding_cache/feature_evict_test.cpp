//
// Created by arron on 2025/5/22.
//
#include "fbgemm_gpu/src/dram_kv_embedding_cache/feature_evict.h"

#include <cstdio>
#include <iostream>

#include <array>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include "fbgemm_gpu/src/dram_kv_embedding_cache/SynchronizedShardedMap.h"

namespace kv_mem {
class FeatureEvictTest : public ::testing::Test {
 protected:
  static constexpr int NUM_SHARDS = 4;
  static constexpr int DIMENSION = 128;
  size_t BLOCK_SIZE = FixedBlockPool::calculate_block_size<float>(DIMENSION);
  size_t BLOCK_ALIGNMENT = FixedBlockPool::calculate_block_alignment<float>();

  void SetUp() override {
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(4);
    kv_store_ = std::make_unique<SynchronizedShardedMap<int64_t, float*>>(
        NUM_SHARDS, BLOCK_SIZE, BLOCK_ALIGNMENT);

    // 插入测试数据

    for (int i = 0; i < 1000; ++i) {
      int shard_id = i % NUM_SHARDS;
      auto wlock = kv_store_->by(shard_id).wlock();
      auto* pool = kv_store_->pool_by(shard_id);
      float* block = pool->allocate_t<float>();
      FixedBlockPool::set_key(block, i);
      FixedBlockPool::set_count(block, 1);  // 初始分数
      FixedBlockPool::set_used(block, true);
      wlock->insert({i, block});
    }

    for (int i = 1000; i < 2000; ++i) {
      int shard_id = i % NUM_SHARDS;
      auto wlock = kv_store_->by(shard_id).wlock();
      auto* pool = kv_store_->pool_by(shard_id);
      float* block = pool->allocate_t<float>();
      FixedBlockPool::set_key(block, i);
      FixedBlockPool::set_count(block, 2);  // 初始分数
      FixedBlockPool::set_used(block, true);
      wlock->insert({i, block});
    }
  }

  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  std::unique_ptr<SynchronizedShardedMap<int64_t, float*>> kv_store_;
};

TEST_F(FeatureEvictTest, BasicEviction) {
  CounterBasedEvict evictor(executor_.get(), *kv_store_.get(), 0.5f, 1);

  // 初始验证
  size_t total_blocks = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    total_blocks += rlock->size();
  }
  ASSERT_EQ(total_blocks, 2000);

  // 执行淘汰
  evictor.trigger_evict();

  // 验证淘汰过程
  while (evictor.is_evicting()) {
    evictor.resume();
    std::this_thread::sleep_for(std::chrono::microseconds(5));
    evictor.pause();
  }

  // 验证结果
  size_t remaining = 0;
  for (int shard_id = 0; shard_id < NUM_SHARDS; ++shard_id) {
    auto rlock = kv_store_->by(shard_id).rlock();
    remaining += rlock->size();
    // 验证分数衰减
    for (const auto& [key, block] : *rlock) {
      ASSERT_EQ(FixedBlockPool::get_count(block), 1);
    }
  }
  std::cout << "remaining: " << remaining << std::endl;
  ASSERT_EQ(remaining, 1000);
}
}  // namespace kv_mem
