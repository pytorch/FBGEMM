//
// Created by root on 25-5-26.
//
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory_resource>
#include <stdexcept>
#include <vector>

#include <cassert>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>

#include "SynchronizedShardedMap.h"

namespace kv_mem {

class FeatureEvictBase {
 public:
  FeatureEvictBase(folly::CPUThreadPoolExecutor* executor,
                   SynchronizedShardedMap<int64_t, float*>& kv_store)
      : executor_(executor),
        kv_store_(kv_store),
        evict_flag_(false),
        evict_interrupt_(false),
        num_shards_(kv_store.getNumShards()) {
    init_shard_status();
    // evict_flag_ 表示是否有任务在进行
    // evict_interrupt_ 表示是否有任务被中断
  }

  virtual ~FeatureEvictBase() {
    // 析构时，需要等待任务执行完成
    wait_completion();  // 等待所有异步任务完成
  };

  // 触发异步淘汰
  // 如果有执行中的任务，直接返回, 防止多次触发
  // 如果没有执行中的任务，初始化任务状态
  void trigger_evict() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (evict_flag_.exchange(true)) return;
    prepare_evict();
  }

  // 恢复任务执行，如果有进行中的任务返回true, 没有返回false
  bool resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!evict_flag_.load()) return false;
    evict_interrupt_.store(false);
    for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
      submit_shard_task(shard_id);
    }
    return true;
  };

  // 暂停淘汰过程，如果有进行中的任务返回true, 没有返回false
  // 在暂停阶段，判断淘汰是否完成
  bool pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!evict_flag_.load()) return false;
    evict_interrupt_.store(true);
    check_and_reset_evict_flag();
    wait_completion();
    return true;
  }

  // 检查是否正在淘汰
  bool is_evicting() {
    std::lock_guard<std::mutex> lock(mutex_);
    check_and_reset_evict_flag();
    return evict_flag_.load();
  }

 protected:
  void init_shard_status() {
    block_cursors_.resize(num_shards_);
    block_nums_snapshot_.resize(num_shards_);
    shards_finished_.clear();
    for (int i = 0; i < num_shards_; ++i) {
      block_cursors_[i] = 0;
      block_nums_snapshot_[i] = 0;
      shards_finished_.emplace_back(std::make_unique<std::atomic<bool>>(false));
    }
  }

  // 初始化分片状态
  void prepare_evict() {
    for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
      auto rlmap = kv_store_.by(shard_id).rlock();
      auto* mempool = kv_store_.pool_by(shard_id);
      block_nums_snapshot_[shard_id] =
          mempool->get_chunks().size() * mempool->get_blocks_per_chunk();
      block_cursors_[shard_id] = 0;
      shards_finished_[shard_id]->store(false);
    }
  }

  void submit_shard_task(int shard_id) {
    if (shards_finished_[shard_id]->load()) return;
    futures_.emplace_back(folly::via(executor_).thenValue(
        [this, shard_id](auto&&) { process_shard(shard_id); }));
  }

  void process_shard(int shard_id) {
    auto wlock = kv_store_.by(shard_id).wlock();
    auto* pool = kv_store_.pool_by(shard_id);
    while (!evict_interrupt_.load() &&
           block_cursors_[shard_id] < block_nums_snapshot_[shard_id]) {
      auto* block = pool->get_block<float>(block_cursors_[shard_id]++);
      if (block && evict_block(block)) {
        int64_t key = FixedBlockPool::get_key(block);
        auto it = wlock->find(key);
        if (it != wlock->end() && block == it->second) {
          wlock->erase(key);
          pool->deallocate_t<float>(block);
        }
      }
    }

    // 判断循环正常结束
    if (block_cursors_[shard_id] >= block_nums_snapshot_[shard_id]) {
      shards_finished_[shard_id]->store(true);
    }
  }

  virtual bool evict_block(float* block) = 0;

  void wait_completion() {
    folly::collectAll(futures_).wait();
    futures_.clear();
  }

  // 检查并重置
  void check_and_reset_evict_flag() {
    bool all_finished = true;
    for (int i = 0; i < num_shards_; ++i) {
      if (!shards_finished_[i]->load()) all_finished = false;
    }
    if (all_finished) evict_flag_.store(false);
  }

  folly::CPUThreadPoolExecutor* executor_;             // 线程池
  SynchronizedShardedMap<int64_t, float*>& kv_store_;  // shard map
  std::vector<std::size_t> block_cursors_;             // 已处理的block 索引
  std::vector<std::size_t> block_nums_snapshot_;  // 触发淘汰时，记录的block总数
  std::vector<std::unique_ptr<std::atomic<bool>>>
      shards_finished_;                              // 已完成的shard标识
  std::atomic<bool> evict_flag_;                     // 表示是否驱逐任务在进行
  std::atomic<bool> evict_interrupt_;                // 表示驱逐任务是否暂停
  std::vector<folly::Future<folly::Unit>> futures_;  // 分片任务记录
  std::mutex mutex_;  // 接口锁，保证 public 接口 线程安全
  int num_shards_;    // 并发任务数
};

class CounterBasedEvict : public FeatureEvictBase {
 public:
  CounterBasedEvict(folly::CPUThreadPoolExecutor* executor,
                    SynchronizedShardedMap<int64_t, float*>& kv_store,
                    float decay_rate,
                    int threshold)
      : FeatureEvictBase(executor, kv_store),
        decay_rate_(decay_rate),
        threshold_(threshold) {}

 protected:
  bool evict_block(float* block) override {
    // 应用衰减并检查阈值
    auto current_count = FixedBlockPool::get_count(block);
    current_count *= decay_rate_;
    FixedBlockPool::set_count(block, current_count);
    return current_count < threshold_;
  }

 private:
  float decay_rate_;
  uint32_t threshold_;
};

class TimeBasedEvict : public FeatureEvictBase {
 public:
  TimeBasedEvict(folly::CPUThreadPoolExecutor* executor,
                 SynchronizedShardedMap<int64_t, float*>& kv_store,
                 uint32_t ttl)
      : FeatureEvictBase(executor, kv_store), ttl_(ttl) {}

 protected:
  bool evict_block(float* block) override {
    auto current_time = FixedBlockPool::current_timestamp();
    return current_time - FixedBlockPool::get_timestamp(block) > ttl_;
  }

 private:
  uint32_t ttl_;
};
}  // namespace kv_mem
