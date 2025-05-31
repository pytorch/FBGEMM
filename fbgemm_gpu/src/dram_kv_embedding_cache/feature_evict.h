#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory_resource>
#include <stdexcept>
#include <vector>

#include <cassert>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>

#include "SynchronizedShardedMap.h"

namespace kv_mem {

enum class EvictTriggerMode {
  DISABLED,   // Do not use feature evict
  ITERATION,  // Trigger based on iteration steps
  MANUAL      // Manually triggered by upstream
};

enum class EvictTriggerStrategy { BY_TIMESTAMP, BY_COUNTER, BY_TIMESTAMP_AND_COUNTER, BY_L2WEIGHT };

struct FeatureEvictConfig {
  EvictTriggerStrategy trigger_strategy;
  EvictTriggerMode trigger_mode;
  int64_t trigger_step_interval;
  uint32_t ttl;
  uint32_t count_threshold;
  float count_decay_rate;
  double l2_weight_threshold;
};

template <typename weight_type>
class FeatureEvict {
 public:
  FeatureEvict(folly::CPUThreadPoolExecutor* executor, SynchronizedShardedMap<int64_t, weight_type*>& kv_store)
      : executor_(executor),
        kv_store_(kv_store),
        evict_flag_(false),
        evict_interrupt_(false),
        num_shards_(kv_store.getNumShards()) {
    init_shard_status();
  }

  virtual ~FeatureEvict() {
    wait_completion();  // Wait for all asynchronous tasks to complete.
  };

  // Trigger asynchronous eviction.
  // If there is an ongoing task, return directly to prevent multiple triggers.
  // If there is no ongoing task, initialize the task state.
  void trigger_evict() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (evict_flag_.exchange(true)) return;
    prepare_evict();
  }

  // Resume task execution. Returns true if there is an ongoing task, false otherwise.
  bool resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!evict_flag_.load()) return false;
    evict_interrupt_.store(false);
    for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
      submit_shard_task(shard_id);
    }
    return true;
  };

  // Pause the eviction process. Returns true if there is an ongoing task, false otherwise.
  // During the pause phase, check whether the eviction is complete.
  bool pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!evict_flag_.load()) return false;
    evict_interrupt_.store(true);
    check_and_reset_evict_flag();
    wait_completion();
    return true;
  }

  // Check whether eviction is ongoing.
  bool is_evicting() {
    std::lock_guard<std::mutex> lock(mutex_);
    check_and_reset_evict_flag();
    return evict_flag_.load();
  }

  virtual void update_feature_statistics(weight_type* block) = 0;

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

  // Initialize shard state.
  void prepare_evict() {
    for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
      auto rlmap = kv_store_.by(shard_id).rlock();
      auto* mempool = kv_store_.pool_by(shard_id);
      block_nums_snapshot_[shard_id] = mempool->get_chunks().size() * mempool->get_blocks_per_chunk();
      block_cursors_[shard_id] = 0;
      shards_finished_[shard_id]->store(false);
    }
  }

  void submit_shard_task(int shard_id) {
    if (shards_finished_[shard_id]->load()) return;
    futures_.emplace_back(folly::via(executor_).thenValue([this, shard_id](auto&&) { process_shard(shard_id); }));
  }

  void process_shard(int shard_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t evicted_count = 0;
    size_t processed_count = 0;

    auto wlock = kv_store_.by(shard_id).wlock();
    auto* pool = kv_store_.pool_by(shard_id);

    while (!evict_interrupt_.load() && block_cursors_[shard_id] < block_nums_snapshot_[shard_id]) {
      auto* block = pool->template get_block<weight_type>(block_cursors_[shard_id]++);
      processed_count++;
      if (block && evict_block(block)) {
        int64_t key = FixedBlockPool::get_key(block);
        auto it = wlock->find(key);
        if (it != wlock->end() && block == it->second) {
          wlock->erase(key);
          pool->template deallocate_t<weight_type>(block);
          evicted_count++;
        }
      }
    }

    // Check whether the loop ends normally.
    if (block_cursors_[shard_id] >= block_nums_snapshot_[shard_id]) {
      shards_finished_[shard_id]->store(true);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    fmt::print(
        "Shard {} completed: \n"
        "  - Time taken: {}ms\n"
        "  - Total blocks processed: {}\n"
        "  - Blocks evicted: {}\n"
        "  - Eviction rate: {:.2f}%\n",
        shard_id,
        duration.count(),
        processed_count,
        evicted_count,
        (evicted_count * 100.0f) / processed_count);
  }

  virtual bool evict_block(weight_type* block) = 0;

  void wait_completion() {
    folly::collectAll(futures_).wait();
    futures_.clear();
  }

  // Check and reset the eviction flag.
  void check_and_reset_evict_flag() {
    bool all_finished = true;
    for (int i = 0; i < num_shards_; ++i) {
      if (!shards_finished_[i]->load()) all_finished = false;
    }
    if (all_finished) evict_flag_.store(false);
  }

  folly::CPUThreadPoolExecutor* executor_;                           // Thread pool.
  SynchronizedShardedMap<int64_t, weight_type*>& kv_store_;          // Sharded map.
  std::vector<std::size_t> block_cursors_;                           // Index of processed blocks.
  std::vector<std::size_t> block_nums_snapshot_;                     // Snapshot of total blocks at eviction trigger.
  std::vector<std::unique_ptr<std::atomic<bool>>> shards_finished_;  // Flags indicating whether shards are finished.
  std::atomic<bool> evict_flag_;                                     // Indicates whether an eviction task is ongoing.
  std::atomic<bool> evict_interrupt_;                                // Indicates whether the eviction task is paused.
  std::vector<folly::Future<folly::Unit>> futures_;                  // Records of shard tasks.
  std::mutex mutex_;  // Interface lock to ensure thread safety for public methods.
  int num_shards_;    // Number of concurrent tasks.
};

template <typename weight_type>
class CounterBasedEvict : public FeatureEvict<weight_type> {
 public:
  CounterBasedEvict(folly::CPUThreadPoolExecutor* executor,
                    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
                    float decay_rate,
                    uint32_t threshold)
      : FeatureEvict<weight_type>(executor, kv_store), decay_rate_(decay_rate), threshold_(threshold) {}

  void update_feature_statistics(weight_type* block) override { FixedBlockPool::update_count(block); }

 protected:
  bool evict_block(weight_type* block) override {
    // Apply decay and check the threshold.
    auto current_count = FixedBlockPool::get_count(block);
    current_count *= decay_rate_;
    FixedBlockPool::set_count(block, current_count);
    return current_count < threshold_;
  }

 private:
  float decay_rate_;    // Decay rate for the block count.
  uint32_t threshold_;  // Threshold for eviction.
};

template <typename weight_type>
class TimeBasedEvict : public FeatureEvict<weight_type> {
 public:
  TimeBasedEvict(folly::CPUThreadPoolExecutor* executor,
                 SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
                 uint32_t ttl)
      : FeatureEvict<weight_type>(executor, kv_store), ttl_(ttl) {}

  void update_feature_statistics(weight_type* block) override { FixedBlockPool::update_timestamp(block); }

 protected:
  bool evict_block(weight_type* block) override {
    auto current_time = FixedBlockPool::current_timestamp();
    return current_time - FixedBlockPool::get_timestamp(block) > ttl_;
  }

 private:
  uint32_t ttl_;  // Time-to-live for eviction.
};

template <typename weight_type>
class TimeCounterBasedEvict : public FeatureEvict<weight_type> {
 public:
  TimeCounterBasedEvict(folly::CPUThreadPoolExecutor* executor,
                        SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
                        uint32_t ttl,
                        float decay_rate,
                        uint32_t threshold)
      : FeatureEvict<weight_type>(executor, kv_store), ttl_(ttl), decay_rate_(decay_rate), threshold_(threshold) {}

  void update_feature_statistics(weight_type* block) override {
    FixedBlockPool::update_timestamp(block);
    FixedBlockPool::update_count(block);
  }

 protected:
  bool evict_block(weight_type* block) override {
    // Apply decay and check the count threshold and ttl.
    auto current_time = FixedBlockPool::current_timestamp();
    auto current_count = FixedBlockPool::get_count(block);
    current_count *= decay_rate_;
    FixedBlockPool::set_count(block, current_count);
    return (current_time - FixedBlockPool::get_timestamp(block) > ttl_) && (current_count < threshold_);
  }

 private:
  uint32_t ttl_;       // Time-to-live for eviction.
  float decay_rate_;   // Decay rate for the block count.
  uint32_t threshold_; // Count threshold for eviction.
};

template <typename weight_type>
class L2WeightBasedEvict : public FeatureEvict<weight_type> {
 public:
  L2WeightBasedEvict(folly::CPUThreadPoolExecutor* executor,
                     SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
                     double threshold,
                     size_t dimension)
      : FeatureEvict<weight_type>(executor, kv_store), threshold_(threshold), dimension_(dimension) {}

  void update_feature_statistics([[maybe_unused]] weight_type* block) override {}

 protected:
  bool evict_block(weight_type* block) override {
    auto l2weight = FixedBlockPool::get_l2weight(block, dimension_);
    return l2weight < threshold_;
  }

 private:
  double threshold_;  // L2 weight threshold for eviction.
  size_t dimension_;  // Embedding dimension
};

template <typename weight_type>
std::unique_ptr<FeatureEvict<weight_type>> create_feature_evict(
    const FeatureEvictConfig& config,
    folly::CPUThreadPoolExecutor* executor,
    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
    size_t dimension) {
  if (executor == nullptr) {
    throw std::invalid_argument("executor cannot be null");
  }

  switch (config.trigger_strategy) {
    case EvictTriggerStrategy::BY_TIMESTAMP: {
      if (config.ttl <= 0) {
        throw std::invalid_argument("ttl must be positive");
      }
      return std::make_unique<TimeBasedEvict<weight_type>>(executor, kv_store, config.ttl);
    }

    case EvictTriggerStrategy::BY_COUNTER: {
      if (config.count_decay_rate <= 0 || config.count_decay_rate > 1) {
        throw std::invalid_argument("count_decay_rate must be in range (0,1]");
      }
      if (config.count_threshold <= 0) {
        throw std::invalid_argument("count_threshold must be positive");
      }
      return std::make_unique<CounterBasedEvict<weight_type>>(
          executor, kv_store, config.count_decay_rate, config.count_threshold);
    }

    case EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER: {
      if (config.ttl <= 0) {
        throw std::invalid_argument("ttl must be positive");
      }
      if (config.count_decay_rate <= 0 || config.count_decay_rate > 1) {
        throw std::invalid_argument("count_decay_rate must be in range (0,1]");
      }
      if (config.count_threshold <= 0) {
        throw std::invalid_argument("count_threshold must be positive");
      }
      return std::make_unique<TimeCounterBasedEvict<weight_type>>(
          executor, kv_store, config.ttl, config.count_decay_rate, config.count_threshold);
    }

    case EvictTriggerStrategy::BY_L2WEIGHT: {
      if (config.l2_weight_threshold <= 0) {
        throw std::invalid_argument("l2_weight_threshold must be positive");
      }
      // TODO: optimizer parameters should not be included in dimension
      return std::make_unique<L2WeightBasedEvict<weight_type>>(
          executor, kv_store, config.l2_weight_threshold, dimension);
    }

    default:
      throw std::runtime_error("Unknown evict trigger strategy");
  }
}

}  // namespace kv_mem
