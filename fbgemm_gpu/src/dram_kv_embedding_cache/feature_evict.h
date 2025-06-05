#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory_resource>
#include <stdexcept>
#include <vector>

#include <ATen/ATen.h>
#include <cassert>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>

#include "SynchronizedShardedMap.h"

namespace kv_mem {

enum class EvictTriggerMode {
  DISABLED,   // Do not use feature evict
  ITERATION,  // Trigger based on iteration steps
  MEM_UTIL,   // Trigger based on memory usage
  MANUAL      // Manually triggered by upstream
};

enum class EvictTriggerStrategy {
  BY_TIMESTAMP,
  BY_COUNTER,
  BY_TIMESTAMP_AND_COUNTER,
  BY_L2WEIGHT
};

struct FeatureEvictConfig {
  EvictTriggerStrategy trigger_strategy;
  EvictTriggerMode trigger_mode;
  int64_t trigger_step_interval;
  int64_t mem_util_threshold_in_GB;
  std::vector<uint32_t> ttls_in_hour;
  std::vector<uint32_t> count_thresholds;
  std::vector<float> count_decay_rates;
  std::vector<double> l2_weight_thresholds;
  std::vector<int> embedding_dims;
};

struct FeatureEvictMetrics {
  explicit FeatureEvictMetrics(int table_num) {
    evicted_counts.resize(table_num, 0);
    processed_counts.resize(table_num, 0);
    exec_duration_ms = 0;
    full_duration_ms = 0;
  }

  void reset() {
    std::fill(evicted_counts.begin(), evicted_counts.end(), 0);
    std::fill(processed_counts.begin(), processed_counts.end(), 0);
    exec_duration_ms = 0;
    full_duration_ms = 0;
    start_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()
                            ).count();
  }

  void update_duration(int num_shards) {
    full_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::high_resolution_clock::now().time_since_epoch()
                           ).count() - start_time_ms;
    // The exec_duration of all shards will be accumulated during the statistics
    // So finally, the number of shards needs to be divided
    exec_duration_ms /= num_shards;
  }

  std::vector<int64_t> evicted_counts;
  std::vector<int64_t> processed_counts;
  int64_t exec_duration_ms;
  int64_t full_duration_ms;
  int64_t start_time_ms;
};

struct FeatureEvictMetricTensors {
  // evicted feature count
  at::Tensor evicted_counts;
  // feature count before evict
  at::Tensor processed_counts;
  // feature evict exec duration
  at::Tensor exec_duration_ms;
  // feature evict full duration(from trigger to finish)
  at::Tensor full_duration_ms;
};

template <typename weight_type>
class FeatureEvict {
 public:
  FeatureEvict(folly::CPUThreadPoolExecutor* executor,
               SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
               const std::vector<int64_t>& sub_table_hash_cumsum)
      : executor_(executor),
        kv_store_(kv_store),
        evict_flag_(false),
        evict_interrupt_(false),
        num_shards_(kv_store.getNumShards()),
        sub_table_hash_cumsum_(sub_table_hash_cumsum),
        metrics_(sub_table_hash_cumsum_.size()) {
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

  // Get feature eviction metric.
  FeatureEvictMetricTensors get_feature_evict_metric() {
    std::lock_guard<std::mutex> lock(metric_mtx_);
    return metric_tensors_;
  }

  // Resume task execution. Returns true if there is an ongoing task, false
  // otherwise.
  bool resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!evict_flag_.load()) return false;
    evict_interrupt_.store(false);
    for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
      submit_shard_task(shard_id);
    }
    return true;
  };

  // Pause the eviction process. Returns true if there is an ongoing task, false
  // otherwise. During the pause phase, check whether the eviction is complete.
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
      block_nums_snapshot_[shard_id] =
          mempool->get_chunks().size() * mempool->get_blocks_per_chunk();
      block_cursors_[shard_id] = 0;
      shards_finished_[shard_id]->store(false);
    }
    metrics_.reset();
  }

  void submit_shard_task(int shard_id) {
    if (shards_finished_[shard_id]->load()) return;
    futures_.emplace_back(folly::via(executor_).thenValue(
        [this, shard_id](auto&&) { process_shard(shard_id); }));
  }

  void process_shard(int shard_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> evicted_counts(sub_table_hash_cumsum_.size(), 0);
    std::vector<int64_t> processed_counts(sub_table_hash_cumsum_.size(), 0);
    std::vector<float> evict_rates(sub_table_hash_cumsum_.size(), 0.0f);

    auto wlock = kv_store_.by(shard_id).wlock();
    auto* pool = kv_store_.pool_by(shard_id);

    while (!evict_interrupt_.load() &&
           block_cursors_[shard_id] < block_nums_snapshot_[shard_id]) {
      auto* block =
          pool->template get_block<weight_type>(block_cursors_[shard_id]++);
      if (block == nullptr) {
        continue;
      }
      int64_t key = FixedBlockPool::get_key(block);
      int sub_table_id = get_sub_table_id(key);
      processed_counts[sub_table_id]++;
      if (evict_block(block, sub_table_id)) {
        auto it = wlock->find(key);
        if (it != wlock->end() && block == it->second) {
          wlock->erase(key);
          pool->template deallocate_t<weight_type>(block);
          evicted_counts[sub_table_id]++;
        }
      }
    }

    // Check whether the loop ends normally.
    if (block_cursors_[shard_id] >= block_nums_snapshot_[shard_id]) {
      shards_finished_[shard_id]->store(true);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    {
      std::lock_guard<std::mutex> lock(metric_mtx_);
      metrics_.exec_duration_ms += duration.count();
      for (size_t i = 0; i < evicted_counts.size(); ++i) {
        metrics_.evicted_counts[i] += evicted_counts[i];
        metrics_.processed_counts[i] += processed_counts[i];
      }
    }

    for (size_t i = 0; i < evicted_counts.size(); ++i) {
      evict_rates[i] = processed_counts[i] > 0
                           ? (evicted_counts[i] * 100.0f) / processed_counts[i]
                           : 0.0f;
    }
    /*
    DLOG(INFO) << fmt::format(
        "Shard {} completed: \n"
        "  - Time taken: {}ms\n"
        "  - Total blocks processed: [{}]\n"
        "  - Blocks evicted: [{}]\n"
        "  - Eviction rate: [{}]%\n",
        shard_id,
        duration.count(),
        fmt::join(processed_counts, ", "),
        fmt::join(evicted_counts, ", "),
        fmt::join(evict_rates, ", "));
    */
  }

  virtual bool evict_block(weight_type* block, int sub_table_id) = 0;

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
    if (all_finished && evict_flag_.exchange(false)) {
      record_metrics_to_report_tensor();
    }
  }

  [[nodiscard]] int get_sub_table_id(int64_t key) const {
    auto it = std::upper_bound(
        sub_table_hash_cumsum_.begin(), sub_table_hash_cumsum_.end(), key);
    if (it == sub_table_hash_cumsum_.end()) {
      CHECK(false) << "key " << key << " doesn't belong to any feature";
    }

    return std::distance(sub_table_hash_cumsum_.begin(), it);
  }

  void record_metrics_to_report_tensor() {
    std::lock_guard<std::mutex> lock(metric_mtx_);
    metrics_.update_duration(num_shards_);
    metric_tensors_.evicted_counts =
        at::from_blob(const_cast<int64_t*>(metrics_.evicted_counts.data()),
                      {static_cast<int64_t>(metrics_.evicted_counts.size())},
                      at::kLong)
            .clone();

    metric_tensors_.processed_counts =
        at::from_blob(const_cast<int64_t*>(metrics_.processed_counts.data()),
                      {static_cast<int64_t>(metrics_.processed_counts.size())},
                      at::kLong)
            .clone();

    metric_tensors_.full_duration_ms = at::scalar_tensor(metrics_.full_duration_ms, at::kLong);
    metric_tensors_.exec_duration_ms = at::scalar_tensor(metrics_.exec_duration_ms, at::kLong);
    std::vector<float> evict_rates(metrics_.evicted_counts.size());
    for (size_t i = 0; i < metrics_.evicted_counts.size(); ++i) {
      evict_rates[i] = metrics_.processed_counts[i] > 0
                           ? (metrics_.evicted_counts[i] * 100.0f) /
                                 metrics_.processed_counts[i]
                           : 0.0f;
    }
    LOG(INFO) << fmt::format(
        "Feature evict completed: \n"
        "  - full Time taken: {}ms\n"
        "  - exec Time taken: {}ms\n"
        "  - exec / full: {:.2f}%\n"
        "  - Total blocks processed: [{}]\n"
        "  - Blocks evicted: [{}]\n"
        "  - Eviction rate: [{}]%\n",
        metrics_.full_duration_ms,
        metrics_.exec_duration_ms,
        metrics_.exec_duration_ms * 100.0f / metrics_.full_duration_ms,
        fmt::join(metrics_.processed_counts, ", "),
        fmt::join(metrics_.evicted_counts, ", "),
        fmt::join(evict_rates, ", "));
  }

  // Thread pool.
  folly::CPUThreadPoolExecutor* executor_;
  // Sharded map.
  SynchronizedShardedMap<int64_t, weight_type*>& kv_store_;
  // Index of processed blocks.
  std::vector<std::size_t> block_cursors_;
  // Snapshot of total blocks at eviction trigger.
  std::vector<std::size_t> block_nums_snapshot_;
  // Flags indicating whether shards are finished.
  std::vector<std::unique_ptr<std::atomic<bool>>> shards_finished_;
  // Indicates whether an eviction task is ongoing.
  std::atomic<bool> evict_flag_;
  // Indicates whether the eviction task is paused.
  std::atomic<bool> evict_interrupt_;
  // Records of shard tasks.
  std::vector<folly::Future<folly::Unit>> futures_;
  // Interface lock to ensure thread safety for public methods.
  std::mutex mutex_;
  // Number of concurrent tasks.
  int num_shards_;
  // used to calculate which sub-table the key belongs to
  const std::vector<int64_t>& sub_table_hash_cumsum_;
  // metric lock
  std::mutex metric_mtx_;
  // record the statistical information of feature_evict
  FeatureEvictMetrics metrics_;
  // report the statistical information of feature_evict
  FeatureEvictMetricTensors metric_tensors_;
};

template <typename weight_type>
class CounterBasedEvict : public FeatureEvict<weight_type> {
 public:
  CounterBasedEvict(folly::CPUThreadPoolExecutor* executor,
                    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
                    const std::vector<int64_t>& sub_table_hash_cumsum,
                    const std::vector<float>& decay_rates,
                    const std::vector<uint32_t>& thresholds)
      : FeatureEvict<weight_type>(executor, kv_store, sub_table_hash_cumsum),
        decay_rates_(decay_rates),
        thresholds_(thresholds) {}

  void update_feature_statistics(weight_type* block) override {
    FixedBlockPool::update_count(block);
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    float decay_rate = decay_rates_[sub_table_id];
    uint32_t threshold = thresholds_[sub_table_id];
    // Apply decay and check the threshold.
    auto current_count = FixedBlockPool::get_count(block);
    current_count *= decay_rate;
    FixedBlockPool::set_count(block, current_count);
    return current_count < threshold;
  }

 private:
  const std::vector<float>& decay_rates_;    // Decay rate for the block count.
  const std::vector<uint32_t>& thresholds_;  // Threshold for eviction.
};

template <typename weight_type>
class TimeBasedEvict : public FeatureEvict<weight_type> {
 public:
  TimeBasedEvict(folly::CPUThreadPoolExecutor* executor,
                 SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
                 const std::vector<int64_t>& sub_table_hash_cumsum,
                 const std::vector<uint32_t>& ttls_in_hour)
      : FeatureEvict<weight_type>(executor, kv_store, sub_table_hash_cumsum),
        ttls_in_hour_(ttls_in_hour) {}

  void update_feature_statistics(weight_type* block) override {
    FixedBlockPool::update_timestamp(block);
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    uint32_t ttl = ttls_in_hour_[sub_table_id];
    auto current_time = FixedBlockPool::current_timestamp();
    return current_time - FixedBlockPool::get_timestamp(block) > ttl * 3600;
  }

 private:
  const std::vector<uint32_t>& ttls_in_hour_;  // Time-to-live for eviction.
};

template <typename weight_type>
class TimeCounterBasedEvict : public FeatureEvict<weight_type> {
 public:
  TimeCounterBasedEvict(folly::CPUThreadPoolExecutor* executor,
                        SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
                        const std::vector<int64_t>& sub_table_hash_cumsum,
                        const std::vector<uint32_t>& ttls_in_hour,
                        const std::vector<float>& decay_rates,
                        const std::vector<uint32_t>& thresholds)
      : FeatureEvict<weight_type>(executor, kv_store, sub_table_hash_cumsum),
        ttls_in_hour_(ttls_in_hour),
        decay_rates_(decay_rates),
        thresholds_(thresholds) {}

  void update_feature_statistics(weight_type* block) override {
    FixedBlockPool::update_timestamp(block);
    FixedBlockPool::update_count(block);
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    uint32_t ttl = ttls_in_hour_[sub_table_id];
    float decay_rate = decay_rates_[sub_table_id];
    uint32_t threshold = thresholds_[sub_table_id];
    // Apply decay and check the count threshold and ttl.
    auto current_time = FixedBlockPool::current_timestamp();
    auto current_count = FixedBlockPool::get_count(block);
    current_count *= decay_rate;
    FixedBlockPool::set_count(block, current_count);
    return (current_time - FixedBlockPool::get_timestamp(block) > ttl * 3600) &&
           (current_count < threshold);
  }

 private:
  const std::vector<uint32_t>& ttls_in_hour_;        // Time-to-live for eviction.
  const std::vector<float>& decay_rates_;    // Decay rate for the block count.
  const std::vector<uint32_t>& thresholds_;  // Count threshold for eviction.
};

template <typename weight_type>
class L2WeightBasedEvict : public FeatureEvict<weight_type> {
 public:
  L2WeightBasedEvict(folly::CPUThreadPoolExecutor* executor,
                     SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
                     const std::vector<int64_t>& sub_table_hash_cumsum,
                     const std::vector<double>& thresholds,
                     const std::vector<int>& sub_table_dims)
      : FeatureEvict<weight_type>(executor, kv_store, sub_table_hash_cumsum),
        thresholds_(thresholds),
        sub_table_dims_(sub_table_dims) {}

  void update_feature_statistics([[maybe_unused]] weight_type* block) override {
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    size_t dimension = sub_table_dims_[sub_table_id];
    double threshold = thresholds_[sub_table_id];
    auto l2weight = FixedBlockPool::get_l2weight(block, dimension);
    return l2weight < threshold;
  }

 private:
  const std::vector<double>& thresholds_;   // L2 weight threshold for eviction.
  const std::vector<int>& sub_table_dims_;  // Embedding dimension
};

template <typename weight_type>
std::unique_ptr<FeatureEvict<weight_type>> create_feature_evict(
    const FeatureEvictConfig& config,
    folly::CPUThreadPoolExecutor* executor,
    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
    const std::vector<int64_t>& sub_table_hash_cumsum) {
  if (executor == nullptr) {
    throw std::invalid_argument("executor cannot be null");
  }

  switch (config.trigger_strategy) {
    case EvictTriggerStrategy::BY_TIMESTAMP: {
      return std::make_unique<TimeBasedEvict<weight_type>>(
          executor, kv_store, sub_table_hash_cumsum, config.ttls_in_hour);
    }

    case EvictTriggerStrategy::BY_COUNTER: {
      for (auto count_decay_rate : config.count_decay_rates) {
        if (count_decay_rate <= 0 || count_decay_rate > 1) {
          throw std::invalid_argument(
              "count_decay_rate must be in range (0,1]");
        }
      }
      return std::make_unique<CounterBasedEvict<weight_type>>(
          executor,
          kv_store,
          sub_table_hash_cumsum,
          config.count_decay_rates,
          config.count_thresholds);
    }

    case EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER: {
      for (auto count_decay_rate : config.count_decay_rates) {
        if (count_decay_rate <= 0 || count_decay_rate > 1) {
          throw std::invalid_argument(
              "count_decay_rate must be in range (0,1]");
        }
      }
      return std::make_unique<TimeCounterBasedEvict<weight_type>>(
          executor,
          kv_store,
          sub_table_hash_cumsum,
          config.ttls_in_hour,
          config.count_decay_rates,
          config.count_thresholds);
    }

    case EvictTriggerStrategy::BY_L2WEIGHT: {
      for (auto l2_weight_threshold : config.l2_weight_thresholds) {
        if (l2_weight_threshold < 0) {
          throw std::invalid_argument("l2_weight_threshold must be positive");
        }
      }
      for (auto embedding_dim : config.embedding_dims) {
        if (embedding_dim <= 0) {
          throw std::invalid_argument("embedding_dim must be positive");
        }
      }

      return std::make_unique<L2WeightBasedEvict<weight_type>>(
          executor,
          kv_store,
          sub_table_hash_cumsum,
          config.l2_weight_thresholds,
          config.embedding_dims);
    }

    default:
      throw std::runtime_error("Unknown evict trigger strategy");
  }
}

}  // namespace kv_mem
