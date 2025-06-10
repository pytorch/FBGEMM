/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory_resource>
#include <stdexcept>
#include <vector>

#include <ATen/ATen.h>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <cassert>

#include "SynchronizedShardedMap.h"

namespace kv_mem {

enum class EvictTriggerMode {
  DISABLED, // Do not use feature evict
  ITERATION, // Trigger based on iteration steps
  MEM_UTIL, // Trigger based on memory usage
  MANUAL // Manually triggered by upstream
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
  std::vector<uint32_t> ttls_in_mins;
  std::vector<uint32_t> counter_thresholds;
  std::vector<float> counter_decay_rates;
  std::vector<double> l2_weight_thresholds;
  std::vector<int64_t> embedding_dims;
};

struct FeatureEvictMetrics {
  explicit FeatureEvictMetrics(int table_num);

  void reset();

  void update_duration(int num_shards);

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
  FeatureEvict(
      folly::CPUThreadPoolExecutor* executor,
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum);

  virtual ~FeatureEvict();

  // Trigger asynchronous eviction.
  // If there is an ongoing task, return directly to prevent multiple triggers.
  // If there is no ongoing task, initialize the task state.
  void trigger_evict();

  // Get feature eviction metric.
  FeatureEvictMetricTensors get_feature_evict_metric();

  // Resume task execution. Returns true if there is an ongoing task, false
  // otherwise.
  bool resume();

  // Pause the eviction process. Returns true if there is an ongoing task, false
  // otherwise. During the pause phase, check whether the eviction is complete.
  bool pause();

  // Check whether eviction is ongoing.
  bool is_evicting();

  virtual void update_feature_statistics(weight_type* block) = 0;

 protected:
  void init_shard_status();

  // Initialize shard state.
  void prepare_evict();

  void submit_shard_task(int shard_id);

  void process_shard(int shard_id);

  virtual bool evict_block(weight_type* block, int sub_table_id) = 0;

  void wait_completion();

  // Check and reset the eviction flag.
  void check_and_reset_evict_flag();

  [[nodiscard]] int get_sub_table_id(int64_t key) const;

  void record_metrics_to_report_tensor();

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
  CounterBasedEvict(
      folly::CPUThreadPoolExecutor* executor,
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      const std::vector<float>& decay_rates,
      const std::vector<uint32_t>& thresholds);

  void update_feature_statistics(weight_type* block) override;

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override;

 private:
  const std::vector<float>& decay_rates_; // Decay rate for the block count.
  const std::vector<uint32_t>& thresholds_; // Threshold for eviction.
};

template <typename weight_type>
class TimeBasedEvict : public FeatureEvict<weight_type> {
 public:
  TimeBasedEvict(
      folly::CPUThreadPoolExecutor* executor,
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      const std::vector<uint32_t>& ttls_in_mins);

  void update_feature_statistics(weight_type* block) override;

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override;

 private:
  const std::vector<uint32_t>& ttls_in_mins_; // Time-to-live for eviction.
};

template <typename weight_type>
class TimeCounterBasedEvict : public FeatureEvict<weight_type> {
 public:
  TimeCounterBasedEvict(
      folly::CPUThreadPoolExecutor* executor,
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      const std::vector<uint32_t>& ttls_in_mins,
      const std::vector<float>& decay_rates,
      const std::vector<uint32_t>& thresholds);

  void update_feature_statistics(weight_type* block) override;

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override;

 private:
  const std::vector<uint32_t>& ttls_in_mins_; // Time-to-live for eviction.
  const std::vector<float>& decay_rates_; // Decay rate for the block count.
  const std::vector<uint32_t>& thresholds_; // Count threshold for eviction.
};

template <typename weight_type>
class L2WeightBasedEvict : public FeatureEvict<weight_type> {
 public:
  L2WeightBasedEvict(
      folly::CPUThreadPoolExecutor* executor,
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      const std::vector<double>& thresholds,
      const std::vector<int64_t>& sub_table_dims);

  void update_feature_statistics([[maybe_unused]] weight_type* block) override;

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override;

 private:
  const std::vector<double>& thresholds_; // L2 weight threshold for eviction.
  const std::vector<int64_t>& sub_table_dims_; // Embedding dimension
};

} // namespace kv_mem
