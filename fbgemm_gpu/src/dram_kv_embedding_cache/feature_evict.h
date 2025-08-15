/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <barrier>
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
#include <gtest/gtest_prod.h>
#include <cassert>

#include "SynchronizedShardedMap.h"

namespace kv_mem {

enum class EvictTriggerMode {
  DISABLED, // Do not use feature evict
  ITERATION, // Trigger based on iteration steps
  MEM_UTIL, // Trigger based on memory usage
  MANUAL // Manually triggered by upstream
};
inline std::string to_string(EvictTriggerMode mode) {
  switch (mode) {
    case EvictTriggerMode::DISABLED:
      return "DISABLED";
    case EvictTriggerMode::ITERATION:
      return "ITERATION";
    case EvictTriggerMode::MEM_UTIL:
      return "MEM_UTIL";
    case EvictTriggerMode::MANUAL:
      return "MANUAL";
  }
}

enum class EvictTriggerStrategy {
  BY_TIMESTAMP,
  BY_COUNTER,
  BY_TIMESTAMP_AND_COUNTER,
  BY_L2WEIGHT,
  BY_TIMESTAMP_THRESHOLD,
  BY_FEATURE_SCORE
};

inline std::string to_string(EvictTriggerStrategy strategy) {
  switch (strategy) {
    case EvictTriggerStrategy::BY_TIMESTAMP:
      return "BY_TIMESTAMP";
    case EvictTriggerStrategy::BY_COUNTER:
      return "BY_COUNTER";
    case EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER:
      return "BY_TIMESTAMP_AND_COUNTER";
    case EvictTriggerStrategy::BY_L2WEIGHT:
      return "BY_L2WEIGHT";
    case EvictTriggerStrategy::BY_TIMESTAMP_THRESHOLD:
      return "BY_TIMESTAMP_THRESHOLD";
    case EvictTriggerStrategy::BY_FEATURE_SCORE:
      return "BY_FEATURE_SCORE";
  }
}

// for UT different corner cases
enum class TestMode {
  DISABLED, // no test mode
  NORMAL, // normal test
  PAUSE_ON_LAST_ITERATION, // pause called on last iteration
};

inline std::string to_string(TestMode mode) {
  switch (mode) {
    case TestMode::DISABLED:
      return "DISABLED";
    case TestMode::NORMAL:
      return "NORMAL";
    case TestMode::PAUSE_ON_LAST_ITERATION:
      return "PAUSE_ON_LAST_ITERATION";
  }
}

struct FeatureEvictConfig : public torch::jit::CustomClassHolder {
  explicit FeatureEvictConfig(
      int64_t trigger_mode,
      int64_t trigger_strategy,
      std::optional<int64_t> trigger_step_interval,
      std::optional<int64_t> mem_util_threshold_in_GB,
      std::optional<std::vector<int64_t>> ttls_in_mins,
      std::optional<std::vector<int64_t>> counter_thresholds,
      std::optional<std::vector<double>> counter_decay_rates,
      std::optional<std::vector<double>> feature_score_counter_decay_rates,
      std::optional<std::vector<int64_t>> max_training_id_num_per_table,
      std::optional<std::vector<double>> target_eviction_percent_per_table,
      std::optional<std::vector<double>> l2_weight_thresholds,
      std::optional<std::vector<int64_t>> embedding_dims,
      std::optional<double> threshold_calculation_bucket_stride = 0.2,
      std::optional<int64_t> threshold_calculation_bucket_num = 1000000, // 1M
      int64_t interval_for_insufficient_eviction_s = 600, // 10 min
      int64_t interval_for_sufficient_eviction_s = 60, // 1 min
      int64_t interval_for_feature_statistics_decay_s = 24 * 3600) // 1 day
      : trigger_mode_(static_cast<EvictTriggerMode>(trigger_mode)),
        trigger_strategy_(static_cast<EvictTriggerStrategy>(trigger_strategy)),
        trigger_step_interval_(trigger_step_interval),
        mem_util_threshold_in_GB_(mem_util_threshold_in_GB),
        ttls_in_mins_(ttls_in_mins),
        counter_thresholds_(counter_thresholds),
        counter_decay_rates_(counter_decay_rates),
        feature_score_counter_decay_rates_(
            std::move(feature_score_counter_decay_rates)),
        max_training_id_num_per_table_(
            std::move(max_training_id_num_per_table)),
        target_eviction_percent_per_table_(
            std::move(target_eviction_percent_per_table)),
        l2_weight_thresholds_(l2_weight_thresholds),
        embedding_dims_(embedding_dims),
        threshold_calculation_bucket_stride_(
            threshold_calculation_bucket_stride),
        threshold_calculation_bucket_num_(threshold_calculation_bucket_num),
        interval_for_insufficient_eviction_s_(
            interval_for_insufficient_eviction_s),
        interval_for_sufficient_eviction_s_(interval_for_sufficient_eviction_s),
        interval_for_feature_statistics_decay_s_(
            interval_for_feature_statistics_decay_s) {
    // verification
    if (trigger_mode_ == EvictTriggerMode::DISABLED) {
      LOG(INFO) << "eviction config, trigger mode is disabled";
      return;
    }
    std::string eviction_trigger_stats_log = "";
    switch (trigger_mode_) {
      case EvictTriggerMode::ITERATION: {
        CHECK(
            trigger_step_interval_.has_value() &&
            trigger_step_interval_.value() > 0);
        eviction_trigger_stats_log = ", trigger_step_interval: " +
            std::to_string(trigger_step_interval_.value());
        break;
      }
      case EvictTriggerMode::MEM_UTIL: {
        CHECK(
            mem_util_threshold_in_GB_.has_value() &&
            mem_util_threshold_in_GB_.value() > 0);
        eviction_trigger_stats_log = ", mem_util_threshold_in_GB: " +
            std::to_string(mem_util_threshold_in_GB_.value());
        break;
      }
      case EvictTriggerMode::MANUAL: {
        break;
      }
      default:
        throw std::runtime_error("Unknown evict trigger mode");
    }

    switch (trigger_strategy_) {
      case EvictTriggerStrategy::BY_COUNTER: {
        CHECK(counter_thresholds_.has_value());
        CHECK(counter_decay_rates_.has_value());
        LOG(INFO) << "eviction config, trigger mode:"
                  << to_string(trigger_mode_) << eviction_trigger_stats_log
                  << ", strategy: " << to_string(trigger_strategy_)
                  << ", counter_thresholds: " << counter_thresholds_.value()
                  << ", counter_decay_rates: " << counter_decay_rates_.value();
        return;
      }

      case EvictTriggerStrategy::BY_FEATURE_SCORE: {
        CHECK(feature_score_counter_decay_rates_.has_value());
        CHECK(max_training_id_num_per_table_.has_value());
        CHECK(target_eviction_percent_per_table_.has_value());
        CHECK(threshold_calculation_bucket_stride_.has_value());
        CHECK(threshold_calculation_bucket_num_.has_value());
        LOG(INFO) << "eviction config, trigger mode:"
                  << to_string(trigger_mode_) << eviction_trigger_stats_log
                  << ", strategy: " << to_string(trigger_strategy_)
                  << ", max_training_id_num_per_table: "
                  << max_training_id_num_per_table_.value()
                  << ", target_eviction_percent_per_table:"
                  << target_eviction_percent_per_table_.value()
                  << ", threshold_calculation_bucket_stride: "
                  << threshold_calculation_bucket_stride_.value()
                  << ", threshold_calculation_bucket_num: "
                  << threshold_calculation_bucket_num_.value()
                  << ", feature_score_counter_decay_rates: "
                  << feature_score_counter_decay_rates_.value();
        return;
      }

      case EvictTriggerStrategy::BY_TIMESTAMP: {
        CHECK(ttls_in_mins_.has_value());
        LOG(INFO) << "eviction config, trigger mode:"
                  << to_string(trigger_mode_) << eviction_trigger_stats_log
                  << ", strategy: " << to_string(trigger_strategy_)
                  << ", ttls_in_mins: " << ttls_in_mins_.value();
        return;
      }

      case EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER: {
        CHECK(counter_thresholds_.has_value());
        CHECK(counter_decay_rates_.has_value());
        CHECK(ttls_in_mins_.has_value());
        LOG(INFO) << "eviction config, trigger mode:"
                  << to_string(trigger_mode_) << eviction_trigger_stats_log
                  << ", strategy: " << to_string(trigger_strategy_)
                  << ", counter_thresholds: " << counter_thresholds_.value()
                  << ", counter_decay_rates: " << counter_decay_rates_.value()
                  << ", ttls_in_mins: " << ttls_in_mins_.value();
        return;
      }

      case EvictTriggerStrategy::BY_L2WEIGHT: {
        CHECK(l2_weight_thresholds_.has_value());
        CHECK(embedding_dims_.has_value());
        LOG(INFO) << "eviction config, trigger mode:"
                  << to_string(trigger_mode_) << eviction_trigger_stats_log
                  << ", strategy: " << to_string(trigger_strategy_)
                  << ", l2_weight_thresholds: " << l2_weight_thresholds_.value()
                  << ", embedding_dims: " << embedding_dims_.value();
        return;
      }

      case EvictTriggerStrategy::BY_TIMESTAMP_THRESHOLD: {
        LOG(INFO) << "eviction config, trigger mode:"
                  << to_string(trigger_mode_) << eviction_trigger_stats_log
                  << ", strategy: " << to_string(trigger_strategy_);
        break;
      }

      default:
        throw std::runtime_error("Unknown evict trigger strategy");
    }
  }
  EvictTriggerMode trigger_mode_;
  EvictTriggerStrategy trigger_strategy_;
  std::optional<int64_t> trigger_step_interval_;
  std::optional<int64_t> mem_util_threshold_in_GB_;
  std::optional<std::vector<int64_t>> ttls_in_mins_;
  std::optional<std::vector<int64_t>> counter_thresholds_;
  std::optional<std::vector<double>> counter_decay_rates_;
  std::optional<std::vector<double>> feature_score_counter_decay_rates_;
  std::optional<std::vector<int64_t>> max_training_id_num_per_table_;
  std::optional<std::vector<double>> target_eviction_percent_per_table_;
  std::optional<std::vector<double>> l2_weight_thresholds_;
  std::optional<std::vector<int64_t>> embedding_dims_;
  std::optional<double> threshold_calculation_bucket_stride_;
  std::optional<int64_t> threshold_calculation_bucket_num_;
  int64_t interval_for_insufficient_eviction_s_;
  int64_t interval_for_sufficient_eviction_s_;
  int64_t interval_for_feature_statistics_decay_s_;
};

struct FeatureEvictMetrics {
  explicit FeatureEvictMetrics(int table_num) {
    evicted_counts.resize(table_num, 0);
    processed_counts.resize(table_num, 0);
    eviction_threshold_with_dry_run.resize(table_num, 0.0);
    exec_duration_ms = 0;
    full_duration_ms = 0;
    dry_run_exec_duration_ms = 0;
  }

  void reset() {
    std::fill(evicted_counts.begin(), evicted_counts.end(), 0);
    std::fill(processed_counts.begin(), processed_counts.end(), 0);
    std::fill(
        eviction_threshold_with_dry_run.begin(),
        eviction_threshold_with_dry_run.end(),
        0.0);
    exec_duration_ms = 0;
    full_duration_ms = 0;
    dry_run_exec_duration_ms = 0;
    start_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
  }

  void update_duration(int num_shards) {
    full_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count() -
        start_time_ms;
    // The exec_duration of all shards will be accumulated during the
    // statistics So finally, the number of shards needs to be divided
    exec_duration_ms /= num_shards;
    dry_run_exec_duration_ms /= num_shards;
  }

  std::vector<int64_t> evicted_counts;
  std::vector<int64_t> processed_counts;
  std::vector<float> eviction_threshold_with_dry_run;
  int64_t exec_duration_ms;
  int64_t full_duration_ms;
  int64_t dry_run_exec_duration_ms;
  int64_t start_time_ms;
};

struct FeatureEvictMetricTensors {
  // Constructor that initializes all members
  explicit FeatureEvictMetricTensors(int64_t table_num)
      : evicted_counts(at::zeros({table_num}, at::kLong)),
        processed_counts(at::zeros({table_num}, at::kLong)),
        eviction_threshold_with_dry_run(at::zeros({table_num}, at::kFloat)),
        exec_duration_ms(at::scalar_tensor(0, at::kLong)),
        dry_run_exec_duration_ms(at::scalar_tensor(0, at::kLong)),
        full_duration_ms(at::scalar_tensor(0, at::kLong)) {}

  // Constructor to initialize from existing tensors
  FeatureEvictMetricTensors(
      at::Tensor evicted,
      at::Tensor processed,
      at::Tensor eviction_threshold_with_dry_run,
      at::Tensor exec_duration,
      at::Tensor dry_run_exec_duration_ms,
      at::Tensor full_duration)
      : evicted_counts(std::move(evicted)),
        processed_counts(std::move(processed)),
        eviction_threshold_with_dry_run(
            std::move(eviction_threshold_with_dry_run)),
        exec_duration_ms(std::move(exec_duration)),
        dry_run_exec_duration_ms(std::move(dry_run_exec_duration_ms)),
        full_duration_ms(std::move(full_duration)) {}

  [[nodiscard]] FeatureEvictMetricTensors clone() const {
    return FeatureEvictMetricTensors{
        evicted_counts.clone(),
        processed_counts.clone(),
        eviction_threshold_with_dry_run.clone(),
        exec_duration_ms.clone(),
        dry_run_exec_duration_ms.clone(),
        full_duration_ms.clone()};
  }

  // evicted feature count
  at::Tensor evicted_counts;
  // feature count before evict
  at::Tensor processed_counts;
  // feature evict threshold with dry run
  at::Tensor eviction_threshold_with_dry_run;
  // feature evict exec duration
  at::Tensor exec_duration_ms;
  // feature evict dry run exec duration
  at::Tensor dry_run_exec_duration_ms;
  // feature evict full duration(from trigger to finish)
  at::Tensor full_duration_ms;
};

enum class EvictState { Idle, Dry_Run_Ongoing, Dry_Run_Done, Evict_Ongoing };

inline std::string to_string(EvictState state) {
  switch (state) {
    case EvictState::Idle:
      return "Idle";
    case EvictState::Dry_Run_Ongoing:
      return "Dry_Run_Ongoing";
    case EvictState::Dry_Run_Done:
      return "Dry_Run_Done";
    case EvictState::Evict_Ongoing:
      return "Evict_Ongoing";
    default:
      return "Unknown";
  }
}

template <typename weight_type>
class FeatureEvict {
 public:
  FeatureEvict(
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      int64_t interval_for_insufficient_eviction_s,
      int64_t interval_for_sufficient_eviction_s,
      int64_t interval_for_feature_statistics_decay_s,
      bool is_training = true,
      bool enable_dry_run = false,
      TestMode test_mode = TestMode::DISABLED)
      : kv_store_(kv_store),
        evict_state_(EvictState::Idle),
        evict_interrupt_(false),
        shutdown_(false),
        num_shards_(kv_store.getNumShards()),
        sub_table_hash_cumsum_(sub_table_hash_cumsum),
        metrics_(sub_table_hash_cumsum_.size()),
        metric_tensors_(sub_table_hash_cumsum_.size()),
        interval_for_insufficient_eviction_s_(
            interval_for_insufficient_eviction_s),
        interval_for_sufficient_eviction_s_(interval_for_sufficient_eviction_s),
        interval_for_feature_statistics_decay_s_(
            interval_for_feature_statistics_decay_s),
        is_training_(is_training),
        enable_dry_run_(enable_dry_run),
        test_mode_(test_mode) {
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(num_shards_);

    init_shard_status();

    // UT specific construction
    reset_ut_specific();
  }

  virtual ~FeatureEvict() {
    std::unique_lock<std::mutex> lock(mutex_);
    // set shutdown flag before setting evict_interrupt to guarantee when
    // evict_interrupt check happens shtudown flag is set already
    shutdown_.store(true);
    evict_interrupt_.store(true);

    evict_cv_.notify_all();

    // wait until futures all finished
    folly::collectAll(dry_run_futures_).wait();
    dry_run_futures_.clear();
    folly::collectAll(futures_).wait();
    futures_.clear();
  };

  void reset_ut_specific() {
    if (last_iter_shards_.size() == 0) {
      last_iter_shards_.reserve(num_shards_);
      for (int i = 0; i < num_shards_; ++i) {
        last_iter_shards_.emplace_back(
            std::make_unique<std::atomic<bool>>(false));
      }
    } else {
      for (int i = 0; i < num_shards_; ++i) {
        last_iter_shards_[i]->store(false);
      }
    }
    should_call_.store(false);
  }

  // Trigger asynchronous eviction.
  // If there is an ongoing task, return directly to prevent multiple
  // triggers. If there is no ongoing task, initialize the task state.
  void trigger_evict() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!reach_interval_to_trigger_new_round()) {
      return;
    }
    if (enable_dry_run_) {
      LOG(INFO) << "trigger new round of eviction with dry run enabled: "
                << to_string(evict_state_);
      auto evict_state = evict_state_.load();
      // if dry run or eviction task is ongoing, return directly
      if (evict_state == EvictState::Dry_Run_Ongoing ||
          evict_state == EvictState::Evict_Ongoing) {
        return;
      }
      // if no dry run running or finished and no evict running, start new
      // round eviction, run dry run first
      if (evict_state == EvictState::Idle) {
        sanity_check_before_new_round();
        evict_state_.store(EvictState::Dry_Run_Ongoing);
        prepare_evict();

        // Decide should decay or not.
        if (reach_interval_to_decay_feature_statistics()) {
          should_decay_.store(true);
        } else {
          should_decay_.store(false);
        }

        LOG(INFO)
            << "Trigger dry run eviction to get the feature evict threshold with decay "
            << should_decay_.load();

        for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
          submit_shard_task(shard_id, true); // Dry run is true
        }
      } else if (evict_state == EvictState::Dry_Run_Done) {
        // if dry run is done, run eviction
        evict_state_.store(EvictState::Evict_Ongoing);
        LOG(INFO) << "Trigger new round eviction after dry run";
        for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
          submit_shard_task(shard_id, false);
        }
        return;
      }

    } else {
      if (evict_state_.load() == EvictState::Evict_Ongoing) {
        return;
      }
      evict_state_.store(EvictState::Evict_Ongoing);
      LOG(INFO) << "trigger new round of eviction";
      sanity_check_before_new_round();
      prepare_evict();
      for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
        submit_shard_task(shard_id, false);
      }
    }
  }

  // Get feature eviction metric.
  FeatureEvictMetricTensors get_feature_evict_metric() {
    std::unique_lock<std::mutex> lock(metric_mtx_);
    return metric_tensors_.clone();
  }

  // resume task execution
  void resume() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto evict_state = evict_state_.load();
    if (evict_state == EvictState::Idle)
      return;
    evict_interrupt_.store(false);
    evict_cv_.notify_all();
    return;
  };

  // Pause the eviction process. Returns true if there is an ongoing task,
  // false otherwise. During the pause phase, check whether the eviction is
  // complete.
  bool pause() {
    // std::unique_lock<std::mutex> lock(mutex_);
    std::unique_lock<std::mutex> lock(mutex_);
    auto evict_state = evict_state_.load();
    if (evict_state == EvictState::Idle)
      return false;
    evict_interrupt_.store(true);
    return true;
  }

  // Check whether a round of eviction is ongoing.
  // True, even if eviction is paused
  bool is_evicting() {
    std::unique_lock<std::mutex> lock(mutex_);
    return evict_state_.load() != EvictState::Idle;
  }

  // wait until eviction round finishes
  void wait_until_eviction_done() {
    resume();

    folly::collectAll(dry_run_futures_).wait();
    dry_run_futures_.clear();

    folly::collectAll(futures_).wait();
    futures_.clear();
  }

  virtual void update_feature_statistics(weight_type* block) = 0;

  void wait_completion() {
    folly::collectAll(dry_run_futures_).wait();
    dry_run_futures_.clear();
    folly::collectAll(futures_).wait();
    futures_.clear();
  }

 protected:
  void sanity_check_before_new_round() {
    CHECK_EQ(num_waiting_evicts_.load(), 0)
        << "found " << num_waiting_evicts_.load()
        << " waiting evicts before triggering new round of evict, this should be 0";
    auto finished_evicts = finished_evictions_.load();
    CHECK(finished_evicts == 0 || finished_evicts == num_shards_)
        << "found " << finished_evicts
        << " finished evicts before triggering new round of evict, "
        << "this should be either 0 or num shards:" << num_shards_;
    CHECK(futures_.size() == 0 || futures_.size() == num_shards_)
        << "found " << futures_.size()
        << " futures before triggering new round of evict, "
        << "this should be either 0 or num shards:" << num_shards_;
    CHECK(
        dry_run_futures_.size() == 0 || dry_run_futures_.size() == num_shards_)
        << "found " << dry_run_futures_.size()
        << " futures before triggering new round of evict, "
        << "this should be either 0 or num shards:" << num_shards_;
  }
  void init_shard_status() {
    block_cursors_.resize(num_shards_);
    dry_run_block_cursors_.resize(num_shards_);
    block_nums_snapshot_.resize(num_shards_);
    for (int i = 0; i < num_shards_; ++i) {
      block_cursors_[i] = 0;
      dry_run_block_cursors_[i] = 0;
      block_nums_snapshot_[i] = 0;
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
      dry_run_block_cursors_[shard_id] = 0;
    }
    metrics_.reset();
    dry_run_futures_.clear();
    futures_.clear();
    finished_evictions_.store(0);
    finished_dry_run_.store(0);
    // make sure we don't start right away, wait until resume() is called
    evict_interrupt_.store(true);
  }

  // submitting eviction job to the executor
  void submit_shard_task(int shard_id, bool dry_run) {
    if (dry_run) {
      dry_run_futures_.emplace_back(
          folly::via(executor_.get())
              .thenValue([this, shard_id, dry_run](auto&&) {
                process_shard(shard_id, true);
                update_evict_finish_flags(shard_id, dry_run);
              }));
    } else {
      futures_.emplace_back(folly::via(executor_.get())
                                .thenValue([this, shard_id, dry_run](auto&&) {
                                  process_shard(shard_id, dry_run);
                                  update_evict_finish_flags(shard_id, dry_run);
                                }));
    }
  }

  bool reach_interval_to_trigger_new_round() {
    auto now_ts =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    auto time_elapsed = now_ts - last_eviction_ts_.load();
    if (is_last_eviction_sufficient_.load()) {
      return time_elapsed >= interval_for_sufficient_eviction_s_;
    } else {
      return time_elapsed >= interval_for_insufficient_eviction_s_;
    }
  }

  bool reach_interval_to_decay_feature_statistics() {
    auto now_ts =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();

    auto time_elapsed = now_ts - last_decay_ts_.load();
    return time_elapsed >= interval_for_feature_statistics_decay_s_;
  }

  // conditions where we need to break the evict loop
  // currently conditions are:
  // 1. evict is paused
  // 2. evict is finished
  // 3. evict is distroyed
  bool should_exit_evict_loop(int shard_id) {
    return evict_interrupt_.load() ||
        block_cursors_[shard_id] >= block_nums_snapshot_[shard_id];
  }

  // conditions where we need to break the dry run evict loop
  bool should_exit_dry_run_loop(int shard_id) {
    return evict_interrupt_.load() ||
        dry_run_block_cursors_[shard_id] >= block_nums_snapshot_[shard_id];
  }

  // check whether there is any evict neither paused nor finished
  bool has_running_evict() {
    return (num_waiting_evicts_.load() + finished_evictions_.load()) !=
        num_shards_;
  }

  // check whether there is any dry run neither paused nor finished
  bool has_running_dry_run() {
    return finished_dry_run_.load() != sub_table_hash_cumsum_.size();
  }

  // the inner loop of each evict that can be paused
  void start_training_eviction_loop(
      int shard_id,
      std::vector<int64_t>& evicted_counts,
      std::vector<int64_t>& processed_counts) {
    auto wlock = kv_store_.by(shard_id).wlock();
    auto* pool = kv_store_.pool_by(shard_id);

    while (!should_exit_evict_loop(shard_id)) {
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
          auto time_elapsed = FixedBlockPool::current_timestamp() -
              FixedBlockPool::get_timestamp(block);
          if (time_elapsed < 1800) { // 30 mins
            LOG_EVERY_N(WARNING, 1000)
                << "Evicting key:" << key << " with " << time_elapsed
                << " seconds, less than 30 mins elapsed since first seen,"
                << " make sure this is expected";
          }
          wlock->erase(key);
          pool->template deallocate_t<weight_type>(block);
          evicted_counts[sub_table_id]++;
        }
      }
    }
  }

  // the inner loop of each evict that can be paused
  void start_inference_eviction_loop(
      int shard_id,
      std::vector<int64_t>& evicted_counts,
      std::vector<int64_t>& processed_counts) {
    auto* pool = kv_store_.pool_by(shard_id);
    auto mem_pool_lock = pool->acquire_lock();

    std::vector<int> evicting_keys;
    evicting_keys.reserve(block_nums_snapshot_[shard_id] / 100);
    while (!should_exit_evict_loop(shard_id)) {
      auto* block =
          pool->template get_block<weight_type>(block_cursors_[shard_id]++);
      if (block == nullptr) {
        continue;
      }
      int64_t key = FixedBlockPool::get_key(block);
      int sub_table_id = get_sub_table_id(key);
      processed_counts[sub_table_id]++;
      if (evict_block(block, sub_table_id)) {
        pool->template deallocate_t<weight_type>(block);
        evicted_counts[sub_table_id]++;
        evicting_keys.push_back(key);
      }
    }
    mem_pool_lock.unlock();

    // lock dram kv shard hash map to remove evicted blocks in the map
    // dedicate map update in a wlock to reduce the blocking time for inference
    // read
    auto shard_map_wlock = kv_store_.by(shard_id).wlock();
    for (auto& key : evicting_keys) {
      shard_map_wlock->erase(key);
    }
  }

  // return false when we should exit, true to continue evicting
  bool wait_until_resume(int shard_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (shutdown_.load()) {
      return false;
    }
    if (!should_exit_evict_loop(shard_id)) {
      return true;
    }

    if (!should_exit_dry_run_loop(shard_id)) {
      return true;
    }

    num_waiting_evicts_++;
    evict_cv_.wait(lock, [this] { return !evict_interrupt_.load(); });
    num_waiting_evicts_--;
    if (shutdown_.load()) {
      // faster shutdown path, skip dealing with metrics
      return false;
    }
    return true;
  }

  // the outer loop of each evict round that only exits when evction round is
  // done
  void process_shard(int shard_id, bool dry_run) {
    std::chrono::milliseconds duration{};

    if (dry_run) {
      auto start_time = std::chrono::high_resolution_clock::now();
      dry_run_calculate_thresholds(shard_id);
      duration += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - start_time);
      {
        std::unique_lock<std::mutex> lock(metric_mtx_);
        metrics_.dry_run_exec_duration_ms += duration.count();
      }
    } else {
      std::vector<int64_t> evicted_counts(sub_table_hash_cumsum_.size(), 0);
      std::vector<int64_t> processed_counts(sub_table_hash_cumsum_.size(), 0);
      // each active eviction round
      while (block_cursors_[shard_id] < block_nums_snapshot_[shard_id]) {
        if (!wait_until_resume(shard_id)) {
          return;
        }
        auto start_time = std::chrono::high_resolution_clock::now();
        if (is_training_) {
          start_training_eviction_loop(
              shard_id, evicted_counts, processed_counts);
        } else {
          start_inference_eviction_loop(
              shard_id, evicted_counts, processed_counts);
        }
        duration += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
      }
      if (test_mode_ == TestMode::PAUSE_ON_LAST_ITERATION &&
          block_cursors_[shard_id] == block_nums_snapshot_[shard_id]) {
        last_iter_shards_[shard_id]->store(true);
        should_call_.store(true);
        // hold on on the last iteration for a while waiting for the UT to call
        // pause before updating the shards_finished
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      {
        std::unique_lock<std::mutex> lock(metric_mtx_);
        metrics_.exec_duration_ms += duration.count();
        for (size_t i = 0; i < evicted_counts.size(); ++i) {
          metrics_.evicted_counts[i] += evicted_counts[i];
          metrics_.processed_counts[i] += processed_counts[i];
        }
      }
    }
  }

  virtual bool evict_block(weight_type* block, int sub_table_id) = 0;

  virtual void dry_run_calculate_thresholds(int shard_id) = 0;

  // Check and reset the eviction state .
  void update_evict_finish_flags(int shard_id, bool dry_run) {
    if (dry_run) {
      bool trigger_evict_after_dry_run = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!has_running_dry_run()) {
          bool all_finished =
              finished_dry_run_.load() == sub_table_hash_cumsum_.size();
          if (all_finished &&
              evict_state_.load() == EvictState::Dry_Run_Ongoing &&
              shard_id == 0) {
            evict_state_.store(EvictState::Dry_Run_Done);
            // dry run is finished, tigger real eviction process;
            // Only change evict_state and tigger evict on shard 0
            trigger_evict_after_dry_run = true;
          }
        }
      }

      // Trigger evict outside the lock
      if (trigger_evict_after_dry_run) {
        trigger_evict();
      }
    } else {
      std::unique_lock<std::mutex> lock(mutex_);
      finished_evictions_++;
      if (!has_running_evict()) {
        bool all_finished = finished_evictions_.load() == num_shards_;
        if (all_finished && evict_state_.load() == EvictState::Evict_Ongoing) {
          record_metrics_to_report_tensor();
          int64_t num_evicts = 0;
          for (long evicted_count : metrics_.evicted_counts) {
            num_evicts += evicted_count;
          }
          is_last_eviction_sufficient_.store(num_evicts > 100);
          last_eviction_ts_ =
              std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::high_resolution_clock::now().time_since_epoch())
                  .count();
          // update evict_state_ in the last place, making sure the future
          // finishes around the same time as evict_state_ reset
          evict_state_.store(EvictState::Idle);
        }
      }
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
    std::unique_lock<std::mutex> lock(metric_mtx_);
    metrics_.update_duration(num_shards_);
    metric_tensors_.evicted_counts =
        at::from_blob(
            const_cast<int64_t*>(metrics_.evicted_counts.data()),
            {static_cast<int64_t>(metrics_.evicted_counts.size())},
            at::kLong)
            .clone();

    metric_tensors_.processed_counts =
        at::from_blob(
            const_cast<int64_t*>(metrics_.processed_counts.data()),
            {static_cast<int64_t>(metrics_.processed_counts.size())},
            at::kLong)
            .clone();

    metric_tensors_.eviction_threshold_with_dry_run =
        at::from_blob(
            const_cast<float*>(metrics_.eviction_threshold_with_dry_run.data()),
            {static_cast<int64_t>(
                metrics_.eviction_threshold_with_dry_run.size())},
            at::kFloat)
            .clone();

    metric_tensors_.full_duration_ms =
        at::scalar_tensor(metrics_.full_duration_ms, at::kLong);
    metric_tensors_.exec_duration_ms =
        at::scalar_tensor(metrics_.exec_duration_ms, at::kLong);
    metric_tensors_.dry_run_exec_duration_ms =
        at::scalar_tensor(metrics_.dry_run_exec_duration_ms, at::kLong);
    std::vector<float> evict_rates(metrics_.evicted_counts.size());
    for (size_t i = 0; i < metrics_.evicted_counts.size(); ++i) {
      evict_rates[i] = metrics_.processed_counts[i] > 0
          ? (metrics_.evicted_counts[i] * 100.0f) / metrics_.processed_counts[i]
          : 0.0f;
    }
    LOG(INFO) << fmt::format(
        "Feature evict completed: \n"
        "  - full Time taken: {}ms\n"
        "  - exec Time taken: {}ms\n"
        "  - exec / full: {:.2f}%\n"
        "  - dryrun Time taken: {}ms\n"
        "  - Total blocks processed: [{}]\n"
        "  - Blocks evicted: [{}]\n"
        "  - Eviction rate: [{}]%\n"
        "  - Eviction threshold dry run: [{}]\n",
        metrics_.full_duration_ms,
        metrics_.exec_duration_ms,
        metrics_.exec_duration_ms * 100.0f / metrics_.full_duration_ms,
        metrics_.dry_run_exec_duration_ms,
        fmt::join(metrics_.processed_counts, ", "),
        fmt::join(metrics_.evicted_counts, ", "),
        fmt::join(evict_rates, ", "),
        fmt::join(metrics_.eviction_threshold_with_dry_run, ", "));
  }

  // Thread pool.
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  // Sharded map.
  SynchronizedShardedMap<int64_t, weight_type*>& kv_store_;
  // Index of processed blocks.
  std::vector<std::size_t> block_cursors_;
  // Index of processed dry run blocks.
  std::vector<std::size_t> dry_run_block_cursors_;
  // Snapshot of total blocks at eviction trigger.
  std::vector<std::size_t> block_nums_snapshot_;
  // Indicates whether an eviction task is ongoing.
  std::atomic<EvictState> evict_state_;
  // Indicates whether the eviction task is paused.
  std::atomic<bool> evict_interrupt_;
  std::atomic<bool> shutdown_;
  // cv to waking up when resume long running tasks
  std::condition_variable evict_cv_;

  // number waiting/finished evicts, used for blocking pause
  std::atomic<int> num_waiting_evicts_{0};
  std::atomic<int> finished_evictions_{0};
  std::atomic<int> finished_dry_run_{0};

  // Records of shard tasks.
  std::vector<folly::Future<folly::Unit>> futures_;
  std::vector<folly::Future<folly::Unit>> dry_run_futures_;
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

  // eviction intervals
  std::atomic<int64_t> last_eviction_ts_{0};
  std::atomic<bool> is_last_eviction_sufficient_{false};

  // decay
  std::atomic<bool> should_decay_ = false;
  std::atomic<int64_t> last_decay_ts_{
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count()};

  const int64_t interval_for_insufficient_eviction_s_;
  const int64_t interval_for_sufficient_eviction_s_;
  const int64_t interval_for_feature_statistics_decay_s_;

  // TODO: use this 2 threshold to help decide whether an eviction is enough
  // absolute amount of evictions for eviction to be considered as "enough"
  const int64_t abs_evicts_enough_threshold{100};
  // pct amount of evictions for eviction to be considered as "enough"
  const double pct_evicts_enough_threshold{0.01}; // 0.01%

  const bool is_training_;
  const bool enable_dry_run_;

  // UT specific mode
  TestMode test_mode_;
  std::atomic<bool> should_call_ = false;
  std::vector<std::unique_ptr<std::atomic<bool>>> last_iter_shards_;

  FRIEND_TEST(FeatureEvictTest, DupAPINoOpCheck);
  FRIEND_TEST(FeatureEvictTest, EdgeCase_NoPause);
  FRIEND_TEST(FeatureEvictTest, EdgeCase_PauseOnLastIter);
};

template <typename weight_type>
class CounterBasedEvict : public FeatureEvict<weight_type> {
 public:
  CounterBasedEvict(
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      const std::vector<double>& decay_rates,
      const std::vector<int64_t>& thresholds,
      int64_t interval_for_insufficient_eviction_s,
      int64_t interval_for_sufficient_eviction_s,
      int64_t interval_for_feature_statistics_decay_s,
      bool is_training,
      bool enable_dry_run_,
      TestMode test_mode = TestMode::DISABLED)
      : FeatureEvict<weight_type>(
            kv_store,
            sub_table_hash_cumsum,
            interval_for_insufficient_eviction_s,
            interval_for_sufficient_eviction_s,
            interval_for_feature_statistics_decay_s,
            is_training,
            enable_dry_run_,
            test_mode),
        decay_rates_(decay_rates),
        thresholds_(thresholds) {
    LOG(WARNING)
        << "CounterBasedEvict is not supported for Non-UT cases for now, "
        << "make sure you know what you are doing";
  }

  void update_feature_statistics(weight_type* block) override {
    FixedBlockPool::update_count(block);
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    double decay_rate = decay_rates_[sub_table_id];
    int64_t threshold = thresholds_[sub_table_id];
    // Apply decay and check the threshold.
    auto current_count = FixedBlockPool::get_count(block);
    current_count *= decay_rate;
    FixedBlockPool::set_count(block, current_count);
    return current_count < threshold;
  }

  void dry_run_calculate_thresholds(int shard_id) override {}

 private:
  const std::vector<double>& decay_rates_; // Decay rate for the block count.
  const std::vector<int64_t>& thresholds_; // Threshold for eviction.
};

template <typename weight_type>
class FeatureScoreBasedEvict : public FeatureEvict<weight_type> {
 public:
  FeatureScoreBasedEvict(
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      const std::vector<double>& decay_rates,
      const std::vector<int64_t>& max_training_id_num_per_table,
      const std::vector<double>& target_eviction_percent_per_table,
      const double threshold_calculation_bucket_stride,
      const int64_t threshold_calculation_bucket_num,
      int64_t interval_for_insufficient_eviction_s,
      int64_t interval_for_sufficient_eviction_s,
      int64_t interval_for_feature_statistics_decay_s,
      bool is_training,
      bool enable_dry_run_,
      TestMode test_mode = TestMode::DISABLED)
      : FeatureEvict<weight_type>(
            kv_store,
            sub_table_hash_cumsum,
            interval_for_insufficient_eviction_s,
            interval_for_sufficient_eviction_s,
            interval_for_feature_statistics_decay_s,
            is_training,
            enable_dry_run_,
            test_mode),
        decay_rates_(decay_rates),
        max_training_id_num_per_table_(max_training_id_num_per_table),
        target_eviction_percent_per_table_(target_eviction_percent_per_table),
        threshold_calculation_bucket_stride_(
            threshold_calculation_bucket_stride),
        num_buckets_(threshold_calculation_bucket_num),
        finalize_barrier_(this->num_shards_) {
    LOG(WARNING)
        << "FeatureScoreBasedEvict is not supported for Non-UT cases for now, "
        << "make sure you know what you are doing";

    local_buckets_per_shard_per_table_ =
        std::vector<std::vector<std::vector<size_t>>>(
            num_tables_,
            std::vector<std::vector<size_t>>(
                this->num_shards_, std::vector<size_t>(num_buckets_, 0)));

    local_blocks_num_per_shard_per_table_ = std::vector<std::vector<size_t>>(
        num_tables_, std::vector<size_t>(this->num_shards_, 0));

    thresholds_.resize(sub_table_hash_cumsum.size(), 0.0);
    evict_modes_.resize(num_tables_);
  }

  void update_feature_statistics(weight_type* block) override {
    FixedBlockPool::update_timestamp(block);
  }

  void update_feature_score_statistics(weight_type* block, double ratio) {
    double old_ratio = FixedBlockPool::get_feature_score_rate(block);
    FixedBlockPool::set_feature_score_rate(block, ratio + old_ratio);
  }

  std::vector<double> get_thresholds() {
    return thresholds_;
  }

  void set_num_buckets(int num_buckets) {
    num_buckets_ = num_buckets;
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    double threshold = thresholds_[sub_table_id];
    double overall_ratio = FixedBlockPool::get_feature_score_rate(block);
    const double EPSILON = 1e-9;
    switch (evict_modes_[sub_table_id]) {
      case EvictMode::NONE:
        return false;
      case EvictMode::ONLY_ZERO:
        return std::abs(overall_ratio) < EPSILON;
      case EvictMode::THRESHOLD:
        return overall_ratio < threshold;
      default:
        LOG(ERROR) << "Invalid evict mode";
        return false;
    }
  }

  void dry_run_calculate_thresholds(int shard_id) override {
    auto collect_buckets_start_time = std::chrono::high_resolution_clock::now();
    while (this->dry_run_block_cursors_[shard_id] <
           this->block_nums_snapshot_[shard_id]) {
      if (!this->wait_until_resume(shard_id)) {
        return;
      }
      // scan every shard to get bucket information
      collect_buckets(shard_id);
    }

    auto lock_start_time = std::chrono::high_resolution_clock::now();
    finalize_barrier_.arrive_and_wait();
    auto lock_end_time = std::chrono::high_resolution_clock::now();
    auto lock_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                             lock_end_time - lock_start_time)
                             .count();
    LOG(INFO) << "[Dry run debug]collect_buckets lock took " << lock_duration
              << " ms";

    auto collect_buckets_end_time = std::chrono::high_resolution_clock::now();

    auto collect_buckets_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            collect_buckets_end_time - collect_buckets_start_time)
            .count();

    LOG(INFO) << "[Dry run debug]collect_buckets for loop took "
              << collect_buckets_duration << " ms";

    if (shard_id == 0) {
      auto collect_buckets_wait_start_time =
          std::chrono::high_resolution_clock::now();
      compute_thresholds_from_buckets();
      finalize_dry_run();
      auto collect_buckets_wait_end_time =
          std::chrono::high_resolution_clock::now();
      auto collect_buckets_wait_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              collect_buckets_wait_end_time - collect_buckets_wait_start_time)
              .count();
      LOG(INFO)
          << "[Dry run debug]collect_buckets wait all shards and compute_thresholds_from_buckets and finalize_dry_run took "
          << collect_buckets_wait_duration << " ms";
    }
  }

 private:
  void collect_buckets(int shard_id) {
    auto* pool = this->kv_store_.pool_by(shard_id);
    auto wlock = this->kv_store_.by(shard_id).wlock();
    while (!this->should_exit_dry_run_loop(shard_id)) {
      auto* block = pool->template get_block<weight_type>(
          this->dry_run_block_cursors_[shard_id]++);
      if (block == nullptr || !FixedBlockPool::get_used(block))
        continue;
      int64_t key = FixedBlockPool::get_key(block);
      int sub_table_id = this->get_sub_table_id(key);
      if (this->should_decay_) {
        auto it = wlock->find(key);
        if (it != wlock->end() && block == it->second) {
          double ratio = FixedBlockPool::get_feature_score_rate(block);
          double decay_rate = decay_rates_[sub_table_id];
          FixedBlockPool::set_feature_score_rate(block, ratio * decay_rate);
        }
      }
      double ratio = FixedBlockPool::get_feature_score_rate(block);
      int64_t idx = 0;
      const double EPSILON = 1e-9;
      if (ratio < 0) {
        continue;
      } else if (std::abs(ratio) < EPSILON) {
        idx = 0;
      } else if (ratio >= num_buckets_ * threshold_calculation_bucket_stride_) {
        idx = num_buckets_ - 1;
      } else {
        idx =
            static_cast<int64_t>(ratio / threshold_calculation_bucket_stride_) +
            1;
      }

      // Adding check to avoid out of bound access
      if (idx < 0 || idx >= num_buckets_) {
        LOG(ERROR) << "[Dry Run Debug]Invalid idx: " << idx
                   << " for key: " << key << " ratio: " << ratio;
        continue;
      }

      local_buckets_per_shard_per_table_[sub_table_id][shard_id][idx]++;
      local_blocks_num_per_shard_per_table_[sub_table_id][shard_id]++;
    }
  }

  void compute_thresholds_from_buckets() {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t table_id = 0; table_id < num_tables_; ++table_id) {
      int64_t total = 0;

      for (int shard_id = 0; shard_id < this->num_shards_; ++shard_id) {
        total +=
            this->local_blocks_num_per_shard_per_table_[table_id][shard_id];
      }

      const double target_keep_ratio =
          1 - target_eviction_percent_per_table_[table_id];
      int64_t max_id =
          static_cast<int64_t>(max_training_id_num_per_table_[table_id]);
      int64_t keep_count = static_cast<int64_t>(max_id * target_keep_ratio);
      int64_t evict_count = total - keep_count;

      int64_t acc_count = 0;
      int64_t threshold_bucket = num_buckets_ - 1;
      int64_t bucket0_count = 0;
      for (int shard_id = 0; shard_id < this->num_shards_; ++shard_id) {
        bucket0_count +=
            local_buckets_per_shard_per_table_[table_id][shard_id][0];
      }
      if (total <= keep_count) {
        // Case 1: If the total blocks do not exceed the retention target,
        // no eviction is needed for this table.
        thresholds_[table_id] = 0.0;
        evict_modes_[table_id] = EvictMode::NONE;
      } else if (bucket0_count >= evict_count) {
        // Case 2: If bucket 0 alone contains sufficient blocks to satisfy the
        // eviction demand, restrict eviction only to bucket 0 (blocks with
        // score == 0).
        thresholds_[table_id] = 0.0;
        evict_modes_[table_id] = EvictMode::ONLY_ZERO;
      } else {
        // Case 3: Need to evict across multiple buckets.
        // Accumulate candidate blocks from bucket 0 and higher until the
        // eviction quota is met.
        acc_count = bucket0_count;
        bool found_threshold_bucket = false;
        for (int bucket_id = 1; bucket_id < num_buckets_; ++bucket_id) {
          for (int shard_id = 0; shard_id < this->num_shards_; ++shard_id) {
            acc_count += local_buckets_per_shard_per_table_[table_id][shard_id]
                                                           [bucket_id];
            if (acc_count >= evict_count && !found_threshold_bucket) {
              threshold_bucket = bucket_id;
              found_threshold_bucket = true;
            }
          }
          if (found_threshold_bucket)
            break;
        }

        double threshold =
            threshold_bucket * threshold_calculation_bucket_stride_;
        thresholds_[table_id] = threshold;
        evict_modes_[table_id] = EvictMode::THRESHOLD;
      }
      LOG(INFO) << "[Dry Run Result]table " << table_id
                << " threshold: " << thresholds_[table_id]
                << " threshold bucket: " << threshold_bucket
                << " acc count: " << acc_count
                << " evict count: " << evict_count << " total: " << total;

      {
        std::unique_lock<std::mutex> lock(this->mutex_);
        this->finished_dry_run_++;
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    LOG(INFO) << "[Dry run debug]compute_thresholds_from_buckets for loop took "
              << duration << " ms";
  }

  void finalize_dry_run() {
    local_buckets_per_shard_per_table_ =
        std::vector<std::vector<std::vector<size_t>>>(
            num_tables_,
            std::vector<std::vector<size_t>>(
                this->num_shards_, std::vector<size_t>(num_buckets_, 0)));

    local_blocks_num_per_shard_per_table_ = std::vector<std::vector<size_t>>(
        num_tables_, std::vector<size_t>(this->num_shards_, 0));

    if (this->should_decay_) {
      this->last_decay_ts_ =
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();
    }

    for (int table_id = 0; table_id < num_tables_; ++table_id) {
      this->metrics_.eviction_threshold_with_dry_run[table_id] =
          thresholds_[table_id];
    }
  }

 private:
  enum class EvictMode {
    NONE, // no evict
    ONLY_ZERO, // only blocks in bucket 0 (score == 0) are considered for
               // eviction
    THRESHOLD // blocks with scores below the computed threshold will be
              // evicted
  };
  std::vector<EvictMode> evict_modes_;

  const int num_tables_ = static_cast<int>(this->sub_table_hash_cumsum_.size());
  const std::vector<double>& decay_rates_; // Decay rate for the block count.
  std::vector<double> thresholds_; // Threshold for eviction.

  const std::vector<int64_t>&
      max_training_id_num_per_table_; // training max id for each table.
  const std::vector<double>&
      target_eviction_percent_per_table_; // target eviction percent for each
                                          // table
  std::vector<std::vector<std::vector<size_t>>>
      local_buckets_per_shard_per_table_;
  std::vector<std::vector<size_t>> local_blocks_num_per_shard_per_table_;

  const double threshold_calculation_bucket_stride_; // stride for bucketing
  int num_buckets_; // number of buckets for threshold calculation

  std::barrier<> finalize_barrier_;
};

template <typename weight_type>
class TimeBasedEvict : public FeatureEvict<weight_type> {
 public:
  TimeBasedEvict(
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      const std::vector<int64_t>& ttls_in_mins,
      int64_t interval_for_insufficient_eviction_s,
      int64_t interval_for_sufficient_eviction_s,
      int64_t interval_for_feature_statistics_decay_s,
      bool is_training,
      bool enable_dry_run_)
      : FeatureEvict<weight_type>(
            kv_store,
            sub_table_hash_cumsum,
            interval_for_insufficient_eviction_s,
            interval_for_sufficient_eviction_s,
            interval_for_feature_statistics_decay_s,
            is_training,
            enable_dry_run_),
        ttls_in_mins_(ttls_in_mins) {}

  void update_feature_statistics(weight_type* block) override {
    FixedBlockPool::update_timestamp(block);
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    int64_t ttl = ttls_in_mins_[sub_table_id];
    if (ttl == 0) {
      // ttl = 0 means no eviction
      return false;
    }
    auto current_time = FixedBlockPool::current_timestamp();
    return current_time - FixedBlockPool::get_timestamp(block) > ttl * 60;
  }

  void dry_run_calculate_thresholds(int shard_id) override {}

 private:
  const std::vector<int64_t>& ttls_in_mins_; // Time-to-live for eviction.
};

template <typename weight_type>
class TimeThresholdBasedEvict : public FeatureEvict<weight_type> {
 public:
  TimeThresholdBasedEvict(
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      int64_t interval_for_insufficient_eviction_s,
      int64_t interval_for_sufficient_eviction_s,
      int64_t interval_for_feature_statistics_decay_s,
      bool is_training,
      bool enable_dry_run_)
      : FeatureEvict<weight_type>(
            kv_store,
            sub_table_hash_cumsum,
            interval_for_insufficient_eviction_s,
            interval_for_sufficient_eviction_s,
            interval_for_feature_statistics_decay_s,
            is_training,
            enable_dry_run_) {}

  void update_feature_statistics(weight_type* block) override {
    FixedBlockPool::update_timestamp(block);
  }

  void set_eviction_timestamp_threshold(uint32_t timestamp) {
    eviction_timestamp_threshold_ = timestamp;
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    return FixedBlockPool::get_timestamp(block) < eviction_timestamp_threshold_;
  }

  void dry_run_calculate_thresholds(int shard_id) override {}

 private:
  uint32_t eviction_timestamp_threshold_ = 0;
};

template <typename weight_type>
class TimeCounterBasedEvict : public FeatureEvict<weight_type> {
 public:
  TimeCounterBasedEvict(
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      const std::vector<int64_t>& ttls_in_mins,
      const std::vector<double>& decay_rates,
      const std::vector<int64_t>& thresholds,
      int64_t interval_for_insufficient_eviction_s,
      int64_t interval_for_sufficient_eviction_s,
      int64_t interval_for_feature_statistics_decay_s,
      bool is_training,
      bool enable_dry_run_)
      : FeatureEvict<weight_type>(
            kv_store,
            sub_table_hash_cumsum,
            interval_for_insufficient_eviction_s,
            interval_for_sufficient_eviction_s,
            interval_for_feature_statistics_decay_s,
            is_training,
            enable_dry_run_),
        ttls_in_mins_(ttls_in_mins),
        decay_rates_(decay_rates),
        thresholds_(thresholds) {}

  void update_feature_statistics(weight_type* block) override {
    FixedBlockPool::update_timestamp(block);
    FixedBlockPool::update_count(block);
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    int64_t ttl = ttls_in_mins_[sub_table_id];
    if (ttl == 0) {
      // ttl = 0 means no eviction
      return false;
    }
    double decay_rate = decay_rates_[sub_table_id];
    int64_t threshold = thresholds_[sub_table_id];
    if (threshold == 0) {
      // threshold = 0 means no eviction
      return false;
    }

    // Apply decay and check the count threshold and ttl.
    auto current_time = FixedBlockPool::current_timestamp();
    auto current_count = FixedBlockPool::get_count(block);
    current_count *= decay_rate;
    FixedBlockPool::set_count(block, current_count);
    return (current_time - FixedBlockPool::get_timestamp(block) > ttl * 60) &&
        (current_count < threshold);
  }

  void dry_run_calculate_thresholds(int shard_id) override {}

 private:
  const std::vector<int64_t>& ttls_in_mins_; // Time-to-live for eviction.
  const std::vector<double>& decay_rates_; // Decay rate for the block count.
  const std::vector<int64_t>& thresholds_; // Count threshold for eviction.
};

template <typename weight_type>
class L2WeightBasedEvict : public FeatureEvict<weight_type> {
 public:
  L2WeightBasedEvict(
      SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      const std::vector<double>& thresholds,
      const std::vector<int64_t>& sub_table_dims,
      int64_t interval_for_insufficient_eviction_s,
      int64_t interval_for_sufficient_eviction_s,
      int64_t interval_for_feature_statistics_decay_s,
      bool is_training,
      bool enable_dry_run_)
      : FeatureEvict<weight_type>(
            kv_store,
            sub_table_hash_cumsum,
            interval_for_insufficient_eviction_s,
            interval_for_sufficient_eviction_s,
            interval_for_feature_statistics_decay_s,
            is_training,
            enable_dry_run_),
        thresholds_(thresholds),
        sub_table_dims_(sub_table_dims) {}

  void update_feature_statistics([[maybe_unused]] weight_type* block) override {
  }

 protected:
  bool evict_block(weight_type* block, int sub_table_id) override {
    size_t dimension = sub_table_dims_[sub_table_id];
    double threshold = thresholds_[sub_table_id];
    if (threshold == 0.0) {
      // threshold = 0 means no eviction
      return false;
    }
    auto l2weight = FixedBlockPool::get_l2weight(block, dimension);
    return l2weight < threshold;
  }

  void dry_run_calculate_thresholds(int shard_id) override {}

 private:
  const std::vector<double>& thresholds_; // L2 weight threshold for eviction.
  const std::vector<int64_t>& sub_table_dims_; // Embedding dimension
};

template <typename weight_type>
std::unique_ptr<FeatureEvict<weight_type>> create_feature_evict(
    c10::intrusive_ptr<FeatureEvictConfig> config,
    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
    const std::vector<int64_t>& sub_table_hash_cumsum,
    bool is_training = true,
    TestMode test_mode = TestMode::DISABLED) {
  switch (config->trigger_strategy_) {
    case EvictTriggerStrategy::BY_TIMESTAMP: {
      return std::make_unique<TimeBasedEvict<weight_type>>(
          kv_store,
          sub_table_hash_cumsum,
          config->ttls_in_mins_.value(),
          config->interval_for_insufficient_eviction_s_,
          config->interval_for_sufficient_eviction_s_,
          config->interval_for_feature_statistics_decay_s_,
          is_training,
          false);
    }

    case EvictTriggerStrategy::BY_COUNTER: {
      for (auto count_decay_rate : config->counter_decay_rates_.value()) {
        if (count_decay_rate <= 0 || count_decay_rate > 1) {
          throw std::invalid_argument(
              "count_decay_rate must be in range (0,1]");
        }
      }
      return std::make_unique<CounterBasedEvict<weight_type>>(
          kv_store,
          sub_table_hash_cumsum,
          config->counter_decay_rates_.value(),
          config->counter_thresholds_.value(),
          config->interval_for_insufficient_eviction_s_,
          config->interval_for_sufficient_eviction_s_,
          config->interval_for_feature_statistics_decay_s_,
          is_training,
          false,
          test_mode);
    }

    case EvictTriggerStrategy::BY_FEATURE_SCORE: {
      for (auto count_decay_rate :
           config->feature_score_counter_decay_rates_.value()) {
        if (count_decay_rate <= 0 || count_decay_rate > 1) {
          throw std::invalid_argument(
              "feature_score_counter_decay_rates_ must be in range (0,1]");
        }
      }
      return std::make_unique<FeatureScoreBasedEvict<weight_type>>(
          kv_store,
          sub_table_hash_cumsum,
          config->feature_score_counter_decay_rates_.value(),
          config->max_training_id_num_per_table_.value(),
          config->target_eviction_percent_per_table_.value(),
          config->threshold_calculation_bucket_stride_.value(),
          config->threshold_calculation_bucket_num_.value(),
          config->interval_for_insufficient_eviction_s_,
          config->interval_for_sufficient_eviction_s_,
          config->interval_for_feature_statistics_decay_s_,
          is_training,
          true,
          test_mode);
    }

    case EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER: {
      for (auto count_decay_rate : config->counter_decay_rates_.value()) {
        if (count_decay_rate <= 0 || count_decay_rate > 1) {
          throw std::invalid_argument(
              "count_decay_rate must be in range (0,1]");
        }
      }
      return std::make_unique<TimeCounterBasedEvict<weight_type>>(
          kv_store,
          sub_table_hash_cumsum,
          config->ttls_in_mins_.value(),
          config->counter_decay_rates_.value(),
          config->counter_thresholds_.value(),
          config->interval_for_insufficient_eviction_s_,
          config->interval_for_sufficient_eviction_s_,
          config->interval_for_feature_statistics_decay_s_,
          is_training,
          false);
    }

    case EvictTriggerStrategy::BY_L2WEIGHT: {
      for (auto l2_weight_threshold : config->l2_weight_thresholds_.value()) {
        if (l2_weight_threshold < 0) {
          throw std::invalid_argument("l2_weight_threshold must be positive");
        }
      }
      for (auto embedding_dim : config->embedding_dims_.value()) {
        if (embedding_dim <= 0) {
          throw std::invalid_argument("embedding_dim must be positive");
        }
      }

      return std::make_unique<L2WeightBasedEvict<weight_type>>(
          kv_store,
          sub_table_hash_cumsum,
          config->l2_weight_thresholds_.value(),
          config->embedding_dims_.value(),
          config->interval_for_insufficient_eviction_s_,
          config->interval_for_sufficient_eviction_s_,
          config->interval_for_feature_statistics_decay_s_,
          is_training,
          false);
    }

    case EvictTriggerStrategy::BY_TIMESTAMP_THRESHOLD: {
      return std::make_unique<TimeThresholdBasedEvict<weight_type>>(
          kv_store,
          sub_table_hash_cumsum,
          config->interval_for_insufficient_eviction_s_,
          config->interval_for_sufficient_eviction_s_,
          config->interval_for_feature_statistics_decay_s_,
          is_training,
          false);
    }

    default:
      throw std::runtime_error("Unknown evict trigger strategy");
  }
}
} // namespace kv_mem
