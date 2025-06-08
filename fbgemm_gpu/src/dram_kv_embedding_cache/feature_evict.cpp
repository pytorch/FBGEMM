// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/feature_evict.h"

namespace kv_mem {

FeatureEvictMetrics::FeatureEvictMetrics(int table_num) {
  evicted_counts.resize(table_num, 0);
  processed_counts.resize(table_num, 0);
  exec_duration_ms = 0;
  full_duration_ms = 0;
}

void FeatureEvictMetrics::reset() {
  std::fill(evicted_counts.begin(), evicted_counts.end(), 0);
  std::fill(processed_counts.begin(), processed_counts.end(), 0);
  exec_duration_ms = 0;
  full_duration_ms = 0;
  start_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
}

void FeatureEvictMetrics::update_duration(int num_shards) {
  full_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count() -
      start_time_ms;
  // The exec_duration of all shards will be accumulated during the statistics
  // So finally, the number of shards needs to be divided
  exec_duration_ms /= num_shards;
}

template <typename weight_type>
FeatureEvict<weight_type>::FeatureEvict(
    folly::CPUThreadPoolExecutor* executor,
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

template <typename weight_type>
FeatureEvict<weight_type>::~FeatureEvict() {
  wait_completion(); // Wait for all asynchronous tasks to complete.
};

// Trigger asynchronous eviction.
// If there is an ongoing task, return directly to prevent multiple triggers.
// If there is no ongoing task, initialize the task state.
template <typename weight_type>
void FeatureEvict<weight_type>::trigger_evict() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (evict_flag_.exchange(true))
    return;
  prepare_evict();
}

// Get feature eviction metric.
template <typename weight_type>
FeatureEvictMetricTensors
FeatureEvict<weight_type>::get_feature_evict_metric() {
  std::lock_guard<std::mutex> lock(metric_mtx_);
  return metric_tensors_;
}

// Resume task execution. Returns true if there is an ongoing task, false
// otherwise.
template <typename weight_type>
bool FeatureEvict<weight_type>::resume() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!evict_flag_.load())
    return false;
  evict_interrupt_.store(false);
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    submit_shard_task(shard_id);
  }
  return true;
};

// Pause the eviction process. Returns true if there is an ongoing task, false
// otherwise. During the pause phase, check whether the eviction is complete.
template <typename weight_type>
bool FeatureEvict<weight_type>::pause() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!evict_flag_.load())
    return false;
  evict_interrupt_.store(true);
  check_and_reset_evict_flag();
  wait_completion();
  return true;
}

// Check whether eviction is ongoing.
template <typename weight_type>
bool FeatureEvict<weight_type>::is_evicting() {
  std::lock_guard<std::mutex> lock(mutex_);
  check_and_reset_evict_flag();
  return evict_flag_.load();
}

template <typename weight_type>
void FeatureEvict<weight_type>::init_shard_status() {
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
template <typename weight_type>
void FeatureEvict<weight_type>::prepare_evict() {
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

template <typename weight_type>
void FeatureEvict<weight_type>::submit_shard_task(int shard_id) {
  if (shards_finished_[shard_id]->load())
    return;
  futures_.emplace_back(folly::via(executor_).thenValue(
      [this, shard_id](auto&&) { process_shard(shard_id); }));
}

template <typename weight_type>
void FeatureEvict<weight_type>::process_shard(int shard_id) {
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

template <typename weight_type>
void FeatureEvict<weight_type>::wait_completion() {
  folly::collectAll(futures_).wait();
  futures_.clear();
}

// Check and reset the eviction flag.
template <typename weight_type>
void FeatureEvict<weight_type>::check_and_reset_evict_flag() {
  bool all_finished = true;
  for (int i = 0; i < num_shards_; ++i) {
    if (!shards_finished_[i]->load())
      all_finished = false;
  }
  if (all_finished && evict_flag_.exchange(false)) {
    record_metrics_to_report_tensor();
  }
}

template <typename weight_type>
[[nodiscard]] int FeatureEvict<weight_type>::get_sub_table_id(
    int64_t key) const {
  auto it = std::upper_bound(
      sub_table_hash_cumsum_.begin(), sub_table_hash_cumsum_.end(), key);
  if (it == sub_table_hash_cumsum_.end()) {
    CHECK(false) << "key " << key << " doesn't belong to any feature";
  }

  return std::distance(sub_table_hash_cumsum_.begin(), it);
}

template <typename weight_type>
void FeatureEvict<weight_type>::record_metrics_to_report_tensor() {
  std::lock_guard<std::mutex> lock(metric_mtx_);
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

  metric_tensors_.full_duration_ms =
      at::scalar_tensor(metrics_.full_duration_ms, at::kLong);
  metric_tensors_.exec_duration_ms =
      at::scalar_tensor(metrics_.exec_duration_ms, at::kLong);
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

template class FeatureEvict<uint8_t>;
template class FeatureEvict<float>;
template class FeatureEvict<at::Half>;

template <typename weight_type>
CounterBasedEvict<weight_type>::CounterBasedEvict(
    folly::CPUThreadPoolExecutor* executor,
    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
    const std::vector<int64_t>& sub_table_hash_cumsum,
    const std::vector<float>& decay_rates,
    const std::vector<uint32_t>& thresholds)
    : FeatureEvict<weight_type>(executor, kv_store, sub_table_hash_cumsum),
      decay_rates_(decay_rates),
      thresholds_(thresholds) {}

template <typename weight_type>
void CounterBasedEvict<weight_type>::update_feature_statistics(
    weight_type* block) {
  FixedBlockPool::update_count(block);
}

template <typename weight_type>
bool CounterBasedEvict<weight_type>::evict_block(
    weight_type* block,
    int sub_table_id) {
  float decay_rate = decay_rates_[sub_table_id];
  uint32_t threshold = thresholds_[sub_table_id];
  // Apply decay and check the threshold.
  auto current_count = FixedBlockPool::get_count(block);
  current_count *= decay_rate;
  FixedBlockPool::set_count(block, current_count);
  return current_count < threshold;
}

template class CounterBasedEvict<uint8_t>;
template class CounterBasedEvict<float>;
template class CounterBasedEvict<at::Half>;

template <typename weight_type>
TimeBasedEvict<weight_type>::TimeBasedEvict(
    folly::CPUThreadPoolExecutor* executor,
    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
    const std::vector<int64_t>& sub_table_hash_cumsum,
    const std::vector<uint32_t>& ttls_in_mins)
    : FeatureEvict<weight_type>(executor, kv_store, sub_table_hash_cumsum),
      ttls_in_mins_(ttls_in_mins) {}

template <typename weight_type>
void TimeBasedEvict<weight_type>::update_feature_statistics(
    weight_type* block) {
  FixedBlockPool::update_timestamp(block);
}

template <typename weight_type>
bool TimeBasedEvict<weight_type>::evict_block(
    weight_type* block,
    int sub_table_id) {
  uint32_t ttl = ttls_in_mins_[sub_table_id];
  auto current_time = FixedBlockPool::current_timestamp();
  return current_time - FixedBlockPool::get_timestamp(block) > ttl * 60;
}

template class TimeBasedEvict<uint8_t>;
template class TimeBasedEvict<float>;
template class TimeBasedEvict<at::Half>;

template <typename weight_type>
TimeCounterBasedEvict<weight_type>::TimeCounterBasedEvict(
    folly::CPUThreadPoolExecutor* executor,
    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
    const std::vector<int64_t>& sub_table_hash_cumsum,
    const std::vector<uint32_t>& ttls_in_mins,
    const std::vector<float>& decay_rates,
    const std::vector<uint32_t>& thresholds)
    : FeatureEvict<weight_type>(executor, kv_store, sub_table_hash_cumsum),
      ttls_in_mins_(ttls_in_mins),
      decay_rates_(decay_rates),
      thresholds_(thresholds) {}

template <typename weight_type>
void TimeCounterBasedEvict<weight_type>::update_feature_statistics(
    weight_type* block) {
  FixedBlockPool::update_timestamp(block);
  FixedBlockPool::update_count(block);
}

template <typename weight_type>
bool TimeCounterBasedEvict<weight_type>::evict_block(
    weight_type* block,
    int sub_table_id) {
  uint32_t ttl = ttls_in_mins_[sub_table_id];
  float decay_rate = decay_rates_[sub_table_id];
  uint32_t threshold = thresholds_[sub_table_id];
  // Apply decay and check the count threshold and ttl.
  auto current_time = FixedBlockPool::current_timestamp();
  auto current_count = FixedBlockPool::get_count(block);
  current_count *= decay_rate;
  FixedBlockPool::set_count(block, current_count);
  return (current_time - FixedBlockPool::get_timestamp(block) > ttl * 60) &&
      (current_count < threshold);
}

template class TimeCounterBasedEvict<uint8_t>;
template class TimeCounterBasedEvict<float>;
template class TimeCounterBasedEvict<at::Half>;

template <typename weight_type>
L2WeightBasedEvict<weight_type>::L2WeightBasedEvict(
    folly::CPUThreadPoolExecutor* executor,
    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
    const std::vector<int64_t>& sub_table_hash_cumsum,
    const std::vector<double>& thresholds,
    const std::vector<int64_t>& sub_table_dims)
    : FeatureEvict<weight_type>(executor, kv_store, sub_table_hash_cumsum),
      thresholds_(thresholds),
      sub_table_dims_(sub_table_dims) {}

template <typename weight_type>
void L2WeightBasedEvict<weight_type>::update_feature_statistics(
    [[maybe_unused]] weight_type* block) {}

template <typename weight_type>
bool L2WeightBasedEvict<weight_type>::evict_block(
    weight_type* block,
    int sub_table_id) {
  size_t dimension = sub_table_dims_[sub_table_id];
  double threshold = thresholds_[sub_table_id];
  auto l2weight = FixedBlockPool::get_l2weight(block, dimension);
  return l2weight < threshold;
}

template class L2WeightBasedEvict<uint8_t>;
template class L2WeightBasedEvict<float>;
template class L2WeightBasedEvict<at::Half>;

} // namespace kv_mem
