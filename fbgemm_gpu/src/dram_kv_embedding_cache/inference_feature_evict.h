/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "InferenceSynchronizedShardedMap.h"
#include "feature_evict.h"
#include "inference_fixed_block_pool.h"

namespace kv_mem {

/// @brief TimeThresholdBasedEvict specialized for inference with 12-byte header
///
/// Extends TimeThresholdBasedEvict but overrides methods to work with
/// InferenceFixedBlockPool's 12-byte header layout.
template <typename weight_type>
class InferenceTimeThresholdBasedEvict
    : public TimeThresholdBasedEvict<weight_type> {
 public:
  using InferenceKVStore = InferenceSynchronizedShardedMap<
      int64_t,
      weight_type*,
      folly::SharedMutexWritePriority>;

  InferenceTimeThresholdBasedEvict(
      InferenceKVStore& kv_store,
      const std::vector<int64_t>& sub_table_hash_cumsum,
      int64_t interval_for_insufficient_eviction_s,
      int64_t interval_for_sufficient_eviction_s,
      int64_t interval_for_feature_statistics_decay_s)
      : TimeThresholdBasedEvict<weight_type>(
            reinterpret_cast<SynchronizedShardedMap<int64_t, weight_type*>&>(
                kv_store),
            sub_table_hash_cumsum,
            interval_for_insufficient_eviction_s,
            interval_for_sufficient_eviction_s,
            interval_for_feature_statistics_decay_s,
            /*is_training=*/false),
        inference_kv_store_(&kv_store) {}

  // Override to use 12-byte header's timestamp field (not 16-byte layout)
  // Using FixedBlockPool::update_timestamp would overwrite the used bit!
  void update_feature_statistics(weight_type* block) override {
    InferenceFixedBlockPool::update_timestamp(block);
  }

 protected:
  // Override to use 12-byte header's timestamp field
  bool evict_block(weight_type* block, int sub_table_id, int shard_id)
      override {
    return InferenceFixedBlockPool::get_timestamp(block) <
        this->eviction_timestamp_threshold_;
  }

  // Override to use InferenceFixedBlockPool's get_block with 12-byte header
  void start_inference_eviction_loop(
      int shard_id,
      std::vector<int64_t>& evicted_counts,
      std::vector<int64_t>& processed_counts) override {
    auto* pool = inference_kv_store_->pool_by(shard_id);

    // Erase keys from the map before freeing blocks; freeing first would leave
    // the map pointing at freed memory a reader (holding rlock) could write to.
    std::vector<std::pair<int64_t, weight_type*>> evicting_blocks;
    {
      // 1st step: pick victims under mempool lock, don't free yet.
      auto mem_pool_lock = pool->acquire_lock();
      evicting_blocks.reserve(this->block_nums_snapshot_[shard_id] / 100);
      while (!this->should_exit_evict_loop(shard_id)) {
        auto* block = pool->template get_block<weight_type>(
            this->block_cursors_[shard_id]++);
        if (block == nullptr) {
          continue;
        }
        int64_t key = InferenceFixedBlockPool::get_key(block);
        int sub_table_id = this->get_sub_table_id(key);
        processed_counts[sub_table_id]++;
        if (this->evict_block(block, sub_table_id, shard_id)) {
          evicting_blocks.emplace_back(key, block);
          evicted_counts[sub_table_id]++;
        }
      }
    }

    // 2nd step: drop keys from the map under wlock (excludes readers).
    {
      auto shard_map_wlock = this->kv_store_.by(shard_id).wlock();
      for (int64_t key : evicting_blocks | std::views::keys) {
        shard_map_wlock->erase(key);
      }
    }

    // 3rd step: keys unreachable and readers drained, now free the blocks.
    auto mem_pool_lock = pool->acquire_lock();
    for (auto* block : evicting_blocks | std::views::values) {
      pool->template deallocate_t<weight_type>(block);
    }
  }

 private:
  InferenceKVStore* inference_kv_store_;
};

/// @brief Factory function for creating FeatureEvict with
/// InferenceSynchronizedShardedMap
///
/// Only supports BY_TIMESTAMP_THRESHOLD strategy for inference.
/// Creates InferenceTimeThresholdBasedEvict to work with 12-byte headers.
template <typename weight_type>
std::unique_ptr<FeatureEvict<weight_type>> create_inference_feature_evict(
    c10::intrusive_ptr<FeatureEvictConfig> config,
    InferenceSynchronizedShardedMap<
        int64_t,
        weight_type*,
        folly::SharedMutexWritePriority>& kv_store,
    const std::vector<int64_t>& sub_table_hash_cumsum,
    TestMode test_mode = TestMode::DISABLED) {
  if (config->trigger_strategy_ !=
      EvictTriggerStrategy::BY_TIMESTAMP_THRESHOLD) {
    throw std::runtime_error(
        "Only BY_TIMESTAMP_THRESHOLD is supported for inference with InferenceSynchronizedShardedMap");
  }
  return std::make_unique<InferenceTimeThresholdBasedEvict<weight_type>>(
      kv_store,
      sub_table_hash_cumsum,
      config->interval_for_insufficient_eviction_s_,
      config->interval_for_sufficient_eviction_s_,
      config->interval_for_feature_statistics_decay_s_);
}

} // namespace kv_mem
