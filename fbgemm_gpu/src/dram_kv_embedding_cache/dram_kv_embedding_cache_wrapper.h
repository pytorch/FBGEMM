/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include "../ssd_split_embeddings_cache/kv_tensor_wrapper.h"
#include "dram_kv_embedding_cache.h"

namespace ssd {
struct EmbeddingSnapshotHandleWrapper;
}

namespace {
using DramKVEmbeddingCacheVariant = std::variant<
    std::shared_ptr<kv_mem::DramKVEmbeddingCache<float>>,
    std::shared_ptr<kv_mem::DramKVEmbeddingCache<at::Half>>>;
}

namespace kv_mem {

class DramKVEmbeddingCacheWrapper : public torch::jit::CustomClassHolder {
 public:
  DramKVEmbeddingCacheWrapper(
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t evict_trigger_mode = 0,
      int64_t trigger_step_interval = 0,
      int64_t mem_util_threshold = 0,
      int64_t evict_trigger_strategy = 1,
      const std::optional<at::Tensor>& count_thresholds = std::nullopt,
      const std::optional<at::Tensor>& ttls = std::nullopt,
      const std::optional<at::Tensor>& count_decay_rates = std::nullopt,
      const std::optional<at::Tensor>& l2_weight_thresholds = std::nullopt,
      const std::optional<at::Tensor>& embedding_dims = std::nullopt,
      int64_t num_shards = 8,
      int64_t num_threads = 32,
      int64_t row_storage_bitwidth = 32,
      const std::optional<at::Tensor>& table_dims = std::nullopt,
      const std::optional<at::Tensor>& hash_size_cumsum = std::nullopt,
      bool enable_async_update = false) {

    if (row_storage_bitwidth == 16) {
      impl_ = std::make_shared<kv_mem::DramKVEmbeddingCache<at::Half>>(
          max_D,
          uniform_init_lower,
          uniform_init_upper,
          evict_trigger_mode,
          trigger_step_interval,
          mem_util_threshold,
          evict_trigger_strategy,
          count_thresholds,
          ttls,
          count_decay_rates,
          l2_weight_thresholds,
          embedding_dims,
          num_shards,
          num_threads,
          row_storage_bitwidth,
          enable_async_update,
          table_dims,
          hash_size_cumsum);
    } else if (row_storage_bitwidth == 32) {
      impl_ = std::make_shared<kv_mem::DramKVEmbeddingCache<float>>(
          max_D,
          uniform_init_lower,
          uniform_init_upper,
          evict_trigger_mode,
          trigger_step_interval,
          mem_util_threshold,
          evict_trigger_strategy,
          count_thresholds,
          ttls,
          count_decay_rates,
          l2_weight_thresholds,
          embedding_dims,
          num_shards,
          num_threads,
          row_storage_bitwidth,
          enable_async_update,
          table_dims,
          hash_size_cumsum);
    } else {
      throw std::runtime_error("Failed to create recording device");
    }
  }

  void set_cuda(
      at::Tensor indices,
      at::Tensor weights,
      at::Tensor count,
      int64_t timestep,
      bool is_bwd) {
    return impl_->set_cuda(indices, weights, count, timestep);
  }

  void get_cuda(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    return impl_->get_cuda(indices, weights, count);
  }

  void set(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    impl_->set(indices, weights, count);
    // when use ITERATION or EvictTriggerMode,
    // trigger evict by trigger_step_interval or mem_util_threshold_GB
    impl_->maybe_evict();
  }

  void flush() {
    return impl_->flush();
  }

  void set_range_to_storage(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length) {
    return impl_->set_range_to_storage(weights, start, length);
  }

  at::Tensor get_keys_in_range_by_snapshot(
      int64_t start_id,
      int64_t end_id,
      int64_t id_offset,
      const std::optional<
          c10::intrusive_ptr<ssd::EmbeddingSnapshotHandleWrapper>>&
      /*snapshot_handle*/) {
    return impl_->get_keys_in_range_impl(start_id, end_id, id_offset);
  }

  void get(
      at::Tensor indices,
      at::Tensor weights,
      at::Tensor count,
      int64_t sleep_ms) {
    impl_->get(indices, weights, count, sleep_ms);
  }

  void wait_util_filling_work_done() {
    return impl_->wait_util_filling_work_done();
  }

  at::Tensor get_keys_in_range(int64_t start, int64_t end) {
    return impl_->get_keys_in_range_impl(start, end, std::nullopt);
  }

  size_t get_map_used_memsize() const {
    return impl_->get_map_used_memsize();
  }

  void get_feature_evict_metric(at::Tensor evicted_counts,
                                at::Tensor processed_counts,
                                at::Tensor duration) {
    FeatureEvictMetricTensors metrics = impl_->get_feature_evict_metric();
    evicted_counts = metrics.evicted_counts;      // evicted_counts (Long)
    processed_counts = metrics.processed_counts;  // processed_counts (Long)
    duration = metrics.duration;                  // duration (unit is ms, Long)
  }

 private:
  // friend class EmbeddingRocksDBWrapper;
  friend class ssd::KVTensorWrapper;

  std::shared_ptr<kv_db::EmbeddingKVDB> impl_;
};

} // namespace kv_mem
