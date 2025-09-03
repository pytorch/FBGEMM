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
      std::optional<c10::intrusive_ptr<kv_mem::FeatureEvictConfig>>
          feature_evict_config,
      int64_t num_shards = 8,
      int64_t num_threads = 32,
      int64_t row_storage_bitwidth = 32,
      const std::optional<at::Tensor>& table_dims = std::nullopt,
      const std::optional<at::Tensor>& hash_size_cumsum = std::nullopt,
      bool backend_return_whole_row = false,
      bool enable_async_update = false,
      bool disable_random_init = false) {
    if (row_storage_bitwidth == 16) {
      impl_ = std::make_shared<kv_mem::DramKVEmbeddingCache<at::Half>>(
          max_D,
          uniform_init_lower,
          uniform_init_upper,
          feature_evict_config,
          num_shards,
          num_threads,
          row_storage_bitwidth,
          backend_return_whole_row,
          enable_async_update,
          table_dims,
          hash_size_cumsum,
          true, // is_training
          disable_random_init);
    } else if (row_storage_bitwidth == 32) {
      impl_ = std::make_shared<kv_mem::DramKVEmbeddingCache<float>>(
          max_D,
          uniform_init_lower,
          uniform_init_upper,
          feature_evict_config,
          num_shards,
          num_threads,
          row_storage_bitwidth,
          backend_return_whole_row,
          enable_async_update,
          table_dims,
          hash_size_cumsum,
          true, // is_training
          disable_random_init);
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
    return impl_->set_cuda(indices, weights, count, timestep, is_bwd);
  }

  void get_cuda(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    impl_->get_cuda(indices, weights, count);
  }

  void set(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    impl_->set(indices, weights, count);
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

  at::Tensor get_kv_zch_eviction_metadata_by_snapshot(
      const at::Tensor& indices,
      const at::Tensor& count,
      const std::optional<
          c10::intrusive_ptr<ssd::EmbeddingSnapshotHandleWrapper>>&
      /*snapshot_handle*/) {
    return impl_->get_kv_zch_eviction_metadata_impl(indices, count);
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

  size_t get_map_used_memsize_in_bytes() const {
    return impl_->get_map_used_memsize_in_bytes();
  }

  std::vector<double> get_dram_kv_perf(
      const int64_t step,
      const int64_t interval) {
    return impl_->get_dram_kv_perf(step, interval);
  }

  void get_feature_evict_metric(
      at::Tensor evicted_counts,
      at::Tensor processed_counts,
      at::Tensor eviction_threshold_with_dry_run,
      at::Tensor full_duration_ms,
      at::Tensor exec_duration_ms) {
    auto metrics = impl_->get_feature_evict_metric();
    if (metrics.has_value()) {
      evicted_counts.copy_(
          metrics.value().evicted_counts); // evicted_counts (Long)
      processed_counts.copy_(
          metrics.value().processed_counts); // processed_counts (Long)
      eviction_threshold_with_dry_run.copy_(
          metrics.value()
              .eviction_threshold_with_dry_run); // eviction threshold with dry
                                                 // run (float)
      full_duration_ms.copy_(
          metrics.value().full_duration_ms); // full duration (Long)
      exec_duration_ms.copy_(
          metrics.value().exec_duration_ms); // exec duration (Long)
    }
  }

  void wait_until_eviction_done() {
    impl_->wait_until_eviction_done();
  }

  void set_backend_return_whole_row(bool backend_return_whole_row) {
    impl_->set_backend_return_whole_row(backend_return_whole_row);
  }

  void set_feature_score_metadata_cuda(
      at::Tensor indices,
      at::Tensor count,
      at::Tensor engage_show_count) {
    impl_->set_feature_score_metadata_cuda(indices, count, engage_show_count);
  }

 private:
  // friend class EmbeddingRocksDBWrapper;
  friend class ssd::KVTensorWrapper;

  std::shared_ptr<kv_db::EmbeddingKVDB> impl_;
};

} // namespace kv_mem
