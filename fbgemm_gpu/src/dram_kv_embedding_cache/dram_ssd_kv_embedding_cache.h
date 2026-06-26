/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/futures/Future.h>

#include "dram_kv_embedding_cache.h"

namespace kv_db {

/// @ingroup KVMemEmbedding
///
/// @brief Composite backend that wraps the DRAM (L2) embedding cache.
///
/// This is the forwarding skeleton: every EmbeddingKVDB method delegates to
/// the wrapped DRAM cache. The SSD (L3) tier and its orchestration (two-phase
/// lookup, async backfill, eviction writeback, flush-to-SSD, and checkpoint
/// reads) are layered on top in a follow-up.
///
template <typename weight_type>
class DramSsdKVEmbeddingCache : public EmbeddingKVDB {
 public:
  /// Construct the composite backend over the DRAM tier (L2).
  ///
  /// @param dram_cache The DRAM tier (L2)
  explicit DramSsdKVEmbeddingCache(
      std::shared_ptr<kv_mem::DramKVEmbeddingCache<weight_type>> dram_cache)
      : EmbeddingKVDB(
            require_dram_cache(dram_cache)->get_num_shards(),
            require_dram_cache(dram_cache)->get_max_D(),
            /*cache_size_gb=*/0,
            /*unique_id=*/0,
            /*ele_size_bytes=*/sizeof(weight_type)),
        dram_cache_(std::move(dram_cache)) {}

  ~DramSsdKVEmbeddingCache() override = default;

  // --- Core EmbeddingKVDB interface (forwarded to DRAM) ---

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    return dram_cache_->get_kv_db_async(indices, weights, count);
  }

  void set_range_to_storage(
      const at::Tensor& weights_with_metaheader,
      const int64_t start,
      const int64_t length) override {
    dram_cache_->set_range_to_storage(weights_with_metaheader, start, length);
  }

  folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const RocksdbWriteMode w_mode =
          RocksdbWriteMode::FWD_ROCKSDB_READ) override {
    return dram_cache_->set_kv_db_async(indices, weights, count, w_mode);
  }

  /// Delegate enrichment metadata to DRAM cache.
  folly::SemiFuture<std::vector<folly::Unit>>
  set_kv_zch_eviction_metadata_async(
      at::Tensor indices,
      at::Tensor count,
      at::Tensor engage_show_count) override {
    return dram_cache_->set_kv_zch_eviction_metadata_async(
        std::move(indices), std::move(count), std::move(engage_show_count));
  }

  /// Delegate enrichment query ID processing to DRAM cache.
  void set_embedding_cache_enrich_query_id_async(
      at::Tensor hashed_indices,
      at::Tensor unhashed_indices,
      at::Tensor count) override {
    dram_cache_->set_embedding_cache_enrich_query_id_async(
        std::move(hashed_indices),
        std::move(unhashed_indices),
        std::move(count));
  }

  void compact() override {
    dram_cache_->compact();
  }

  /// Delegate sync SID fetch to the DRAM cache. Without this override the
  /// composite would fall back to the base-class no-op and silently return
  /// empty tensors.
  std::tuple<at::Tensor, at::Tensor> fetch_sids_sync(
      at::Tensor hashed_indices,
      at::Tensor unhashed_indices,
      at::Tensor count) override {
    return dram_cache_->fetch_sids_sync(
        std::move(hashed_indices),
        std::move(unhashed_indices),
        std::move(count));
  }

  // --- Eviction delegation ---

  void maybe_evict() override {
    dram_cache_->maybe_evict();
  }

  void trigger_feature_evict() override {
    dram_cache_->trigger_feature_evict();
  }

  bool is_evicting() override {
    return dram_cache_->is_evicting();
  }

  void set_backend_return_whole_row(bool backend_return_whole_row) override {
    dram_cache_->set_backend_return_whole_row(backend_return_whole_row);
  }

  size_t get_map_used_memsize_in_bytes() const override {
    return dram_cache_->get_map_used_memsize_in_bytes();
  }

  void pause_ongoing_eviction(bool force_resume = false) override {
    dram_cache_->pause_ongoing_eviction(force_resume);
  }

  void resume_ongoing_eviction(bool force_pause = false) override {
    dram_cache_->resume_ongoing_eviction(force_pause);
  }

  void wait_until_eviction_done() override {
    dram_cache_->wait_until_eviction_done();
  }

  std::optional<kv_mem::FeatureEvictMetricTensors> get_feature_evict_metric()
      const override {
    return dram_cache_->get_feature_evict_metric();
  }

  std::vector<double> get_dram_kv_perf(
      const int64_t step,
      const int64_t interval) override {
    // Call through base class pointer to access the private override via
    // virtual dispatch.
    EmbeddingKVDB* base = dram_cache_.get();
    return base->get_dram_kv_perf(step, interval);
  }

  /// Delegate to the DRAM cache via base-class pointer to dispatch through the
  /// virtual interface (the override is private in DramKVEmbeddingCache).
  bool get_backend_return_whole_row() override {
    EmbeddingKVDB* base = dram_cache_.get();
    return base->get_backend_return_whole_row();
  }

  int64_t get_metaheader_width_in_front() override {
    EmbeddingKVDB* base = dram_cache_.get();
    return base->get_metaheader_width_in_front();
  }

  void get_range_from_snapshot(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length,
      const ssd::SnapshotHandle* snapshot_handle,
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override {
    dram_cache_->get_range_from_snapshot(
        weights, start, length, snapshot_handle, width_offset, width_length);
  }

  void get_kv_from_storage_by_snapshot(
      const at::Tensor& ids,
      const at::Tensor& weights,
      const ssd::SnapshotHandle* snapshot_handle,
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override {
    dram_cache_->get_kv_from_storage_by_snapshot(
        ids, weights, snapshot_handle, width_offset, width_length);
  }

  // --- Key range and metadata ---

  at::Tensor get_keys_in_range_impl(
      int64_t start,
      int64_t end,
      std::optional<int64_t> offset = std::nullopt) override {
    return dram_cache_->get_keys_in_range_impl(start, end, offset);
  }

  at::Tensor get_kv_zch_eviction_metadata_impl(
      const at::Tensor& indices,
      const at::Tensor& count) override {
    return dram_cache_->get_kv_zch_eviction_metadata_impl(indices, count);
  }

  // --- Accessors ---

  const std::shared_ptr<kv_mem::DramKVEmbeddingCache<weight_type>>& dram_cache()
      const {
    return dram_cache_;
  }

 private:
  static const std::shared_ptr<kv_mem::DramKVEmbeddingCache<weight_type>>&
  require_dram_cache(
      const std::shared_ptr<kv_mem::DramKVEmbeddingCache<weight_type>>&
          dram_cache) {
    TORCH_CHECK(dram_cache != nullptr, "dram_cache must not be null");
    return dram_cache;
  }

  void flush_or_compact(const int64_t /*timestep*/) override {
    // No-op; flush is explicit and added with the SSD tier.
  }

  std::shared_ptr<kv_mem::DramKVEmbeddingCache<weight_type>> dram_cache_;
}; // class DramSsdKVEmbeddingCache

} // namespace kv_db
