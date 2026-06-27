/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <fmt/format.h>
#include <functional>
#include "../ssd_split_embeddings_cache/kv_tensor_wrapper.h"
#include "../ssd_split_embeddings_cache/ssd_table_batched_embeddings.h"
#include "dram_kv_embedding_cache.h"
#include "dram_ssd_kv_embedding_cache.h"

namespace ssd {
struct EmbeddingSnapshotHandleWrapper;
}

namespace kv_mem {

/// @brief TorchScript wrapper for the DramSsdKVEmbeddingCache composite
/// backend.
///
/// This wrapper is the sole initializer of the DRAM+SSD backend: it builds the
/// DRAM (L2) cache, constructs the SSD (L3) RocksDB tier internally, and
/// assembles the composite DramSsdKVEmbeddingCache from both. The wrapper layer
/// keeps a single DRAM-backed composite (impl_) and a single RocksDB storage
/// (rocksdb_impl_); the weight-type (Half/float) selection is confined to the
/// templated init_composite() helper. Because RocksDB is always created in the
/// constructor, rocksdb_impl_ is guaranteed non-null for all
/// checkpoint/snapshot code paths.
class DramSsdKVEmbeddingCacheWrapper : public torch::jit::CustomClassHolder {
 public:
  DramSsdKVEmbeddingCacheWrapper(
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
      bool disable_random_init = false,
      bool enable_raw_embedding_streaming = false,
      int64_t res_store_shards = 0,
      int64_t res_server_port = 0,
      std::vector<std::string> table_names = {},
      std::vector<int64_t> table_offsets = {},
      std::vector<int64_t> table_sizes = {},
      int64_t writeback_queue_size = 1024,
      int64_t writeback_batch_size = 1024,
      std::optional<c10::intrusive_ptr<kv_mem::EnrichmentConfig>>
          enrichment_config = std::nullopt,
      // SSD (RocksDB L3 tier) construction params; mirror EmbeddingRocksDB.
      std::string ssd_path = "",
      int64_t memtable_flush_period = 0,
      int64_t memtable_flush_offset = 0,
      int64_t l0_files_per_compact = 0,
      int64_t rate_limit_mbps = 0,
      int64_t size_ratio = 10,
      int64_t compaction_ratio = 0,
      int64_t write_buffer_size = 0,
      int64_t max_write_buffer_num = 0,
      int64_t block_cache_size = 0,
      bool use_passed_in_path = true,
      int64_t tbe_unique_id = 0,
      int64_t l2_cache_size_gb = 0,
      bool ssd_enable_async_update = false,
      int64_t flushing_block_size = 2000000000 /*2GB*/,
      bool enable_blob_db = false) {
    row_storage_bitwidth_ = row_storage_bitwidth;
    writeback_queue_size_ = writeback_queue_size;
    writeback_batch_size_ = writeback_batch_size;

    if (row_storage_bitwidth == 16) {
      init_composite<at::Half>(
          max_D,
          uniform_init_lower,
          uniform_init_upper,
          feature_evict_config,
          num_shards,
          num_threads,
          row_storage_bitwidth,
          table_dims,
          hash_size_cumsum,
          backend_return_whole_row,
          enable_async_update,
          disable_random_init,
          enable_raw_embedding_streaming,
          res_store_shards,
          res_server_port,
          table_names,
          table_offsets,
          table_sizes,
          enrichment_config,
          ssd_path,
          memtable_flush_period,
          memtable_flush_offset,
          l0_files_per_compact,
          rate_limit_mbps,
          size_ratio,
          compaction_ratio,
          write_buffer_size,
          max_write_buffer_num,
          block_cache_size,
          use_passed_in_path,
          tbe_unique_id,
          l2_cache_size_gb,
          ssd_enable_async_update,
          flushing_block_size,
          enable_blob_db);
    } else if (row_storage_bitwidth == 32) {
      init_composite<float>(
          max_D,
          uniform_init_lower,
          uniform_init_upper,
          feature_evict_config,
          num_shards,
          num_threads,
          row_storage_bitwidth,
          table_dims,
          hash_size_cumsum,
          backend_return_whole_row,
          enable_async_update,
          disable_random_init,
          enable_raw_embedding_streaming,
          res_store_shards,
          res_server_port,
          table_names,
          table_offsets,
          table_sizes,
          enrichment_config,
          ssd_path,
          memtable_flush_period,
          memtable_flush_offset,
          l0_files_per_compact,
          rate_limit_mbps,
          size_ratio,
          compaction_ratio,
          write_buffer_size,
          max_write_buffer_num,
          block_cache_size,
          use_passed_in_path,
          tbe_unique_id,
          l2_cache_size_gb,
          ssd_enable_async_update,
          flushing_block_size,
          enable_blob_db);
    } else {
      throw std::runtime_error(
          fmt::format(
              "Unsupported row_storage_bitwidth={}; expected 16 or 32",
              row_storage_bitwidth));
    }
  }

  // --- Core methods (forwarded to composite backend) ---

  void set_cuda(
      at::Tensor indices,
      at::Tensor weights,
      at::Tensor count,
      int64_t timestep,
      bool is_bwd) {
    impl_->set_cuda(indices, weights, count, timestep, is_bwd);
  }

  void get_cuda(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    impl_->get_cuda(indices, weights, count);
  }

  void set(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    impl_->set(indices, weights, count);
  }

  void flush() {
    // EmbeddingKVDB::flush() is NOT virtual; calling impl_->flush() would
    // dispatch to the base version (which only flushes l2_cache_). Invoke the
    // typed composite's flush() (captured at construction) to flush dirty DRAM
    // blocks to SSD and drain the writeback queue.
    if (flush_composite_) {
      flush_composite_();
    } else {
      impl_->flush();
    }
  }

  void set_range_to_storage(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length) {
    impl_->set_range_to_storage(weights, start, length);
    XLOG(INFO) << "DramSsdKVEmbeddingCacheWrapper::set_range_to_storage()"
               << " start=" << start << " length=" << length
               << " impl_type=" << typeid(*impl_).name();
  }

  at::Tensor get_keys_in_range_by_snapshot(
      int64_t start_id,
      int64_t end_id,
      int64_t id_offset,
      const std::optional<
          c10::intrusive_ptr<ssd::EmbeddingSnapshotHandleWrapper>>&
          snapshot_handle) {
    if (rocksdb_impl_) {
      // DRAM_SSD: read keys directly from SSD (which has all data after flush)
      const ssd::SnapshotHandle* snap = nullptr;
      if (snapshot_handle.has_value() && snapshot_handle.value()) {
        snap = snapshot_handle.value()->handle;
      }
      return rocksdb_impl_->get_keys_in_range_by_snapshot(
          start_id, end_id, id_offset, snap);
    }
    return impl_->get_keys_in_range_impl(start_id, end_id, id_offset);
  }

  at::Tensor get_kv_zch_eviction_metadata_by_snapshot(
      const at::Tensor& indices,
      const at::Tensor& count,
      const std::optional<
          c10::intrusive_ptr<ssd::EmbeddingSnapshotHandleWrapper>>&
          snapshot_handle) {
    if (rocksdb_impl_) {
      const ssd::SnapshotHandle* snap = nullptr;
      if (snapshot_handle.has_value() && snapshot_handle.value()) {
        snap = snapshot_handle.value()->handle;
      }
      return rocksdb_impl_->get_kv_zch_eviction_metadata_by_snapshot(
          indices, count, snap);
    }
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
    impl_->wait_util_filling_work_done();
  }

  at::Tensor get_keys_in_range(int64_t start, int64_t end) {
    CHECK(rocksdb_impl_) << "SSD backend not initialized";
    // DRAM_SSD: read keys from SSD (all data after flush)
    return rocksdb_impl_->get_keys_in_range_by_snapshot(
        start, end, /*id_offset=*/0, /*snapshot_handle=*/nullptr);
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
      evicted_counts.copy_(metrics.value().evicted_counts);
      processed_counts.copy_(metrics.value().processed_counts);
      eviction_threshold_with_dry_run.copy_(
          metrics.value().eviction_threshold_with_dry_run);
      full_duration_ms.copy_(metrics.value().full_duration_ms);
      exec_duration_ms.copy_(metrics.value().exec_duration_ms);
    }
  }

  void wait_until_eviction_done() {
    impl_->wait_until_eviction_done();
  }

  void set_backend_return_whole_row(bool backend_return_whole_row) {
    impl_->set_backend_return_whole_row(backend_return_whole_row);
  }

  void trigger_feature_evict() {
    impl_->trigger_feature_evict();
  }

  bool is_evicting() {
    return impl_->is_evicting();
  }

  void set_feature_score_metadata_cuda(
      at::Tensor indices,
      at::Tensor count,
      at::Tensor engage_show_count) {
    impl_->set_feature_score_metadata_cuda(indices, count, engage_show_count);
  }

  void set_embedding_cache_enrich_query_id_cuda(
      at::Tensor hashed_indices,
      at::Tensor unhashed_indices,
      at::Tensor count) {
    impl_->set_embedding_cache_enrich_query_id_cuda(
        hashed_indices, unhashed_indices, count);
  }

  std::tuple<at::Tensor, at::Tensor> fetch_sids_sync(
      at::Tensor hashed_indices,
      at::Tensor unhashed_indices,
      at::Tensor count) {
    return impl_->fetch_sids_sync(
        std::move(hashed_indices),
        std::move(unhashed_indices),
        std::move(count));
  }

  /// Create a RocksDB snapshot for consistent checkpoint reads.
  /// Must be called AFTER flush() so SSD has all data.
  c10::intrusive_ptr<ssd::EmbeddingSnapshotHandleWrapper> create_snapshot() {
    CHECK(rocksdb_impl_) << "SSD backend not initialized";
    auto handle = rocksdb_impl_->create_snapshot();
    return c10::make_intrusive<ssd::EmbeddingSnapshotHandleWrapper>(
        handle, rocksdb_impl_);
  }

  /// Get the number of active RocksDB snapshots.
  int64_t get_snapshot_count() {
    CHECK(rocksdb_impl_) << "SSD backend not initialized";
    return rocksdb_impl_->get_snapshot_count();
  }

  /// Create a RocksDB hard-link checkpoint for cross-process access.
  /// Delegates to the underlying RocksDB backend.
  void create_rocksdb_hard_link_snapshot(int64_t global_step) {
    CHECK(rocksdb_impl_) << "SSD backend not initialized";
    rocksdb_impl_->create_checkpoint(global_step);
  }

  /// Get the active checkpoint UUID for cross-process checkpoint reading.
  /// Returns nullopt if no checkpoint is available for the given step.
  std::optional<c10::intrusive_ptr<ssd::RocksdbCheckpointHandleWrapper>>
  get_active_checkpoint_uuid(int64_t global_step) {
    CHECK(rocksdb_impl_) << "SSD backend not initialized";
    auto uuid_opt = rocksdb_impl_->get_active_checkpoint_uuid(global_step);
    if (uuid_opt.has_value()) {
      return c10::make_intrusive<ssd::RocksdbCheckpointHandleWrapper>(
          uuid_opt.value(), rocksdb_impl_);
    } else {
      return std::nullopt;
    }
  }

  void delete_rocksdb_checkpoint_dir() {
    CHECK(rocksdb_impl_) << "SSD backend not initialized";
    rocksdb_impl_->delete_rocksdb_checkpoint_dir();
  }

 private:
  friend class ssd::KVTensorWrapper;

  /// Build the DRAM (L2) cache, the SSD (L3) RocksDB tier, and the composite
  /// DramSsdKVEmbeddingCache that orchestrates them. The weight-type template
  /// parameter is the only place Half/float is differentiated; all persistent
  /// state is stored as a single composite (impl_) plus a single RocksDB tier
  /// (rocksdb_impl_).
  template <typename weight_type>
  void init_composite(
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      const std::optional<c10::intrusive_ptr<kv_mem::FeatureEvictConfig>>&
          feature_evict_config,
      int64_t num_shards,
      int64_t num_threads,
      int64_t row_storage_bitwidth,
      const std::optional<at::Tensor>& table_dims,
      const std::optional<at::Tensor>& hash_size_cumsum,
      bool backend_return_whole_row,
      bool enable_async_update,
      bool disable_random_init,
      bool enable_raw_embedding_streaming,
      int64_t res_store_shards,
      int64_t res_server_port,
      std::vector<std::string> table_names,
      std::vector<int64_t> table_offsets,
      std::vector<int64_t> table_sizes,
      const std::optional<c10::intrusive_ptr<kv_mem::EnrichmentConfig>>&
          enrichment_config,
      std::string ssd_path,
      int64_t memtable_flush_period,
      int64_t memtable_flush_offset,
      int64_t l0_files_per_compact,
      int64_t rate_limit_mbps,
      int64_t size_ratio,
      int64_t compaction_ratio,
      int64_t write_buffer_size,
      int64_t max_write_buffer_num,
      int64_t block_cache_size,
      bool use_passed_in_path,
      int64_t tbe_unique_id,
      int64_t l2_cache_size_gb,
      bool ssd_enable_async_update,
      int64_t flushing_block_size,
      bool enable_blob_db) {
    // DRAM (L2) cache. enable_ssd_backend=true turns on dirty-bit tracking so
    // the composite can write back dirty DRAM blocks to SSD.
    auto dram_cache =
        std::make_shared<kv_mem::DramKVEmbeddingCache<weight_type>>(
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
            /*is_training=*/true,
            disable_random_init,
            enable_raw_embedding_streaming,
            res_store_shards,
            res_server_port,
            table_names,
            table_offsets,
            table_sizes,
            enrichment_config,
            /*enable_ssd_backend=*/true);

    // The SSD tier stores the full DRAM block including the MetaHeader, so the
    // RocksDB max_D / table_dims must be widened by the MetaHeader dimension.
    const int64_t metaheader_dim = static_cast<int64_t>(
        kv_mem::FixedBlockPool::get_metaheader_dim<weight_type>());
    const int64_t ssd_max_D = max_D + metaheader_dim;
    std::optional<at::Tensor> ssd_table_dims = std::nullopt;
    if (table_dims.has_value()) {
      ssd_table_dims = table_dims.value() + metaheader_dim;
    }

    // SSD (L3) RocksDB tier. enable_metadata_cf=true + metadata_dim create the
    // metadata column family used to store per-row metadata separately.
    rocksdb_impl_ = std::make_shared<ssd::EmbeddingRocksDB>(
        std::move(ssd_path),
        num_shards,
        num_threads,
        memtable_flush_period,
        memtable_flush_offset,
        l0_files_per_compact,
        ssd_max_D,
        rate_limit_mbps,
        size_ratio,
        compaction_ratio,
        write_buffer_size,
        max_write_buffer_num,
        static_cast<float>(uniform_init_lower),
        static_cast<float>(uniform_init_upper),
        row_storage_bitwidth,
        block_cache_size,
        use_passed_in_path,
        tbe_unique_id,
        l2_cache_size_gb,
        ssd_enable_async_update,
        enable_raw_embedding_streaming,
        res_store_shards,
        res_server_port,
        std::move(table_names),
        std::move(table_offsets),
        table_sizes,
        ssd_table_dims,
        hash_size_cumsum,
        flushing_block_size,
        disable_random_init,
        enable_blob_db,
        /*enable_metadata_cf=*/true,
        /*metadata_dim=*/metaheader_dim);

    // Assemble the composite (DRAM L2 + RocksDB L3).
    auto composite =
        std::make_shared<kv_db::DramSsdKVEmbeddingCache<weight_type>>(
            dram_cache,
            rocksdb_impl_,
            writeback_queue_size_,
            writeback_batch_size_);
    impl_ = composite;
    flush_composite_ = [composite]() { composite->flush(); };
  }

  // Single composite backend (DRAM L2 + RocksDB L3).
  std::shared_ptr<kv_db::EmbeddingKVDB> impl_;

  // Single RocksDB storage tier, kept for snapshot/checkpoint operations.
  std::shared_ptr<ssd::EmbeddingRocksDB> rocksdb_impl_;

  // Typed composite flush(), captured at construction (flush() is non-virtual).
  std::function<void()> flush_composite_;

  int64_t row_storage_bitwidth_;
  int64_t writeback_queue_size_;
  int64_t writeback_batch_size_;
};

} // namespace kv_mem
