/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/MPMCQueue.h>
#include <folly/coro/BlockingWait.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/logging/xlog.h>
#include <thread>

#include "dram_kv_embedding_cache.h"

namespace kv_db {

/// A batch of dirty blocks evicted from DRAM, to be written to SSD.
struct WriteBatchItem {
  std::vector<std::pair<int64_t, std::string>> blocks; // key -> serialized
};

/// @ingroup KVMemEmbedding
///
/// @brief Composite backend that orchestrates DRAM (L2) + RocksDB SSD (L3).
///
/// Lookup path: DRAM first -> SSD for misses -> backfill DRAM on SSD hits.
/// Write path: Write to DRAM only (dirty bit marks for eventual SSD writeback).
/// Eviction path: Dirty blocks evicted from DRAM are enqueued to a writeback
/// queue and asynchronously flushed to SSD by a background thread.
///
/// Checkpoint: flush() drains all dirty DRAM blocks to SSD, then the caller
/// can create a RocksDB snapshot for consistent reads.
///
template <typename weight_type>
class DramSsdKVEmbeddingCache : public EmbeddingKVDB {
 public:
  /// Construct the composite backend.
  ///
  /// @param dram_cache The DRAM tier (L2)
  /// @param rocksdb_backend The SSD tier (L3), an EmbeddingKVDB implementation
  /// @param writeback_queue_size Capacity of the writeback queue
  /// @param writeback_batch_size Max items to batch-write to SSD per iteration
  DramSsdKVEmbeddingCache(
      std::shared_ptr<kv_mem::DramKVEmbeddingCache<weight_type>> dram_cache,
      std::shared_ptr<EmbeddingKVDB> rocksdb_backend,
      int64_t writeback_queue_size = 1024,
      int64_t writeback_batch_size = 1024)
      : EmbeddingKVDB(
            require_dram_cache(dram_cache)->get_num_shards(),
            require_dram_cache(dram_cache)->get_max_D(),
            /*cache_size_gb=*/0,
            /*unique_id=*/0,
            /*ele_size_bytes=*/sizeof(weight_type)),
        dram_cache_(std::move(dram_cache)),
        rocksdb_backend_(std::move(rocksdb_backend)),
        writeback_queue_(writeback_queue_size),
        writeback_batch_size_(writeback_batch_size),
        metaheader_dim_(
            kv_mem::FixedBlockPool::get_metaheader_dim<weight_type>()) {
    TORCH_CHECK(
        dram_cache_->get_backend_return_whole_row(),
        "DramSsdKVEmbeddingCache requires the DRAM cache to be constructed "
        "with backend_return_whole_row=true; checkpoint restore is not "
        "supported otherwise.");

    // Wire eviction writeback callback (Step 2)
    // When DRAM evicts dirty blocks, they are enqueued to writeback queue.
    auto* feature_evict = dram_cache_->get_feature_evict();
    if (feature_evict) {
      feature_evict->set_writeback_callback(
          [this](std::vector<std::pair<int64_t, std::string>> batch) {
            enqueue_writeback(WriteBatchItem{std::move(batch)});
          });
    }

    // Dedicated executor for async DRAM backfill (won't compete with
    // L2 cache or forward/backward thread pools).
    backfill_executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(2);

    // Start background writeback thread
    start_writeback_thread();
  }

  ~DramSsdKVEmbeddingCache() override {
    stop_writeback_thread();
  }

  // --- SSD metrics accessors ---

  int64_t get_ssd_num_lookups() const {
    return ssd_num_lookups_.load(std::memory_order_relaxed);
  }
  int64_t get_ssd_num_hits() const {
    return ssd_num_hits_.load(std::memory_order_relaxed);
  }
  int64_t get_ssd_num_writes() const {
    return ssd_num_writes_.load(std::memory_order_relaxed);
  }

  // --- Core EmbeddingKVDB interface ---

  /// Two-phase lookup: DRAM first, then SSD for misses, backfill DRAM on hits.
  ///
  /// Phase 1: Call dram_cache_->get_kv_db_async(indices, weights, count)
  ///   DRAM hits fill weights directly. Misses remain as zero rows.
  ///
  /// Phase 2: For zero rows (DRAM misses), batch-lookup in SSD.
  ///   SSD hits fill the weights.
  ///
  /// Phase 3: Backfill DRAM with SSD hits so future lookups are fast.
  ///   This marks them dirty — feature score eviction handles thrash.
  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    // Phase 1: DRAM lookup
    dram_cache_->get_kv_db_async(indices, weights, count).wait();

    auto num = count.scalar_type() == at::ScalarType::Long
        ? *(count.const_data_ptr<int64_t>())
        : static_cast<int64_t>(*(count.const_data_ptr<int32_t>()));

    if (num <= 0) {
      return folly::makeSemiFuture(std::vector<folly::Unit>());
    }

    // Phase 2: Check for DRAM misses (all-zero rows)
    auto* idx_ptr = indices.const_data_ptr<int64_t>();
    auto* w_ptr = weights.template data_ptr<weight_type>();
    int64_t stride = weights.size(1);

    std::vector<int64_t> miss_indices;
    std::vector<int64_t> miss_positions; // position in original tensor
    miss_indices.reserve(num);
    miss_positions.reserve(num);

    for (int64_t i = 0; i < num; ++i) {
      if (idx_ptr[i] < 0) {
        continue;
      }
      bool all_zero = true;
      for (int64_t j = 0; j < stride; ++j) {
        if (w_ptr[i * stride + j] != static_cast<weight_type>(0)) {
          all_zero = false;
          break;
        }
      }
      if (all_zero) {
        miss_indices.push_back(idx_ptr[i]);
        miss_positions.push_back(i);
      }
    }

    if (miss_indices.empty()) {
      return folly::makeSemiFuture(std::vector<folly::Unit>());
    }

    // Track SSD lookups
    ssd_num_lookups_.fetch_add(miss_indices.size(), std::memory_order_relaxed);

    // Create tensors for SSD lookup.
    // Use weights-only read (skip MetaHeader) for the critical path.
    auto miss_count_val = static_cast<int64_t>(miss_indices.size());
    auto miss_idx_tensor = at::from_blob(
                               miss_indices.data(),
                               {miss_count_val},
                               at::TensorOptions().dtype(at::kLong))
                               .clone();
    auto ssd_weights = at::zeros(
        {miss_count_val, stride},
        at::TensorOptions().dtype(
            c10::CppTypeToScalarType<weight_type>::value));
    auto miss_count_tensor = at::tensor({miss_count_val}, at::ScalarType::Long);

    // SSD lookup for misses (weights only, skipping MetaHeader prefix)
    rocksdb_backend_
        ->get_kv_db_weights_only_async(
            miss_idx_tensor, ssd_weights, miss_count_tensor)
        .wait();

    // Phase 3: Merge SSD hits into output weights and backfill DRAM.
    // ssd_weights is already weights-only (no MetaHeader prefix).
    auto* ssd_w_ptr = ssd_weights.template data_ptr<weight_type>();
    std::vector<int64_t> backfill_indices;
    std::vector<int64_t> backfill_positions; // positions in ssd_weights
    backfill_indices.reserve(miss_count_val);
    backfill_positions.reserve(miss_count_val);

    for (int64_t m = 0; m < miss_count_val; ++m) {
      bool has_nonzero = false;
      for (int64_t j = 0; j < stride; ++j) {
        if (ssd_w_ptr[m * stride + j] != static_cast<weight_type>(0)) {
          has_nonzero = true;
          break;
        }
      }
      if (has_nonzero) {
        // Copy weights directly into the output tensor
        int64_t orig_pos = miss_positions[m];
        std::copy(
            ssd_w_ptr + m * stride,
            ssd_w_ptr + (m + 1) * stride,
            w_ptr + orig_pos * stride);
        backfill_indices.push_back(miss_indices[m]);
        backfill_positions.push_back(m);
      }
    }

    // Track SSD hits
    ssd_num_hits_.fetch_add(backfill_indices.size(), std::memory_order_relaxed);

    // Backfill DRAM with SSD hits asynchronously (fire-and-forget).
    // The metadata will be read from SSD (likely cache-hot) on the backfill
    // thread. This unblocks the critical path.
    XLOG_EVERY_MS(INFO, 30000)
        << "[DramSsdKVEmbeddingCache] SSD->DRAM backfill: "
        << backfill_indices.size() << " hits out of " << miss_indices.size()
        << " SSD lookups";
    if (!backfill_indices.empty()) {
      auto bf_count_val = static_cast<int64_t>(backfill_indices.size());
      auto bf_idx_tensor = at::from_blob(
                               backfill_indices.data(),
                               {bf_count_val},
                               at::TensorOptions().dtype(at::kLong))
                               .clone();
      // Build backfill weights from SSD results (already weights-only)
      auto bf_weights = at::zeros(
          {bf_count_val, stride},
          at::TensorOptions().dtype(
              c10::CppTypeToScalarType<weight_type>::value));
      auto* bf_w_ptr = bf_weights.template data_ptr<weight_type>();
      for (int64_t b = 0; b < bf_count_val; ++b) {
        int64_t m = backfill_positions[b];
        std::copy(
            ssd_w_ptr + m * stride,
            ssd_w_ptr + (m + 1) * stride,
            bf_w_ptr + b * stride);
      }
      auto bf_count_tensor = at::tensor({bf_count_val}, at::ScalarType::Long);

      // Fire-and-forget: backfill DRAM on dedicated backfill executor
      auto dram_cache = dram_cache_;
      folly::via(
          backfill_executor_.get(),
          [dram_cache,
           bf_idx = std::move(bf_idx_tensor),
           bf_wts = std::move(bf_weights),
           bf_cnt = std::move(bf_count_tensor)]() {
            dram_cache->set_kv_db_async(bf_idx, bf_wts, bf_cnt).wait();
            ;
          });
    }

    return folly::makeSemiFuture(std::vector<folly::Unit>());
  }

  /// Checkpoint restore writes directly to SSD, bypassing DRAM allocation.
  /// For KVZCH (backend_return_whole_row), the correct linearized IDs are
  /// already embedded in the first 8 bytes of each weight row by
  /// KVTensorWrapper::set_range() via replace_weights_id(). We extract
  /// them here rather than using the `start` parameter, which is
  /// PMT_offset + row_offset_ and can be very negative for later shards.
  void set_range_to_storage(
      const at::Tensor& weights_with_metaheader,
      const int64_t start,
      const int64_t length) override {
    // Extract keys from the embedded IDs in the first 8 bytes of each row
    // (set by replace_weights_id in KVTensorWrapper::set_range).
    // Follows the same pattern as write_blocks_to_ssd().
    std::vector<int64_t> keys(weights_with_metaheader.size(0), 0);
    for (int64_t i = 0; i < weights_with_metaheader.size(0); ++i) {
      keys[i] = kv_mem::FixedBlockPool::get_key(
          weights_with_metaheader[i].data_ptr());
    }
    auto indices =
        torch::from_blob(keys.data(), {int64_t(keys.size())}, torch::kInt64);
    const auto count =
        at::tensor({weights_with_metaheader.size(0)}, at::ScalarType::Long);
    folly::coro::blockingWait(rocksdb_backend_->set_kv_db_async(
        indices, weights_with_metaheader, count));
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
  /// The DRAM cache already has SSD existence check wired (Step 3).
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
    rocksdb_backend_->compact();
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
    auto ret = dram_cache_->get_dram_kv_perf(step, interval);

    // Append the SSD-tier metrics after the DRAM block, deriving the offset
    // from the base vector size so it stays aligned if the DRAM metric set
    // changes (Python mirrors this with _DRAM_SSD_PERF_OFFSET).
    const size_t ssd_offset = ret.size();
    constexpr size_t kNumSsdMetrics = 5;
    ret.resize(ssd_offset + kNumSsdMetrics, 0);
    if (step > 0 && step % interval == 0) {
      int64_t reset_val = 0;
      auto lookups = ssd_num_lookups_.exchange(reset_val);
      auto hits = ssd_num_hits_.exchange(reset_val);
      auto writes = ssd_num_writes_.exchange(reset_val);

      ret[ssd_offset + 0] = static_cast<double>(lookups) / interval;
      ret[ssd_offset + 1] = static_cast<double>(hits) / interval;
      ret[ssd_offset + 2] = static_cast<double>(writes) / interval;
      // SSD estimated num keys (absolute). Read from the cached atomic to avoid
      // blocking the training thread on RocksDB mutex contention.
      ret[ssd_offset + 3] = static_cast<double>(cached_ssd_num_keys_.load());
      // Cumulative actual rows written to SSD (absolute).
      ret[ssd_offset + 4] =
          static_cast<double>(rocksdb_backend_->get_total_rows_written());
    }
    return ret;
  }

  /// DRAM_SSD stores the full DRAM block in SSD (MetaHeader + weight +
  /// optimizer) so that checkpoint can read the complete row including
  /// eviction metadata.  Delegate to the DRAM cache via base-class pointer
  /// to dispatch through the virtual interface (the override is private in
  /// DramKVEmbeddingCache).
  bool get_backend_return_whole_row() override {
    return dram_cache_->get_backend_return_whole_row();
  }

  /// SSD stores MetaHeader as a prefix, matching the DRAM block layout.
  int64_t get_metaheader_width_in_front() override {
    return dram_cache_->get_metaheader_width_in_front();
  }

  /// Checkpoint: read weights from SSD after flush-then-snapshot.
  /// After flush(), SSD has all data — read only from SSD for consistency.
  void get_range_from_snapshot(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length,
      const ssd::SnapshotHandle* snapshot_handle,
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override {
    rocksdb_backend_->get_range_from_snapshot(
        weights, start, length, snapshot_handle, width_offset, width_length);
  }

  /// Checkpoint: read weights by IDs from SSD after flush-then-snapshot.
  void get_kv_from_storage_by_snapshot(
      const at::Tensor& ids,
      const at::Tensor& weights,
      const ssd::SnapshotHandle* snapshot_handle,
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override {
    rocksdb_backend_->get_kv_from_storage_by_snapshot(
        ids, weights, snapshot_handle, width_offset, width_length);
  }

  // --- Key range and metadata ---

  /// Get keys in range. For checkpoint use, the wrapper routes directly to
  /// EmbeddingRocksDB::get_keys_in_range_by_snapshot() which is the proper
  /// SSD key iteration method. This fallback delegates to DRAM cache for
  /// non-checkpoint use cases (e.g. get_keys_in_range without snapshot).
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

  // --- Flush: drain all dirty DRAM blocks to SSD ---

  /// Flush dirty DRAM blocks to SSD for checkpoint consistency.
  ///
  /// Steps:
  /// 1. Iterate all DRAM shards under rlock
  /// 2. Collect dirty blocks (key + serialized data)
  /// 3. Clear dirty bits
  /// 4. Batch write collected blocks to SSD
  /// 5. Drain any pending writeback queue items
  ///
  /// Only dirty blocks are written. All DRAM write paths that produce
  /// new data (enrichment, checkpoint load, metadata updates) set the
  /// dirty bit. SSD backfill paths skip set_dirty since SSD already
  /// has the data.
  void flush() {
    auto flush_start_us = facebook::WallClockUtil::NowInUsecFast();
    int64_t num_shards = dram_cache_->get_num_shards();
    auto& kv_store = dram_cache_->get_kv_store();
    int64_t block_size = dram_cache_->get_block_size();

    std::vector<int64_t> shard_entry_counts(num_shards, 0);

    // Per-shard timing (microseconds) for diagnosing QPS regression.
    std::vector<int64_t> shard_rlock_us(num_shards, 0);
    std::vector<int64_t> shard_dirty_write_us(num_shards, 0);
    std::vector<int64_t> shard_metadata_read_us(num_shards, 0);
    std::vector<int64_t> shard_metadata_write_us(num_shards, 0);
    std::vector<int64_t> shard_total_keys(num_shards, 0);

    // Step 1: Collect and write dirty blocks from all DRAM shards in parallel.
    // Each shard's rlock is independent, so shards can be processed
    // concurrently. rlock allows training reads to proceed during flush.
    // SSD writes are done per-shard since RocksDB supports concurrent
    // writes.
    // After dirty blocks, also flush ALL metadata for this shard to SSD.
    std::atomic<int64_t> total_metadata_keys{0};
    std::vector<std::thread> workers;
    workers.reserve(num_shards);
    for (int64_t shard_id = 0; shard_id < num_shards; ++shard_id) {
      workers.emplace_back([this,
                            &kv_store,
                            &shard_entry_counts,
                            &total_metadata_keys,
                            &shard_rlock_us,
                            &shard_dirty_write_us,
                            &shard_metadata_read_us,
                            &shard_metadata_write_us,
                            &shard_total_keys,
                            shard_id,
                            block_size]() {
        ShardData shard;
        std::vector<int64_t> non_dirty_keys;
        {
          auto rlock_start = facebook::WallClockUtil::NowInUsecFast();
          auto rlmap = kv_store.by(shard_id).rlock();
          auto* pool = kv_store.pool_by(shard_id);
          shard.keys.reserve(rlmap->size());
          shard.blocks.reserve(rlmap->size());
          non_dirty_keys.reserve(rlmap->size());
          for (auto& [key, block] : *rlmap) {
            if (pool->get_dirty(block)) {
              shard.keys.push_back(key);
              shard.blocks.emplace_back(
                  reinterpret_cast<const char*>(block), block_size);
              pool->clear_dirty(block);
            } else {
              non_dirty_keys.push_back(key);
            }
          }
          shard_rlock_us[shard_id] =
              facebook::WallClockUtil::NowInUsecFast() - rlock_start;
        }
        shard_total_keys[shard_id] =
            static_cast<int64_t>(shard.keys.size() + non_dirty_keys.size());
        shard_entry_counts[shard_id] = shard.keys.size();
        if (!shard.keys.empty()) {
          auto write_start = facebook::WallClockUtil::NowInUsecFast();
          write_blocks_to_ssd(shard.keys, shard.blocks);
          shard_dirty_write_us[shard_id] =
              facebook::WallClockUtil::NowInUsecFast() - write_start;
        }

        // Flush metadata for non-dirty keys to SSD.
        // Dirty keys already have their metadata written atomically
        // with embeddings via write_blocks_to_ssd/set_kv_db_async.
        // Non-dirty keys may still have metadata updates
        // (set_kv_zch_eviction_metadata_async only writes to DRAM
        // without setting the dirty bit).
        if (rocksdb_backend_ && !non_dirty_keys.empty()) {
          auto num_keys = static_cast<int64_t>(non_dirty_keys.size());
          auto md_read_start = facebook::WallClockUtil::NowInUsecFast();
          auto indices = at::from_blob(
                             non_dirty_keys.data(),
                             {num_keys},
                             at::TensorOptions().dtype(at::kLong))
                             .clone();
          auto count = at::tensor({num_keys}, at::ScalarType::Long);
          auto metadata = dram_cache_->get_kv_metadata_rows(indices, count);
          shard_metadata_read_us[shard_id] =
              facebook::WallClockUtil::NowInUsecFast() - md_read_start;

          auto md_write_start = facebook::WallClockUtil::NowInUsecFast();
          rocksdb_backend_->set_kv_metadata_async(indices, metadata, count)
              .wait();
          shard_metadata_write_us[shard_id] =
              facebook::WallClockUtil::NowInUsecFast() - md_write_start;
          total_metadata_keys.fetch_add(num_keys);
        }
      });
    }
    for (auto& t : workers) {
      t.join();
    }

    auto workers_done_us = facebook::WallClockUtil::NowInUsecFast();

    int64_t total_entries = 0;
    for (auto c : shard_entry_counts) {
      total_entries += c;
    }

    // Log per-shard timing breakdown for QPS regression diagnosis.
    int64_t max_rlock = 0, max_dirty_write = 0, max_md_read = 0,
            max_md_write = 0;
    for (int64_t s = 0; s < num_shards; ++s) {
      max_rlock = std::max(max_rlock, shard_rlock_us[s]);
      max_dirty_write = std::max(max_dirty_write, shard_dirty_write_us[s]);
      max_md_read = std::max(max_md_read, shard_metadata_read_us[s]);
      max_md_write = std::max(max_md_write, shard_metadata_write_us[s]);
    }
    LOG(INFO) << "[DramSsdKVEmbeddingCache] flush() max across shards:"
              << " rlock_ms=" << max_rlock / 1000.0
              << " dirty_write_ms=" << max_dirty_write / 1000.0
              << " metadata_read_ms=" << max_md_read / 1000.0
              << " metadata_write_ms=" << max_md_write / 1000.0;

    LOG(INFO) << "[DramSsdKVEmbeddingCache] flush() wrote " << total_entries
              << " dirty entries (with metadata) and metadata for "
              << total_metadata_keys.load() << " non-dirty keys to SSD across "
              << num_shards << " shards, block_size=" << block_size
              << " metaheader_dim=" << metaheader_dim_
              << " max_D=" << dram_cache_->get_max_D() << " workers_elapsed_ms="
              << (workers_done_us - flush_start_us) / 1000.0;

    // Drain the writeback queue
    auto drain_start_us = facebook::WallClockUtil::NowInUsecFast();
    drain_writeback_queue();
    auto drain_us = facebook::WallClockUtil::NowInUsecFast() - drain_start_us;
    auto total_us = facebook::WallClockUtil::NowInUsecFast() - flush_start_us;
    LOG(INFO) << "[DramSsdKVEmbeddingCache] flush() done"
              << " drain_writeback_ms=" << drain_us / 1000.0
              << " total_flush_ms=" << total_us / 1000.0;
  }

  // --- Writeback thread management ---

  void start_writeback_thread() {
    if (writeback_running_.load()) {
      return;
    }
    writeback_running_.store(true);
    writeback_thread_ =
        std::make_unique<std::thread>([this]() { writeback_loop(); });
  }

  void stop_writeback_thread() {
    if (!writeback_running_.load()) {
      return;
    }
    writeback_running_.store(false);
    if (writeback_thread_ && writeback_thread_->joinable()) {
      writeback_thread_->join();
    }
    writeback_thread_.reset();
  }

  // --- Accessors ---

  const std::shared_ptr<kv_mem::DramKVEmbeddingCache<weight_type>>& dram_cache()
      const {
    return dram_cache_;
  }
  auto& ssd_backend() {
    return rocksdb_backend_;
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
    // No-op; flush is explicit via flush() method
  }

  /// Enqueue a batch of dirty evicted blocks for async SSD writeback.
  /// Non-blocking: if queue is full, log a warning instead of blocking
  /// eviction.
  void enqueue_writeback(WriteBatchItem item) {
    if (!writeback_queue_.write(std::move(item))) {
      XLOG_EVERY_MS(WARNING, 60000)
          << "[DramSsdKVEmbeddingCache] Writeback queue full, "
          << "dropping evicted dirty blocks. Consider increasing queue size.";
    }
  }

  /// Background writeback loop: consume from queue, write to SSD in batches.
  void writeback_loop() {
    while (writeback_running_.load()) {
      WriteBatchItem item;
      auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
      if (writeback_queue_.tryReadUntil(deadline, item)) {
        if (!item.blocks.empty()) {
          std::vector<int64_t> keys;
          std::vector<std::string> blocks;
          keys.reserve(item.blocks.size());
          blocks.reserve(item.blocks.size());
          for (auto& [key, data] : item.blocks) {
            keys.push_back(key);
            blocks.push_back(std::move(data));
          }
          write_blocks_to_ssd(keys, blocks);
        }
      }
    }
  }

  /// Drain all remaining items in the writeback queue.
  void drain_writeback_queue() {
    WriteBatchItem item;
    while (writeback_queue_.read(item)) {
      if (!item.blocks.empty()) {
        std::vector<int64_t> keys;
        std::vector<std::string> blocks;
        keys.reserve(item.blocks.size());
        blocks.reserve(item.blocks.size());
        for (auto& [key, data] : item.blocks) {
          keys.push_back(key);
          blocks.push_back(std::move(data));
        }
        write_blocks_to_ssd(keys, blocks);
      }
    }
  }

  /// Write serialized blocks to SSD via rocksdb_backend_->set_kv_db_async().
  /// Writes the FULL DRAM block (MetaHeader + weight + optimizer) so that
  /// checkpoint can read the complete row including eviction metadata.
  void write_blocks_to_ssd(
      const std::vector<int64_t>& keys,
      const std::vector<std::string>& serialized_blocks) {
    if (keys.empty()) {
      return;
    }

    // Track SSD writes
    ssd_num_writes_.fetch_add(keys.size(), std::memory_order_relaxed);
    XLOG_EVERY_MS(INFO, 30000)
        << "[DramSsdKVEmbeddingCache] write_blocks_to_ssd: writing "
        << keys.size() << " blocks, total_writes="
        << ssd_num_writes_.load(std::memory_order_relaxed);

    int64_t max_D = dram_cache_->get_max_D();
    // Total SSD row width: MetaHeader (as weight_type elements) + data
    int64_t total_width = metaheader_dim_ + max_D;
    int64_t num_keys = static_cast<int64_t>(keys.size());

    // Process in batches of writeback_batch_size_
    for (int64_t offset = 0; offset < num_keys;
         offset += writeback_batch_size_) {
      int64_t batch_size = std::min(writeback_batch_size_, num_keys - offset);

      auto indices =
          at::zeros({batch_size}, at::TensorOptions().dtype(at::kLong));
      auto weights = at::zeros(
          {batch_size, total_width},
          at::TensorOptions().dtype(
              c10::CppTypeToScalarType<weight_type>::value));

      auto* idx_ptr = indices.data_ptr<int64_t>();
      auto* w_ptr = weights.template data_ptr<weight_type>();

      for (int64_t i = 0; i < batch_size; ++i) {
        int64_t global_idx = offset + i;
        idx_ptr[i] = keys[global_idx];

        // Copy the full DRAM block (MetaHeader + weight + optimizer),
        // treating MetaHeader bytes as weight_type elements.
        const auto& block_data = serialized_blocks[global_idx];
        const auto* block_start =
            reinterpret_cast<const weight_type*>(block_data.data());
        int64_t copy_len = std::min(
            total_width,
            static_cast<int64_t>(block_data.size() / sizeof(weight_type)));
        std::copy(block_start, block_start + copy_len, w_ptr + i * total_width);
      }

      auto count = at::tensor({batch_size}, at::ScalarType::Long);
      try {
        rocksdb_backend_->set_kv_db_async(indices, weights, count).wait();
      } catch (const std::exception& e) {
        XLOG_EVERY_MS(WARNING, 60000)
            << "[DramSsdKVEmbeddingCache] SSD write failed: " << e.what();
      }
    }
    cached_ssd_num_keys_.store(rocksdb_backend_->get_estimated_num_keys());
  }

  struct ShardData {
    std::vector<int64_t> keys;
    std::vector<std::string> blocks;
  };

  std::shared_ptr<kv_mem::DramKVEmbeddingCache<weight_type>> dram_cache_;
  std::shared_ptr<EmbeddingKVDB> rocksdb_backend_;

  // Writeback queue: evicted dirty blocks -> SSD
  folly::MPMCQueue<WriteBatchItem> writeback_queue_;
  std::unique_ptr<std::thread> writeback_thread_;
  std::atomic<bool> writeback_running_{false};

  int64_t writeback_batch_size_;

  // MetaHeader width in weight_type elements
  int64_t metaheader_dim_;

  // Dedicated executor for async DRAM backfill
  std::unique_ptr<folly::CPUThreadPoolExecutor> backfill_executor_;

  // SSD metrics (per-interval, reset on read via exchange)
  std::atomic<int64_t> ssd_num_lookups_{0}; // DRAM misses sent to SSD
  std::atomic<int64_t> ssd_num_hits_{0}; // SSD hits (backfilled to DRAM)
  std::atomic<int64_t> ssd_num_writes_{0}; // blocks written to SSD
  std::atomic<int64_t> cached_ssd_num_keys_{0}; // refreshed by writeback thread
}; // class DramSsdKVEmbeddingCache

} // namespace kv_db
