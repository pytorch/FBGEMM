/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "kv_tensor_wrapper.h"
#include "ssd_table_batched_embeddings.h"

namespace ssd {

class EmbeddingRocksDBWrapper : public torch::jit::CustomClassHolder {
 public:
  EmbeddingRocksDBWrapper(
      std::string path,
      int64_t num_shards,
      int64_t num_threads,
      int64_t memtable_flush_period,
      int64_t memtable_flush_offset,
      int64_t l0_files_per_compact,
      int64_t max_D,
      int64_t rate_limit_mbps,
      int64_t size_ratio,
      int64_t compaction_ratio,
      int64_t write_buffer_size,
      int64_t max_write_buffer_num,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t row_storage_bitwidth = 32,
      int64_t cache_size = 0,
      bool use_passed_in_path = false,
      int64_t tbe_unique_id = 0,
      int64_t l2_cache_size_gb = 0,
      bool enable_async_update = false,
      bool enable_raw_embedding_streaming = false,
      int64_t res_store_shards = 0,
      int64_t res_server_port = 0,
      std::vector<std::string> table_names = {},
      std::vector<int64_t> table_offsets = {},
      const std::vector<int64_t>& table_sizes = {},
      std::optional<at::Tensor> table_dims = std::nullopt,
      std::optional<at::Tensor> hash_size_cumsum = std::nullopt,
      int64_t flushing_block_size = 2000000000 /*2GB*/,
      bool disable_random_init = false)
      : impl_(std::make_shared<ssd::EmbeddingRocksDB>(
            path,
            num_shards,
            num_threads,
            memtable_flush_period,
            memtable_flush_offset,
            l0_files_per_compact,
            max_D,
            rate_limit_mbps,
            size_ratio,
            compaction_ratio,
            write_buffer_size,
            max_write_buffer_num,
            uniform_init_lower,
            uniform_init_upper,
            row_storage_bitwidth,
            cache_size,
            use_passed_in_path,
            tbe_unique_id,
            l2_cache_size_gb,
            enable_async_update,
            enable_raw_embedding_streaming,
            res_store_shards,
            res_server_port,
            std::move(table_names),
            std::move(table_offsets),
            table_sizes,
            table_dims,
            hash_size_cumsum,
            flushing_block_size,
            disable_random_init)) {}

  void set_cuda(
      at::Tensor indices,
      at::Tensor weights,
      at::Tensor count,
      int64_t timestep,
      bool is_bwd) {
    return impl_->set_cuda(indices, weights, count, timestep, is_bwd);
  }

  void stream_cuda(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      bool blocking_tensor_copy = true) {
    return impl_->stream_cuda(indices, weights, count, blocking_tensor_copy);
  }

  void set_feature_score_metadata_cuda(
      const at::Tensor& indices,
      const at::Tensor& count,
      const at::Tensor& engage_show_count) {
    LOG(INFO) << "set_feature_score_metadata_cuda";
    impl_->set_feature_score_metadata_cuda(indices, count, engage_show_count);
  }

  void stream_sync_cuda() {
    return impl_->stream_sync_cuda();
  }

  void get_cuda(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    return impl_->get_cuda(indices, weights, count);
  }

  void set(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    return impl_->set(indices, weights, count);
  }

  void set_kv_to_storage(const at::Tensor& ids, const at::Tensor& weights) {
    return impl_->set_kv_to_storage(ids, weights);
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
      std::optional<c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper>>
          snapshot_handle) {
    return impl_->get_keys_in_range_by_snapshot(
        start_id,
        end_id,
        id_offset,
        snapshot_handle.has_value() ? snapshot_handle.value()->handle
                                    : nullptr);
  }

  at::Tensor get_kv_zch_eviction_metadata_by_snapshot(
      const at::Tensor& indices,
      const at::Tensor& count,
      std::optional<c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper>>
          snapshot_handle) {
    return impl_->get_kv_zch_eviction_metadata_by_snapshot(
        indices,
        count,
        snapshot_handle.has_value() ? snapshot_handle.value()->handle
                                    : nullptr);
  }

  void toggle_compaction(bool enable) {
    impl_->toggle_compaction(enable);
  }

  bool is_auto_compaction_enabled() {
    return impl_->is_auto_compaction_enabled();
  }

  void get(
      at::Tensor indices,
      at::Tensor weights,
      at::Tensor count,
      int64_t sleep_ms) {
    return impl_->get(indices, weights, count, sleep_ms);
  }

  std::vector<int64_t> get_mem_usage() {
    return impl_->get_mem_usage();
  }

  std::vector<double> get_rocksdb_io_duration(
      const int64_t step,
      const int64_t interval) {
    return impl_->get_rocksdb_io_duration(step, interval);
  }

  std::vector<double> get_l2cache_perf(
      const int64_t step,
      const int64_t interval) {
    return impl_->get_l2cache_perf(step, interval);
  }

  void compact() {
    return impl_->compact();
  }

  void flush() {
    return impl_->flush();
  }

  void reset_l2_cache() {
    return impl_->reset_l2_cache();
  }

  void wait_util_filling_work_done() {
    return impl_->wait_util_filling_work_done();
  }

  c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper> create_snapshot() {
    auto handle = impl_->create_snapshot();
    return c10::make_intrusive<EmbeddingSnapshotHandleWrapper>(handle, impl_);
  }

  void release_snapshot(
      c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper> snapshot_handle) {
    auto handle = snapshot_handle->handle;
    CHECK_NE(handle, nullptr);
    impl_->release_snapshot(handle);
  }

  int64_t get_snapshot_count() const {
    return impl_->get_snapshot_count();
  }

  void create_rocksdb_hard_link_snapshot(int64_t global_step) {
    impl_->create_checkpoint(global_step);
  }

  std::optional<c10::intrusive_ptr<RocksdbCheckpointHandleWrapper>>
  get_active_checkpoint_uuid(int64_t global_step) {
    auto uuid_opt = impl_->get_active_checkpoint_uuid(global_step);
    if (uuid_opt.has_value()) {
      return c10::make_intrusive<RocksdbCheckpointHandleWrapper>(
          uuid_opt.value(), impl_);
    } else {
      return std::nullopt;
    }
  }

  void set_backend_return_whole_row(bool backend_return_whole_row) {
    impl_->set_backend_return_whole_row(backend_return_whole_row);
  }

 private:
  friend class KVTensorWrapper;

  // shared pointer since we use shared_from_this() in callbacks.
  std::shared_ptr<ssd::EmbeddingRocksDB> impl_;
};

} // namespace ssd
