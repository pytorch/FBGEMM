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
      bool enable_async_update = false)
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
            enable_async_update)) {}

  void set_cuda(
      at::Tensor indices,
      at::Tensor weights,
      at::Tensor count,
      int64_t timestep,
      bool is_bwd) {
    return impl_->set_cuda(indices, weights, count, timestep, is_bwd);
  }

  void get_cuda(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    return impl_->get_cuda(indices, weights, count);
  }

  void set(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    return impl_->set(indices, weights, count);
  }

  void set_range_to_storage(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length) {
    return impl_->set_range_to_storage(weights, start, length);
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

 private:
  friend class KVTensorWrapper;

  // shared pointer since we use shared_from_this() in callbacks.
  std::shared_ptr<ssd::EmbeddingRocksDB> impl_;
};

} // namespace ssd
