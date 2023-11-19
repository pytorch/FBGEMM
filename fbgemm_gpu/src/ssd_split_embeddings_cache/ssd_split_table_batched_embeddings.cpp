/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include <torch/custom_class.h>

#include "./ssd_table_batched_embeddings.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using namespace at;

std::tuple<Tensor, Tensor, Tensor, Tensor> ssd_cache_populate_actions_cuda(
    Tensor linear_indices,
    int64_t total_hash_size,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    int64_t prefetch_dist,
    Tensor lru_state);

Tensor
masked_index_put_cuda(Tensor self, Tensor indices, Tensor values, Tensor count);

Tensor masked_index_put_byte_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count);

namespace {
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
      int64_t row_storage_bitwidth = 32)
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
            row_storage_bitwidth)) {}

  void
  set_cuda(Tensor indices, Tensor weights, Tensor count, int64_t timestep) {
    return impl_->set_cuda(indices, weights, count, timestep);
  }

  void get_cuda(Tensor indices, Tensor weights, Tensor count) {
    return impl_->get_cuda(indices, weights, count);
  }

  void set(Tensor indices, Tensor weights, Tensor count) {
    return impl_->set(indices, weights, count);
  }

  void get(Tensor indices, Tensor weights, Tensor count) {
    return impl_->get(indices, weights, count);
  }

  void compact() {
    return impl_->compact();
  }

  void flush() {
    return impl_->flush();
  }

 private:
  // shared pointer since we use shared_from_this() in callbacks.
  std::shared_ptr<ssd::EmbeddingRocksDB> impl_;
};

static auto embedding_rocks_db_wrapper =
    torch::class_<EmbeddingRocksDBWrapper>("fbgemm", "EmbeddingRocksDBWrapper")
        .def(torch::init<
             std::string,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             double,
             double,
             int64_t>())
        .def("set_cuda", &EmbeddingRocksDBWrapper::set_cuda)
        .def("get_cuda", &EmbeddingRocksDBWrapper::get_cuda)
        .def("compact", &EmbeddingRocksDBWrapper::compact)
        .def("flush", &EmbeddingRocksDBWrapper::flush)
        .def("set", &EmbeddingRocksDBWrapper::set)
        .def("get", &EmbeddingRocksDBWrapper::get);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "masked_index_put(Tensor self, Tensor indices, Tensor values, Tensor count) -> Tensor");
  DISPATCH_TO_CUDA("masked_index_put", masked_index_put_cuda);
  m.def(
      "ssd_cache_populate_actions(Tensor linear_indices, int total_hash_size, Tensor lxu_cache_state, int time_stamp, int prefetch_dist, Tensor lru_state) -> (Tensor, Tensor, Tensor, Tensor)");
  DISPATCH_TO_CUDA(
      "ssd_cache_populate_actions", ssd_cache_populate_actions_cuda);
}
} // namespace
