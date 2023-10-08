/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>
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
            row_storage_bitwidth)),
        path(path),
        num_shards(num_shards),
        num_threads(num_threads),
        memtable_flush_period(memtable_flush_period),
        memtable_flush_offset(memtable_flush_offset),
        l0_files_per_compact(l0_files_per_compact),
        max_D(max_D),
        rate_limit_mbps(rate_limit_mbps),
        size_ratio(size_ratio),
        compaction_ratio(compaction_ratio),
        write_buffer_size(write_buffer_size),
        max_write_buffer_num(max_write_buffer_num),
        uniform_init_lower(uniform_init_lower),
        uniform_init_upper(uniform_init_upper),
        row_storage_bitwidth(row_storage_bitwidth) {}

  explicit EmbeddingRocksDBWrapper(std::string serialized) {
    std::string path;
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(serialized.data(), serialized.size());
    torch::IValue value;
    input_archive.read("path", value);
    path = value.toStringRef();
    input_archive.read("num_shards", value);
    num_shards = value.toInt();
    input_archive.read("num_threads", value);
    num_threads = value.toInt();
    input_archive.read("memtable_flush_period", value);
    memtable_flush_period = value.toInt();
    input_archive.read("memtable_flush_offset", value);
    memtable_flush_period = value.toInt();
    input_archive.read("l0_files_per_compact", value);
    l0_files_per_compact = value.toInt();
    input_archive.read("max_D", value);
    max_D = value.toInt();
    input_archive.read("rate_limit_mbps", value);
    rate_limit_mbps = value.toInt();
    input_archive.read("size_ratio", value);
    size_ratio = value.toInt();
    input_archive.read("compaction_ratio", value);
    compaction_ratio = value.toInt();
    input_archive.read("write_buffer_size", value);
    write_buffer_size = value.toInt();
    input_archive.read("max_write_buffer_num", value);
    max_write_buffer_num = value.toInt();
    input_archive.read("uniform_init_lower", value);
    uniform_init_lower = value.toDouble();
    input_archive.read("uniform_init_upper", value);
    uniform_init_upper = value.toDouble();
    input_archive.read("row_storage_bitwidth", value);
    row_storage_bitwidth = value.toInt();

    impl_ = std::make_shared<ssd::EmbeddingRocksDB>(
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
        row_storage_bitwidth);
  }

  std::string serialize() const {
    std::cout << "Enter serialize function in EmbeddingRocksDBWrapper"
              << std::endl;
    torch::serialize::OutputArchive output_archive(
        std::make_shared<torch::jit::CompilationUnit>());
    output_archive.write("path", torch::IValue(path));
    output_archive.write("num_shards", torch::IValue(num_shards));
    output_archive.write(
        "memtable_flush_period", torch::IValue(memtable_flush_period));
    output_archive.write(
        "memtable_flush_offset", torch::IValue(memtable_flush_offset));
    output_archive.write(
        "l0_files_per_compact", torch::IValue(l0_files_per_compact));
    output_archive.write("max_D", torch::IValue(max_D));
    output_archive.write("rate_limit_mbps", torch::IValue(rate_limit_mbps));
    output_archive.write("size_ratio", torch::IValue(size_ratio));
    output_archive.write("compaction_ratio", torch::IValue(compaction_ratio));
    output_archive.write("write_buffer_size", torch::IValue(write_buffer_size));
    output_archive.write(
        "max_write_buffer_num", torch::IValue(max_write_buffer_num));
    output_archive.write(
        "uniform_init_lower", torch::IValue(uniform_init_lower));
    output_archive.write(
        "uniform_init_upper", torch::IValue(uniform_init_upper));
    output_archive.write(
        "row_storage_bitwidth", torch::IValue(row_storage_bitwidth));
    std::ostringstream oss;
    output_archive.save_to(oss);
    std::cout << "serialized string: " << oss.str() << std::endl;
    return oss.str();
  }

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
  std::string path;
  int64_t num_shards;
  int64_t num_threads;
  int64_t memtable_flush_period;
  int64_t memtable_flush_offset;
  int64_t l0_files_per_compact;
  int64_t max_D;
  int64_t rate_limit_mbps;
  int64_t size_ratio;
  int64_t compaction_ratio;
  int64_t write_buffer_size;
  int64_t max_write_buffer_num;
  double uniform_init_lower;
  double uniform_init_upper;
  int64_t row_storage_bitwidth;
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
        .def("get", &EmbeddingRocksDBWrapper::get)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<EmbeddingRocksDBWrapper>& self)
                -> std::string { return self->serialize(); },
            // __setstate__
            [](std::string data)
                -> c10::intrusive_ptr<EmbeddingRocksDBWrapper> {
              return c10::make_intrusive<EmbeddingRocksDBWrapper>(
                  std::move(data));
            });

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
