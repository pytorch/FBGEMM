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
#include "fbgemm_gpu/utils/ops_utils.h"

using namespace at;
using namespace ssd;

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
ssd_cache_populate_actions_cuda(
    Tensor linear_indices,
    int64_t total_hash_size,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    int64_t prefetch_dist,
    Tensor lru_state,
    bool gather_cache_stats,
    std::optional<Tensor> ssd_cache_stats,
    const bool lock_cache_line,
    const c10::optional<Tensor>& lxu_cache_locking_counter);

/// @ingroup embedding-ssd
///
/// @brief Similar to `torch.Tensor.index_put` but ignore `indices < 0`
///
/// `masked_index_put_cuda` only supports 2D input `values`. It puts
/// `count` rows in `values` into `self` using the row indices that
/// are >= 0 in `indices`.
///
/// ```python
/// # Equivalent PyTorch Python code
/// indices = indices[:count]
/// filter_ = indices >= 0
/// indices_ = indices[filter_]
/// self[indices_] = values[filter_.nonzero().flatten()]
/// ```
///
/// @param self The 2D output tensor (the tensor that is indexed)
/// @param indices The 1D index tensor
/// @param values The 2D input tensor
/// @param count The tensor that contains the length of `indices` to
///            process
/// @param use_pipeline A flag that indicates that this kernel will
///            overlap with other kernels. If it is true, then use a
///            fraction of SMs to reduce resource competition
/// @param preferred_sms The number of preferred SMs for the kernel to
///            use when use_pipeline=true. This value is ignored when
///            use_pipeline=false.
///
/// @return The `self` tensor
Tensor masked_index_put_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count,
    const bool use_pipeline,
    const int64_t preferred_sms);

/// @ingroup embedding-ssd
///
/// @brief Similar to `torch.index_select` but ignore `indices < 0`
///
/// `masked_index_select_cuda` only supports 2D input `values`. It
/// puts `count` rows that are specified in `indices` (where `indices`
/// >= 0) from `values` into `self`
///
/// ```python
/// # Equivalent PyTorch Python code
/// indices = indices[:count]
/// filter_ = indices >= 0
/// indices_ = indices[filter_]
/// self[filter_.nonzero().flatten()] = values[indices_]
/// ```
///
/// @param self The 2D output tensor
/// @param indices The 1D index tensor
/// @param values The 2D input tensor (the tensor that is indexed)
/// @param count The tensor that contains the length of `indices` to
///            process
/// @param use_pipeline A flag that indicates that this kernel will
///            overlap with other kernels. If it is true, then use a
///            fraction of SMs to reduce resource competition
///// @param preferred_sms The number of preferred SMs for the kernel to
///            use when use_pipeline=true. This value is ignored when
///            use_pipeline=false.
///
/// @return The `self` tensor
Tensor masked_index_select_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count,
    const bool use_pipeline,
    const int64_t preferred_sms);

Tensor masked_index_put_byte_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count);

/// @ingroup embedding-ssd
///
/// @brief Generate memory addresses for SSD TBE data
///
/// The data retrieved from SSD can be stored in either a scratch pad
/// (HBM) or LXU cache (also HBM). `lxu_cache_locations` is used to
/// specify the location of the data. If the location is -1, the data
/// for the associated index is in the scratch pad; otherwise, it is
/// in the cache. To enable TBE kernels to access the data
/// conveniently, this operator generates memory addresses of the
/// first byte for each index. When accessing data, a TBE kernel only
/// needs to convert addresses into pointers.
///
/// Moreover, this operator also generate the list of post backward
/// evicted indices which are basically the indices that their data
/// is in the scratch pad.
///
/// @param lxu_cache_locations The tensor that contains cache slots
///                            where data is stored for the *full* list
///                            of indices. -1 is a sentinel value that
///                            indicates that data is not in cache.
/// @param assigned_cache_slots The tensor that contains cache slots
///                             for the *unique* list of indices. -1
///                             indicates that data is not in cache
/// @param linear_index_inverse_indices The tensor that contains
///                                     the original position of
///                                     linear indices before being
///                                     sorted
/// @param unique_indices_count_cumsum The tensor that contains the
///                                    the exclusive prefix sum
///                                    results of the counts of unique
///                                    indices
/// @param cache_set_inverse_indices The tensor that contains the
///                                  original positions of cache sets
///                                  before being sorted
/// @param lxu_cache_weights The LXU cache tensor
/// @param inserted_ssd_weights The scratch pad tensor
/// @param unique_indices_length The tensor that contains the number
///                              of unique indices (GPU tensor)
/// @param cache_set_sorted_unique_indices The tensor that contains
///                                        associated unique indices
///                                        for the sorted unique cache
///                                        sets
///
/// @return A tuple of tensors (the SSD row address tensor and the
///         post backward evicted index tensor)
std::tuple<Tensor, Tensor> ssd_generate_row_addrs_cuda(
    const Tensor& lxu_cache_locations,
    const Tensor& assigned_cache_slots,
    const Tensor& linear_index_inverse_indices,
    const Tensor& unique_indices_count_cumsum,
    const Tensor& cache_set_inverse_indices,
    const Tensor& lxu_cache_weights,
    const Tensor& inserted_ssd_weights,
    const Tensor& unique_indices_length,
    const Tensor& cache_set_sorted_unique_indices);

/// @ingroup embedding-ssd
///
/// @brief Update memory addresses for SSD TBE data
///
/// When pipeline prefetching is enabled, data in a scratch pad of the
/// current iteration can be moved to L1 or a scratch pad of the next
/// iteration during the prefetch step. This operator updates the
/// memory addresses of data that is relocated to the correct
/// location.
///
/// @param ssd_row_addrs_curr The tensor that contains the row address
///            of the current iteration
/// @param inserted_ssd_weights_curr_next_map The tensor that contains
///            mapping between the location of each index in the
///            current iteration in the scratch pad of the next
///            iteration. (-1 = the data has not been moved).
///            inserted_ssd_weights_curr_next_map[i] is the location
//             of index i in the next iteration's scratch pad.
/// @param lxu_cache_locations_curr The tensor that contains cache
///            slots where data is stored for the *full* list of
///            indices for the current iteration. -1 is a sentinel
///            value that indicates that data is not in cache.
/// @param linear_index_inverse_indices_curr The tensor that contains
///            the original position of linear indices before being
///            sorted for the current iteration
/// @param unique_indices_count_cumsum_curr The tensor that contains
///            the the exclusive prefix sum results of the counts of
///            unique indices for the current iteration
/// @param cache_set_inverse_indices_curr The tensor that contains the
///            original positions of cache sets before being sorted
///            for the current iteration
/// @param lxu_cache_weights The LXU cache tensor
/// @param inserted_ssd_weights_next The scratch pad tensor for the
///            next iteration
/// @param unique_indices_length_curr The tensor that contains the
///            number of unique indices (GPU tensor) for the current
///            iteration
///
/// @return None
void ssd_update_row_addrs_cuda(
    const Tensor& ssd_row_addrs_curr,
    const Tensor& inserted_ssd_weights_curr_next_map,
    const Tensor& lxu_cache_locations_curr,
    const Tensor& linear_index_inverse_indices_curr,
    const Tensor& unique_indices_count_cumsum_curr,
    const Tensor& cache_set_inverse_indices_curr,
    const Tensor& lxu_cache_weights,
    const Tensor& inserted_ssd_weights_next,
    const Tensor& unique_indices_length_curr);

/// @ingroup embedding-ssd
///
/// @brief Compact the given list of indices
///
/// This operator compact the given list of indices based on the given
/// masks (a tensor that contains either 0 or 1). The operater removes
/// the indices that their corresponding mask is 0.  It only operates
/// on `count` number of elements (not the full tensor).
///
/// Example:
///
/// ```
/// indices = [[0, 3, -1, 3, -1, -1, 7], [0, 2, 2, 3, -1, 9, 7]]
/// masks = [1, 1, 0, 1, 0, 0, 1]
/// count = 5
///
/// # x represents an arbitrary value
/// compact_indices = [[0, 3, 3, x, x, x, x], [0, 2, 3, x, x, x, x]]
/// compact_count = 3
/// ```
///
/// @param compact_indices A list of compact indices (output indices).
/// @param compact_count A tensor that contains the number of elements
///            after being compacted
/// @param indices An input list of indices to be compacted
/// @param masks A tensor that contains 0 or 1 to indicate whether to
///            remove/keep the element. 0 = remove the corresponding
///            index. 1 = keep the corresponding index.
/// @count count A tensor that contains the number of elements to be
///            compacted
void compact_indices_cuda(
    std::vector<Tensor> compact_indices,
    Tensor compact_count,
    std::vector<Tensor> indices,
    Tensor masks,
    Tensor count);

namespace {
class KVTensorWrapper;

struct EmbeddingSnapshotHandleWrapper : public torch::jit::CustomClassHolder {
  explicit EmbeddingSnapshotHandleWrapper(
      const EmbeddingRocksDB::SnapshotHandle* handle,
      std::shared_ptr<EmbeddingRocksDB> db)
      : handle(handle), db(std::move(db)) {}

  ~EmbeddingSnapshotHandleWrapper() {
    db->release_snapshot(handle);
  }

  const EmbeddingRocksDB::SnapshotHandle* handle;
  std::shared_ptr<EmbeddingRocksDB> db;
};

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
      int64_t l2_cache_size_gb = 0)
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
            l2_cache_size_gb)) {}

  void set_cuda(
      Tensor indices,
      Tensor weights,
      Tensor count,
      int64_t timestep,
      bool is_bwd) {
    return impl_->set_cuda(indices, weights, count, timestep, is_bwd);
  }

  void get_cuda(Tensor indices, Tensor weights, Tensor count) {
    return impl_->get_cuda(indices, weights, count);
  }

  void set(Tensor indices, Tensor weights, Tensor count) {
    return impl_->set(indices, weights, count);
  }

  void get(Tensor indices, Tensor weights, Tensor count, int64_t sleep_ms) {
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

  int64_t get_max_D() const {
    return impl_->get_max_D();
  }

 private:
  friend class KVTensorWrapper;

  // shared pointer since we use shared_from_this() in callbacks.
  std::shared_ptr<ssd::EmbeddingRocksDB> impl_;
};

class KVTensorWrapper : public torch::jit::CustomClassHolder {
 public:
  explicit KVTensorWrapper(
      c10::intrusive_ptr<EmbeddingRocksDBWrapper> db,
      c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper> snapshot_handle,
      std::vector<int64_t> shape,
      int64_t dtype,
      int64_t row_offset)
      : db_(db->impl_),
        snapshot_handle_(std::move(snapshot_handle)),
        shape_(std::move(shape)),
        row_offset_(row_offset) {
    CHECK_EQ(shape_.size(), 2) << "Only 2D emb tensors are supported";
    options_ = at::TensorOptions()
                   .dtype(static_cast<c10::ScalarType>(dtype))
                   .device(at::kCPU)
                   .layout(at::kStrided);
  }

  at::Tensor narrow(int64_t dim, int64_t start, int64_t length) {
    CHECK_EQ(dim, 0) << "Only narrow on dim 0 is supported";
    CHECK_EQ(db_->get_max_D(), shape_[1]);
    auto t = at::empty(c10::IntArrayRef({length, db_->get_max_D()}), options_);
    db_->get_range_from_snapshot(t, start, length, snapshot_handle_->handle);
    // TBE may have multiple embeddings in one table padded to max D
    // narrow to the actual shape here before returning
    return t.narrow(1, 0, shape_[1]);
  }

  c10::IntArrayRef size() {
    return shape_;
  }

  c10::ScalarType dtype() {
    return options_.dtype().toScalarType();
  }

 private:
  std::shared_ptr<EmbeddingRocksDB> db_;
  c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper> snapshot_handle_;
  at::TensorOptions options_;
  std::vector<int64_t> shape_;
  int64_t row_offset_;
};

static auto embedding_snapshot_handle_wrapper =
    torch::class_<EmbeddingSnapshotHandleWrapper>(
        "fbgemm",
        "EmbeddingSnapshotHandleWrapper");

static auto embedding_rocks_db_wrapper =
    torch::class_<EmbeddingRocksDBWrapper>("fbgemm", "EmbeddingRocksDBWrapper")
        .def(
            torch::init<
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
                int64_t,
                int64_t,
                bool,
                int64_t,
                int64_t>(),
            "",
            {
                torch::arg("path"),
                torch::arg("num_shards"),
                torch::arg("num_threads"),
                torch::arg("memtable_flush_period"),
                torch::arg("memtable_flush_offset"),
                torch::arg("l0_files_per_compact"),
                torch::arg("max_D"),
                torch::arg("rate_limit_mbps"),
                torch::arg("size_ratio"),
                torch::arg("compaction_ratio"),
                torch::arg("write_buffer_size"),
                torch::arg("max_write_buffer_num"),
                torch::arg("uniform_init_lower"),
                torch::arg("uniform_init_upper"),
                torch::arg("row_storage_bitwidth"),
                torch::arg("cache_size"),
                torch::arg("use_passed_in_path") = true,
                torch::arg("tbe_unique_id") = 0,
                torch::arg("l2_cache_size_gb") = 0,
            })
        .def(
            "set_cuda",
            &EmbeddingRocksDBWrapper::set_cuda,
            "",
            {
                torch::arg("indices"),
                torch::arg("weights"),
                torch::arg("count"),
                torch::arg("timestep"),
                torch::arg("is_bwd") = false,
            })
        .def("get_cuda", &EmbeddingRocksDBWrapper::get_cuda)
        .def("compact", &EmbeddingRocksDBWrapper::compact)
        .def("flush", &EmbeddingRocksDBWrapper::flush)
        .def("get_mem_usage", &EmbeddingRocksDBWrapper::get_mem_usage)
        .def(
            "get_rocksdb_io_duration",
            &EmbeddingRocksDBWrapper::get_rocksdb_io_duration)
        .def("get_l2cache_perf", &EmbeddingRocksDBWrapper::get_l2cache_perf)
        .def("set", &EmbeddingRocksDBWrapper::set)
        .def(
            "get",
            &EmbeddingRocksDBWrapper::get,
            "",
            {
                torch::arg("indices"),
                torch::arg("weights"),
                torch::arg("count"),
                torch::arg("sleep_ms") = 0,
            })
        .def("reset_l2_cache", &EmbeddingRocksDBWrapper::reset_l2_cache)
        .def(
            "wait_util_filling_work_done",
            &EmbeddingRocksDBWrapper::wait_util_filling_work_done)
        .def("create_snapshot", &EmbeddingRocksDBWrapper::create_snapshot)
        .def("release_snapshot", &EmbeddingRocksDBWrapper::release_snapshot)
        .def(
            "get_snapshot_count",
            &EmbeddingRocksDBWrapper::get_snapshot_count);

static auto kv_tensor_wrapper =
    torch::class_<KVTensorWrapper>("fbgemm", "KVTensorWrapper")
        .def(
            torch::init<
                c10::intrusive_ptr<EmbeddingRocksDBWrapper>,
                c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper>,
                std::vector<int64_t>,
                int64_t,
                int64_t>(),
            "",
            {torch::arg("db"),
             torch::arg("snapshot_handle"),
             torch::arg("shape"),
             torch::arg("dtype"),
             torch::arg("row_offset")})
        .def("narrow", &KVTensorWrapper::narrow)
        .def_property(
            "shape",
            &KVTensorWrapper::size,
            std::string(
                "Returns the shape of the original tensor. Only the narrowed part is materialized."));

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "masked_index_put("
      "    Tensor self, "
      "    Tensor indices, "
      "    Tensor values, "
      "    Tensor count, "
      "    bool use_pipeline=False, "
      "    int preferred_sms=-1"
      ") -> Tensor");
  DISPATCH_TO_CUDA("masked_index_put", masked_index_put_cuda);
  m.def(
      "masked_index_select("
      "    Tensor self, "
      "    Tensor indices, "
      "    Tensor values, "
      "    Tensor count, "
      "    bool use_pipeline=False, "
      "    int preferred_sms=-1"
      ") -> Tensor");
  DISPATCH_TO_CUDA("masked_index_select", masked_index_select_cuda);
  m.def(
      "ssd_cache_populate_actions("
      "    Tensor linear_indices, "
      "    int total_hash_size, "
      "    Tensor lxu_cache_state, "
      "    int time_stamp, "
      "    int prefetch_dist, "
      "    Tensor lru_state, "
      "    bool gather_cache_stats=False, "
      "    Tensor? ssd_cache_stats=None, "
      "    bool lock_cache_line=False, "
      "    Tensor? lxu_cache_locking_counter=None"
      ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  DISPATCH_TO_CUDA(
      "ssd_cache_populate_actions", ssd_cache_populate_actions_cuda);
  m.def(
      "ssd_generate_row_addrs("
      "    Tensor lxu_cache_locations, "
      "    Tensor assigned_cache_slots, "
      "    Tensor linear_index_inverse_indices, "
      "    Tensor unique_indices_count_cumsum, "
      "    Tensor cache_set_inverse_indices, "
      "    Tensor lxu_cache_weights, "
      "    Tensor inserted_ssd_weights, "
      "    Tensor unique_indices_length, "
      "    Tensor cache_set_sorted_unique_indices"
      ") -> (Tensor, Tensor)");
  DISPATCH_TO_CUDA("ssd_generate_row_addrs", ssd_generate_row_addrs_cuda);
  m.def(
      "ssd_update_row_addrs("
      "    Tensor ssd_row_addrs_curr, "
      "    Tensor inserted_ssd_weights_curr_next_map, "
      "    Tensor lxu_cache_locations_curr, "
      "    Tensor linear_index_inverse_indices_curr, "
      "    Tensor unique_indices_count_cumsum_curr, "
      "    Tensor cache_set_inverse_indices_curr, "
      "    Tensor lxu_cache_weights, "
      "    Tensor inserted_ssd_weights_next, "
      "    Tensor unique_indices_length_curr"
      ") -> ()");
  DISPATCH_TO_CUDA("ssd_update_row_addrs", ssd_update_row_addrs_cuda);
  m.def(
      "compact_indices("
      "    Tensor[] compact_indices, "
      "    Tensor compact_count, "
      "    Tensor[] indices, "
      "    Tensor masks, "
      "    Tensor count) -> ()");
  DISPATCH_TO_CUDA("compact_indices", compact_indices_cuda);
}
} // namespace
