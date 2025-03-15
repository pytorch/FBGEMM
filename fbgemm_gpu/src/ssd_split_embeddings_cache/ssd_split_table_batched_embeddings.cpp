/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/library.h>

#include <torch/custom_class.h>

#include "./ssd_table_batched_embeddings.h"
#include "embedding_rocksdb_wrapper.h"
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
    const std::optional<Tensor>& lxu_cache_locking_counter);

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

namespace ssd {

SnapshotHandle::SnapshotHandle(EmbeddingRocksDB* db) : db_(db) {
  auto num_shards = db->num_shards();
  CHECK_GT(num_shards, 0);
  shard_snapshots_.reserve(num_shards);
  for (auto shard = 0; shard < num_shards; ++shard) {
    const auto* snapshot = db->dbs_[shard]->GetSnapshot();
    CHECK(snapshot != nullptr)
        << "ERROR: create_snapshot fails to create a snapshot "
        << "for db shard " << shard << ". Please make sure that "
        << "inplace_update_support is set to false" << std::endl;
    shard_snapshots_.push_back(snapshot);
  }
}

SnapshotHandle::~SnapshotHandle() {
  for (auto shard = 0; shard < db_->dbs_.size(); ++shard) {
    snapshot_ptr_t snapshot = shard_snapshots_[shard];
    CHECK(snapshot != nullptr) << "Unexpected nullptr for snapshot " << shard;
    db_->dbs_[shard]->ReleaseSnapshot(snapshot);
  }
}

void SnapshotHandle::release() {
  db_->release_snapshot(this);
}

snapshot_ptr_t SnapshotHandle::get_snapshot_for_shard(size_t shard) const {
  CHECK_LE(shard, shard_snapshots_.size());
  return shard_snapshots_[shard];
}

EmbeddingSnapshotHandleWrapper::EmbeddingSnapshotHandleWrapper(
    const SnapshotHandle* handle,
    std::shared_ptr<EmbeddingRocksDB> db)
    : handle(handle), db(std::move(db)) {}

EmbeddingSnapshotHandleWrapper::~EmbeddingSnapshotHandleWrapper() {
  db->release_snapshot(handle);
}

KVTensorWrapper::KVTensorWrapper(
    c10::intrusive_ptr<EmbeddingRocksDBWrapper> db,
    std::vector<int64_t> shape,
    int64_t dtype,
    int64_t row_offset,
    std::optional<c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper>>
        snapshot_handle)
    : db_(db->impl_), shape_(std::move(shape)), row_offset_(row_offset) {
  CHECK_EQ(shape_.size(), 2) << "Only 2D emb tensors are supported";
  options_ = at::TensorOptions()
                 .dtype(static_cast<c10::ScalarType>(dtype))
                 .device(at::kCPU)
                 .layout(at::kStrided);
  if (snapshot_handle.has_value()) {
    snapshot_handle_ = std::move(snapshot_handle.value());
  }
  // derive strides details assuming contiguous tensor
  strides_ = std::vector<int64_t>(shape_.size(), 1);
  for (auto dim = shape_.size() - 1; dim > 0; --dim) {
    strides_[dim - 1] = strides_[dim] * shape_[dim];
  }
}

at::Tensor KVTensorWrapper::narrow(int64_t dim, int64_t start, int64_t length) {
  CHECK_EQ(dim, 0) << "Only narrow on dim 0 is supported";
  CHECK_GE(db_->get_max_D(), shape_[1]);
  CHECK_TRUE(snapshot_handle_ != nullptr);
  auto t = at::empty(c10::IntArrayRef({length, db_->get_max_D()}), options_);
  db_->get_range_from_snapshot(
      t, start + row_offset_, length, snapshot_handle_->handle);
  // TBE may have multiple embeddings in one table padded to max D
  // narrow to the actual shape here before returning
  return t.narrow(1, 0, shape_[1]);
}

void KVTensorWrapper::set_range(
    int64_t dim,
    const int64_t start,
    const int64_t length,
    const at::Tensor& weights) {
  CHECK_EQ(dim, 0) << "Only set_range on dim 0 is supported";
  CHECK_GE(db_->get_max_D(), shape_[1]);
  int pad_right = db_->get_max_D() - weights.size(1);
  if (pad_right == 0) {
    db_->set_range_to_storage(weights, start + row_offset_, length);
  } else {
    std::vector<int64_t> padding = {0, pad_right, 0, 0};
    auto padded_weights = torch::constant_pad_nd(weights, padding, 0);
    CHECK_EQ(db_->get_max_D(), padded_weights.size(1));
    db_->set_range_to_storage(padded_weights, start + row_offset_, length);
  }
}

c10::IntArrayRef KVTensorWrapper::sizes() {
  return shape_;
}

c10::IntArrayRef KVTensorWrapper::strides() {
  return strides_;
}

c10::ScalarType KVTensorWrapper::dtype() {
  return options_.dtype().toScalarType();
}

std::string_view KVTensorWrapper::dtype_str() {
  return scalarTypeToTypeMeta(dtype()).name();
}

c10::Device KVTensorWrapper::device() {
  return options_.device();
}

std::string KVTensorWrapper::device_str() {
  return device().str();
}

std::string KVTensorWrapper::layout_str() {
  std::ostringstream oss;
  oss << options_.layout();
  return oss.str();
}
} // namespace ssd

namespace {

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
                int64_t,
                bool>(),
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
                torch::arg("enable_async_update") = true,
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
            "set_range_to_storage",
            &EmbeddingRocksDBWrapper::set_range_to_storage)
        .def("toggle_compaction", &EmbeddingRocksDBWrapper::toggle_compaction)
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
                std::vector<int64_t>,
                int64_t,
                int64_t,
                std::optional<
                    c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper>>>(),
            "",
            {torch::arg("db"),
             torch::arg("shape"),
             torch::arg("dtype"),
             torch::arg("row_offset"),
             // snapshot must be provided for reading
             // not needed for writing
             torch::arg("snapshot_handle") = std::nullopt})
        .def(
            "narrow",
            &KVTensorWrapper::narrow,
            "",
            {torch::arg("dim"), torch::arg("start"), torch::arg("length")})
        .def("set_range", &KVTensorWrapper::set_range)
        .def_property("dtype_str", &KVTensorWrapper::dtype_str)
        .def_property("device_str", &KVTensorWrapper::device_str)
        .def_property("layout_str", &KVTensorWrapper::layout_str)
        .def_property(
            "shape",
            &KVTensorWrapper::sizes,
            std::string(
                "Returns the shape of the original tensor. Only the narrowed part is materialized."))
        .def_property("strides", &KVTensorWrapper::strides);

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
