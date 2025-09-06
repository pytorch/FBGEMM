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
#include <nlohmann/json.hpp>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <mutex>
#include "../dram_kv_embedding_cache/dram_kv_embedding_cache_wrapper.h"
#include "./ssd_table_batched_embeddings.h"
#include "embedding_rocksdb_wrapper.h"
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "rocksdb/utilities/checkpoint.h"
using namespace at;
using namespace ssd;
using namespace kv_mem;

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

// Inline method to replace first 8 bytes of weights with linearized_ids
// when backend returns whole row
inline void replace_weights_id(
    at::Tensor& weights,
    const at::Tensor& linearized_ids) {
  // Calculate how many bytes we need to copy
  auto weights_dtype_id_2d =
      linearized_ids.view({-1, 1}).view(weights.dtype().toScalarType());
  auto weights_first_cols = weights.slice(1, 0, weights_dtype_id_2d.size(1));
  weights_first_cols.copy_(weights_dtype_id_2d);
}

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

CheckpointHandle::CheckpointHandle(
    EmbeddingRocksDB* db,
    const std::string& tbe_uuid,
    const std::string& ckpt_uuid,
    const std::string& base_path,
    bool use_default_ssd_path)
    : db_(db), ckpt_uuid_(ckpt_uuid) {
  auto num_shards = db->num_shards();
  CHECK_GT(num_shards, 0);
  shard_checkpoints_.reserve(num_shards);
  for (auto shard = 0; shard < num_shards; ++shard) {
    std::string curr_base_path = db->paths_[shard % db->paths_.size()];
    auto rocksdb_path = kv_db_utils::get_rocksdb_path(
        curr_base_path, shard, tbe_uuid, use_default_ssd_path);
    auto checkpoint_shard_dir =
        kv_db_utils::get_rocksdb_checkpoint_dir(shard, rocksdb_path);
    kv_db_utils::create_dir(checkpoint_shard_dir);
    rocksdb::Checkpoint* checkpoint = nullptr;
    rocksdb::Status s =
        rocksdb::Checkpoint::Create(db->dbs_[shard].get(), &checkpoint);
    CHECK(s.ok()) << "ERROR: Checkpoint init for tbe_uuid " << tbe_uuid
                  << ", db shard " << shard << " failed, " << s.code() << ", "
                  << s.ToString();
    std::string checkpoint_shard_path = checkpoint_shard_dir + "/" + ckpt_uuid_;
    s = checkpoint->CreateCheckpoint(checkpoint_shard_path);
    CHECK(s.ok()) << "ERROR: Checkpoint creation for tbe_uuid " << tbe_uuid
                  << ", db shard " << shard << " failed, " << s.code() << ", "
                  << s.ToString();
    shard_checkpoints_.push_back(checkpoint_shard_path);
  }

  XLOG(INFO) << "rocksdb checkpoint shards paths: " << shard_checkpoints_;
}

std::vector<std::string> CheckpointHandle::get_shard_checkpoints() const {
  return shard_checkpoints_;
}

EmbeddingSnapshotHandleWrapper::EmbeddingSnapshotHandleWrapper(
    const SnapshotHandle* handle,
    std::shared_ptr<EmbeddingRocksDB> db)
    : handle(handle), db(std::move(db)) {}

EmbeddingSnapshotHandleWrapper::~EmbeddingSnapshotHandleWrapper() {
  db->release_snapshot(handle);
}

RocksdbCheckpointHandleWrapper::RocksdbCheckpointHandleWrapper(
    const std::string& checkpoint_uuid,
    std::shared_ptr<EmbeddingRocksDB> db)
    : uuid(checkpoint_uuid), db(std::move(db)) {}

// do not release uuid when RocksdbCheckpointHandleWrapper is destroyed
// subsequent get_active_checkpoint_uuid() calls need to retrieve
// the checkpoint uuid
// RocksdbCheckpointHandleWrapper::~RocksdbCheckpointHandleWrapper() {
//   db->release_checkpoint(uuid);
// }

KVTensorWrapper::KVTensorWrapper(
    std::vector<int64_t> shape,
    int64_t dtype,
    int64_t row_offset,
    const std::optional<c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper>>
        snapshot_handle,
    std::optional<at::Tensor> sorted_indices,
    int64_t width_offset_,
    const std::optional<c10::intrusive_ptr<RocksdbCheckpointHandleWrapper>>
        checkpoint_handle,
    bool read_only)
    : db_(nullptr),
      shape_(std::move(shape)),
      row_offset_(row_offset),
      width_offset_(width_offset_),
      read_only_(read_only) {
  CHECK_GE(width_offset_, 0);
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

  if (sorted_indices.has_value()) {
    sorted_indices_ = sorted_indices;
  }
  if (checkpoint_handle.has_value()) {
    checkpoint_handle_ = checkpoint_handle.value();
  }
}

std::string KVTensorWrapper::serialize() const {
  // auto call to_json()
  ssd::json json_serialized = *this;
  return json_serialized.dump();
}

std::vector<std::string> KVTensorWrapper::get_kvtensor_serializable_metadata()
    const {
  std::vector<std::string> metadata;
  auto* db = dynamic_cast<EmbeddingRocksDB*>(db_.get());
  auto checkpoint_paths = db->get_checkpoints(checkpoint_handle_->uuid);
  metadata.push_back(std::to_string(checkpoint_paths.size()));
  for (const auto& path : checkpoint_paths) {
    metadata.push_back(path);
  }
  metadata.push_back(db->get_tbe_uuid());
  metadata.push_back(std::to_string(db->num_shards()));
  metadata.push_back(std::to_string(db->num_threads()));
  metadata.push_back(std::to_string(db->get_max_D()));
  metadata.push_back(std::to_string(row_offset_));
  CHECK_EQ(shape_.size(), 2);
  metadata.push_back(std::to_string(shape_[0]));
  metadata.push_back(std::to_string(shape_[1]));
  metadata.push_back(
      std::to_string(static_cast<int64_t>(options_.dtype().toScalarType())));
  metadata.push_back(checkpoint_handle_->uuid);
  return metadata;
}

std::string KVTensorWrapper::logs() const {
  std::stringstream ss;
  if (db_) {
    CHECK(readonly_db_ == nullptr) << "rdb logs, ro_rdb must be nullptr";
    ss << "from ckpt paths: " << std::endl;
    // Required to cast as the KVTensorWrapper.db_ is a pointer for the
    // EmbeddingKVDB class which is inherited by the EmbeddingRocksDB class
    auto* db = dynamic_cast<EmbeddingRocksDB*>(db_.get());
    auto ckpts = db->get_checkpoints(checkpoint_handle_->uuid);
    for (int i = 0; i < ckpts.size(); i++) {
      ss << "  shard:" << i << ", ckpt_path:" << ckpts[i] << std::endl;
    }
    ss << "  tbe_uuid: " << db->get_tbe_uuid() << std::endl;
    ss << "  num_shards: " << db->num_shards() << std::endl;
    ss << "  num_threads: " << db->num_threads() << std::endl;
    ss << "  max_D: " << db->get_max_D() << std::endl;
    ss << "  row_offset: " << row_offset_ << std::endl;
    ss << "  shape: " << shape_ << std::endl;
    ss << "  dtype: " << static_cast<int64_t>(options_.dtype().toScalarType())
       << std::endl;
    ss << "  checkpoint_uuid: " << checkpoint_handle_->uuid << std::endl;
  } else {
    CHECK(readonly_db_) << "ro_rdb logs, ro_rdb must be valid";
    ss << "from ckpt paths: " << std::endl;
    auto* db = dynamic_cast<ReadOnlyEmbeddingKVDB*>(readonly_db_.get());
    auto rdb_shard_checkpoint_paths = db->get_rdb_shard_checkpoint_paths();
    for (int i = 0; i < rdb_shard_checkpoint_paths.size(); i++) {
      ss << "  shard:" << i << ", ckpt_path:" << rdb_shard_checkpoint_paths[i]
         << std::endl;
    }
    ss << "  tbe_uuid: " << db->get_tbe_uuid() << std::endl;
    ss << "  num_shards: " << db->num_shards() << std::endl;
    ss << "  num_threads: " << db->num_threads() << std::endl;
    ss << "  max_D: " << db->get_max_D() << std::endl;
    ss << "  row_offset: " << row_offset_ << std::endl;
    ss << "  shape: " << shape_ << std::endl;
    ss << "  dtype: " << static_cast<int64_t>(options_.dtype().toScalarType())
       << std::endl;
    ss << "  checkpoint_uuid: " << checkpoint_uuid << std::endl;
  }
  return ss.str();
}

void KVTensorWrapper::deserialize(const std::string& serialized) {
  ssd::json json_serialized = ssd::json::parse(serialized);
  from_json(json_serialized, *this);
}

KVTensorWrapper::KVTensorWrapper(const std::string& serialized) {
  deserialize(serialized);
}

void KVTensorWrapper::set_embedding_rocks_dp_wrapper(
    c10::intrusive_ptr<EmbeddingRocksDBWrapper> db) {
  db_ = db->impl_;
}

void KVTensorWrapper::set_dram_db_wrapper(
    c10::intrusive_ptr<kv_mem::DramKVEmbeddingCacheWrapper> db) {
  db_ = db->impl_;
}

at::Tensor KVTensorWrapper::narrow(int64_t dim, int64_t start, int64_t length) {
  CHECK_EQ(dim, 0) << "Only narrow on dim 0 is supported";
  if (db_) {
    CHECK_TRUE(db_ != nullptr);
    CHECK_GE(
        db_->get_max_D() + db_->get_metaheader_width_in_front(), shape_[1]);
    TORCH_CHECK(
        (snapshot_handle_ == nullptr) ==
            (std::dynamic_pointer_cast<EmbeddingRocksDB>(db_).get() == nullptr),
        "snapshot handler must be valid for rocksdb and nullptr for emb kvdb");
    if (!sorted_indices_.has_value()) {
      auto t = at::empty(c10::IntArrayRef({length, shape_[1]}), options_);
      db_->get_range_from_snapshot(
          t,
          start + row_offset_,
          length,
          snapshot_handle_ != nullptr ? snapshot_handle_->handle : nullptr,
          width_offset_,
          shape_[1]);
      CHECK(t.is_contiguous());
      return t;
    } else {
      at::Tensor sliced_ids =
          sorted_indices_.value().slice(0, start, start + length);
      auto weights = get_weights_by_ids(sliced_ids);

      if (db_->get_backend_return_whole_row() && width_offset_ == 0) {
        // backend returns whole row, so we need to replace the first 8 bytes
        // with the sliced_ids
        replace_weights_id(weights, sliced_ids);
      }
      return weights;
    }
  } else {
    CHECK(readonly_db_)
        << "ReadOnlyEmbeddingKVDB pointer must be valid to read tensor";
    CHECK_GE(readonly_db_->get_max_D(), shape_[1]);
    CHECK_EQ(width_offset_, 0)
        << "Width offset must be 0 for ro_rdb becuase the functionality is not supported yet";
    auto t = at::empty(c10::IntArrayRef({length, shape_[1]}), options_);
    readonly_db_->get_range_from_rdb_checkpoint(
        t, start + row_offset_, length, width_offset_);
    // TBE may have multiple embeddings in one table padded to max D
    // narrow to the actual shape here before returning
    return t.narrow(1, 0, shape_[1]);
  }
}

void KVTensorWrapper::set_range(
    int64_t dim,
    const int64_t start,
    const int64_t length,
    at::Tensor& weights) {
  if (read_only_) {
    XLOG(INFO) << "KVTensorWrapper is read only, set_range() is no-op";
    return;
  }
  // Mutex lock for disabling concurrent writes to the same KVTensor
  std::lock_guard<std::mutex> lock(mtx);
  CHECK_EQ(weights.device(), at::kCPU);
  CHECK(db_) << "EmbeddingRocksDB must be a valid pointer to call set_range";
  CHECK_EQ(dim, 0) << "Only set_range on dim 0 is supported";
  CHECK_TRUE(db_ != nullptr);
  CHECK_GE(db_->get_max_D() + db_->get_metaheader_width_in_front(), shape_[1]);

  if (db_->get_backend_return_whole_row()) {
    // backend returns whole row, so we need to replace the first 8 bytes with
    // the sliced_ids
    TORCH_CHECK(
        sorted_indices_.has_value(),
        "sorted_indices_ must be valid to set range when backend returns whole row");
    at::Tensor sliced_ids =
        sorted_indices_.value().slice(0, start, start + length);
    auto linearized_ids = sliced_ids + row_offset_;
    replace_weights_id(weights, linearized_ids);
  }

  int pad_right =
      db_->get_max_D() + db_->get_metaheader_width_in_front() - weights.size(1);
  if (pad_right == 0) {
    db_->set_range_to_storage(weights, start + row_offset_, length);
  } else {
    std::vector<int64_t> padding = {0, pad_right, 0, 0};
    auto padded_weights = torch::constant_pad_nd(weights, padding, 0);
    CHECK_EQ(
        db_->get_max_D() + db_->get_metaheader_width_in_front(),
        padded_weights.size(1));
    db_->set_range_to_storage(padded_weights, start + row_offset_, length);
  }
}

void KVTensorWrapper::set_weights_and_ids(
    const at::Tensor& weights,
    const at::Tensor& ids) {
  if (read_only_) {
    XLOG(INFO)
        << "KVTensorWrapper is read only, set_weights_and_ids() is no-op";
    return;
  }
  CHECK_EQ(weights.device(), at::kCPU);
  CHECK_TRUE(db_ != nullptr);
  CHECK_EQ(ids.size(0), weights.size(0))
      << "ids and weights must have same # rows";
  CHECK_GE(db_->get_max_D() + db_->get_metaheader_width_in_front(), shape_[1]);
  auto linearized_ids = ids + row_offset_;
  int pad_right =
      db_->get_max_D() + db_->get_metaheader_width_in_front() - weights.size(1);
  if (pad_right == 0) {
    db_->set_kv_to_storage(linearized_ids, weights);
  } else {
    std::vector<int64_t> padding = {0, pad_right, 0, 0};
    auto padded_weights = torch::constant_pad_nd(weights, padding, 0);
    CHECK_EQ(
        db_->get_max_D() + db_->get_metaheader_width_in_front(),
        padded_weights.size(1));
    db_->set_kv_to_storage(linearized_ids, padded_weights);
  }
}

void to_json(ssd::json& j, const KVTensorWrapper& kvt) {
  // Required to cast as the KVTensorWrapper.db_ is a pointer for the
  // EmbeddingKVDB class which is inherited by the EmbeddingRocksDB class
  std::shared_ptr<EmbeddingRocksDB> db =
      std::dynamic_pointer_cast<EmbeddingRocksDB>(kvt.db_);
  j = ssd::json{
      {"rdb_shard_checkpoint_paths",
       db->get_checkpoints(kvt.checkpoint_handle_->uuid)},
      {"tbe_uuid", db->get_tbe_uuid()},
      {"num_shards", db->num_shards()},
      {"num_threads", db->num_threads()},
      {"max_D", db->get_max_D()},
      {"row_offset", kvt.row_offset_},
      {"shape", kvt.shape_},
      {"dtype", static_cast<int64_t>(kvt.options_.dtype().toScalarType())},
      {"checkpoint_uuid", kvt.checkpoint_handle_->uuid},
      {"width_offset", kvt.width_offset_}};
}

void from_json(const ssd::json& j, KVTensorWrapper& kvt) {
  std::vector<std::string> rdb_shard_checkpoint_paths;
  std::string tbe_uuid;
  int64_t num_shards;
  int64_t num_threads;
  int64_t max_D;
  int64_t dtype;
  int64_t width_offset;
  j.at("rdb_shard_checkpoint_paths").get_to(rdb_shard_checkpoint_paths);
  j.at("tbe_uuid").get_to(tbe_uuid);
  j.at("num_shards").get_to(num_shards);
  j.at("num_threads").get_to(num_threads);
  j.at("max_D").get_to(max_D);
  j.at("dtype").get_to(dtype);
  j.at("width_offset").get_to(width_offset);

  // initialize ro rdb during KV tensor deserialization
  // one rdb checkpoint is related to # tables of KVT, this way each KVT will
  // hold their own rdb instance link to the same checkpoint during
  // destruction, they will delete the same checkpoint, but since ckpt path
  // has been opened during ro rdb init, OS will not delete the file until all
  // file handles are closed
  kvt.readonly_db_ = std::make_shared<ReadOnlyEmbeddingKVDB>(
      rdb_shard_checkpoint_paths, tbe_uuid, num_shards, num_threads, max_D);
  j.at("checkpoint_uuid").get_to(kvt.checkpoint_uuid);
  j.at("row_offset").get_to(kvt.row_offset_);
  j.at("shape").get_to(kvt.shape_);
  j.at("width_offset").get_to(kvt.width_offset_);
  kvt.options_ = at::TensorOptions()
                     .dtype(static_cast<at::ScalarType>(dtype))
                     .device(at::kCPU)
                     .layout(at::kStrided);
}

at::Tensor KVTensorWrapper::get_weights_by_ids(const at::Tensor& ids) {
  CHECK_TRUE(db_ != nullptr);
  CHECK_GE(db_->get_max_D() + db_->get_metaheader_width_in_front(), shape_[1]);
  CHECK_GE(
      db_->get_max_D() + db_->get_metaheader_width_in_front(),
      shape_[1] + width_offset_);
  TORCH_CHECK(
      (snapshot_handle_ == nullptr) ==
          (std::dynamic_pointer_cast<EmbeddingRocksDB>(db_).get() == nullptr),
      "snapshot handler must be valid for rocksdb and nullptr for emb kvdb");
  auto weights =
      at::empty(c10::IntArrayRef({ids.size(0), shape_[1]}), options_);
  auto linearized_ids = ids + row_offset_;
  db_->get_kv_from_storage_by_snapshot(
      linearized_ids,
      weights,
      snapshot_handle_ != nullptr ? snapshot_handle_->handle : nullptr,
      width_offset_,
      shape_[1]);
  CHECK(weights.is_contiguous());
  return weights;
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

static auto feature_evict_config =
    torch::class_<kv_mem::FeatureEvictConfig>("fbgemm", "FeatureEvictConfig")
        .def(
            torch::init<
                int64_t,
                int64_t,
                std::optional<int64_t>,
                std::optional<int64_t>,
                std::optional<std::vector<int64_t>>,
                std::optional<std::vector<int64_t>>,
                std::optional<std::vector<double>>,
                std::optional<std::vector<double>>,
                std::optional<std::vector<int64_t>>,
                std::optional<std::vector<int64_t>>,
                std::optional<std::vector<double>>,
                std::optional<std::vector<int64_t>>,
                std::optional<double>,
                std::optional<int64_t>,
                int64_t,
                int64_t,
                int64_t>(),
            "",
            {
                torch::arg("trigger_mode"),
                torch::arg("trigger_strategy"),
                torch::arg("trigger_step_interval") = std::nullopt,
                torch::arg("mem_util_threshold_in_GB") = std::nullopt,
                torch::arg("ttls_in_mins") = std::nullopt,
                torch::arg("counter_thresholds") = std::nullopt,
                torch::arg("counter_decay_rates") = std::nullopt,
                torch::arg("feature_score_counter_decay_rates") = std::nullopt,
                torch::arg("training_id_eviction_trigger_count") = std::nullopt,
                torch::arg("training_id_keep_count") = std::nullopt,
                torch::arg("l2_weight_thresholds") = std::nullopt,
                torch::arg("embedding_dims") = std::nullopt,
                torch::arg("threshold_calculation_bucket_stride") = 0.2,
                torch::arg("threshold_calculation_bucket_num") = 1000000,
                torch::arg("interval_for_insufficient_eviction_s") = 600,
                torch::arg("interval_for_sufficient_eviction_s") = 60,
                torch::arg("interval_for_feature_statistics_decay_s") =
                    24 * 3600,
            });

static auto embedding_snapshot_handle_wrapper =
    torch::class_<EmbeddingSnapshotHandleWrapper>(
        "fbgemm",
        "EmbeddingSnapshotHandleWrapper");

static auto rocksdb_checkpoint_handle_wrapper =
    torch::class_<RocksdbCheckpointHandleWrapper>(
        "fbgemm",
        "RocksdbCheckpointHandleWrapper");

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
                bool,
                bool,
                int64_t,
                int64_t,
                std::vector<std::string>,
                std::vector<int64_t>,
                std::vector<int64_t>,
                std::optional<at::Tensor>,
                std::optional<at::Tensor>,
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
                torch::arg("enable_raw_embedding_streaming") = false,
                torch::arg("res_store_shards") = 0,
                torch::arg("res_server_port") = 0,
                torch::arg("table_names") = torch::List<std::string>(),
                torch::arg("table_offsets") = torch::List<int64_t>(),
                torch::arg("table_sizes") = torch::List<int64_t>(),
                torch::arg("table_dims") = std::nullopt,
                torch::arg("hash_size_cumsum") = std::nullopt,
                torch::arg("flushing_block_size") = 2000000000 /* 2GB */,
                torch::arg("disable_random_init") = false,
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
        .def(
            "set_feature_score_metadata_cuda",
            &EmbeddingRocksDBWrapper::set_feature_score_metadata_cuda,
            "",
            {
                torch::arg("indices"),
                torch::arg("count"),
                torch::arg("engage_rates"),
            })
        .def(
            "stream_cuda",
            &EmbeddingRocksDBWrapper::stream_cuda,
            "",
            {
                torch::arg("indices"),
                torch::arg("weights"),
                torch::arg("count"),
                torch::arg("blocking_tensor_copy"),
            })
        .def(
            "set_backend_return_whole_row",
            &EmbeddingRocksDBWrapper::set_backend_return_whole_row,
            "",
            {
                torch::arg("backend_return_whole_row"),
            })
        .def("stream_sync_cuda", &EmbeddingRocksDBWrapper::stream_sync_cuda)
        .def("get_cuda", &EmbeddingRocksDBWrapper::get_cuda)
        .def("compact", &EmbeddingRocksDBWrapper::compact)
        .def("flush", &EmbeddingRocksDBWrapper::flush)
        .def("get_mem_usage", &EmbeddingRocksDBWrapper::get_mem_usage)
        .def(
            "get_rocksdb_io_duration",
            &EmbeddingRocksDBWrapper::get_rocksdb_io_duration)
        .def("get_l2cache_perf", &EmbeddingRocksDBWrapper::get_l2cache_perf)
        .def("set", &EmbeddingRocksDBWrapper::set)
        .def("set_kv_to_storage", &EmbeddingRocksDBWrapper::set_kv_to_storage)
        .def(
            "set_range_to_storage",
            &EmbeddingRocksDBWrapper::set_range_to_storage)
        .def("toggle_compaction", &EmbeddingRocksDBWrapper::toggle_compaction)
        .def(
            "is_auto_compaction_enabled",
            &EmbeddingRocksDBWrapper::is_auto_compaction_enabled)
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
        .def("get_snapshot_count", &EmbeddingRocksDBWrapper::get_snapshot_count)
        .def(
            "get_keys_in_range_by_snapshot",
            &EmbeddingRocksDBWrapper::get_keys_in_range_by_snapshot)
        .def(
            "get_kv_zch_eviction_metadata_by_snapshot",
            &EmbeddingRocksDBWrapper::get_kv_zch_eviction_metadata_by_snapshot)
        .def(
            "create_rocksdb_hard_link_snapshot",
            &EmbeddingRocksDBWrapper::create_rocksdb_hard_link_snapshot)
        .def(
            "get_active_checkpoint_uuid",
            &EmbeddingRocksDBWrapper::get_active_checkpoint_uuid);

static auto dram_kv_embedding_cache_wrapper =
    torch::class_<DramKVEmbeddingCacheWrapper>(
        "fbgemm",
        "DramKVEmbeddingCacheWrapper")
        .def(
            torch::init<
                int64_t,
                double,
                double,
                std::optional<c10::intrusive_ptr<kv_mem::FeatureEvictConfig>>,
                int64_t,
                int64_t,
                int64_t,
                std::optional<at::Tensor>,
                std::optional<at::Tensor>,
                bool,
                bool,
                bool>(),
            "",
            {
                torch::arg("max_D"),
                torch::arg("uniform_init_lower"),
                torch::arg("uniform_init_upper"),
                torch::arg("feature_evict_config") = std::nullopt,
                torch::arg("num_shards") = 8,
                torch::arg("num_threads") = 32,
                torch::arg("row_storage_bitwidth") = 32,
                torch::arg("table_dims") = std::nullopt,
                torch::arg("hash_size_cumsum") = std::nullopt,
                torch::arg("backend_return_whole_row") = false,
                torch::arg("enable_async_update") = false,
                torch::arg("disable_random_init") = false,
            })
        .def(
            "set_cuda",
            &DramKVEmbeddingCacheWrapper::set_cuda,
            "",
            {
                torch::arg("indices"),
                torch::arg("weights"),
                torch::arg("count"),
                torch::arg("timestep"),
                torch::arg("is_bwd") = false,
            })
        .def("get_cuda", &DramKVEmbeddingCacheWrapper::get_cuda)
        .def(
            "set_backend_return_whole_row",
            &DramKVEmbeddingCacheWrapper::set_backend_return_whole_row,
            "",
            {
                torch::arg("backend_return_whole_row"),
            })
        .def("set", &DramKVEmbeddingCacheWrapper::set)
        .def(
            "set_range_to_storage",
            &DramKVEmbeddingCacheWrapper::set_range_to_storage)
        .def(
            "get",
            &DramKVEmbeddingCacheWrapper::get,
            "",
            {
                torch::arg("indices"),
                torch::arg("weights"),
                torch::arg("count"),
                torch::arg("sleep_ms") = 0,
            })
        .def(
            "wait_util_filling_work_done",
            &DramKVEmbeddingCacheWrapper::wait_util_filling_work_done)
        .def(
            "wait_until_eviction_done",
            &DramKVEmbeddingCacheWrapper::wait_until_eviction_done)
        .def(
            "get_keys_in_range",
            &DramKVEmbeddingCacheWrapper::get_keys_in_range,
            "",
            {
                torch::arg("start"),
                torch::arg("end"),
            })
        .def(
            "set_feature_score_metadata_cuda",
            &DramKVEmbeddingCacheWrapper::set_feature_score_metadata_cuda,
            "",
            {
                torch::arg("indices"),
                torch::arg("count"),
                torch::arg("engage_rates"),
            })
        .def("flush", &DramKVEmbeddingCacheWrapper::flush)
        .def(
            "get_keys_in_range_by_snapshot",
            &DramKVEmbeddingCacheWrapper::get_keys_in_range_by_snapshot)
        .def(
            "get_kv_zch_eviction_metadata_by_snapshot",
            &DramKVEmbeddingCacheWrapper::
                get_kv_zch_eviction_metadata_by_snapshot)
        .def(
            "get_feature_evict_metric",
            &DramKVEmbeddingCacheWrapper::get_feature_evict_metric)
        .def(
            "get_dram_kv_perf",
            &DramKVEmbeddingCacheWrapper::get_dram_kv_perf);
static auto embedding_rocks_db_read_only_wrapper =
    torch::class_<ReadOnlyEmbeddingKVDB>("fbgemm", "ReadOnlyEmbeddingKVDB")
        .def(
            torch::init<
                std::vector<std::string>,
                std::string,
                int64_t,
                int64_t,
                int64_t,
                int64_t>(),
            "",
            {torch::arg("rdb_shard_checkpoint_paths"),
             torch::arg("tbe_uuid"),
             torch::arg("num_shards"),
             torch::arg("num_threads"),
             torch::arg("max_D"),
             torch::arg("cache_size") = 0})
        .def(
            "get_range_from_rdb_checkpoint",
            &ReadOnlyEmbeddingKVDB::get_range_from_rdb_checkpoint)
        .def(
            "delete_rocksdb_checkpoint_dir",
            &ReadOnlyEmbeddingKVDB::delete_rocksdb_checkpoint_dir);

static auto kv_tensor_wrapper =
    torch::class_<KVTensorWrapper>("fbgemm", "KVTensorWrapper")
        .def(
            torch::init<
                std::vector<int64_t>,
                int64_t,
                int64_t,
                std::optional<
                    c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper>>,
                std::optional<at::Tensor>,
                int64_t,
                std::optional<
                    c10::intrusive_ptr<RocksdbCheckpointHandleWrapper>>,
                bool>(),
            "",
            {torch::arg("shape"),
             torch::arg("dtype"),
             torch::arg("row_offset"),
             // snapshot must be provided for reading
             // not needed for writing
             torch::arg("snapshot_handle") = std::nullopt,
             torch::arg("sorted_indices") = std::nullopt,
             torch::arg("width_offset") = 0,
             torch::arg("checkpoint_handle") = std::nullopt,
             torch::arg("read_only") = false})
        .def(
            "set_embedding_rocks_dp_wrapper",
            &KVTensorWrapper::set_embedding_rocks_dp_wrapper,
            "",
            {torch::arg("db")})
        .def(
            "set_dram_db_wrapper",
            &KVTensorWrapper::set_dram_db_wrapper,
            "",
            {torch::arg("db")})
        .def(
            "narrow",
            &KVTensorWrapper::narrow,
            "",
            {torch::arg("dim"), torch::arg("start"), torch::arg("length")})
        .def("set_range", &KVTensorWrapper::set_range)
        .def("set_weights_and_ids", &KVTensorWrapper::set_weights_and_ids)
        .def("get_weights_by_ids", &KVTensorWrapper::get_weights_by_ids)
        .def_property("dtype_str", &KVTensorWrapper::dtype_str)
        .def_property("device_str", &KVTensorWrapper::device_str)
        .def_property("layout_str", &KVTensorWrapper::layout_str)
        .def_property(
            "shape",
            &KVTensorWrapper::sizes,
            std::string(
                "Returns the shape of the original tensor. Only the narrowed part is materialized."))
        .def_property("strides", &KVTensorWrapper::strides)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<KVTensorWrapper>& self) -> std::string {
              return self->serialize();
            },
            // __setstate__
            [](std::string data) -> c10::intrusive_ptr<KVTensorWrapper> {
              return c10::make_intrusive<KVTensorWrapper>(data);
            })
        .def("logs", &KVTensorWrapper::logs, "")
        .def(
            "get_kvtensor_serializable_metadata",
            &KVTensorWrapper::get_kvtensor_serializable_metadata);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "get_bucket_sorted_indices_and_bucket_tensor("
      "    Tensor unordered_indices,"
      "    int hash_mode,"
      "    int bucket_start,"
      "    int bucket_end,"
      "    int? bucket_size=None,"
      "    int? total_num_buckets=None"
      ") -> (Tensor, Tensor)");
  DISPATCH_TO_CPU(
      "get_bucket_sorted_indices_and_bucket_tensor",
      kv_db_utils::get_bucket_sorted_indices_and_bucket_tensor);
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
