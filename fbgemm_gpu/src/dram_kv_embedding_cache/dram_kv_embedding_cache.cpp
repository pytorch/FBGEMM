/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_cache.h"

#include <caffe2/torch/fb/distributed/wireSerializer/WireSerializer.h>
#include <common/base/Proc.h>

#include <folly/SocketAddress.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>

#include <folly/executors/FunctionScheduler.h>

#include <servicerouter/client/cpp2/ServiceRouter.h>

#include <torch/script.h>
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "feature_evict_util.h"
#include "fixed_block_pool.h"

namespace kv_mem {

template <typename weight_type>
DramKVEmbeddingCache<weight_type>::DramKVEmbeddingCache(
    int64_t max_D,
    double uniform_init_lower,
    double uniform_init_upper,
    int64_t evict_trigger_mode,
    int64_t trigger_step_interval,
    int64_t mem_util_threshold_in_GB,
    int64_t evict_trigger_strategy,
    const std::optional<at::Tensor>& counter_thresholds,
    const std::optional<at::Tensor>& ttls_in_mins,
    const std::optional<at::Tensor>& counter_decay_rates,
    const std::optional<at::Tensor>& l2_weight_thresholds,
    int64_t num_shards,
    int64_t num_threads,
    int64_t row_storage_bitwidth,
    bool enable_async_update,
    std::optional<at::Tensor> table_dims,
    std::optional<at::Tensor> hash_size_cumsum)
    : DramKVEmbeddingBase(
          num_shards,
          max_D,
          0, // l2_cache_size_gb =0 to disable l2 cache
          0, // tbe_unqiue_id
          2, // ele_size_bytes
          enable_async_update),
      max_D_(max_D),
      num_shards_(num_shards),
      block_size_(FixedBlockPool::calculate_block_size<weight_type>(max_D)),
      block_alignment_(
          FixedBlockPool::calculate_block_alignment<weight_type>()),
      kv_store_(SynchronizedShardedMap<int64_t, weight_type*>(
          num_shards_,
          block_size_,
          block_alignment_,
          /*blocks_per_chunk=*/8192)),
      elem_size_(row_storage_bitwidth / 8) {
  executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(
      std::max<size_t>(num_threads, facebook::Proc::getCpuInfo().numCpuCores));
  initialize_initializers(
      num_shards,
      max_D,
      uniform_init_lower,
      uniform_init_upper,
      row_storage_bitwidth);
  if (table_dims.has_value()) {
    TORCH_CHECK_TENSOR_PROPERTIES(table_dims, at::ScalarType::Long);
    TORCH_CHECK_TENSOR_PROPERTIES(hash_size_cumsum, at::ScalarType::Long);
    TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
        table_dims, sub_table_dims_, int64_t);
    sub_table_hash_cumsum_.assign(
        hash_size_cumsum->data_ptr<int64_t>() + 1, // skip the first 0
        hash_size_cumsum->data_ptr<int64_t>() + hash_size_cumsum->numel());
  }

  // feature evict config
  feature_evict_config_.trigger_mode =
      static_cast<EvictTriggerMode>(evict_trigger_mode);
  if (feature_evict_config_.trigger_mode == EvictTriggerMode::DISABLED) {
    return;
  }
  // feature evict must need hash_size_cumsum!
  // only support mutli table config now.
  TORCH_CHECK(hash_size_cumsum.has_value());
  feature_evict_config_.trigger_strategy =
      static_cast<EvictTriggerStrategy>(evict_trigger_strategy);
  feature_evict_config_.trigger_step_interval = trigger_step_interval;
  feature_evict_config_.mem_util_threshold_in_GB = mem_util_threshold_in_GB;

  if (feature_evict_config_.trigger_strategy ==
          EvictTriggerStrategy::BY_COUNTER ||
      feature_evict_config_.trigger_strategy ==
          EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER) {
    TORCH_CHECK_TENSOR_PROPERTIES(counter_thresholds, at::ScalarType::UInt32);
    TORCH_CHECK_TENSOR_PROPERTIES(counter_decay_rates, at::ScalarType::Float);
    TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
        counter_thresholds, feature_evict_config_.counter_thresholds, uint32_t);
    TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
        counter_decay_rates, feature_evict_config_.counter_decay_rates, float);
  }

  if (feature_evict_config_.trigger_strategy ==
          EvictTriggerStrategy::BY_TIMESTAMP ||
      feature_evict_config_.trigger_strategy ==
          EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER) {
    TORCH_CHECK_TENSOR_PROPERTIES(ttls_in_mins, at::ScalarType::UInt32);
    TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
        ttls_in_mins, feature_evict_config_.ttls_in_mins, uint32_t);
  }

  if (feature_evict_config_.trigger_strategy ==
      EvictTriggerStrategy::BY_L2WEIGHT) {
    TORCH_CHECK_TENSOR_PROPERTIES(l2_weight_thresholds, at::ScalarType::Double);
    TORCH_CHECK_TENSOR_PROPERTIES(table_dims, at::ScalarType::Long);
    TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
        l2_weight_thresholds,
        feature_evict_config_.l2_weight_thresholds,
        double);
    TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
        table_dims, feature_evict_config_.embedding_dims, int64_t);
  }
  feature_evict_ = create_feature_evict(
      feature_evict_config_,
      executor_.get(),
      kv_store_,
      sub_table_hash_cumsum_);
}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::initialize_initializers(
    int64_t num_shards,
    int64_t max_D,
    double uniform_init_lower,
    double uniform_init_upper,
    int64_t row_storage_bitwidth) {
  for (auto i = 0; i < num_shards; ++i) {
    auto* gen = at::check_generator<at::CPUGeneratorImpl>(
        at::detail::getDefaultCPUGenerator());
    {
      std::lock_guard<std::mutex> lock(gen->mutex_);
      initializers_.push_back(std::make_unique<ssd ::Initializer>(
          gen->random64(),
          max_D,
          uniform_init_lower,
          uniform_init_upper,
          row_storage_bitwidth));
    }
  }
}

/// get all ids in the kvstore
///
/// @return a Tensor contained ids
template <typename weight_type>
at::Tensor DramKVEmbeddingCache<weight_type>::get_keys_in_range_impl(
    int64_t start,
    int64_t end,
    std::optional<int64_t> offset) {
  std::vector<std::vector<int64_t>> ids;
  ids.reserve(num_shards_);
  for (int i = 0; i < num_shards_; i++) {
    ids.emplace_back();
  }
  std::vector<folly::Future<folly::Unit>> futures;
  for (int shard_id = 0; shard_id < num_shards_; shard_id++) {
    auto f =
        folly::via(executor_.get())
            .thenValue([this, shard_id, start, end, offset, &ids](folly::Unit) {
              auto rlmap = kv_store_.by(shard_id).rlock();
              for (auto iter = rlmap->begin(); iter != rlmap->end(); iter++) {
                if (iter->first >= start && iter->first < end) {
                  if (offset.has_value()) {
                    ids[shard_id].push_back(iter->first - offset.value());
                  } else {
                    ids[shard_id].push_back(iter->first);
                  }
                }
              }
            });
    futures.push_back(std::move(f));
  }
  folly::collect(futures).get();

  auto all_ids_ptr = std::make_shared<std::vector<int64_t>>();
  for (auto& sub : ids) {
    all_ids_ptr->insert(
        all_ids_ptr->end(),
        std::make_move_iterator(sub.begin()),
        std::make_move_iterator(sub.end()));
  }

  return torch::from_blob(
             all_ids_ptr->data(),
             {int64_t(all_ids_ptr->size())},
             [all_ids_ptr](void* p) mutable { all_ids_ptr.reset(); },
             torch::kInt64 // data type
             )
      .view({-1, 1});
}

/// insert embeddings into kvstore.
/// current underlying memory management is done through F14FastMap
/// key value pair will be sharded into multiple shards to increase
/// parallelism.
///
/// @param indices The 1D embedding index tensor, should skip on negative
/// value
/// @param weights The 2D tensor that each row(embeddings) is paired up with
/// relative element in <indices>
/// @param count A single element tensor that contains the number of indices
/// to be processed
///
/// @return None
template <typename weight_type>
folly::SemiFuture<std::vector<folly::Unit>>
DramKVEmbeddingCache<weight_type>::set_kv_db_async(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    const kv_db::RocksdbWriteMode /*unused*/) {
  if (feature_evict_config_.trigger_mode != EvictTriggerMode::DISABLED) {
    feature_evict_pause();
  }
  std::vector<folly::Future<folly::Unit>> futures;
  auto shardid_to_indexes = shard_input(indices, count);
  for (auto iter = shardid_to_indexes.begin(); iter != shardid_to_indexes.end();
       iter++) {
    const auto shard_id = iter->first;
    const auto indexes = iter->second;
    auto f =
        folly::via(executor_.get())
            .thenValue(
                [this, shard_id, indexes, &indices, &weights](folly::Unit) {
                  FBGEMM_DISPATCH_INTEGRAL_TYPES(
                      indices.scalar_type(),
                      "dram_kv_set",
                      [this, shard_id, indexes, &indices, &weights] {
                        using index_t = scalar_t;
                        CHECK(indices.is_contiguous());
                        CHECK(weights.is_contiguous());
                        CHECK_EQ(indices.size(0), weights.size(0));
                        int64_t stride = weights.size(1);
                        auto indices_data_ptr = indices.data_ptr<index_t>();
                        auto weights_data_ptr = weights.data_ptr<weight_type>();
                        {
                          auto wlmap = kv_store_.by(shard_id).wlock();
                          auto* pool = kv_store_.pool_by(shard_id);
                          for (auto index_iter = indexes.begin();
                               index_iter != indexes.end();
                               index_iter++) {
                            const auto& id_index = *index_iter;
                            auto id = int64_t(indices_data_ptr[id_index]);
                            // use mempool
                            weight_type* block = nullptr;
                            // First check if the key already exists
                            auto it = wlmap->find(id);
                            if (it != wlmap->end()) {
                              block = it->second;
                            } else {
                              // Key doesn't exist, allocate new block and
                              // insert.
                              block = pool->template allocate_t<weight_type>();
                              FixedBlockPool::set_key(block, id);
                              wlmap->insert({id, block});
                            }
                            if (feature_evict_config_.trigger_mode !=
                                    EvictTriggerMode::DISABLED &&
                                feature_evict_) {
                              feature_evict_->update_feature_statistics(block);
                            }
                            auto* data_ptr =
                                FixedBlockPool::data_ptr<weight_type>(block);
                            std::copy(
                                weights_data_ptr + id_index * stride,
                                weights_data_ptr + (id_index + 1) * stride,
                                data_ptr);
                          }
                        }
                      });
                });
    futures.push_back(std::move(f));
  }
  auto result = folly::collect(futures);
  if (feature_evict_config_.trigger_mode != EvictTriggerMode::DISABLED) {
    feature_evict_resume();
  }
  return result;
}

/// Get embeddings from kvstore.
///
/// @param indices The 1D embedding index tensor, should skip on negative
/// value
/// @param weights The 2D tensor that each row(embeddings) is paired up with
/// relative element in <indices>. This tensor will be filled up with the
/// returned embeddings from KVstore.
/// @param count A single element tensor that contains the number of indices
/// to be processed
///
/// @return None
template <typename weight_type>
folly::SemiFuture<std::vector<folly::Unit>>
DramKVEmbeddingCache<weight_type>::get_kv_db_async_impl(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    int64_t width_offset,
    std::optional<int64_t> width_length) {
  if (feature_evict_config_.trigger_mode != EvictTriggerMode::DISABLED) {
    feature_evict_pause();
  }
  std::vector<folly::Future<folly::Unit>> futures;
  auto row_width = weights.size(1);
  auto copy_width = width_length.value_or(row_width);
  CHECK_LE(row_width, max_D_);
  CHECK_EQ(copy_width, row_width);
  auto shardid_to_indexes = shard_input(indices, count);

  for (auto iter = shardid_to_indexes.begin(); iter != shardid_to_indexes.end();
       iter++) {
    const auto shard_id = iter->first;
    const auto indexes = iter->second;
    auto f =
        folly::via(executor_.get())
            .thenValue([this,
                        shard_id,
                        indexes,
                        &indices,
                        &weights,
                        width_offset,
                        row_width](folly::Unit) {
              FBGEMM_DISPATCH_INTEGRAL_TYPES(
                  indices.scalar_type(),
                  "dram_kvstore_set",
                  [this,
                   shard_id,
                   indexes,
                   &indices,
                   &weights,
                   width_offset,
                   row_width] {
                    using index_t = scalar_t;
                    CHECK(indices.is_contiguous());
                    CHECK(weights.is_contiguous());
                    CHECK_EQ(indices.size(0), weights.size(0));

                    const auto& init_storage =
                        initializers_[shard_id]->row_storage_;
                    TORCH_CHECK(
                        init_storage.scalar_type() == weights.scalar_type(),
                        "init_storage (",
                        toString(init_storage.scalar_type()),
                        ") and weights scalar (",
                        toString(weights.scalar_type()),
                        ") types mismatch");
                    auto row_storage_data_ptr =
                        init_storage.template data_ptr<weight_type>();
                    auto wlmap = kv_store_.by(shard_id).wlock();
                    auto indices_data_ptr = indices.data_ptr<index_t>();
                    {
                      for (auto index_iter = indexes.begin();
                           index_iter != indexes.end();
                           index_iter++) {
                        auto weights_data_ptr = weights.data_ptr<weight_type>();
                        const auto weights_row_index = *index_iter;
                        auto weight_idx =
                            int64_t(indices_data_ptr[weights_row_index]);
                        const auto cached_iter = wlmap->find(weight_idx);
                        if (cached_iter == wlmap->end()) {
                          auto weight_width = get_width_for_weights(
                              weight_idx, width_offset, row_width);
                          fill_from_row_storage(
                              shard_id,
                              reinterpret_cast<unsigned char*>(
                                  weights_data_ptr),
                              weights_row_index,
                              reinterpret_cast<unsigned char*>(
                                  row_storage_data_ptr),
                              width_offset,
                              row_width,
                              weight_width);
                          continue;
                        }
                        // use mempool
                        const auto* data_ptr =
                            FixedBlockPool::data_ptr<weight_type>(
                                cached_iter->second);
                        std::copy(
                            data_ptr + width_offset,
                            data_ptr + width_offset + row_width,
                            &(weights_data_ptr
                                  [weights_row_index *
                                   row_width])); // dst_start
                      }
                    }
                  });
            });
    futures.push_back(std::move(f));
  }
  auto result = folly::collect(futures);
  if (feature_evict_config_.trigger_mode != EvictTriggerMode::DISABLED) {
    feature_evict_resume();
  }
  return result;
};

template <typename weight_type>
folly::SemiFuture<std::vector<folly::Unit>>
DramKVEmbeddingCache<weight_type>::get_kv_db_async(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  return get_kv_db_async_impl(indices, weights, count);
}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::set_range_to_storage(
    const at::Tensor& weights,
    const int64_t start,
    const int64_t length) {
  const auto seq_indices =
      at::arange(start, start + length, at::TensorOptions().dtype(at::kLong));
  const auto count = at::tensor({length}, at::ScalarType::Long);
  folly::coro::blockingWait(set_kv_db_async(seq_indices, weights, count));
}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::get_range_from_snapshot(
    const at::Tensor& weights,
    const int64_t start,
    const int64_t length,
    const ssd::SnapshotHandle* snapshot_handle, // should be nullptr for dram
    int64_t width_offset,
    std::optional<int64_t> width_length) {
  CHECK(snapshot_handle == nullptr);
  const auto seq_indices =
      at::arange(start, start + length, at::TensorOptions().dtype(at::kLong));
  const auto count = at::tensor({length}, at::ScalarType::Long);
  get_kv_db_async_impl(seq_indices, weights, count, width_offset, width_length)
      .wait();
}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::get_kv_from_storage_by_snapshot(
    const at::Tensor& ids,
    const at::Tensor& weights,
    const ssd::SnapshotHandle* snapshot_handle, // should be nullptr for dram
    int64_t width_offset,
    std::optional<int64_t> width_length) {
  CHECK(snapshot_handle == nullptr);
  const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
  get_kv_db_async_impl(ids, weights, count, width_offset, width_length).wait();
}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::compact() {}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::trigger_feature_evict() {
  if (feature_evict_) {
    feature_evict_->trigger_evict();
  }
}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::maybe_evict() {
  switch (feature_evict_config_.trigger_mode) {
    case EvictTriggerMode::ITERATION: {
      if (feature_evict_config_.trigger_step_interval > 0 &&
          ++current_iter_ % feature_evict_config_.trigger_step_interval == 0) {
        trigger_feature_evict();
      }
      break;
    }
    case EvictTriggerMode::MEM_UTIL: {
      auto mem_util = get_map_used_memsize() / (1024 * 1024 * 1024);
      if (mem_util > feature_evict_config_.mem_util_threshold_in_GB) {
        trigger_feature_evict();
      }
      break;
    }
    default:
      break;
  }
}

template <typename weight_type>
size_t DramKVEmbeddingCache<weight_type>::get_map_used_memsize() const {
  return kv_store_.getUsedMemSize();
}

template <typename weight_type>
FeatureEvictMetricTensors
DramKVEmbeddingCache<weight_type>::get_feature_evict_metric() const {
  if (feature_evict_config_.trigger_mode == EvictTriggerMode::DISABLED) {
    throw std::runtime_error("feature evict is disabled");
  }
  return feature_evict_->get_feature_evict_metric();
}

template <typename weight_type>
int64_t DramKVEmbeddingCache<weight_type>::get_dim_from_index(
    int64_t weight_idx) const {
  if (sub_table_dims_.empty()) {
    return max_D_;
  }
  auto it = std::upper_bound(
      sub_table_hash_cumsum_.begin(), sub_table_hash_cumsum_.end(), weight_idx);
  if (it != sub_table_hash_cumsum_.end()) {
    int index = std::distance(sub_table_hash_cumsum_.begin(), it);
    return sub_table_dims_[index];
  }
  CHECK(false) << "weight_idx " << weight_idx
               << " doesn't belong to any feature";
  return max_D_;
}

template <typename weight_type>
int64_t DramKVEmbeddingCache<weight_type>::get_width_for_weights(
    int64_t weight_idx,
    int64_t width_offset,
    int64_t row_width) const {
  // when init an untouch embedding, we only want to init the weights part
  // and set the optimizer part to 0. This function helps us to get the dim
  // for each sub table, calculate the max bytes we should copy to the passed
  // in weights_data_ptr before optimizer section
  auto feature_dim = get_dim_from_index(weight_idx);
  CHECK_GT(feature_dim, width_offset);
  auto feature_width = feature_dim - width_offset;
  return std::min(feature_width, row_width);
}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::fill_from_row_storage(
    int shard_id,
    unsigned char* weights_data_ptr,
    int64_t weights_row_index,
    unsigned char* row_storage_data_ptr,
    int64_t width_offset,
    int64_t row_width,
    int64_t copied_width) {
  CHECK_GE(row_width, copied_width);
  CHECK_GE(max_D_, row_width);
  int64_t storage_row_bytes = elem_size_ * max_D_;
  int64_t row_bytes = row_width * elem_size_;
  auto copied_bytes = elem_size_ * copied_width;
  int64_t start_offset_bytes = elem_size_ * width_offset;
  int64_t row_index;
  initializers_[shard_id]->producer_queue_.dequeue(row_index);
  // TODO: fill the opt state as zeros for init value?
  std::copy(
      &(row_storage_data_ptr
            [row_index * storage_row_bytes + start_offset_bytes]),
      &(row_storage_data_ptr
            [row_index * storage_row_bytes + start_offset_bytes +
             copied_bytes]),
      &(weights_data_ptr[weights_row_index * row_bytes]));
  if (row_bytes > copied_bytes) {
    std::memset(
        &(weights_data_ptr[weights_row_index * row_bytes + copied_bytes]),
        0,
        row_bytes - copied_bytes);
  }
  initializers_[shard_id]->consumer_queue_.enqueue(row_index);
}

/// shard input ids into multiple shards based on hash function
///
/// @param indices The 1D embedding index tensor, should skip on negative
/// value
/// @param count A single element tensor that contains the number of indices
/// to be processed
///
/// @return a map of shard id to a vector of id indexes
template <typename weight_type>
folly::F14FastMap<int, std::vector<int64_t>>
DramKVEmbeddingCache<weight_type>::shard_input(
    const at::Tensor& indices,
    const at::Tensor& count) {
  folly::F14FastMap<int, std::vector<int64_t>> shardid_to_indexes;

  FBGEMM_DISPATCH_INTEGRAL_TYPES(
      indices.scalar_type(),
      "dram_shard_input",
      [this, &indices, &shardid_to_indexes, &count] {
        using index_t = scalar_t;
        // Due to duplicate indicies, we only need to get/set the first count
        // of
        // entries.
        auto conv_count = count.scalar_type() == at::ScalarType::Long
            ? *(count.data_ptr<int64_t>())
            : *(count.data_ptr<int32_t>());
        auto indices_data_ptr = indices.data_ptr<index_t>();
        // There could be negative indices, which we should skipp
        for (int i = 0; i < conv_count; i++) {
          auto index = int64_t(indices_data_ptr[i]);
          if (index < 0) {
            continue;
          }

          const auto shard_id = kv_db_utils::hash_shard(index, num_shards_);

          if (shardid_to_indexes.find(shard_id) == shardid_to_indexes.end()) {
            shardid_to_indexes[shard_id] = std::vector<int64_t>();
          }
          shardid_to_indexes[shard_id].push_back(i);
        }
      });

  // chunk request based on bucket sharding
  return shardid_to_indexes;
}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::flush_or_compact(
    const int64_t /* timestep */) {}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::feature_evict_resume() {
  if (feature_evict_) {
    feature_evict_->resume();
  }
}

template <typename weight_type>
void DramKVEmbeddingCache<weight_type>::feature_evict_pause() {
  if (feature_evict_) {
    feature_evict_->pause();
  }
}

template class DramKVEmbeddingCache<uint8_t>;
template class DramKVEmbeddingCache<float>;
template class DramKVEmbeddingCache<at::Half>;

} // namespace kv_mem
