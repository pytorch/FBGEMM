/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <ATen/core/ivalue.h>
#include <caffe2/torch/fb/distributed/wireSerializer/WireSerializer.h>
#include <common/base/Proc.h>
#include <common/stats/Stats.h>
#include <folly/SocketAddress.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Invoke.h>
#include <folly/coro/Task.h>
#include <folly/executors/FunctionScheduler.h>
#include <folly/logging/xlog.h>
#include <servicerouter/client/cpp2/ServiceRouter.h>
#include <thrift/lib/cpp2/protocol/CompactProtocol.h>
#include <thrift/lib/cpp2/protocol/Serializer.h>
#include <torch/script.h>

#include "../ssd_split_embeddings_cache/kv_db_table_batched_embeddings.h"
#include "SynchronizedShardedMap.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/initializer.h"
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "feature_evict.h"
#include "fixed_block_pool.h"

namespace kv_mem {

#define TORCH_CHECK_TENSOR_PROPERTIES(tensor, scalar_type) \
  do {                                                     \
    TORCH_CHECK((tensor).has_value());                     \
    TORCH_CHECK((tensor)->dim() == 1);                     \
    TORCH_CHECK((tensor)->dtype() == (scalar_type));       \
    TORCH_CHECK((tensor)->is_contiguous());                \
    TORCH_CHECK((tensor)->device().is_cpu());              \
  } while (0)

#define TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(                         \
    source_tensor, target_container, data_type)                         \
  do {                                                                  \
    TORCH_CHECK(                                                        \
        (source_tensor)->numel() + 1 == hash_size_cumsum->numel(),      \
        "hash_size_cumsum length must be one more than " #source_tensor \
        " length, but got ",                                            \
        hash_size_cumsum->numel(),                                      \
        " and ",                                                        \
        (source_tensor)->numel());                                      \
    (target_container)                                                  \
        .assign((source_tensor)->data_ptr<data_type>(),                 \
                (source_tensor)->data_ptr<data_type>() +                \
                    (source_tensor)->numel());                          \
  } while (0)

/// @ingroup KVMemEmbedding
///
/// @brief An implementation of EmbeddingKVDB for ZCH v.Next
///
template <typename weight_type>
class DramKVEmbeddingCache : public kv_db::EmbeddingKVDB {
 public:
  /// DramKVEmbeddingCache constructor
  ///
  /// @param max_D the maximum dimension of of embedding tensor
  /// @param uniform_init_lower the lower bound of the uniform distribution
  /// @param uniform_init_upper the upper bound of the uniform distribution
  /// @param evict_trigger_mode  evict trigger mode, reference EvictTriggerMode
  /// @param trigger_step_interval trigger step interval
  /// (EvictTriggerMode::ITERATION used)
  /// @param mem_util_threshold_in_GB mem util threshold (.GB
  /// EvictTriggerMode::MEM_UTIL used)
  /// @param evict_trigger_strategy evict trigger strategy, reference
  /// EvictTriggerStrategy
  /// @param count_thresholds count threshold for each table,
  /// at::ScalarType::UInt32
  /// @param ttls_in_hour the time to feature live for each table,(.hour)
  /// at::ScalarType::UInt32
  /// @param count_decay_rates count decay rate for each table,
  /// at::ScalarType::Float
  /// @param l2_weight_thresholds l2 weight threshold for each table,
  /// at::ScalarType::Double
  /// @param num_shards number of shards for the kvstore. This is to improve
  /// parallelization. Each key value pair will be sharded into one shard.
  /// @param num_threads num of threads that kvstore needs to be run upon for
  /// parallelization. This is to improve read and write performance.
  /// @param row_storage_bitwidth storage bitwidth for each row of embedding for
  /// initializers. 32 kFloat || 16 kHalf || 8 kByte
  /// @param enable_async_update whether to enable async update for the cache
  /// @param table_dims the table dimension for each table
  /// @param hash_size_cumsum the hash size cumulative sum for each table
  /// @return None
  explicit DramKVEmbeddingCache(
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t evict_trigger_mode = 0,
      int64_t trigger_step_interval = 0,
      int64_t mem_util_threshold_in_GB = 0.0,
      int64_t evict_trigger_strategy = 1,
      const std::optional<at::Tensor>& count_thresholds = std::nullopt,
      const std::optional<at::Tensor>& ttls_in_hour = std::nullopt,
      const std::optional<at::Tensor>& count_decay_rates = std::nullopt,
      const std::optional<at::Tensor>& l2_weight_thresholds = std::nullopt,
      int64_t num_shards = 8,
      int64_t num_threads = 32,
      int64_t row_storage_bitwidth = 32,
      bool enable_async_update = false,
      std::optional<at::Tensor> table_dims = std::nullopt,
      std::optional<at::Tensor> hash_size_cumsum = std::nullopt)
      : kv_db::EmbeddingKVDB(num_shards,
                             max_D,
                             0,  // l2_cache_size_gb =0 to disable l2 cache
                             0,  // tbe_unqiue_id
                             2,  // ele_size_bytes
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
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(std::max<size_t>(
        num_threads, facebook::Proc::getCpuInfo().numCpuCores));
    initialize_initializers(num_shards,
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
          hash_size_cumsum->data_ptr<int64_t>() + 1,  // skip the first 0
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
      TORCH_CHECK_TENSOR_PROPERTIES(count_thresholds, at::ScalarType::UInt32);
      TORCH_CHECK_TENSOR_PROPERTIES(count_decay_rates, at::ScalarType::Float);
      TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
          count_thresholds, feature_evict_config_.count_thresholds, uint32_t);
      TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
          count_decay_rates, feature_evict_config_.count_decay_rates, float);
    }

    if (feature_evict_config_.trigger_strategy ==
            EvictTriggerStrategy::BY_TIMESTAMP ||
        feature_evict_config_.trigger_strategy ==
            EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER) {
      TORCH_CHECK_TENSOR_PROPERTIES(ttls_in_hour, at::ScalarType::UInt32);
      TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
          ttls_in_hour, feature_evict_config_.ttls_in_hour, uint32_t);
    }

    if (feature_evict_config_.trigger_strategy ==
        EvictTriggerStrategy::BY_L2WEIGHT) {
      TORCH_CHECK_TENSOR_PROPERTIES(l2_weight_thresholds,
                                    at::ScalarType::Double);
      TORCH_CHECK_TENSOR_PROPERTIES(table_dims, at::ScalarType::Long);
      TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
          l2_weight_thresholds,
          feature_evict_config_.l2_weight_thresholds,
          double);
      TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
          table_dims, feature_evict_config_.embedding_dims, int);
    }
    feature_evict_ = create_feature_evict(feature_evict_config_,
                                          executor_.get(),
                                          kv_store_,
                                          sub_table_hash_cumsum_);
  }

  void initialize_initializers(int64_t num_shards,
                               int64_t max_D,
                               double uniform_init_lower,
                               double uniform_init_upper,
                               int64_t row_storage_bitwidth) {
    for (auto i = 0; i < num_shards; ++i) {
      auto* gen = at::check_generator<at::CPUGeneratorImpl>(
          at::detail::getDefaultCPUGenerator());
      {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        initializers_.push_back(
            std::make_unique<ssd ::Initializer>(gen->random64(),
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
  at::Tensor get_keys_in_range_impl(
      int64_t start,
      int64_t end,
      std::optional<int64_t> offset = std::nullopt) override {
    std::vector<std::vector<int64_t>> ids;
    for (int i = 0; i < num_shards_; i++) {
      ids.push_back(std::vector<int64_t>());
    }
    std::vector<folly::Future<folly::Unit>> futures;
    for (int shard_id = 0; shard_id < num_shards_; shard_id++) {
      auto f =
          folly::via(executor_.get())
              .thenValue([this, shard_id, start, end, offset, &ids](
                             folly::Unit) {
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
      all_ids_ptr->insert(all_ids_ptr->end(),
                          std::make_move_iterator(sub.begin()),
                          std::make_move_iterator(sub.end()));
    }

    return torch::from_blob(
               all_ids_ptr->data(),
               {int64_t(all_ids_ptr->size())},
               [all_ids_ptr](void* p) mutable { all_ids_ptr.reset(); },
               torch::kInt64  // data type
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
  folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const kv_db::RocksdbWriteMode w_mode =
          kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ /*unused*/) override {
    if (feature_evict_config_.trigger_mode != EvictTriggerMode::DISABLED) {
      feature_evict_pause();
    }
    std::vector<folly::Future<folly::Unit>> futures;
    auto shardid_to_indexes = shard_input(indices, count);
    for (auto iter = shardid_to_indexes.begin();
         iter != shardid_to_indexes.end();
         iter++) {
      const auto shard_id = iter->first;
      const auto indexes = iter->second;
      auto f =
          folly::via(executor_.get())
              .thenValue([this, shard_id, indexes, &indices, &weights](
                             folly::Unit) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kv_set",
                    [this, shard_id, indexes, &indices, &weights] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      CHECK(weights.is_contiguous());
                      CHECK_EQ(indices.size(0), weights.size(0));
                      {
                        auto wlmap = kv_store_.by(shard_id).wlock();
                        auto indices_data_ptr = indices.data_ptr<index_t>();
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
                            // Key doesn't exist, allocate new block and insert.
                            block = pool->allocate_t();
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
                          std::copy(weights[id_index]
                                        .template data_ptr<weight_type>(),
                                    weights[id_index]
                                            .template data_ptr<weight_type>() +
                                        weights[id_index].numel(),
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
  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async_impl(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) {
    if (feature_evict_config_.trigger_mode != EvictTriggerMode::DISABLED) {
      feature_evict_pause();
    }
    std::vector<folly::Future<folly::Unit>> futures;
    auto row_width = weights.size(1);
    auto copy_width = width_length.value_or(row_width);
    CHECK_LE(row_width, max_D_);
    CHECK_EQ(copy_width, row_width);
    auto shardid_to_indexes = shard_input(indices, count);

    for (auto iter = shardid_to_indexes.begin();
         iter != shardid_to_indexes.end();
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
                          auto weights_data_ptr =
                              weights.data_ptr<weight_type>();
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
                              &(weights_data_ptr[weights_row_index *
                                                 row_width]));  // dst_start
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

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    return get_kv_db_async_impl(indices, weights, count);
  }

  void set_range_to_storage(const at::Tensor& weights,
                            const int64_t start,
                            const int64_t length) {
    const auto seq_indices =
        at::arange(start, start + length, at::TensorOptions().dtype(at::kLong));
    const auto count = at::tensor({length}, at::ScalarType::Long);
    folly::coro::blockingWait(set_kv_db_async(seq_indices, weights, count));
  }

  void get_range_from_snapshot(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length,
      const ssd::SnapshotHandle* snapshot_handle,  // should be nullptr for dram
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override {
    CHECK(snapshot_handle == nullptr);
    const auto seq_indices =
        at::arange(start, start + length, at::TensorOptions().dtype(at::kLong));
    const auto count = at::tensor({length}, at::ScalarType::Long);
    get_kv_db_async_impl(
        seq_indices, weights, count, width_offset, width_length)
        .wait();
  }

  void get_kv_from_storage_by_snapshot(
      const at::Tensor& ids,
      const at::Tensor& weights,
      const ssd::SnapshotHandle* snapshot_handle,  // should be nullptr for dram
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override {
    CHECK(snapshot_handle == nullptr);
    const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
    get_kv_db_async_impl(ids, weights, count, width_offset, width_length)
        .wait();
  }

  void compact() override {}

  void trigger_feature_evict() {
    if (feature_evict_) {
      feature_evict_->trigger_evict();
    }
  }

  void maybe_evict() {
    switch (feature_evict_config_.trigger_mode) {
      case EvictTriggerMode::ITERATION: {
        if (feature_evict_config_.trigger_step_interval > 0 &&
            ++current_iter_ % feature_evict_config_.trigger_step_interval ==
                0) {
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

  size_t get_map_used_memsize() const { return kv_store_.getUsedMemSize(); }

  FeatureEvictMetricTensors get_feature_evict_metric() const {
    if (feature_evict_config_.trigger_mode == EvictTriggerMode::DISABLED) {
      throw std::runtime_error("feature evict is disabled");
    }
    return feature_evict_->get_feature_evict_metric();
  }

 private:
  int64_t get_dim_from_index(int64_t weight_idx) const {
    if (sub_table_dims_.empty()) {
      return max_D_;
    }
    auto it = std::upper_bound(sub_table_hash_cumsum_.begin(),
                               sub_table_hash_cumsum_.end(),
                               weight_idx);
    if (it != sub_table_hash_cumsum_.end()) {
      int index = std::distance(sub_table_hash_cumsum_.begin(), it);
      return sub_table_dims_[index];
    }
    CHECK(false) << "weight_idx " << weight_idx
                 << " doesn't belong to any feature";
    return max_D_;
  }

  int64_t get_width_for_weights(int64_t weight_idx,
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

  void fill_from_row_storage(int shard_id,
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
    std::copy(&(row_storage_data_ptr[row_index * storage_row_bytes +
                                     start_offset_bytes]),
              &(row_storage_data_ptr[row_index * storage_row_bytes +
                                     start_offset_bytes + copied_bytes]),
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
  folly::F14FastMap<int, std::vector<int64_t>> shard_input(
      const at::Tensor& indices, const at::Tensor& count) {
    folly::F14FastMap<int, std::vector<int64_t>> shardid_to_indexes;

    // Due to duplicate indicies, we only need to get/set the first count of
    // entries.
    auto conv_count = count.scalar_type() == at::ScalarType::Long
                          ? *(count.data_ptr<int64_t>())
                          : *(count.data_ptr<int32_t>());

    // There could be negative indices, which we should skipp
    for (int i = 0; i < conv_count; i++) {
      if (indices[i].item<int64_t>() < 0) {
        continue;
      }

      const auto shard_id =
          kv_db_utils::hash_shard(indices[i].item<int64_t>(), num_shards_);

      if (shardid_to_indexes.find(shard_id) == shardid_to_indexes.end()) {
        shardid_to_indexes[shard_id] = std::vector<int64_t>();
      }
      shardid_to_indexes[shard_id].push_back(i);
    }
    // chunk request based on bucket sharding
    return shardid_to_indexes;
  }

  void flush_or_compact(const int64_t timestep) override {}

  void feature_evict_resume() {
    if (feature_evict_) {
      feature_evict_->resume();
    }
  }

  void feature_evict_pause() {
    if (feature_evict_) {
      feature_evict_->pause();
    }
  }

  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  // background thread
  folly::FunctionScheduler scheduler_;
  int64_t max_D_;
  int64_t num_shards_;
  // mempool params
  size_t block_size_;
  size_t block_alignment_;
  SynchronizedShardedMap<int64_t, weight_type*> kv_store_;
  std::atomic_bool is_eviction_ongoing_ = false;
  std::vector<std::unique_ptr<ssd::Initializer>> initializers_;
  int64_t elem_size_;
  std::vector<int64_t> sub_table_dims_;
  std::vector<int64_t> sub_table_hash_cumsum_;
  FeatureEvictConfig feature_evict_config_;
  std::unique_ptr<FeatureEvict<weight_type>> feature_evict_;
  int current_iter_ = 0;
};  // class DramKVEmbeddingCache

}  // namespace kv_mem
