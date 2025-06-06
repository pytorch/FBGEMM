/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "../ssd_split_embeddings_cache/kv_db_table_batched_embeddings.h"

#include <folly/coro/BlockingWait.h>
#include <folly/coro/Task.h>
#include <folly/executors/FunctionScheduler.h>

#include "SynchronizedShardedMap.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/initializer.h"
#include "store_value_utils.h"

#include <ATen/core/ivalue.h>
#include <caffe2/torch/fb/distributed/wireSerializer/WireSerializer.h>
#include <common/base/Proc.h>
#include <common/stats/Stats.h>
#include <common/time/Time.h>
#include <folly/SocketAddress.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Invoke.h>
#include <folly/logging/xlog.h>
#include <servicerouter/client/cpp2/ServiceRouter.h>
#include <thrift/lib/cpp2/protocol/CompactProtocol.h>
#include <thrift/lib/cpp2/protocol/Serializer.h>
#include <torch/script.h>
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"

namespace kv_mem {

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
  /// @param num_shards number of shards for the kvstore. This is to improve
  /// parallelization. Each key value pair will be sharded into one shard.
  /// @param num_threads num of threads that kvstore needs to be run upon for
  /// parallelization. This is to improve read and write performance.
  /// @param row_storage_bitwidth storage bitwidth for each row of embedding for
  /// initializers. 32 kFloat || 16 kHalf || 8 kByte
  /// @param weight_ttl_in_hours ttl in hours for each entry before it being
  /// evicted
  /// @param enable_async_update whether to enable async update for the cache
  /// @param table_dims the table dimension for each table
  /// @param hash_size_cumsum the hash size cumulative sum for each table
  /// @return None
  explicit DramKVEmbeddingCache(
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t num_shards = 8,
      int64_t num_threads = 32,
      int64_t row_storage_bitwidth = 32,
      int64_t weight_ttl_in_hours = 2,
      bool enable_async_update = false,
      std::optional<at::Tensor> table_dims = std::nullopt,
      std::optional<at::Tensor> hash_size_cumsum = std::nullopt)
      : kv_db::EmbeddingKVDB(
            num_shards,
            max_D,
            0, // l2_cache_size_gb =0 to disable l2 cache
            0, // tbe_unqiue_id
            2, // ele_size_bytes
            enable_async_update),
        max_D_(max_D),
        num_shards_(num_shards),
        weight_ttl_in_hours_(weight_ttl_in_hours),
        block_size_(StoreValueUtils::calculate_block_size<weight_type>(max_D)),
        block_alignment_(
            StoreValueUtils::calculate_block_alignment<weight_type>()),
        kv_store_(SynchronizedShardedMap<int64_t, weight_type*>(
            num_shards_,
            block_size_,
            block_alignment_,
            /*blocks_per_chunk=*/8192)),
        elem_size_(row_storage_bitwidth / 8) {
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(std::max<size_t>(
        num_threads, facebook::Proc::getCpuInfo().numCpuCores));
    initialize_initializers(
        num_shards,
        max_D,
        uniform_init_lower,
        uniform_init_upper,
        row_storage_bitwidth);
    if (table_dims.has_value()) {
      TORCH_CHECK(table_dims->dim() == 1);
      TORCH_CHECK(table_dims->dtype() == at::ScalarType::Long);
      TORCH_CHECK(table_dims->is_contiguous());
      TORCH_CHECK(table_dims->device().is_cpu());
      TORCH_CHECK(hash_size_cumsum.has_value());
      TORCH_CHECK(hash_size_cumsum->dim() == 1);
      TORCH_CHECK(hash_size_cumsum->dtype() == at::ScalarType::Long);
      TORCH_CHECK(hash_size_cumsum->is_contiguous());
      TORCH_CHECK(hash_size_cumsum->device().is_cpu());
      TORCH_CHECK(
          table_dims->numel() + 1 == hash_size_cumsum->numel(),
          "hash_size_cumsum length must be one more than table_dims length, but got ",
          hash_size_cumsum->numel(),
          " and ",
          table_dims->numel());
      sub_table_dims_.assign(
          table_dims->data_ptr<int64_t>(),
          table_dims->data_ptr<int64_t>() + table_dims->numel());
      sub_table_hash_cumsum_.assign(
          hash_size_cumsum->data_ptr<int64_t>() + 1, // skip the first 0
          hash_size_cumsum->data_ptr<int64_t>() + hash_size_cumsum->numel());
    }
  }

  void initialize_initializers(
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
  folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const kv_db::RocksdbWriteMode w_mode =
          kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ /*unused*/) override {
    std::vector<folly::Future<folly::Unit>> futures;
    auto shardid_to_indexes = shard_input(indices, count);
    auto now = facebook::WallClockUtil::NowInUsecFast();

    for (auto iter = shardid_to_indexes.begin();
         iter != shardid_to_indexes.end();
         iter++) {
      const auto shard_id = iter->first;
      const auto indexes = iter->second;
      auto f =
          folly::via(executor_.get())
              .thenValue([this, shard_id, indexes, &indices, &weights, &now](
                             folly::Unit) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kv_set",
                    [this, shard_id, indexes, &indices, &weights, &now] {
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
                            // Key doesn't exist, allocate new block and insert.
                            block = StoreValueUtils::allocate<weight_type>(
                                block_size_, block_alignment_, pool);
                            wlmap->insert({id, block});
                          }
                          StoreValueUtils::update_timestamp<weight_type>(
                              block, now);
                          auto* data_ptr =
                              StoreValueUtils::data_ptr<weight_type>(block);
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
    return folly::collect(futures);
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
    std::vector<folly::Future<folly::Unit>> futures;
    auto row_width = weights.size(1);
    auto copy_width = width_length.value_or(row_width);
    CHECK_LE(row_width, max_D_);
    CHECK_EQ(copy_width, row_width);
    auto shardid_to_indexes = shard_input(indices, count);
    auto now = facebook::WallClockUtil::NowInUsecFast();
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
                          row_width,
                          &now](folly::Unit) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kvstore_set",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &weights,
                     width_offset,
                     row_width,
                     &now] {
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
                              StoreValueUtils::data_ptr<weight_type>(
                                  cached_iter->second);
                          StoreValueUtils::update_timestamp(
                              cached_iter->second, now);
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
    return folly::collect(futures);
  };

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    return get_kv_db_async_impl(indices, weights, count);
  }

  void set_range_to_storage(
      const at::Tensor& weights,
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
      const ssd::SnapshotHandle* snapshot_handle, // should be nullptr for dram
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
      const ssd::SnapshotHandle* snapshot_handle, // should be nullptr for dram
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override {
    CHECK(snapshot_handle == nullptr);
    const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
    get_kv_db_async_impl(ids, weights, count, width_offset, width_length)
        .wait();
  }

  void compact() override {}

 private:
  int64_t get_dim_from_index(int64_t weight_idx) const {
    if (sub_table_dims_.empty()) {
      return max_D_;
    }
    for (int i = 0; i < sub_table_hash_cumsum_.size(); i++) {
      if (weight_idx < sub_table_hash_cumsum_[i]) {
        return sub_table_dims_[i];
      }
    }
    CHECK(false) << "weight_idx " << weight_idx
                 << " doesn't belong to any feature";
    return max_D_;
  }

  int64_t get_width_for_weights(
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

  void fill_from_row_storage(
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
  folly::F14FastMap<int, std::vector<int64_t>> shard_input(
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

  void flush_or_compact(const int64_t timestep) override {}

  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  // background thread
  folly::FunctionScheduler scheduler_;
  int64_t max_D_;
  int64_t num_shards_;
  int64_t weight_ttl_in_hours_;
  // mempool params
  size_t block_size_;
  size_t block_alignment_;
  SynchronizedShardedMap<int64_t, weight_type*> kv_store_;
  std::atomic_bool is_eviction_ongoing_ = false;
  std::vector<std::unique_ptr<ssd::Initializer>> initializers_;
  int64_t elem_size_;
  std::vector<int64_t> sub_table_dims_;
  std::vector<int64_t> sub_table_hash_cumsum_;
}; // class DramKVEmbeddingCache

} // namespace kv_mem
