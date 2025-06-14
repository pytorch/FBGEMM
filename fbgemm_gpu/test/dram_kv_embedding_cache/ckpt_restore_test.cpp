/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <iostream>

#include <ATen/ATen.h>
#include <array>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <folly/SocketAddress.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Invoke.h>
#include <folly/coro/Task.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/FunctionScheduler.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "fbgemm_gpu/src/dram_kv_embedding_cache/SynchronizedShardedMap.h"
#include "fbgemm_gpu/src/dram_kv_embedding_cache/fixed_block_pool.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"

namespace kv_mem {

// Mock to test ckpt restore
template <typename weight_type>
class SimpleDramKVEmbeddingCache {
 public:
  explicit SimpleDramKVEmbeddingCache(int64_t max_D,
                                      int64_t num_shards = 8,
                                      int64_t num_threads = 32,
                                      int64_t row_storage_bitwidth = 32)
      : max_D_(max_D),
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
        std::max<size_t>(num_threads, 32));
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

  /// get all ids in the kvstore
  ///
  /// @return a Tensor contained ids
  at::Tensor get_keys_in_range_impl(
      int64_t start,
      int64_t end,
      std::optional<int64_t> offset = std::nullopt) {
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
               [all_ids_ptr](void* p [[maybe_unused]]) mutable {
                 all_ids_ptr.reset();
               },
               torch::kInt64  // data type
               )
        .view({-1, 1});
  }

  /// Get embeddings and metaheader from kvstore.
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights_with_metaheader The 2D tensor that each row(embeddings) is
  /// paired up with relative element in <indices>. This tensor will be
  /// filled up with the returned embeddings and metaheader from KVstore.
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return None
  folly::SemiFuture<std::vector<folly::Unit>>
  get_kv_db_with_metaheader_async_impl(
      const at::Tensor& indices,
      const at::Tensor& weights_with_metaheader,
      const at::Tensor& count) {
    std::vector<folly::Future<folly::Unit>> futures;
    auto row_width = weights_with_metaheader.size(1) *
                     weights_with_metaheader.element_size();
    CHECK_EQ(row_width, block_size_);
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
                          &weights_with_metaheader,
                          row_width](folly::Unit) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kvstore_set",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &weights_with_metaheader,
                     row_width] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      CHECK(weights_with_metaheader.is_contiguous());
                      CHECK_EQ(indices.size(0),
                               weights_with_metaheader.size(0));
                      auto wlmap = kv_store_.by(shard_id).wlock();
                      auto indices_data_ptr = indices.data_ptr<index_t>();
                      {
                        for (auto index_iter = indexes.begin();
                             index_iter != indexes.end();
                             index_iter++) {
                          void* weights_data_ptr =
                              weights_with_metaheader.mutable_data_ptr();
                          const auto weights_row_index = *index_iter;
                          auto weight_idx =
                              int64_t(indices_data_ptr[weights_row_index]);
                          const auto cached_iter = wlmap->find(weight_idx);
                          // Defensive programming
                          // it shouldn't occur under normal circumstances
                          if (cached_iter == wlmap->end()) {
                            std::memset(static_cast<char*>(weights_data_ptr) +
                                            weights_row_index * row_width,
                                        0,
                                        row_width);
                            continue;
                          }
                          std::memcpy(static_cast<char*>(weights_data_ptr) +
                                          weights_row_index * row_width,
                                      cached_iter->second,
                                      row_width);
                        }
                      }
                    });
              });
      futures.push_back(std::move(f));
    }
    return folly::collect(futures);
  };

  /// insert embeddings and metaheader into kvstore.
  /// current underlying memory management is done through F14FastMap
  /// key value pair will be sharded into multiple shards to increase
  /// parallelism.
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights_with_metaheader The 2D tensor that each row(embeddings with
  /// metaheader) is paired up with relative element in <indices>
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return None
  folly::SemiFuture<std::vector<folly::Unit>>
  set_kv_db_with_metaheader_async_impl(
      const at::Tensor& indices,
      const at::Tensor& weights_with_metaheader,
      const at::Tensor& count) {
    std::vector<folly::Future<folly::Unit>> futures;
    auto shardid_to_indexes = shard_input(indices, count);
    for (auto iter = shardid_to_indexes.begin();
         iter != shardid_to_indexes.end();
         iter++) {
      const auto shard_id = iter->first;
      const auto indexes = iter->second;
      auto f =
          folly::via(executor_.get())
              .thenValue(
                  [this, shard_id, indexes, &indices, &weights_with_metaheader](
                      folly::Unit) {
                    FBGEMM_DISPATCH_INTEGRAL_TYPES(
                        indices.scalar_type(),
                        "dram_kv_set",
                        [this,
                         shard_id,
                         indexes,
                         &indices,
                         &weights_with_metaheader] {
                          using index_t = scalar_t;
                          CHECK(indices.is_contiguous());
                          CHECK(weights_with_metaheader.is_contiguous());
                          CHECK_EQ(indices.size(0),
                                   weights_with_metaheader.size(0));
                          int64_t stride =
                              weights_with_metaheader.size(1) *
                              weights_with_metaheader.element_size();
                          CHECK_EQ(stride, block_size_);
                          auto indices_data_ptr = indices.data_ptr<index_t>();
                          void* weights_data_ptr =
                              weights_with_metaheader.data_ptr();
                          {
                            auto wlmap = kv_store_.by(shard_id).wlock();
                            auto* pool = kv_store_.pool_by(shard_id);

                            for (auto index_iter = indexes.begin();
                                 index_iter != indexes.end();
                                 index_iter++) {
                              const auto& id_index = *index_iter;
                              auto id = int64_t(indices_data_ptr[id_index]);
                              // Defensive programming
                              // it shouldn't occur under normal circumstances
                              auto used = FixedBlockPool::get_used(
                                  static_cast<char*>(weights_data_ptr) +
                                  id_index * stride);
                              if (!used) {
                                continue;
                              }
                              // use mempool
                              weight_type* block = nullptr;
                              // First check if the key already exists
                              auto it = wlmap->find(id);
                              if (it != wlmap->end()) {
                                block = it->second;
                              } else {
                                // Key doesn't exist, allocate new block and
                                // insert.
                                block =
                                    pool->template allocate_t<weight_type>();
                                wlmap->insert({id, block});
                              }
                              std::memcpy(block,
                                          static_cast<char*>(weights_data_ptr) +
                                              id_index * stride,
                                          block_size_);
                            }
                          }
                        });
                  });
      futures.push_back(std::move(f));
    }
    return folly::collect(futures);
  }

  // used for ckpt, get kv with metaheader from storage
  void get_kv_with_metaheader_from_storage(
      const at::Tensor& ids, const at::Tensor& weights_with_metaheader) {
    const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
    get_kv_db_with_metaheader_async_impl(ids, weights_with_metaheader, count)
        .wait();
  }

  void set_kv_with_metaheader_to_storage(
      const at::Tensor& weights_with_metaheader) {
    std::vector<int64_t> keys(weights_with_metaheader.size(0), 0);
    for (int64_t i = 0; i < weights_with_metaheader.size(0); ++i) {
      keys[i] = FixedBlockPool::get_key(weights_with_metaheader[i].data_ptr());
    }
    auto indices =
        torch::from_blob(keys.data(), {int64_t(keys.size())}, torch::kInt64);
    const auto count =
        at::tensor({weights_with_metaheader.size(0)}, at::ScalarType::Long);
    set_kv_db_with_metaheader_async_impl(
        indices, weights_with_metaheader, count)
        .wait();
  }

  size_t get_block_size() const { return block_size_; }

  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  int64_t max_D_;
  int64_t num_shards_;
  // mempool params
  size_t block_size_;
  size_t block_alignment_;
  SynchronizedShardedMap<int64_t, weight_type*> kv_store_;
  int64_t elem_size_;
};

std::vector<float> generateFixedEmbedding(int dimension, float value) {
  return std::vector<float>(dimension, value);
}

void insertEmbeddingsWithMetaHeader(
    SynchronizedShardedMap<int64_t, float*>& embeddingMap,
    int64_t dimension,
    size_t numInserts,
    uint32_t count,
    uint32_t timestamp) {
  std::vector<float> fixedEmbedding = generateFixedEmbedding(dimension, 1.0);

  auto wlock = embeddingMap.by(0).wlock();
  auto* pool = embeddingMap.pool_by(0);

  for (size_t i = 0; i < numInserts; i++) {
    auto* block = pool->allocate_t<float>();
    auto* data_ptr = FixedBlockPool::data_ptr<float>(block);
    std::copy(fixedEmbedding.begin(), fixedEmbedding.end(), data_ptr);
    wlock->insert_or_assign(i, block);
    FixedBlockPool::set_key(block, i);
    FixedBlockPool::set_count(block, count);
    FixedBlockPool::set_timestamp(block, timestamp);
  }
}

TEST(SimpleDramKVEmbeddingCache, CKPTANDRESTORE) {
  // init cache
  auto cache = std::make_unique<SimpleDramKVEmbeddingCache<float>>(1024, 1);
  insertEmbeddingsWithMetaHeader(cache->kv_store_, 1024, 1000, 2, 3);

  // test get_block_size
  int64_t block_size = cache->get_block_size();
  EXPECT_EQ(block_size, 4112);

  // get key from map and check init result
  auto key_tensor = cache->get_keys_in_range_impl(0, 1000);
  EXPECT_EQ(key_tensor.size(0), 1000);
  EXPECT_EQ(key_tensor.size(1), 1);
  auto key_data = key_tensor.data_ptr<int64_t>();

  std::vector<int64_t> keys(key_data, key_data + 1000);
  std::sort(keys.begin(), keys.end());
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(keys[i], i) << "Key at index " << i << " should be " << i;
  }

  // test get_kv_with_metaheader_from_storage
  auto ids = key_tensor;
  auto weights_with_metaheader = torch::zeros({key_tensor.size(0), block_size},
                                              torch::dtype(torch::kUInt8));

  cache->get_kv_with_metaheader_from_storage(ids, weights_with_metaheader);

  for (int i = 0; i < 1000; ++i) {
    auto weights_with_metaheader_data = weights_with_metaheader[i].data_ptr();
    EXPECT_EQ(FixedBlockPool::get_key(weights_with_metaheader_data),
              key_data[i]);
    EXPECT_EQ(FixedBlockPool::get_count(weights_with_metaheader_data), 2);
    EXPECT_EQ(FixedBlockPool::get_timestamp(weights_with_metaheader_data), 3);
  }
  // test set_kv_with_metaheader_to_storage
  auto cache_restore =
      std::make_unique<SimpleDramKVEmbeddingCache<float>>(1024, 1);
  cache_restore->set_kv_with_metaheader_to_storage(weights_with_metaheader);
  auto weights_with_metaheader_restore = torch::zeros(
      {key_tensor.size(0), block_size}, torch::dtype(torch::kUInt8));
  cache_restore->get_kv_with_metaheader_from_storage(
      ids, weights_with_metaheader_restore);
  for (int i = 0; i < 1000; ++i) {
    auto weights_with_metaheader_data =
        weights_with_metaheader_restore[i].data_ptr();
    EXPECT_EQ(FixedBlockPool::get_key(weights_with_metaheader_data),
              key_data[i]);
    EXPECT_EQ(FixedBlockPool::get_count(weights_with_metaheader_data), 2);
    EXPECT_EQ(FixedBlockPool::get_timestamp(weights_with_metaheader_data), 3);
  }
}
}  // namespace kv_mem
