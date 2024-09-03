/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <ATen/ATen.h>
#include <cachelib/allocator/CacheAllocator.h>
#include <cachelib/facebook/admin/CacheAdmin.h>
#include "deeplearning/fbgemm/fbgemm_gpu/include/fbgemm_gpu/utils/dispatch_macros.h"

#include <cstdint>
#include <iostream>
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"

namespace l2_cache {

/// @ingroup embedding-ssd
///
/// @brief A Cachelib wrapper class for Cachlib interaction
///
/// It is for maintaining all the cache related operations, including
/// initialization, insertion, lookup and eviction.
/// It is stateful for eviction logic that caller has to specifically
/// fetch and reset eviction related states.
/// Cachelib related optimization will be captured inside this class
/// e.g. fetch and delayed markUseful to boost up get performance
///
/// @note that this class only handles single Cachelib read/update.
/// parallelism is done on the caller side
class CacheLibCache {
 public:
  using Cache = facebook::cachelib::LruAllocator;
  struct CacheConfig {
    size_t cacheSizeBytes;
  };

  explicit CacheLibCache(size_t cacheSizeBytes, int64_t num_shards)
      : cacheConfig_(CacheConfig{.cacheSizeBytes = cacheSizeBytes}),
        cache_(initializeCacheLib(cacheConfig_)),
        admin_(createCacheAdmin(*cache_)) {
    for (int i = 0; i < num_shards; i++) {
      pool_ids_.push_back(cache_->addPool(
          fmt::format("shard_{}", i),
          cache_->getCacheMemoryStats().ramCacheSize / num_shards));
    }
  }

  std::unique_ptr<Cache> initializeCacheLib(const CacheConfig& config) {
    auto eviction_cb =
        [this](const facebook::cachelib::LruAllocator::RemoveCbData& data) {
          FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
              evicted_weights_ptr_->scalar_type(), "l2_eviction_handling", [&] {
                if (data.context ==
                    facebook::cachelib::RemoveContext::kEviction) {
                  auto indices_data_ptr =
                      evicted_indices_ptr_->data_ptr<int64_t>();
                  auto weights_data_ptr =
                      evicted_weights_ptr_->data_ptr<scalar_t>();
                  auto row_id = eviction_row_id++;
                  auto weight_dim = evicted_weights_ptr_->size(1);
                  const auto key_ptr = reinterpret_cast<const int64_t*>(
                      data.item.getKey().data());
                  indices_data_ptr[row_id] = *key_ptr;

                  std::copy(
                      reinterpret_cast<const scalar_t*>(data.item.getMemory()),
                      reinterpret_cast<const scalar_t*>(data.item.getMemory()) +
                          weight_dim,
                      &weights_data_ptr[row_id * weight_dim]); // dst_start
                }
              });
        };
    Cache::Config cacheLibConfig;
    cacheLibConfig.setCacheSize(static_cast<uint64_t>(config.cacheSizeBytes))
        .setRemoveCallback(eviction_cb)
        .setCacheName("TBEL2Cache")
        .setAccessConfig({25 /* bucket power */, 10 /* lock power */})
        .setFullCoredump(false)
        .validate();
    return std::make_unique<Cache>(cacheLibConfig);
  }

  std::unique_ptr<facebook::cachelib::CacheAdmin> createCacheAdmin(
      Cache& cache) {
    facebook::cachelib::CacheAdmin::Config adminConfig;
    adminConfig.oncall = "mvai";
    return std::make_unique<facebook::cachelib::CacheAdmin>(
        cache, std::move(adminConfig));
  }

  /// Find the stored embeddings from a given embedding indices, aka key
  ///
  /// @param key embedding index to look up
  ///
  /// @return an optional value, return none on cache misses, if cache hit
  /// return a pointer to the cachelib underlying storage of associated
  /// embeddings
  ///
  /// @note that this is not thread safe, caller needs to make sure the data is
  /// fully processed before doing cache insertion, otherwise the returned space
  /// might be overwritten if cache is full
  std::optional<void*> get(int64_t key) {
    auto key_str = folly::StringPiece(
        reinterpret_cast<const char*>(&key), sizeof(int64_t));
    auto item = cache_->find(key_str);
    if (!item) {
      return std::nullopt;
    }
    return const_cast<void*>(item->getMemory());
  }

  /// Cachelib wrapper specific hash function
  ///
  /// @param key embedding index to get hashed
  ///
  /// @return an hashed value ranges from [0, num_pools)
  size_t get_shard_id(int64_t key) {
    return kv_db_utils::hash_shard(key, pool_ids_.size());
  }

  /// get pool id given an embedding index
  ///
  /// @param key embedding index to get pool id
  ///
  /// @return a pool id associated with the given key, this is to build a
  /// deterministic mapping from a embedding index to a specific pool id
  facebook::cachelib::PoolId get_pool_id(int64_t key) {
    return pool_ids_[get_shard_id(key)];
  }

  /// Add an embedding index and embeddings into cachelib
  ///
  /// @param key embedding index to insert
  ///
  /// @return true on success insertion, false on failure insertion, a failure
  /// insertion could happen if the refcount of bottom K items in LRU queue
  /// isn't 0.

  /// @note In training use case, this is not expected to happen as we do
  /// bulk read and bluk write sequentially
  ///
  /// @note cache_->allocation will trigger eviction callback func
  bool put(int64_t key, const at::Tensor& data) {
    auto key_str = folly::StringPiece(
        reinterpret_cast<const char*>(&key), sizeof(int64_t));
    auto item = cache_->allocate(get_pool_id(key), key_str, data.nbytes());
    if (!item) {
      XLOG(ERR) << fmt::format(
          "Failed to allocate item {} in cache, skip", key);
      return false;
    }
    std::memcpy(item->getMemory(), data.data_ptr(), data.nbytes());
    cache_->insertOrReplace(std::move(item));
    return true;
  }

  /// instantiate eviction related indices and weights tensors(size of <count>)
  /// for L2 eviction using the same dtype and device from <indices> and
  /// <weights> , managed on the caller side
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights The 2D tensor that each row(embeddings) is paired up with
  /// relative element in <indices>
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return None
  void init_tensor_for_l2_eviction(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) {
    auto num_lookups = count.item<long>();
    evicted_indices_ptr_ = std::make_shared<at::Tensor>(
        at::ones(
            num_lookups,
            at::TensorOptions()
                .device(indices.device())
                .dtype(indices.dtype())) *
        -1);
    evicted_weights_ptr_ = std::make_shared<at::Tensor>(at::empty(
        {num_lookups, weights.size(1)},
        at::TensorOptions().device(weights.device()).dtype(weights.dtype())));
  }

  /// reset slot pointer that points to the next available slot in the eviction
  /// tensors
  void reset_eviction_states() {
    eviction_row_id = 0;
  }

  /// get the filled indices and weights tensors from L2 eviction, could be all
  /// invalid if no eviction happened
  folly::Optional<std::pair<at::Tensor, at::Tensor>>
  get_evicted_indices_and_weights() {
    if (evicted_indices_ptr_) {
      assert(evicted_weights_ptr_ != nullptr);
      return std::make_pair(*evicted_indices_ptr_, *evicted_weights_ptr_);
    } else {
      return folly::none;
    }
  }

 private:
  const CacheConfig cacheConfig_;
  std::unique_ptr<Cache> cache_;
  std::vector<facebook::cachelib::PoolId> pool_ids_;
  std::unique_ptr<facebook::cachelib::CacheAdmin> admin_;

  std::shared_ptr<at::Tensor> evicted_indices_ptr_{nullptr};
  std::shared_ptr<at::Tensor> evicted_weights_ptr_{nullptr};
  std::atomic<int64_t> eviction_row_id{0};
};

} // namespace l2_cache
