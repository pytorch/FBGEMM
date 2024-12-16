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

#include <cstdint>
#include <iostream>

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
    size_t cache_size_bytes;
    size_t item_size_bytes;
    size_t num_shards;
    int64_t max_D_;
  };

  explicit CacheLibCache(
      const CacheConfig& cache_config,
      int64_t unique_tbe_id);

  std::unique_ptr<Cache> initializeCacheLib(const CacheConfig& config);

  std::unique_ptr<facebook::cachelib::CacheAdmin> createCacheAdmin(
      Cache& cache);

  /// Find the stored embeddings from a given embedding indices, aka key
  ///
  /// @param key_tensor embedding index(tensor with only one element) to look up
  ///
  /// @return an optional value, return none on cache misses, if cache hit
  /// return a pointer to the cachelib underlying storage of associated
  /// embeddings
  ///
  /// @note that this is not thread safe, caller needs to make sure the data is
  /// fully processed before doing cache insertion, otherwise the returned space
  /// might be overwritten if cache is full
  folly::Optional<void*> get(const at::Tensor& key_tensor);

  /// Cachelib wrapper specific hash function
  ///
  /// @param key embedding index to get hashed
  ///
  /// @return an hashed value ranges from [0, num_pools)
  size_t get_shard_id(int64_t key);

  /// get pool id given an embedding index
  ///
  /// @param key embedding index to get pool id
  ///
  /// @return a pool id associated with the given key, this is to build a
  /// deterministic mapping from a embedding index to a specific pool id
  facebook::cachelib::PoolId get_pool_id(int64_t key);

  /// update the LRU queue in cachelib, this is detached from cache->find()
  /// so that we could boost up the lookup perf without worrying about LRU queue
  /// contention
  ///
  /// @param read_handles the read handles that record what cache item has been
  /// accessed
  void batchMarkUseful(const std::vector<Cache::ReadHandle>& read_handles);

  /// Add an embedding index and embeddings into cachelib
  ///
  /// @param key_tensor embedding index(tensor with only one element) to insert
  /// @param data embedding weights to insert
  ///
  /// @return true on success insertion, false on failure insertion, a failure
  /// insertion could happen if the refcount of bottom K items in LRU queue
  /// isn't 0.

  /// @note In training use case, this is not expected to happen as we do
  /// bulk read and bluk write sequentially
  ///
  /// @note cache_->allocation will trigger eviction callback func
  bool put(const at::Tensor& key_tensor, const at::Tensor& data);

  /// iterate through all items in L2 cache, fill them in indices and weights
  /// respectively and return indices, weights and count
  ///
  /// @return optional value, if cache is empty return none
  /// @return indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @return weights The 2D tensor that each row(embeddings) is paired up with
  /// relative element in <indices>
  /// @return count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @note this isn't thread safe, caller needs to make sure put isn't called
  /// while this is executed.
  folly::Optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
  get_all_items();

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
      const at::Tensor& count);

  /// reset slot pointer that points to the next available slot in the eviction
  /// tensors and returns number of slots filled
  void reset_eviction_states();

  /// get the filled indices and weights tensors from L2 eviction, could be all
  /// invalid if no eviction happened
  folly::Optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
  get_tensors_and_reset();

  /// get L2 cache utilization stats
  std::vector<int64_t> get_cache_usage();

 private:
  const CacheConfig cache_config_;
  const int64_t unique_tbe_id_;
  std::unique_ptr<Cache> cache_;
  std::vector<facebook::cachelib::PoolId> pool_ids_;
  std::unique_ptr<facebook::cachelib::CacheAdmin> admin_;

  folly::Optional<at::Tensor> evicted_indices_opt_{folly::none};
  folly::Optional<at::Tensor> evicted_weights_opt_{folly::none};
  folly::Optional<at::ScalarType> index_dtype_{folly::none};
  folly::Optional<at::ScalarType> weights_dtype_{folly::none};
  std::atomic<int64_t> eviction_row_id{0};
};

} // namespace l2_cache
