/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/split_embeddings_cache/cachelib_cache.h"
#include <folly/Conv.h>
#include <cmath>
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"

namespace l2_cache {

using Cache = facebook::cachelib::LruAllocator;

CacheLibCache::CacheLibCache(
    const CacheConfig& cache_config,
    int64_t unique_tbe_id)
    : cache_config_(cache_config), unique_tbe_id_(unique_tbe_id) {
  // Initialize cache - this creates either regular Cache or ObjectCache
  // For ObjectCache mode, this sets object_cache_ and returns nullptr
  // For regular mode, this returns the Cache and object_cache_ remains nullptr
  cache_ = initializeCacheLib(cache_config_);

  // Create admin for the underlying cache (works for both modes)
  if (cache_config_.use_object_cache) {
    // ObjectCache mode: create admin for the ObjectCache directly
    admin_ = createCacheAdmin(*object_cache_, true);
  } else {
    // Regular mode: create admin for the cache
    admin_ = createCacheAdmin(*cache_, false);

    // Initialize pools only for regular allocator mode
    for (size_t i = 0; i < cache_config_.num_shards; i++) {
      pool_ids_.push_back(cache_->addPool(
          fmt::format("shard_{}", i),
          cache_->getCacheMemoryStats().ramCacheSize / cache_config_.num_shards,
          std::set<uint32_t>{},
          Cache::MMConfig{
              0, /* promote on every access*/
              true, /*enable promotion on write*/
              true /*enable promotion on read*/}));
    }
  }
}

size_t CacheLibCache::get_cache_item_size() const {
  return cache_config_.item_size_bytes;
}

Cache::AccessIterator CacheLibCache::begin() {
  if (cache_config_.use_object_cache) {
    // ObjectCache has its own iterator - use the underlying L1 cache
    return object_cache_->begin();
  }
  return cache_->begin();
}

std::unique_ptr<Cache> CacheLibCache::initializeCacheLib(
    const CacheConfig& config) {
  // Setup eviction callback (used for both regular Cache and ObjectCache's
  // underlying cache)
  auto eviction_cb = [this](
                         const facebook::cachelib::LruAllocator::RemoveCbData&
                             data) {
    if (evicted_weights_opt_.has_value()) {
      FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
          evicted_weights_opt_->scalar_type(), "l2_eviction_handling", [&] {
            using value_t = scalar_t;
            FBGEMM_DISPATCH_INTEGRAL_TYPES(
                evicted_indices_opt_->scalar_type(),
                "l2_eviction_handling",
                [&] {
                  using index_t = scalar_t;
                  if (data.context ==
                      facebook::cachelib::RemoveContext::kEviction) {
                    auto indices_data_ptr =
                        evicted_indices_opt_->data_ptr<index_t>();
                    auto weights_data_ptr =
                        evicted_weights_opt_->data_ptr<value_t>();
                    auto row_id = eviction_row_id++;
                    auto weight_dim = evicted_weights_opt_->size(1);
                    const auto key_ptr = reinterpret_cast<const index_t*>(
                        data.item.getKey().data());
                    indices_data_ptr[row_id] = *key_ptr;

                    std::copy(
                        reinterpret_cast<const value_t*>(data.item.getMemory()),
                        reinterpret_cast<const value_t*>(
                            data.item.getMemory()) +
                            weight_dim,
                        &weights_data_ptr[row_id * weight_dim]); // dst_start
                  }
                });
          });
    }
  };

  int64_t rough_num_items =
      cache_config_.cache_size_bytes / cache_config_.item_size_bytes;
  unsigned int bucket_power = std::log(rough_num_items) / std::log(2) + 1;
  unsigned int lock_power =
      std::log(cache_config_.num_shards * 15) / std::log(2) + 1;

  XLOG(INFO) << fmt::format(
      "[TBE_ID{}] Setting up Cachelib for L2 cache, capacity: {}GB, "
      "item_size: {}B, max_num_items: {}, bucket_power: {}, lock_power: {}, "
      "use_object_cache: {}",
      unique_tbe_id_,
      config.cache_size_bytes / 1024 / 1024 / 1024,
      cache_config_.item_size_bytes,
      rough_num_items,
      bucket_power,
      lock_power,
      cache_config_.use_object_cache);

  // For ObjectCache mode, create ObjectCache which manages its own cache
  if (cache_config_.use_object_cache) {
    initializeObjectCacheInternal(rough_num_items);
    // Return nullptr for cache_ since ObjectCache manages its own
    // We'll access the underlying cache via object_cache_->getL1Cache() when
    // needed
    return nullptr;
  }

  // Regular allocator mode - create Cache with eviction callback
  Cache::Config cacheLibConfig;
  cacheLibConfig.setCacheSize(static_cast<uint64_t>(config.cache_size_bytes))
      .setRemoveCallback(eviction_cb)
      .setCacheName("TBEL2Cache")
      .setAccessConfig({bucket_power, lock_power})
      .setFullCoredump(false);

  cacheLibConfig.validate();
  return std::make_unique<Cache>(cacheLibConfig);
}

void CacheLibCache::initializeObjectCacheInternal(int64_t rough_num_items) {
  XLOG(INFO) << fmt::format(
      "[TBE_ID{}] ObjectCache mode enabled, initializing object cache",
      unique_tbe_id_);

  // Configure ObjectCache
  typename ObjectCache::Config objCacheConfig;
  objCacheConfig.setCacheName(
      fmt::format("TBEL2ObjectCache_{}", unique_tbe_id_));
  objCacheConfig.setCacheCapacity(rough_num_items);

  // Enable object size tracking to support getTotalObjectSize()
  objCacheConfig.objectSizeTrackingEnabled = true;

  // Set up item destructor for ObjectCache (handles evictions and removals)
  objCacheConfig.setItemDestructor([this](
                                       facebook::cachelib::objcache2::
                                           ObjectCacheDestructorData data) {
    // Handle evictions with the same callback logic
    if (data.context ==
        facebook::cachelib::objcache2::ObjectCacheDestructorContext::kEvicted) {
      // Track evictions for L2 cache eviction handling
      if (evicted_weights_opt_.has_value()) {
        FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
            evicted_weights_opt_->scalar_type(), "l2_eviction_handling", [&] {
              using value_t = scalar_t;
              FBGEMM_DISPATCH_INTEGRAL_TYPES(
                  evicted_indices_opt_->scalar_type(),
                  "l2_eviction_handling",
                  [&] {
                    using index_t = scalar_t;
                    auto indices_data_ptr =
                        evicted_indices_opt_->data_ptr<index_t>();
                    auto weights_data_ptr =
                        evicted_weights_opt_->data_ptr<value_t>();
                    auto row_id = eviction_row_id++;
                    auto weight_dim = evicted_weights_opt_->size(1);

                    // Parse key from the data.key (StringPiece)
                    const auto key_ptr =
                        reinterpret_cast<const index_t*>(data.key.data());
                    indices_data_ptr[row_id] = *key_ptr;

                    // Get the embedding value and copy data
                    auto* embedding_val =
                        reinterpret_cast<const EmbeddingValue*>(data.objectPtr);
                    std::copy(
                        reinterpret_cast<const value_t*>(
                            embedding_val->data.data()),
                        reinterpret_cast<const value_t*>(
                            embedding_val->data.data()) +
                            weight_dim,
                        &weights_data_ptr[row_id * weight_dim]);
                  });
            });
      }
    }

    // Clean up the EmbeddingValue object
    if (data.context ==
            facebook::cachelib::objcache2::ObjectCacheDestructorContext::
                kEvicted ||
        data.context ==
            facebook::cachelib::objcache2::ObjectCacheDestructorContext::
                kRemoved) {
      data.deleteObject<EmbeddingValue>();
    }
  });

  // Enable background item reaper for ObjectCache mode
  objCacheConfig.setItemReaperInterval(std::chrono::seconds{1});

  // Create the ObjectCache
  object_cache_ = ObjectCache::create(objCacheConfig);
}

std::unique_ptr<facebook::cachelib::CacheAdmin> CacheLibCache::createCacheAdmin(
    Cache& cache,
    bool is_object_cache) {
  facebook::cachelib::CacheAdmin::Config adminConfig;
  adminConfig.oncall = "mvai";
  // Disable background stats exporters for ObjectCache mode to avoid
  // race conditions and crashes during initialization
  if (is_object_cache) {
    adminConfig.globalOdsInterval = std::chrono::seconds{0};
    adminConfig.serviceDataStatsInterval = std::chrono::seconds{0};
    adminConfig.poolRebalancerStatsInterval = std::chrono::seconds{0};
  }
  return std::make_unique<facebook::cachelib::CacheAdmin>(
      cache, std::move(adminConfig));
}

// Template implementation for ObjectCache
template <typename CacheType>
std::unique_ptr<facebook::cachelib::CacheAdmin> CacheLibCache::createCacheAdmin(
    CacheType& cache,
    bool is_object_cache) {
  facebook::cachelib::CacheAdmin::Config adminConfig;
  adminConfig.oncall = "mvai";
  // Disable background stats exporters for ObjectCache mode to avoid
  // race conditions and crashes during initialization
  if (is_object_cache) {
    adminConfig.globalOdsInterval = std::chrono::seconds{0};
    adminConfig.serviceDataStatsInterval = std::chrono::seconds{0};
    adminConfig.poolRebalancerStatsInterval = std::chrono::seconds{0};
  }
  return std::make_unique<facebook::cachelib::CacheAdmin>(
      cache, std::move(adminConfig));
}

// Explicit template instantiation for ObjectCache
template std::unique_ptr<facebook::cachelib::CacheAdmin>
CacheLibCache::createCacheAdmin<CacheLibCache::ObjectCache>(
    CacheLibCache::ObjectCache& cache,
    bool is_object_cache);

folly::Optional<void*> CacheLibCache::get(
    const at::Tensor& key_tensor,
    std::shared_ptr<EmbeddingValue>* object_cache_value_out) {
  // Use ObjectCache if enabled
  if (cache_config_.use_object_cache) {
    return getFromObjectCache(key_tensor, object_cache_value_out);
  }

  // Fallback to regular allocator mode
  folly::Optional<void*> res;
  FBGEMM_DISPATCH_INTEGRAL_TYPES(key_tensor.scalar_type(), "get", [&] {
    using index_t = scalar_t;
    auto key = *(key_tensor.const_data_ptr<index_t>());
    auto key_str = folly::StringPiece(
        reinterpret_cast<const char*>(&key), sizeof(index_t));
    auto item = cache_->find(key_str);
    if (!item) {
      res = folly::none;
      return;
    }
    res = const_cast<void*>(item->getMemory());
  });
  return res;
}

size_t CacheLibCache::get_shard_id(int64_t key) {
  return kv_db_utils::hash_shard(key, pool_ids_.size());
}

facebook::cachelib::PoolId CacheLibCache::get_pool_id(int64_t key) {
  return pool_ids_[get_shard_id(key)];
}

void CacheLibCache::batchMarkUseful(
    const std::vector<Cache::ReadHandle>& read_handles) {
  if (read_handles.empty()) {
    return;
  }

  auto* cache_ptr = cache_config_.use_object_cache
      ? &(object_cache_->getL1Cache())
      : cache_.get();

  for (auto& handle : read_handles) {
    if (handle) {
      cache_ptr->markUseful(handle, facebook::cachelib::AccessMode::kRead);
    }
  }
}

bool CacheLibCache::put(const at::Tensor& key_tensor, const at::Tensor& data) {
  if (!index_dtype_.has_value()) {
    index_dtype_ = key_tensor.scalar_type();
  }
  if (!weights_dtype_.has_value()) {
    weights_dtype_ = data.scalar_type();
  }

  // Use ObjectCache if enabled
  if (cache_config_.use_object_cache) {
    return putToObjectCache(key_tensor, data);
  }

  // Fallback to regular allocator mode
  bool res;
  FBGEMM_DISPATCH_INTEGRAL_TYPES(key_tensor.scalar_type(), "put", [&] {
    using index_t = scalar_t;
    auto key = *(key_tensor.const_data_ptr<index_t>());
    auto key_str = folly::StringPiece(
        reinterpret_cast<const char*>(&key), sizeof(index_t));
    auto item = cache_->findToWrite(key_str);
    if (!item) {
      auto alloc_item =
          cache_->allocate(get_pool_id(key), key_str, data.nbytes());
      if (!alloc_item) {
        XLOG(ERR) << fmt::format(
            "[TBE_ID{}]Failed to allocate item {} in cache, skip",
            unique_tbe_id_,
            key);
        res = false;
        return;
      }
      std::memcpy(
          alloc_item->getMemory(), data.const_data_ptr(), data.nbytes());
      cache_->insertOrReplace(std::move(alloc_item));
    } else {
      std::memcpy(item->getMemory(), data.const_data_ptr(), data.nbytes());
    }
    res = true;
  });
  return res;
}

folly::Optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
CacheLibCache::get_n_items(int n, Cache::AccessIterator& itr) {
  if (!index_dtype_.has_value() || !weights_dtype_.has_value()) {
    return folly::none;
  }

  auto* cache_ptr = cache_config_.use_object_cache
      ? &(object_cache_->getL1Cache())
      : cache_.get();

  auto weight_dim = cache_config_.max_D_;
  auto indices = at::empty(
      n, at::TensorOptions().dtype(index_dtype_.value()).device(at::kCPU));
  auto weights = at::empty(
      {n, weight_dim},
      at::TensorOptions().dtype(weights_dtype_.value()).device(at::kCPU));
  int cnt = 0;
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "get_n_items", [&] {
        using value_t = scalar_t;
        FBGEMM_DISPATCH_INTEGRAL_TYPES(
            indices.scalar_type(), "get_n_items", [&] {
              using index_t = scalar_t;
              auto indices_data_ptr = indices.mutable_data_ptr<index_t>();
              auto weights_data_ptr = weights.data_ptr<value_t>();
              for (; itr != cache_ptr->end() && cnt < n; ++itr, ++cnt) {
                const auto key_ptr =
                    reinterpret_cast<const index_t*>(itr->getKey().data());
                indices_data_ptr[cnt] = *key_ptr;
                std::copy(
                    reinterpret_cast<const value_t*>(itr->getMemory()),
                    reinterpret_cast<const value_t*>(itr->getMemory()) +
                        weight_dim,
                    &weights_data_ptr[cnt * weight_dim]); // dst_start
              }
            });
      });
  return std::make_tuple(
      indices,
      weights,
      at::tensor({cnt}, at::TensorOptions().dtype(at::kLong).device(at::kCPU)));
}

void CacheLibCache::init_tensor_for_l2_eviction(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  CHECK_EQ(count.numel(), 1);
  auto num_lookups = count.scalar_type() == at::ScalarType::Long
      ? *(count.const_data_ptr<int64_t>())
      : *(count.const_data_ptr<int32_t>());
  evicted_indices_opt_ =
      at::ones(
          num_lookups, at::TensorOptions().device(at::kCPU).dtype(at::kLong)) *
      -1;
  evicted_weights_opt_ = at::empty(
      {num_lookups, weights.size(1)},
      at::TensorOptions().device(at::kCPU).dtype(weights.dtype()));
}

void CacheLibCache::reset_eviction_states() {
  evicted_indices_opt_.reset();
  evicted_weights_opt_.reset();
  eviction_row_id = 0;
  return;
}

folly::Optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
CacheLibCache::get_tensors_and_reset() {
  folly::Optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ret =
      folly::none;
  if (evicted_indices_opt_.has_value()) {
    CHECK(evicted_weights_opt_.has_value());
    if (eviction_row_id > 0) {
      ret = std::make_tuple(
          evicted_indices_opt_.value(),
          evicted_weights_opt_.value(),
          at::tensor(eviction_row_id, evicted_indices_opt_->options()));
    }
  }
  reset_eviction_states();
  return ret;
}

std::vector<int64_t> CacheLibCache::get_cache_usage() {
  std::vector<int64_t> cache_mem_stats(2, 0); // freeBytes, capacity
  cache_mem_stats[1] = cache_config_.cache_size_bytes;

  if (cache_config_.use_object_cache) {
    // For ObjectCache mode, use cache-level stats instead of pool stats
    // since ObjectCache manages its own internal pool structure
    int64_t used_mem = object_cache_->getTotalObjectSize();
    cache_mem_stats[0] = cache_config_.cache_size_bytes - used_mem;
  } else {
    // For regular allocator mode, use the pool_ids we created
    for (auto& pool_id : pool_ids_) {
      auto pool_stats = cache_->getPoolStats(pool_id);
      cache_mem_stats[0] += pool_stats.freeMemoryBytes();
    }
  }

  return cache_mem_stats;
}

folly::Optional<void*> CacheLibCache::getFromObjectCache(
    const at::Tensor& key_tensor,
    std::shared_ptr<EmbeddingValue>* object_cache_value_out) {
  folly::Optional<void*> res;
  FBGEMM_DISPATCH_INTEGRAL_TYPES(key_tensor.scalar_type(), "get", [&] {
    using index_t = scalar_t;
    auto key = *(key_tensor.data_ptr<index_t>());

    // Convert integer key to string key for ObjectCache
    auto key_str = folly::StringPiece(
        reinterpret_cast<const char*>(&key), sizeof(index_t));

    // Try to find the value in object cache
    auto found = object_cache_->find<EmbeddingValue>(key_str);
    if (!found) {
      res = folly::none;
      return;
    }

    // Store the value to keep it alive and prevent use-after-free
    // Cast away const since we need mutable access to the data
    auto value_ptr = std::const_pointer_cast<EmbeddingValue>(found);

    // Store in output parameter if provided (for multi-threaded contexts)
    // Otherwise fall back to member variable (for backward compatibility)
    if (object_cache_value_out) {
      *object_cache_value_out = value_ptr;
    } else {
      last_retrieved_value_ = value_ptr;
    }

    res = static_cast<void*>(value_ptr->data.data());
  });
  return res;
}

bool CacheLibCache::putToObjectCache(
    const at::Tensor& key_tensor,
    const at::Tensor& data) {
  bool res = false;
  FBGEMM_DISPATCH_INTEGRAL_TYPES(key_tensor.scalar_type(), "put", [&] {
    using index_t = scalar_t;
    auto key = *(key_tensor.data_ptr<index_t>());

    // Convert integer key to string key for ObjectCache
    auto key_str = folly::StringPiece(
        reinterpret_cast<const char*>(&key), sizeof(index_t));

    // Create EmbeddingValue from tensor data
    auto embedding_value =
        std::make_unique<EmbeddingValue>(data.data_ptr(), data.nbytes());

    // Insert or replace in object cache
    auto [insert_status, new_obj, old_obj] =
        object_cache_->insertOrReplace<EmbeddingValue>(
            key_str, std::move(embedding_value), data.nbytes());
    res = (insert_status == ObjectCache::AllocStatus::kSuccess);
  });
  return res;
}

} // namespace l2_cache
