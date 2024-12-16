/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/split_embeddings_cache/cachelib_cache.h"
#include <cmath>
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"

namespace l2_cache {

using Cache = facebook::cachelib::LruAllocator;

CacheLibCache::CacheLibCache(
    const CacheConfig& cache_config,
    int64_t unique_tbe_id)
    : cache_config_(cache_config),
      unique_tbe_id_(unique_tbe_id),
      cache_(initializeCacheLib(cache_config_)),
      admin_(createCacheAdmin(*cache_)) {
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

std::unique_ptr<Cache> CacheLibCache::initializeCacheLib(
    const CacheConfig& config) {
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
  Cache::Config cacheLibConfig;
  int64_t rough_num_items =
      cache_config_.cache_size_bytes / cache_config_.item_size_bytes;
  unsigned int bucket_power = std::log(rough_num_items) / std::log(2) + 1;
  // 15 here is a magic number between 10 and 20
  unsigned int lock_power =
      std::log(cache_config_.num_shards * 15) / std::log(2) + 1;
  XLOG(INFO) << fmt::format(
      "[TBE_ID{}] Setting up Cachelib for L2 cache, capacity: {}GB, "
      "item_size: {}B, max_num_items: {}, bucket_power: {}, lock_power: {}",
      unique_tbe_id_,
      config.cache_size_bytes / 1024 / 1024 / 1024,
      cache_config_.item_size_bytes,
      rough_num_items,
      bucket_power,
      lock_power);
  cacheLibConfig.setCacheSize(static_cast<uint64_t>(config.cache_size_bytes))
      .setRemoveCallback(eviction_cb)
      .setCacheName("TBEL2Cache")
      .setAccessConfig({bucket_power, lock_power})
      .setFullCoredump(false)
      .validate();
  return std::make_unique<Cache>(cacheLibConfig);
}

std::unique_ptr<facebook::cachelib::CacheAdmin> CacheLibCache::createCacheAdmin(
    Cache& cache) {
  facebook::cachelib::CacheAdmin::Config adminConfig;
  adminConfig.oncall = "mvai";
  return std::make_unique<facebook::cachelib::CacheAdmin>(
      cache, std::move(adminConfig));
}

folly::Optional<void*> CacheLibCache::get(const at::Tensor& key_tensor) {
  folly::Optional<void*> res;
  FBGEMM_DISPATCH_INTEGRAL_TYPES(key_tensor.scalar_type(), "get", [&] {
    using index_t = scalar_t;
    auto key = *(key_tensor.data_ptr<index_t>());
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
  for (auto& handle : read_handles) {
    if (handle) {
      cache_->markUseful(handle, facebook::cachelib::AccessMode::kRead);
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
  bool res;
  FBGEMM_DISPATCH_INTEGRAL_TYPES(key_tensor.scalar_type(), "put", [&] {
    using index_t = scalar_t;
    auto key = *(key_tensor.data_ptr<index_t>());
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
      std::memcpy(alloc_item->getMemory(), data.data_ptr(), data.nbytes());
      cache_->insertOrReplace(std::move(alloc_item));
    } else {
      std::memcpy(item->getMemory(), data.data_ptr(), data.nbytes());
    }
    res = true;
  });
  return res;
}

folly::Optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
CacheLibCache::get_all_items() {
  if (!index_dtype_.has_value() || !weights_dtype_.has_value()) {
    return folly::none;
  }
  int total_num_items = 0;
  for (auto& pool_id : pool_ids_) {
    total_num_items += cache_->getPoolStats(pool_id).numItems();
  }
  auto weight_dim = cache_config_.max_D_;
  auto indices = at::empty(
      total_num_items,
      at::TensorOptions().dtype(index_dtype_.value()).device(at::kCPU));
  auto weights = at::empty(
      {total_num_items, weight_dim},
      at::TensorOptions().dtype(weights_dtype_.value()).device(at::kCPU));
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "get_all_items", [&] {
        using value_t = scalar_t;
        FBGEMM_DISPATCH_INTEGRAL_TYPES(
            indices.scalar_type(), "get_all_items", [&] {
              using index_t = scalar_t;
              auto indices_data_ptr = indices.data_ptr<index_t>();
              auto weights_data_ptr = weights.data_ptr<value_t>();
              int64_t item_idx = 0;
              for (auto itr = cache_->begin(); itr != cache_->end(); ++itr) {
                const auto key_ptr =
                    reinterpret_cast<const index_t*>(itr->getKey().data());
                indices_data_ptr[item_idx] = *key_ptr;
                std::copy(
                    reinterpret_cast<const value_t*>(itr->getMemory()),
                    reinterpret_cast<const value_t*>(itr->getMemory()) +
                        weight_dim,
                    &weights_data_ptr[item_idx * weight_dim]); // dst_start
                item_idx++;
              }
              CHECK_EQ(total_num_items, item_idx);
            });
      });
  return std::make_tuple(
      indices,
      weights,
      at::tensor(
          {total_num_items},
          at::TensorOptions().dtype(at::kLong).device(at::kCPU)));
}

void CacheLibCache::init_tensor_for_l2_eviction(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  CHECK_EQ(count.numel(), 1);
  auto num_lookups = count.scalar_type() == at::ScalarType::Long
      ? *(count.data_ptr<int64_t>())
      : *(count.data_ptr<int32_t>());
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
  for (auto& pool_id : pool_ids_) {
    auto pool_stats = cache_->getPoolStats(pool_id);
    cache_mem_stats[0] += pool_stats.freeMemoryBytes();
  }
  return cache_mem_stats;
}

} // namespace l2_cache
