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

class CacheLibCache {
 public:
  using Cache = facebook::cachelib::LruAllocator;
  struct CacheConfig {
    size_t cacheSizeBytes;
  };

  explicit CacheLibCache(size_t cacheSizeBytes)
      : cacheConfig_(CacheConfig{.cacheSizeBytes = cacheSizeBytes}),
        cache_(initializeCacheLib(cacheConfig_)),
        defaultPool_(cache_->addPool(
            "default",
            cache_->getCacheMemoryStats().ramCacheSize)),
        admin_(createCacheAdmin(*cache_)) {}

  std::unique_ptr<Cache> initializeCacheLib(const CacheConfig& config) {
    Cache::Config cacheLibConfig;
    cacheLibConfig.setCacheSize(static_cast<uint64_t>(config.cacheSizeBytes))
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

  std::optional<std::pair<void*, uint32_t>> get(const std::string& key) {
    auto item = cache_->find(key);
    if (!item) {
      return std::nullopt;
    }
    void* dataPtr = const_cast<void*>(item->getMemory());
    return std::make_pair(dataPtr, item->getSize());
  }

  bool put(const std::string& key, const at::Tensor& data) {
    auto item = cache_->allocate(defaultPool_, key, data.nbytes());
    if (!item) {
      XLOG_EVERY_N(INFO, 1000)
          << fmt::format("Failed to allocate item {} in cache, skip", key);
      return false;
    }
    std::memcpy(item->getMemory(), data.data_ptr(), data.nbytes());
    cache_->insertOrReplace(std::move(item));
    return true;
  }

 private:
  const CacheConfig cacheConfig_;
  std::unique_ptr<Cache> cache_;
  const facebook::cachelib::PoolId defaultPool_;
  std::unique_ptr<facebook::cachelib::CacheAdmin> admin_;
};

} // namespace l2_cache
