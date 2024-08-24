/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <mkl.h>
#endif
#include <random>

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <folly/container/F14Map.h>
#include <glog/logging.h>

#include <folly/Random.h>
#include <folly/concurrency/UnboundedQueue.h>
#include <folly/futures/Future.h>
#include <folly/hash/Hash.h>

#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/rate_limiter.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <rocksdb/table_properties.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <folly/experimental/coro/Task.h>
#include "deeplearning/fbgemm/fbgemm_gpu/src/split_embeddings_cache/CacheLibCache.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "kv_db_utils.h"

namespace kv_db {

class EmbeddingKVDB : public std::enable_shared_from_this<EmbeddingKVDB> {
 public:
  struct CacheContext {
    std::vector<int64_t> missed_indices;
    std::unordered_map<int64_t, int> missed_indices_map;
    // pair: start address, data size
    std::unordered_map<int, std::pair<void*, uint32_t>> cached_address_pairs;
  };
  explicit EmbeddingKVDB(int64_t cache_size_gb = 0) {
    l2Cache_ = cache_size_gb > 0 ? std::make_unique<l2_cache::CacheLibCache>(
                                       cache_size_gb * 1024 * 1024 * 1024)
                                 : nullptr;
  }
  virtual ~EmbeddingKVDB() = default;

  void set(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const bool is_bwd = false);

  void get(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count);

  virtual folly::coro::Task<void> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) = 0;

  virtual folly::coro::Task<void> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const bool is_bwd = false) = 0;

  virtual void compact() = 0;

  virtual void flush() = 0;

  // The function attaches the CUDA callback logic to the compute
  // stream to ensure that the data retrieval is carried out properly.
  // It internally invokes get to fetch values from the KV database.
  void get_cuda(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count);

  void set_cuda(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const int64_t timestep,
      const bool is_bwd = false);

 private:
  std::shared_ptr<CacheContext> get_cache(
      const at::Tensor& indices,
      const at::Tensor& count);
  std::shared_ptr<CacheContext> set_cache(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count);
  folly::coro::Task<void> cache_memcpy(
      const at::Tensor& weights,
      std::unordered_map<int, std::pair<void*, uint32_t>>& address_pairs);
  void lookup_memcpy(
      const at::Tensor& src_weights,
      const at::Tensor& dst_weights,
      const std::unordered_map<int64_t, int>& indices_map);
  void from_cache_miss(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      at::Tensor& missed_indices,
      at::Tensor& missed_weights,
      at::Tensor& missed_count,
      std::shared_ptr<CacheContext>& cache_context);
  virtual void flush_or_compact(const int64_t timestep) = 0;
  std::unique_ptr<l2_cache::CacheLibCache> l2Cache_;
}; // class EmbeddingKVDB

} // namespace kv_db
