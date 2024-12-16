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
#include <folly/executors/CPUThreadPoolExecutor.h>
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

#include <folly/coro/Task.h>
#include "fbgemm_gpu/split_embeddings_cache/cachelib_cache.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"

namespace kv_db {

/// @ingroup embedding-ssd
///
/// @brief It holds l2cache lookup results
///
/// num_misses is the number of misses in the l2 cache lookup
/// cached_addr_list is preallocated with number of lookups for better
/// parallelism
///   and invalid spot(cache misses) will stay as sentinel value
class CacheContext {
 public:
  explicit CacheContext(size_t num_keys) {
    cached_addr_list = std::vector<void*>(num_keys, nullptr);
  }
  // invalid spot will stay as sentinel value, this is trading space for better
  // parallelism
  std::atomic<int> num_misses{0};
  std::vector<void*> cached_addr_list;
};

/// @ingroup embedding-ssd
///
/// @brief rocksdb write mode
///
/// In SSD offloading there are 3 writes in each train iteration
/// FWD_ROCKSDB_READ: cache lookup will move uncached data from rocksdb into L2
/// cache on fwd path
///
/// FWD_L1_EVICTION: L1 cache eviciton will evict data into L2 cache on fwd path
///
/// BWD_L1_CNFLCT_MISS_WRITE_BACK: L1 conflict miss will insert into L2 for
/// embedding update on bwd path
///
/// All the L2 cache filling above will potentially trigger rocksdb write once
/// L2 cache is full
///
/// Additionally we will do ssd io on L2 flush
enum RocksdbWriteMode {
  FWD_ROCKSDB_READ = 0,
  FWD_L1_EVICTION = 1,
  BWD_L1_CNFLCT_MISS_WRITE_BACK = 2,
  FLUSH = 3,
};

/// @ingroup embedding-ssd
///
/// @brief queue item for background L2/rocksdb update
///
/// indices/weights/count are the corresponding set() params
///
/// read_handles is cachelib abstracted indices/embedding pair metadata, will be
/// later used on updating cachelib LRU queue as we separate it from
/// EmbeddingKVDB::get_cache()
///
/// mode is used for monitoring rocksdb write, checkout RocksdbWriteMode for
/// detailed explanation
struct QueueItem {
  at::Tensor indices;
  at::Tensor weights;
  at::Tensor count;
  RocksdbWriteMode mode;
  QueueItem(
      at::Tensor src_indices,
      at::Tensor src_weights,
      at::Tensor src_count,
      RocksdbWriteMode src_mode) {
    indices = src_indices;
    weights = src_weights;
    count = src_count;
    mode = src_mode;
  }
};

/// @ingroup embedding-ssd
///
/// @brief A class for interacting with different cache layers and storage
/// layers, public calls are executed on cuda stream
///
/// Currently it is used by TBE to offload Key(Embedding Index)
/// Value(Embeddings) to DRAM, SSD or remote storage, to provide better
/// scalability without blowing up HBM resources
class EmbeddingKVDB : public std::enable_shared_from_this<EmbeddingKVDB> {
 public:
  explicit EmbeddingKVDB(
      int64_t num_shards,
      int64_t max_D,
      int64_t cache_size_gb = 0,
      int64_t unique_id = 0,
      int64_t ele_size_bytes = 2 /*assume by default fp16*/,
      bool enable_async_update = false);

  virtual ~EmbeddingKVDB();

  /// Insert non-negative elements in <indices> and its paired embeddings
  /// from <weights> for the first <count> elements in the tensor.
  /// It will simply copy all 3 tensors and inject them into the background
  /// working item queue to be picked up for eviction/promotion handling between
  /// L2 cache and storage tier
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights The 2D tensor that each row(embeddings) is paired up with
  /// relative element in <indices>
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return None
  void set(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const bool is_bwd = false);

  /// Find non-negative embedding indices in <indices> and fill up the
  /// related embeddings into <weights> for the first <count> elements in the
  /// tensor.
  /// It contains 3 steps:
  /// - L2 cache lookup
  /// - filling weights concurrently from L2 cache(cache hit) and
  /// storage tier(Rocksdb/PS) on cache miss
  /// - for cache misses, copy 3 tensors and inject them into the background
  /// working item queue to be picked up for eviction/promotion handling between
  /// L2 cache and storage tier
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights The 2D tensor that each row(embeddings) is paired up with
  /// relative element in <indices>
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  /// @param sleep_ms this is used to specifically sleep in get function, this
  /// is needed to reproduce synchronization situation deterministicly, in prod
  /// case this will be 0 for sure
  ///
  /// @return None
  /// @note weights will be updated from either L2 cache or storage tier
  void get(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      int64_t sleep_ms = 0);

  /// storage tier counterpart of function get()
  virtual folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) = 0;

  /// storage tier counterpart of function set()
  virtual folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const RocksdbWriteMode w_mode = RocksdbWriteMode::FWD_ROCKSDB_READ) = 0;

  virtual void compact() = 0;

  /// Flush L2 cache into backend storage
  /// @return None
  /// @note caller side should mananger the timing to make sure flush doens't
  /// happen at the same time as get/set
  /// @note flush only flushes L2 cache, if there is cache on the backend
  /// storage, that flush should be called as well
  void flush();

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

  /// export internally collected L2 performance metrics out
  ///
  /// @param step the training step that caller side wants to report the stats
  /// @param interval report interval in terms of training step
  ///
  /// @return a list of doubles with predefined order for each metrics
  std::vector<double> get_l2cache_perf(
      const int64_t step,
      const int64_t interval);

  // reset L2 cache, this is used for unittesting to bypass l2 cache
  void reset_l2_cache();

  // block waiting for working items in queue to be finished, this is called by
  // get_cache() as embedding read should wait until previous write to be
  // finished, it could also be called in unitest to sync
  void wait_util_filling_work_done();

 private:
  /// Find non-negative embedding indices in <indices> and shard them into
  /// #cachelib_pools pieces to be lookedup in parallel
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return preallocated list of memory pointer with <count> size, cache miss
  /// or invalid embedding indices will have sentinel pointer(nullptr)
  /// @note element in <indices> will be updated to sentinel value on cache hit
  std::shared_ptr<CacheContext> get_cache(
      const at::Tensor& indices,
      const at::Tensor& count);

  /// Find non-negative embedding indices in <indices> and shard them into
  /// #cachelib_pools pieces, insert into Cachelib in parallel with their paired
  /// embeddings from <weights>
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights The 2D tensor that each row(embeddings) is paired up with
  /// relative element in <indices>
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return None if L2 is missing or no eviction, other wise return tuple of
  /// tensors with length of <count> containing L2 evicted embedding indices and
  /// embeddings, invalid pairs will have sentinel value(-1) on <indices>
  folly::Optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> set_cache(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count);

  /// Find valid cache pointer from <cached_addr_list> and memcopy the
  /// embeddings from L2 cache to the same slot(row id) in <weights>
  ///
  /// @param cached_addr_list The 1D vector containing L2 cache addr pointer,
  /// sentinel pointer on cache miss in the related slot value
  /// @param weights The 2D tensor that each row(embeddings) is paired up with
  /// relative slot in <cached_addr_list>
  ///
  /// @return None
  /// @note weigths will be updated on the slot that paired up with valid cache
  /// addr pointer
  folly::SemiFuture<std::vector<folly::Unit>> cache_memcpy(
      const at::Tensor& weights,
      const std::vector<void*>& cached_addr_list);

  /// update the L2 cache and backend storage with the given
  /// indices/weights/count
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights The 2D tensor that each row(embeddings) is paired up with
  /// relative element in <indices>
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  /// @param mode is used for monitoring rocksdb write, checkout
  /// RocksdbWriteMode for detailed explanation
  /// @param require_locking whether function will require locking, this is
  /// avoiding deadlock on get() since there will be a lock held on get()
  /// stack before calling update_cache_and_storage
  ///
  /// @return None
  void update_cache_and_storage(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      kv_db::RocksdbWriteMode mode,
      bool require_locking = true);

  virtual void flush_or_compact(const int64_t timestep) = 0;

  void check_tensor_type_consistency(
      const at::Tensor& indices,
      const at::Tensor& weights);

  std::unique_ptr<l2_cache::CacheLibCache> l2_cache_;
  const int64_t unique_id_;
  const int64_t num_shards_;
  const int64_t max_D_;
  folly::Optional<at::ScalarType> index_dtype_{folly::none};
  folly::Optional<at::ScalarType> weights_dtype_{folly::none};
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_tp_;
  bool enable_async_update_;
  std::unique_ptr<std::thread> cache_filling_thread_;
  std::atomic<bool> stop_{false};
  // buffer queue that stores all the needed indices/weights/action_count to
  // fill up cache
  folly::USPSCQueue<QueueItem, true> weights_to_fill_queue_;
  // In non pipelining mode, the sequence is
  //   - get_cuda(): L2 read and insert L2 cache misses into queue for
  //                 bg L2 write
  //   - L1 cache eviction: insert into bg queue for L2 write
  //   - ScratchPad update: insert into bg queue for L2 write
  // in non-prefetch pipeline, cuda synchronization guarantee get_cuda() happen
  // after SP update
  // in prefetch pipeline, cuda sync only guarantee get_cuda() happen after L1
  // cache eviction pipeline case, SP bwd update could happen in parallel with
  // L2 read mutex is used for l2 cache to do read / write exclusively
  std::mutex l2_cache_mtx_;

  // perf stats
  // --  perf of get() function
  // cache miss rate(cmr) is avged on cmr per iteration
  // instead of SUM(cache miss per interval) / SUM(lookups per interval)
  std::atomic<int64_t> num_cache_misses_{0};
  std::atomic<int64_t> num_lookups_{0};
  std::atomic<int64_t> num_evictions_{0};
  std::atomic<int64_t> get_total_duration_{0};
  std::atomic<int64_t> get_cache_lookup_total_duration_{0};
  std::atomic<int64_t> get_cache_lookup_wait_filling_thread_duration_{0};
  std::atomic<int64_t> get_weights_fillup_total_duration_{0};
  std::atomic<int64_t> get_cache_memcpy_duration_{0};
  std::atomic<int64_t> get_tensor_copy_for_cache_update_{0};
  std::atomic<int64_t> get_cache_lock_wait_duration_{0};

  // -- perf of set() function
  std::atomic<int64_t> set_tensor_copy_for_cache_update_{0};
  std::atomic<int64_t> set_cache_lock_wait_duration_{0};

  // -- commone path
  std::atomic<int64_t> total_cache_update_duration_{0};
}; // class EmbeddingKVDB

} // namespace kv_db
