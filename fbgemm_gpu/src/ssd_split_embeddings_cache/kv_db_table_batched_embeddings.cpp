/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kv_db_table_batched_embeddings.h"
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <algorithm>
#include "common/time/Time.h"
#include "kv_db_cuda_utils.h"
#include "torch/csrc/autograd/record_function_ops.h"
#ifdef FBGEMM_FBCODE
#include <folly/stop_watch.h>
#include "aiplatform/gmpp/experimental/training_ps/gen-cpp2/TrainingParameterServerService.h"
#include "caffe2/torch/fb/distributed/wireSerializer/WireSerializer.h"
#include "servicerouter/client/cpp2/ClientParams.h"
#include "servicerouter/client/cpp2/ServiceRouter.h"
#include "torch/types.h"
#endif

namespace kv_db {

namespace {

/// Read a scalar value from a tensor that is maybe a UVM tensor
/// Note that `tensor.item<type>()` is not allowed on a UVM tensor in
/// PyTorch
inline int64_t get_maybe_uvm_scalar(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Long
      ? *(tensor.data_ptr<int64_t>())
      : *(tensor.data_ptr<int32_t>());
}

#ifdef FBGEMM_FBCODE
/*
 * Get the thrift client to the training parameter server service
 * There is a destruction double free issue when wrapping the member
 * variable under ifdef, and creating client is relatively cheap, so create this
 * helper function to get the client just before sending requests.
 */
std::unique_ptr<
    apache::thrift::Client<aiplatform::gmpp::experimental::training_ps::
                               TrainingParameterServerService>>
get_res_client(int64_t res_server_port) {
  auto& factory = facebook::servicerouter::cpp2::getClientFactory();
  auto& params = facebook::servicerouter::ClientParams().setSingleHost(
      "::", res_server_port);
  return factory.getSRClientUnique<
      apache::thrift::Client<aiplatform::gmpp::experimental::training_ps::
                                 TrainingParameterServerService>>(
      "realtime.delta.publish.esr", params);
}
#endif

}; // namespace

QueueItem tensor_copy(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    kv_db::RocksdbWriteMode mode) {
  auto num_sets = get_maybe_uvm_scalar(count);
  auto new_indices = at::empty(
      num_sets, at::TensorOptions().device(at::kCPU).dtype(indices.dtype()));
  auto new_weights = at::empty(
      {num_sets, weights.size(1)},
      at::TensorOptions().device(at::kCPU).dtype(weights.dtype()));
  auto new_count =
      at::empty({1}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "tensor_copy", [&] {
        using value_t = scalar_t;
        FBGEMM_DISPATCH_INTEGRAL_TYPES(
            indices.scalar_type(), "tensor_copy", [&] {
              using index_t = scalar_t;
              auto indices_addr = indices.data_ptr<index_t>();
              auto new_indices_addr = new_indices.data_ptr<index_t>();
              std::copy(
                  indices_addr,
                  indices_addr + num_sets,
                  new_indices_addr); // dst_start

              auto weights_addr = weights.data_ptr<value_t>();
              auto new_weightss_addr = new_weights.data_ptr<value_t>();
              std::copy(
                  weights_addr,
                  weights_addr + num_sets * weights.size(1),
                  new_weightss_addr); // dst_start
            });
      });
  *new_count.data_ptr<int64_t>() = num_sets;
  return QueueItem{new_indices, new_weights, new_count, mode};
}

EmbeddingKVDB::EmbeddingKVDB(
    int64_t num_shards,
    int64_t max_D,
    int64_t cache_size_gb,
    int64_t unique_id,
    int64_t ele_size_bytes,
    bool enable_async_update,
    bool enable_raw_embedding_streaming,
    int64_t res_store_shards,
    int64_t res_server_port,
    std::vector<std::string> table_names,
    std::vector<int64_t> table_offsets,
    const std::vector<int64_t>& table_sizes)
    : unique_id_(unique_id),
      num_shards_(num_shards),
      max_D_(max_D),
      executor_tp_(std::make_unique<folly::CPUThreadPoolExecutor>(num_shards)),
      enable_async_update_(enable_async_update),
      enable_raw_embedding_streaming_(enable_raw_embedding_streaming),
      res_store_shards_(res_store_shards),
      res_server_port_(res_server_port),
      table_names_(std::move(table_names)),
      table_offsets_(std::move(table_offsets)),
      table_sizes_(at::tensor(table_sizes)) {
  CHECK(num_shards > 0);
  if (cache_size_gb > 0) {
    l2_cache::CacheLibCache::CacheConfig cache_config;
    cache_config.cache_size_bytes = cache_size_gb * 1024 * 1024 * 1024;
    cache_config.num_shards = num_shards_;
    cache_config.item_size_bytes = max_D_ * ele_size_bytes;
    cache_config.max_D_ = max_D_;
    l2_cache_ =
        std::make_unique<l2_cache::CacheLibCache>(cache_config, unique_id);
  } else {
    l2_cache_ = nullptr;
  }
  XLOG(INFO) << "[TBE_ID" << unique_id_ << "] L2 created with " << num_shards_
             << " shards, dimension:" << max_D_
             << ", enable_async_update_:" << enable_async_update_
             << ", enable_raw_embedding_streaming_:"
             << enable_raw_embedding_streaming_;

  if (enable_async_update_) {
    cache_filling_thread_ = std::make_unique<std::thread>([=] {
      while (!stop_) {
        auto filling_item_ptr = weights_to_fill_queue_.try_peek();
        if (!filling_item_ptr) {
          // TODO: make it tunable interval for background thread
          // only sleep on empty queue
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        if (stop_) {
          return;
        }
        auto& indices = filling_item_ptr->indices;
        auto& weights = filling_item_ptr->weights;
        auto& count = filling_item_ptr->count;
        auto& rocksdb_wmode = filling_item_ptr->mode;

        update_cache_and_storage(indices, weights, count, rocksdb_wmode);

        weights_to_fill_queue_.dequeue();
      }
    });
  }
#ifdef FBGEMM_FBCODE
  if (enable_raw_embedding_streaming_) {
    XLOG(INFO) << "[TBE_ID" << unique_id_
               << "] Raw embedding streaming enabled with res_server_port at"
               << res_server_port;
    // The first call to get the client is expensive, so eagerly get it here
    auto _eager_client = get_res_client(res_server_port_);

    weights_stream_thread_ = std::make_unique<std::thread>([=, this] {
      while (!stop_) {
        auto stream_item_ptr = weights_to_stream_queue_.try_peek();
        if (!stream_item_ptr) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        if (stop_) {
          return;
        }
        auto& indices = stream_item_ptr->indices;
        auto& weights = stream_item_ptr->weights;
        folly::stop_watch<std::chrono::milliseconds> stop_watch;
        folly::coro::blockingWait(tensor_stream(indices, weights));

        weights_to_stream_queue_.dequeue();
        XLOG_EVERY_MS(INFO, 60000)
            << "[TBE_ID" << unique_id_
            << "] end stream queue size: " << weights_to_stream_queue_.size()
            << " stream takes " << stop_watch.elapsed().count() << "ms";
      }
    });
  }
#endif
}

EmbeddingKVDB::~EmbeddingKVDB() {
  stop_ = true;
  if (enable_async_update_) {
    cache_filling_thread_->join();
  }
#ifdef FBGEMM_FBCODE
  if (enable_raw_embedding_streaming_) {
    weights_stream_thread_->join();
  }
#endif
}

#ifdef FBGEMM_FBCODE
folly::coro::Task<void> EmbeddingKVDB::tensor_stream(
    const at::Tensor& indices,
    const at::Tensor& weights) {
  using namespace ::aiplatform::gmpp::experimental::training_ps;
  if (indices.size(0) != weights.size(0)) {
    XLOG(ERR) << "[TBE_ID" << unique_id_
              << "] Indices and weights size mismatched " << indices.size(0)
              << " " << weights.size(0);
    co_return;
  }
  folly::stop_watch<std::chrono::milliseconds> stop_watch;
  XLOG_EVERY_MS(INFO, 60000)
      << "[TBE_ID" << unique_id_
      << "] send streaming request: indices = " << indices.size(0)
      << ", weights = " << weights.size(0);

  auto biggest_idx = table_sizes_.index({table_sizes_.size(0) - 1});
  auto mask =
      at::logical_and(indices >= 0, indices < biggest_idx).nonzero().squeeze();
  auto filtered_indices = indices.index_select(0, mask);
  auto filtered_weights = weights.index_select(0, mask);
  auto num_invalid_indices = indices.size(0) - filtered_indices.size(0);
  if (num_invalid_indices > 0) {
    XLOG(INFO) << "[TBE_ID" << unique_id_
               << "] number of invalid indices: " << num_invalid_indices;
  }
  // 1. Transform local row indices to embedding table global row indices
  at::Tensor table_indices =
      (at::searchsorted(table_sizes_, filtered_indices, false, true) - 1)
          .to(torch::kInt8);
  auto tb_ac = table_indices.accessor<int8_t, 1>();
  auto indices_ac = filtered_indices.accessor<int64_t, 1>();
  auto tb_sizes_ac = table_sizes_.accessor<int64_t, 1>();
  std::vector<int64_t> global_indices(tb_ac.size(0), 0);
  std::vector<int16_t> shard_indices(tb_ac.size(0), 0);

  for (int i = 0; i < tb_ac.size(0); ++i) {
    int tb_idx = tb_ac[i];
    global_indices[i] =
        indices_ac[i] - tb_sizes_ac[tb_idx] + table_offsets_[tb_idx];
    // hash to shard
    // if we do row range sharding, also shard here.
    auto fqn = table_names_[tb_idx];
    auto hash_key = folly::to<std::string>(fqn, global_indices[i]);
    auto shard_id =
        furcHash(hash_key.data(), hash_key.size(), res_store_shards_);
    shard_indices[i] = shard_id;
  }
  auto global_indices_tensor = at::tensor(global_indices);
  auto shard_indices_tensor = at::tensor(shard_indices);
  auto total_rows = global_indices_tensor.size(0);
  XLOG_EVERY_MS(INFO, 60000)
      << "[TBE_ID" << unique_id_ << "] hash and gloablize rows " << total_rows
      << " in: " << stop_watch.elapsed().count() << "ms";
  stop_watch.reset();

  auto res_client = get_res_client(res_server_port_);
  // 2. Split by shards
  for (int i = 0; i < res_store_shards_; ++i) {
    auto shrad_mask = shard_indices_tensor.eq(i).nonzero().squeeze();
    auto table_indices_masked = table_indices.index_select(0, shrad_mask);
    auto rows_in_shard = table_indices_masked.numel();
    if (rows_in_shard == 0) {
      continue;
    }
    auto global_indices_masked =
        global_indices_tensor.index_select(0, shrad_mask);
    auto weights_masked = filtered_weights.index_select(0, shrad_mask);

    if (weights_masked.size(0) != rows_in_shard ||
        global_indices_masked.numel() != rows_in_shard) {
      XLOG(ERR)
          << "[TBE_ID" << unique_id_
          << "] don't send the request for size mismatched tensors table: "
          << rows_in_shard << " weights: " << weights_masked.size(0)
          << " global_indices: " << global_indices_masked.numel();
      continue;
    }
    SetEmbeddingsRequest req;
    req.shardId() = i;
    req.fqns() = table_names_;

    req.tableIndices() =
        torch::distributed::wireDumpTensor(table_indices_masked);
    req.rowIndices() =
        torch::distributed::wireDumpTensor(global_indices_masked);
    req.weights() = torch::distributed::wireDumpTensor(weights_masked);
    co_await res_client->co_setEmbeddings(req);
  }
  co_return;
}
#endif

void EmbeddingKVDB::update_cache_and_storage(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    kv_db::RocksdbWriteMode mode,
    bool require_locking) {
  std::unique_lock<std::mutex> lock(l2_cache_mtx_, std::defer_lock);
  if (require_locking) {
    auto cache_update_start_ts = facebook::WallClockUtil::NowInUsecFast();
    lock.lock();
    set_cache_lock_wait_duration_ +=
        facebook::WallClockUtil::NowInUsecFast() - cache_update_start_ts;
  }
  if (l2_cache_) {
    auto evicted_tuples_opt = set_cache(indices, weights, count);
    if (evicted_tuples_opt.has_value()) {
      auto& evicted_indices = std::get<0>(evicted_tuples_opt.value());
      auto& evicted_weights = std::get<1>(evicted_tuples_opt.value());
      auto& evicted_count = std::get<2>(evicted_tuples_opt.value());

      set_kv_db_async(evicted_indices, evicted_weights, evicted_count, mode)
          .wait();
    }
  } else {
    set_kv_db_async(indices, weights, count, mode).wait();
  }
}

void EmbeddingKVDB::flush() {
  wait_util_filling_work_done();
  if (l2_cache_) {
    auto tensor_tuple_opt = l2_cache_->get_all_items();
    if (!tensor_tuple_opt.has_value()) {
      XLOG(INFO) << "[TBE_ID" << unique_id_
                 << "]no items exist in L2 cache, flush nothing";
      return;
    }
    auto& indices = std::get<0>(tensor_tuple_opt.value());
    auto& weights = std::get<1>(tensor_tuple_opt.value());
    auto& count = std::get<2>(tensor_tuple_opt.value());
    set_kv_db_async(indices, weights, count, kv_db::RocksdbWriteMode::FLUSH)
        .wait();
  }
}

void EmbeddingKVDB::get_cuda(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## EmbeddingKVDB::get_cuda ##");
  check_tensor_type_consistency(indices, weights);
  // take reference to self to avoid lifetime issues.
  auto self = shared_from_this();
  std::function<void()>* functor =
      new std::function<void()>([=]() { self->get(indices, weights, count); });
  AT_CUDA_CHECK(cudaStreamAddCallback(
      at::cuda::getCurrentCUDAStream(),
      kv_db_utils::cuda_callback_func,
      functor,
      0));
  rec->record.end();
}

void EmbeddingKVDB::set_cuda(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    const int64_t timestep,
    const bool is_bwd) {
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## EmbeddingKVDB::set_cuda ##");
  check_tensor_type_consistency(indices, weights);
  // take reference to self to avoid lifetime issues.
  auto self = shared_from_this();
  std::function<void()>* functor = new std::function<void()>([=]() {
    self->set(indices, weights, count, is_bwd);
    self->flush_or_compact(timestep);
  });
  AT_CUDA_CHECK(cudaStreamAddCallback(
      at::cuda::getCurrentCUDAStream(),
      kv_db_utils::cuda_callback_func,
      functor,
      0));
  rec->record.end();
}

std::vector<double> EmbeddingKVDB::get_l2cache_perf(
    const int64_t step,
    const int64_t interval) {
  std::vector<double> ret(15, 0); // num metrics
  if (step > 0 && step % interval == 0) {
    int reset_val = 0;
    auto num_cache_misses = num_cache_misses_.exchange(reset_val);
    auto num_lookups = num_lookups_.exchange(reset_val);
    auto num_evictions = num_evictions_.exchange(reset_val);
    auto get_total_duration = get_total_duration_.exchange(reset_val);
    auto get_cache_lookup_total_duration =
        get_cache_lookup_total_duration_.exchange(reset_val);
    auto get_cache_lookup_wait_filling_thread_duration =
        get_cache_lookup_wait_filling_thread_duration_.exchange(reset_val);
    auto get_weights_fillup_total_duration =
        get_weights_fillup_total_duration_.exchange(reset_val);
    auto get_cache_memcpy_duration =
        get_cache_memcpy_duration_.exchange(reset_val);

    auto total_cache_update_duration =
        total_cache_update_duration_.exchange(reset_val);
    auto get_tensor_copy_for_cache_update_dur =
        get_tensor_copy_for_cache_update_.exchange(reset_val);
    auto set_tensor_copy_for_cache_update_dur =
        set_tensor_copy_for_cache_update_.exchange(reset_val);

    auto set_cache_lock_wait_duration =
        set_cache_lock_wait_duration_.exchange(reset_val);
    auto get_cache_lock_wait_duration =
        get_cache_lock_wait_duration_.exchange(reset_val);

    ret[0] = (double(num_cache_misses) / interval);
    ret[1] = (double(num_lookups) / interval);
    ret[2] = (double(get_total_duration) / interval);
    ret[3] = (double(get_cache_lookup_total_duration) / interval);
    ret[4] = (double(get_cache_lookup_wait_filling_thread_duration) / interval);
    ret[5] = (double(get_weights_fillup_total_duration) / interval);
    ret[6] = (double(get_cache_memcpy_duration) / interval);
    ret[7] = (double(total_cache_update_duration) / interval);
    ret[8] = (double(get_tensor_copy_for_cache_update_dur) / interval);
    ret[9] = (double(set_tensor_copy_for_cache_update_dur) / interval);
    ret[10] = (double(num_evictions) / interval);
    if (l2_cache_) {
      auto cache_mem_stats = l2_cache_->get_cache_usage();
      ret[11] = (cache_mem_stats[0]); // free cache in bytes
      ret[12] = (cache_mem_stats[1]); // total cache capacity in bytes
    }
    ret[13] = (double(set_cache_lock_wait_duration) / interval);
    ret[14] = (double(get_cache_lock_wait_duration) / interval);
  }
  return ret;
}

void EmbeddingKVDB::reset_l2_cache() {
  l2_cache_ = nullptr;
}

void EmbeddingKVDB::set(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    const bool is_bwd) {
  if (auto num_evictions = get_maybe_uvm_scalar(count); num_evictions <= 0) {
    XLOG_EVERY_MS(INFO, 60000)
        << "[TBE_ID" << unique_id_
        << "]skip set_cuda since number evictions is " << num_evictions;
    return;
  }

  // defer the L2 cache/rocksdb update to the background thread as it could
  // be parallelized with other cuda kernels, as long as all updates are
  // finished before the next L2 cache lookup
  kv_db::RocksdbWriteMode write_mode = is_bwd
      ? kv_db::RocksdbWriteMode::BWD_L1_CNFLCT_MISS_WRITE_BACK
      : kv_db::RocksdbWriteMode::FWD_L1_EVICTION;
  if (enable_async_update_) {
    auto tensor_copy_start_ts = facebook::WallClockUtil::NowInUsecFast();
    auto new_item = tensor_copy(indices, weights, count, write_mode);
    weights_to_fill_queue_.enqueue(new_item);
    set_tensor_copy_for_cache_update_ +=
        facebook::WallClockUtil::NowInUsecFast() - tensor_copy_start_ts;
  } else {
    update_cache_and_storage(indices, weights, count, write_mode);
  }
}

void EmbeddingKVDB::get(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    int64_t sleep_ms) {
  if (auto num_lookups = get_maybe_uvm_scalar(count); num_lookups <= 0) {
    XLOG_EVERY_MS(INFO, 60000)
        << "[TBE_ID" << unique_id_ << "]skip get_cuda since number lookups is "
        << num_lookups;
    return;
  }
  CHECK_GE(max_D_, weights.size(1));
  auto start_ts = facebook::WallClockUtil::NowInUsecFast();
  wait_util_filling_work_done();

  std::unique_lock<std::mutex> lock(l2_cache_mtx_);
  get_cache_lock_wait_duration_ +=
      facebook::WallClockUtil::NowInUsecFast() - start_ts;

  // this is for unittest to repro synchronization situation
  // deterministically
  if (sleep_ms > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    XLOG(INFO) << "get sleep end";
  }

  auto cache_context = get_cache(indices, count);
  if (cache_context != nullptr) {
    if (cache_context->num_misses > 0) {
      // std::vector<folly::coro::Task<void>> tasks;

      auto weight_fillup_start_ts = facebook::WallClockUtil::NowInUsecFast();
      // tasks.emplace_back(get_kv_db_async(indices, weights, count));
      // tasks.emplace_back(
      //     cache_memcpy(weights, cache_context->cached_addr_list));
      auto rocksdb_fut_vec = get_kv_db_async(indices, weights, count);
      auto l2_fut_vec = cache_memcpy(weights, cache_context->cached_addr_list);
      rocksdb_fut_vec.wait();
      l2_fut_vec.wait();
      // folly::coro::blockingWait(folly::coro::collectAllRange(std::move(tasks)));
      get_weights_fillup_total_duration_ +=
          facebook::WallClockUtil::NowInUsecFast() - weight_fillup_start_ts;

      // defer the L2 cache/rocksdb update to the background thread as it
      // could be parallelized with other cuda kernels, as long as all
      // updates are finished before the next L2 cache lookup
      if (enable_async_update_) {
        auto tensor_copy_start_ts = facebook::WallClockUtil::NowInUsecFast();
        auto new_item = tensor_copy(
            indices, weights, count, kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ);
        weights_to_fill_queue_.enqueue(new_item);
        get_tensor_copy_for_cache_update_ +=
            facebook::WallClockUtil::NowInUsecFast() - tensor_copy_start_ts;
      } else {
        update_cache_and_storage(
            indices,
            weights,
            count,
            kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ,
            false /*require_locking=false*/);
      }
    } else {
      auto weight_fillup_start_ts = facebook::WallClockUtil::NowInUsecFast();
      cache_memcpy(weights, cache_context->cached_addr_list).wait();
      get_weights_fillup_total_duration_ +=
          facebook::WallClockUtil::NowInUsecFast() - weight_fillup_start_ts;
    }
  } else { // no l2 cache
    get_kv_db_async(indices, weights, count).wait();
  }
  get_total_duration_ += facebook::WallClockUtil::NowInUsecFast() - start_ts;
}

std::shared_ptr<CacheContext> EmbeddingKVDB::get_cache(
    const at::Tensor& indices,
    const at::Tensor& count) {
  if (l2_cache_ == nullptr) {
    return nullptr;
  }
  auto start_ts = facebook::WallClockUtil::NowInUsecFast();

  auto num_lookups = get_maybe_uvm_scalar(count);
  auto cache_context = std::make_shared<CacheContext>(num_lookups);
  FBGEMM_DISPATCH_INTEGRAL_TYPES(indices.scalar_type(), "get_cache", [&] {
    using index_t = scalar_t;
    auto indices_addr = indices.data_ptr<index_t>();
    auto num_shards = executor_tp_->numThreads();

    std::vector<folly::Future<folly::Unit>> futures;
    std::vector<std::vector<int>> row_ids_per_shard(num_shards);
    for (int i = 0; i < num_shards; i++) {
      row_ids_per_shard[i].reserve(num_lookups / num_shards * 2);
    }
    for (uint32_t row_id = 0; row_id < num_lookups; ++row_id) {
      row_ids_per_shard[l2_cache_->get_shard_id(indices_addr[row_id])]
          .emplace_back(row_id);
    }
    for (uint32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
      auto f = folly::via(executor_tp_.get())
                   .thenValue([=, &indices_addr, &indices, &row_ids_per_shard](
                                  folly::Unit) {
                     for (const auto& row_id : row_ids_per_shard[shard_id]) {
                       auto emb_idx = indices_addr[row_id];
                       if (emb_idx < 0) {
                         continue;
                       }
                       auto cached_addr_opt = l2_cache_->get(indices[row_id]);
                       if (cached_addr_opt.has_value()) { // cache hit
                         cache_context->cached_addr_list[row_id] =
                             cached_addr_opt.value();
                         indices_addr[row_id] = -1; // mark to sentinel value
                       } else { // cache miss
                         cache_context->num_misses += 1;
                       }
                     }
                   });
      futures.push_back(std::move(f));
    }
    folly::collect(futures).wait();

    // the following metrics added here as the current assumption is
    // get_cache will only be called in get_cuda path, if assumption no
    // longer true, we should wrap this up on the caller side
    auto dur = facebook::WallClockUtil::NowInUsecFast() - start_ts;
    get_cache_lookup_total_duration_ += dur;
    auto cache_misses = cache_context->num_misses.load();
    if (num_lookups > 0) {
      num_cache_misses_ += cache_misses;
      num_lookups_ += num_lookups;
    } else {
      XLOG_EVERY_MS(INFO, 60000)
          << "[TBE_ID" << unique_id_
          << "]num_lookups is 0, skip collecting the L2 cache miss stats";
    }
  });
  return cache_context;
}

void EmbeddingKVDB::wait_util_filling_work_done() {
  auto start_ts = facebook::WallClockUtil::NowInUsecFast();
  if (enable_async_update_) {
    int total_wait_time_ms = 0;
    while (!weights_to_fill_queue_.empty()) {
      // need to wait until all the pending weight-filling actions to finish
      // otherwise might get incorrect embeddings
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      total_wait_time_ms += 1;
      if (total_wait_time_ms > 100) {
        XLOG_EVERY_MS(ERR, 1000)
            << "[TBE_ID" << unique_id_
            << "]get_cache: waiting for L2 caching filling embeddings for "
            << total_wait_time_ms << " ms, somethings is likely wrong";
      }
    }
  }
  get_cache_lookup_wait_filling_thread_duration_ +=
      facebook::WallClockUtil::NowInUsecFast() - start_ts;
}

folly::Optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
EmbeddingKVDB::set_cache(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  if (l2_cache_ == nullptr) {
    return folly::none;
  }
  // TODO: consider whether need to reconstruct indices/weights/count and
  // free
  //       the original tensor since most of the tensor elem will be
  //       invalid, this will trade some perf for peak DRAM util saving

  auto cache_update_start_ts = facebook::WallClockUtil::NowInUsecFast();

  l2_cache_->init_tensor_for_l2_eviction(indices, weights, count);
  std::vector<folly::Future<folly::Unit>> futures;

  FBGEMM_DISPATCH_INTEGRAL_TYPES(indices.scalar_type(), "set_cache", [&] {
    using index_t = scalar_t;
    auto indices_addr = indices.data_ptr<index_t>();
    const int64_t num_lookups = get_maybe_uvm_scalar(count);
    auto num_shards = executor_tp_->numThreads();

    // std::vector<folly::coro::TaskWithExecutor<void>> tasks;
    std::vector<std::vector<int>> row_ids_per_shard(num_shards);

    for (int i = 0; i < num_shards; i++) {
      row_ids_per_shard[i].reserve(num_lookups / num_shards * 2);
    }
    for (uint32_t row_id = 0; row_id < num_lookups; ++row_id) {
      row_ids_per_shard[l2_cache_->get_shard_id(indices_addr[row_id])]
          .emplace_back(row_id);
    }

    for (uint32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
      auto f =
          folly::via(executor_tp_.get())
              .thenValue([=,
                          &indices_addr,
                          &indices,
                          &weights,
                          &row_ids_per_shard](folly::Unit) {
                for (const auto& row_id : row_ids_per_shard[shard_id]) {
                  auto emb_idx = indices_addr[row_id];
                  if (emb_idx < 0) {
                    continue;
                  }
                  if (!l2_cache_->put(indices[row_id], weights[row_id])) {
                    XLOG_EVERY_MS(ERR, 1000)
                        << "[TBE_ID" << unique_id_
                        << "]Failed to insert into cache, this shouldn't happen";
                  }
                }
              });
      futures.push_back(std::move(f));
    }
    // folly::coro::blockingWait(folly::coro::collectAllRange(std::move(tasks)));
    folly::collect(futures).wait();
  });
  total_cache_update_duration_ +=
      facebook::WallClockUtil::NowInUsecFast() - cache_update_start_ts;
  auto tensor_tuple_opt = l2_cache_->get_tensors_and_reset();
  if (tensor_tuple_opt.has_value()) {
    auto& num_evictions_tensor = std::get<2>(tensor_tuple_opt.value());
    auto num_evictions = get_maybe_uvm_scalar(num_evictions_tensor);
    num_evictions_ += num_evictions;
  }
  return tensor_tuple_opt;
}

folly::SemiFuture<std::vector<folly::Unit>> EmbeddingKVDB::cache_memcpy(
    const at::Tensor& weights,
    const std::vector<void*>& cached_addr_list) {
  auto cache_memcpy_start_ts = facebook::WallClockUtil::NowInUsecFast();
  std::vector<folly::Future<folly::Unit>> futures;
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "cache_memcpy", [&] {
        // std::vector<folly::coro::TaskWithExecutor<void>> tasks;
        auto weights_data_ptr = weights.data_ptr<scalar_t>();
        auto num_shards = executor_tp_->numThreads();
        for (uint32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
          auto f = folly::via(executor_tp_.get()).thenValue([=](folly::Unit) {
            for (int row_id = 0; row_id < cached_addr_list.size(); row_id++) {
              if (row_id % num_shards != shard_id) {
                continue;
              }
              if (cached_addr_list[row_id] == nullptr) {
                continue;
              }
              std::copy(
                  reinterpret_cast<const scalar_t*>(cached_addr_list[row_id]),
                  reinterpret_cast<const scalar_t*>(cached_addr_list[row_id]) +
                      max_D_,
                  &weights_data_ptr[row_id * max_D_]); // dst_start
            }
          });
          futures.push_back(std::move(f));
        }
      });
  get_cache_memcpy_duration_ +=
      facebook::WallClockUtil::NowInUsecFast() - cache_memcpy_start_ts;
  return folly::collect(futures);
}

void EmbeddingKVDB::check_tensor_type_consistency(
    const at::Tensor& indices,
    const at::Tensor& weights) {
  if (index_dtype_.has_value()) {
    CHECK(index_dtype_.value() == indices.scalar_type());
  } else {
    index_dtype_ = indices.scalar_type();
    XLOG(INFO) << "[TBE_ID" << unique_id_ << "]L2 cache index dtype is "
               << index_dtype_.value();
  }

  if (weights_dtype_.has_value()) {
    CHECK(weights_dtype_.value() == weights.scalar_type());
  } else {
    weights_dtype_ = weights.scalar_type();
    XLOG(INFO) << "[TBE_ID" << unique_id_ << "]L2 cache weights dtype is "
               << weights_dtype_.value();
  }
}

} // namespace kv_db
