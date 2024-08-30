/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kv_db_table_batched_embeddings.h"
#include <folly/coro/Collect.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <algorithm>
#include "common/time/Time.h"
#include "kv_db_cuda_utils.h"
#include "torch/csrc/autograd/record_function_ops.h"

namespace kv_db {

std::tuple<at::Tensor, at::Tensor, at::Tensor> tensor_copy(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  auto num_sets = count.item<long>();
  auto new_indices = at::empty(
      num_sets, at::TensorOptions().device(at::kCPU).dtype(indices.dtype()));
  auto new_weights = at::empty(
      {num_sets, weights.size(1)},
      at::TensorOptions().device(at::kCPU).dtype(weights.dtype()));
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "cache_memcpy", [&] {
        auto indices_addr = indices.data_ptr<int64_t>();
        auto new_indices_addr = new_indices.data_ptr<int64_t>();
        std::copy(
            indices_addr,
            indices_addr + num_sets,
            new_indices_addr); // dst_start

        auto weights_addr = weights.data_ptr<scalar_t>();
        auto new_weightss_addr = new_weights.data_ptr<scalar_t>();
        std::copy(
            weights_addr,
            weights_addr + num_sets * weights.size(1),
            new_weightss_addr); // dst_start
      });
  return std::make_tuple(new_indices, new_weights, count.clone());
}

void EmbeddingKVDB::get_cuda(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## EmbeddingKVDB::get_cuda ##");
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
  std::vector<double> ret;
  ret.reserve(9); // num metrics
  if (step > 0 && step % interval == 0) {
    int reset_val = 0;
    auto num_cache_misses = num_cache_misses_.exchange(reset_val);
    auto num_lookups = num_lookups_.exchange(reset_val);
    auto get_total_duration = get_total_duration_.exchange(reset_val);
    auto get_cache_lookup_total_duration =
        get_cache_lookup_total_duration_.exchange(reset_val);
    auto get_cache_lookup_wait_filling_thread_duration =
        get_cache_lookup_wait_filling_thread_duration_.exchange(reset_val);
    auto get_weights_fillup_total_duration =
        get_weights_fillup_total_duration_.exchange(reset_val);
    auto get_cache_update_total_duration =
        get_cache_update_total_duration_.exchange(reset_val);
    auto get_tensor_copy_for_cache_update_dur =
        get_tensor_copy_for_cache_update_.exchange(reset_val);
    auto set_tensor_copy_for_cache_update_dur =
        set_tensor_copy_for_cache_update_.exchange(reset_val);
    ret.push_back(double(num_cache_misses) / interval);
    ret.push_back(double(num_lookups) / interval);
    ret.push_back(double(get_total_duration) / interval);
    ret.push_back(double(get_cache_lookup_total_duration) / interval);
    ret.push_back(
        double(get_cache_lookup_wait_filling_thread_duration) / interval);
    ret.push_back(double(get_weights_fillup_total_duration) / interval);
    ret.push_back(double(get_cache_update_total_duration) / interval);
    ret.push_back(double(get_tensor_copy_for_cache_update_dur) / interval);
    ret.push_back(double(set_tensor_copy_for_cache_update_dur) / interval);
  }
  return ret;
}

void EmbeddingKVDB::set(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    const bool is_bwd) {
  if (auto num_evictions = count.item<long>(); num_evictions <= 0) {
    XLOG_EVERY_MS(INFO, 60000)
        << "[" << unique_id_ << "]skip set_cuda since number evictions is "
        << num_evictions;
    return;
  }
  auto start_ts = facebook::WallClockUtil::NowInUsecFast();

  // defer the L2 cache/rocksdb update to the background thread as it could be
  // parallelized with other cuda kernels, as long as all updates are finished
  // before the next L2 cache lookup
  auto tensor_copy_start_ts = facebook::WallClockUtil::NowInUsecFast();
  auto new_tuple = tensor_copy(indices, weights, count);
  weights_to_fill_queue_.enqueue(new_tuple);
  set_tensor_copy_for_cache_update_ +=
      facebook::WallClockUtil::NowInUsecFast() - tensor_copy_start_ts;
}

void EmbeddingKVDB::get(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  if (auto num_lookups = count.item<long>(); num_lookups <= 0) {
    XLOG_EVERY_MS(INFO, 60000)
        << "[" << unique_id_ << "]skip get_cuda since number lookups is "
        << num_lookups;
    return;
  }
  ASSERT_EQ(max_D_, weights.size(1));
  auto start_ts = facebook::WallClockUtil::NowInUsecFast();
  wait_util_filling_work_done();
  auto cache_context = get_cache(indices, count);
  if (cache_context != nullptr) {
    if (cache_context->num_misses > 0) {
      std::vector<folly::coro::Task<void>> tasks;

      auto weight_fillup_start_ts = facebook::WallClockUtil::NowInUsecFast();
      tasks.emplace_back(get_kv_db_async(indices, weights, count));
      tasks.emplace_back(
          cache_memcpy(weights, cache_context->cached_addr_list));
      folly::coro::blockingWait(folly::coro::collectAllRange(std::move(tasks)));
      get_weights_fillup_total_duration_ +=
          facebook::WallClockUtil::NowInUsecFast() - weight_fillup_start_ts;

      auto cache_update_start_ts = facebook::WallClockUtil::NowInUsecFast();
      // defer the L2 cache/rocksdb update to the background thread as it could
      // be parallelized with other cuda kernels, as long as all updates are
      // finished before the next L2 cache lookup
      auto tensor_copy_start_ts = facebook::WallClockUtil::NowInUsecFast();
      auto new_tuple = tensor_copy(indices, weights, count);
      weights_to_fill_queue_.enqueue(new_tuple);
      get_tensor_copy_for_cache_update_ +=
          facebook::WallClockUtil::NowInUsecFast() - tensor_copy_start_ts;
      get_cache_update_total_duration_ +=
          facebook::WallClockUtil::NowInUsecFast() - cache_update_start_ts;
    } else {
      auto weight_fillup_start_ts = facebook::WallClockUtil::NowInUsecFast();
      folly::coro::blockingWait(
          cache_memcpy(weights, cache_context->cached_addr_list));
      get_weights_fillup_total_duration_ +=
          facebook::WallClockUtil::NowInUsecFast() - weight_fillup_start_ts;
    }
  } else { // no l2 cache
    folly::coro::blockingWait(get_kv_db_async(indices, weights, count));
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
  auto indices_addr = indices.data_ptr<int64_t>();
  auto num_lookups = count.item<long>();
  auto cache_context = std::make_shared<CacheContext>(num_lookups);

  auto num_shards = executor_tp_->numThreads();

  std::vector<folly::coro::TaskWithExecutor<void>> tasks;
  std::vector<std::vector<int>> row_ids_per_shard(num_shards);
  for (int i = 0; i < num_shards; i++) {
    row_ids_per_shard[i].reserve(num_lookups / num_shards * 2);
  }
  for (uint32_t row_id = 0; row_id < num_lookups; ++row_id) {
    row_ids_per_shard[l2_cache_->get_shard_id(indices_addr[row_id])]
        .emplace_back(row_id);
  }
  for (uint32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
    tasks.emplace_back(
        folly::coro::co_invoke(
            [this,
             &indices_addr,
             cache_context,
             shard_id,
             &row_ids_per_shard]() mutable -> folly::coro::Task<void> {
              for (const auto& row_id : row_ids_per_shard[shard_id]) {
                auto emb_idx = indices_addr[row_id];
                if (emb_idx < 0) {
                  continue;
                }
                auto cached_addr_opt = l2_cache_->get(emb_idx);
                if (cached_addr_opt.has_value()) { // cache hit
                  cache_context->cached_addr_list[row_id] =
                      cached_addr_opt.value();
                  indices_addr[row_id] = -1; // mark to sentinel value
                } else { // cache miss
                  cache_context->num_misses += 1;
                }
              }
              co_return;
            })
            .scheduleOn(executor_tp_.get()));
  }
  folly::coro::blockingWait(folly::coro::collectAllRange(std::move(tasks)));

  // the following metrics added here as the current assumption is
  // get_cache will only be called in get_cuda path, if assumption no longer
  // true, we should wrap this up on the caller side
  auto dur = facebook::WallClockUtil::NowInUsecFast() - start_ts;
  get_cache_lookup_total_duration_ += dur;
  auto cache_misses = cache_context->num_misses.load();
  if (num_lookups > 0) {
    num_cache_misses_ += cache_misses;
    num_lookups_ += num_lookups;
  } else {
    XLOG_EVERY_MS(INFO, 60000)
        << "[" << unique_id_
        << "]num_lookups is 0, skip collecting the L2 cache miss stats";
  }
  return cache_context;
}

void EmbeddingKVDB::wait_util_filling_work_done() {
  auto start_ts = facebook::WallClockUtil::NowInUsecFast();
  int total_wait_time_ms = 0;
  while (!weights_to_fill_queue_.empty()) {
    // need to wait until all the pending weight-filling actions to finish
    // otherwise might get incorrect embeddings
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    total_wait_time_ms += 1;
    if (total_wait_time_ms > 100) {
      XLOG_EVERY_MS(ERR, 1000)
          << "get_cache: waiting for L2 caching filling embeddings for "
          << total_wait_time_ms << " ms, somethings is likely wrong";
    }
  }
  get_cache_lookup_wait_filling_thread_duration_ +=
      facebook::WallClockUtil::NowInUsecFast() - start_ts;
}

void EmbeddingKVDB::set_cache(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  // TODO: consider whether need to reconstruct indices/weights/count and free
  //       the original tensor since most of the tensor elem will be invalid,
  //       this will trade some perf for peak DRAM util saving
  if (l2_cache_ == nullptr) {
    return;
  }
  auto indices_addr = indices.data_ptr<int64_t>();
  auto num_lookups = count.item<long>();
  for (auto i = 0; i < num_lookups; i++) {
    if (indices_addr[i] < 0) {
      continue;
    }
    if (!l2_cache_->put(indices_addr[i], weights[i])) {
      XLOG(ERR) << "[" << unique_id_
                << "]Failed to insert into cache, this shouldn't happen";
    }
  }
}

folly::coro::Task<void> EmbeddingKVDB::cache_memcpy(
    const at::Tensor& weights,
    const std::vector<void*>& cached_addr_list) {
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "cache_memcpy", [&] {
        auto weights_data_ptr = weights.data_ptr<scalar_t>();
        for (int row_id = 0; row_id < cached_addr_list.size(); row_id++) {
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
  co_return;
}

} // namespace kv_db
