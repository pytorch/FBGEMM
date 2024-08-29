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

void EmbeddingKVDB::set(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    const bool is_bwd) {
  folly::stop_watch<std::chrono::microseconds> timer;
  set_cache(indices, weights, count);
  XLOG_EVERY_N(INFO, 1000) << "set_cuda: finished set embeddings in "
                           << timer.elapsed().count() << " us.";
}

void EmbeddingKVDB::get(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  ASSERT_EQ(max_D_, weights.size(1));
  folly::stop_watch<std::chrono::microseconds> timer;
  auto cache_context = get_cache(indices, count);
  if (cache_context != nullptr) {
    if (cache_context->num_misses > 0) {
      XLOG(INFO) << "[" << unique_id_
                 << "]cache miss: " << cache_context->num_misses << " out of "
                 << count.item().toLong() << " lookups";
      std::vector<folly::coro::Task<void>> tasks;

      tasks.emplace_back(get_kv_db_async(indices, weights, count));
      tasks.emplace_back(
          cache_memcpy(weights, cache_context->cached_addr_list));
      folly::coro::blockingWait(folly::coro::collectAllRange(std::move(tasks)));
      set_cache(indices, weights, count);
    } else {
      XLOG_EVERY_N(INFO, 1000) << "[" << unique_id_ << "]cache hit 100%";
      folly::coro::blockingWait(
          cache_memcpy(weights, cache_context->cached_addr_list));
    }
  } else { // no l2 cache
    folly::coro::blockingWait(get_kv_db_async(indices, weights, count));
  }
  XLOG_EVERY_N(INFO, 1000) << "[" << unique_id_
                           << "]get_cuda: finished get embeddings in "
                           << timer.elapsed().count() << " us.";
}

std::shared_ptr<CacheContext> EmbeddingKVDB::get_cache(
    const at::Tensor& indices,
    const at::Tensor& count) {
  if (l2_cache_ == nullptr) {
    return nullptr;
  }
  folly::stop_watch<std::chrono::microseconds> timer;
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
  XLOG(INFO) << "[" << unique_id_ << "]finished get cache in "
             << timer.elapsed().count() << " us. " << cache_context->num_misses
             << " cache misses out of " << num_lookups << " lookups";
  return cache_context;
}

void EmbeddingKVDB::set_cache(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
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
      XLOG(ERR) << "Failed to insert into cache, this shouldn't happen";
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
