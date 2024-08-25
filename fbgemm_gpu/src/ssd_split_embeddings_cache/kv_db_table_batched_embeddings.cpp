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
#include "kv_db_utils.h"
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
  auto cacheContext = set_cache(indices, weights, count);

  if (cacheContext != nullptr) {
    if (cacheContext->missed_indices.size() > 0) {
      at::Tensor missed_indices;
      at::Tensor missed_weights;
      at::Tensor missed_count;
      from_cache_miss(
          indices,
          weights,
          count,
          missed_indices,
          missed_weights,
          missed_count,
          cacheContext);
      folly::coro::blockingWait(set_kv_db_async(
          missed_indices, missed_weights, missed_count, is_bwd));
    }
  } else { // no l2 cache
    folly::coro::blockingWait(set_kv_db_async(indices, weights, count, is_bwd));
  }
  XLOG_EVERY_N(INFO, 1000) << "set_cuda: finished set embeddings in "
                           << timer.elapsed().count() << " us.";
}

void EmbeddingKVDB::get(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  folly::stop_watch<std::chrono::microseconds> timer;
  auto cacheContext = get_cache(indices, count);
  if (cacheContext != nullptr) {
    if (cacheContext->missed_indices.size() > 0) {
      XLOG(INFO) << "cache miss: "
                 << cacheContext->missed_indices.size() / count.item().toLong();
      at::Tensor missed_indices;
      at::Tensor missed_weights;
      at::Tensor missed_count;
      from_cache_miss(
          indices,
          weights,
          count,
          missed_indices,
          missed_weights,
          missed_count,
          cacheContext);
      std::vector<folly::coro::Task<void>> tasks;
      tasks.emplace_back(
          get_kv_db_async(missed_indices, missed_weights, missed_count));
      tasks.emplace_back(
          cache_memcpy(weights, cacheContext->cached_address_pairs));
      folly::coro::blockingWait(folly::coro::collectAllRange(std::move(tasks)));
      set_cache(missed_indices, missed_weights, missed_count);
      lookup_memcpy(missed_weights, weights, cacheContext->missed_indices_map);
    } else {
      XLOG_EVERY_N(INFO, 1000) << "cache hit 100%";
      folly::coro::blockingWait(
          cache_memcpy(weights, cacheContext->cached_address_pairs));
    }
  } else { // no l2 cache
    folly::coro::blockingWait(get_kv_db_async(indices, weights, count));
  }
  XLOG_EVERY_N(INFO, 1000) << "get_cuda: finished get embeddings in "
                           << timer.elapsed().count() << " us.";
}

std::shared_ptr<EmbeddingKVDB::CacheContext> EmbeddingKVDB::get_cache(
    const at::Tensor& indices,
    const at::Tensor& count) {
  if (l2Cache_ == nullptr) {
    return nullptr;
  }
  auto cacheContext = std::make_shared<CacheContext>();
  auto indicesAcc = indices.accessor<int64_t, 1>();
  auto num_lookups = count.item<long>();
  int rowId = 0;
  for (auto i = 0; i < num_lookups; ++i) {
    auto index = indicesAcc[i];
    auto key = folly::to<std::string>(index);
    auto cachedItem = l2Cache_->get(key);
    if (cachedItem.has_value()) { // cache hit
      cacheContext->cached_address_pairs[rowId] = cachedItem.value();
    } else { // cache miss
      cacheContext->missed_indices.emplace_back(index);
      cacheContext->missed_indices_map[index] = rowId;
    }
    rowId += 1;
  }
  return cacheContext;
}

std::shared_ptr<EmbeddingKVDB::CacheContext> EmbeddingKVDB::set_cache(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  if (l2Cache_ == nullptr) {
    return nullptr;
  }
  auto cacheContext = std::make_shared<CacheContext>();
  auto indicesAcc = indices.accessor<int64_t, 1>();
  auto num_lookups = count.item<long>();
  int rowId = 0;
  for (auto i = 0; i < num_lookups; i++) {
    auto index = indicesAcc[i];
    if (!l2Cache_->put(folly::to<std::string>(index), weights[i])) {
      cacheContext->missed_indices.emplace_back(index);
      cacheContext->missed_indices_map[index] = rowId;
    }
    rowId += 1;
  }
  return cacheContext;
}

folly::coro::Task<void> EmbeddingKVDB::cache_memcpy(
    const at::Tensor& weights,
    std::unordered_map<int, std::pair<void*, uint32_t>>& address_pairs) {
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "cache_memcpy", [&] {
        auto weightsDataPtr = weights.data_ptr<scalar_t>();
        auto D = weights.size(1);
        for (auto& [rowId, address_pair] : address_pairs) {
          std::copy(
              reinterpret_cast<const scalar_t*>(address_pair.first),
              reinterpret_cast<const scalar_t*>(address_pair.first) +
                  address_pair.second,
              &weightsDataPtr[rowId * D]); // dst_start
        }
      });
  co_return;
}

void EmbeddingKVDB::lookup_memcpy(
    const at::Tensor& src_weights,
    const at::Tensor& dst_weights,
    const std::unordered_map<int64_t, int>& indices_map) {
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      dst_weights.scalar_type(), "lookup_memcpy", [&] {
        auto srcWeightsDataPtr = src_weights.data_ptr<scalar_t>();
        auto dstWeightsDataPtr = dst_weights.data_ptr<scalar_t>();
        auto D = dst_weights.size(1);
        int i = 0;
        for (auto& [index, rowId] : indices_map) {
          auto* src = srcWeightsDataPtr + i * D;
          std::move(
              reinterpret_cast<const scalar_t*>(src),
              reinterpret_cast<const scalar_t*>(src) + D,
              &dstWeightsDataPtr[rowId * D]); // dst_start
          i += 1;
        }
      });
}

void EmbeddingKVDB::from_cache_miss(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    at::Tensor& missed_indices,
    at::Tensor& missed_weights,
    at::Tensor& missed_count,
    std::shared_ptr<CacheContext>& cache_context) {
  missed_indices = at::from_blob(
      cache_context->missed_indices.data(),
      cache_context->missed_indices.size(),
      at::TensorOptions()
          .device(at::kCPU)
          .dtype(indices.dtype())
          .requires_grad(false));
  missed_weights = at::empty(
      {static_cast<int64_t>(cache_context->missed_indices.size()),
       weights.size(1)},
      at::TensorOptions().device(at::kCPU).dtype(weights.dtype()));
  missed_count =
      at::scalar_tensor(cache_context->missed_indices.size(), count.options());
}

} // namespace kv_db
