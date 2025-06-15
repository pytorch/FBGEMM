/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"

#include <common/base/Proc.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <optional>

namespace kv_db_utils {

int64_t _get_bucket_id(
    int64_t id,
    int64_t hash_mode,
    std::optional<int64_t> bucket_size = std::nullopt,
    std::optional<int64_t> total_num_buckets = std::nullopt) {
  if (hash_mode == 0) {
    CHECK(bucket_size.has_value());
    // chunk-based hashing
    return id / bucket_size.value();
  } else {
    // interleave-based hashing
    CHECK(total_num_buckets.has_value());
    return id % total_num_buckets.value();
  }
}

std::tuple<at::Tensor, at::Tensor> get_bucket_sorted_indices_and_bucket_tensor(
    const at::Tensor& unordered_indices,
    int64_t hash_mode,
    int64_t bucket_start,
    int64_t bucket_end,
    std::optional<int64_t> bucket_size,
    std::optional<int64_t> total_num_buckets) {
  TORCH_CHECK(unordered_indices.is_contiguous());
  TORCH_CHECK(
      hash_mode == 0 || hash_mode == 1,
      "only support hash by chunk-based or interleaved-based hashing for now");
  TORCH_CHECK(
      bucket_start <= bucket_end,
      "bucket_start:",
      bucket_start,
      " must be less than bucket_end:",
      bucket_end);

  if (bucket_end == bucket_start) {
    // likely this is the last rank containing 0 bucket
    TORCH_CHECK(
        unordered_indices.numel() == 0,
        "bucket_end:",
        bucket_end,
        " equals bucket_start:,",
        bucket_start,
        " the input data should be empty but get:",
        unordered_indices.numel());
    return std::make_tuple(
        at::empty(unordered_indices.sizes(), unordered_indices.options()),
        at::zeros({0, 1}, unordered_indices.options()));
  }

  auto indices_data_ptr = unordered_indices.data_ptr<int64_t>();
  auto num_indices = unordered_indices.numel();

  // first loop to figure out size per bucket id in ascending order
  std::map<int64_t, int64_t> bucket_id_to_cnt;
  for (int64_t i = 0; i < num_indices; ++i) {
    auto global_bucket_id = _get_bucket_id(
        indices_data_ptr[i], hash_mode, bucket_size, total_num_buckets);
    TORCH_CHECK(
        global_bucket_id >= bucket_start && global_bucket_id < bucket_end,
        "indices: ",
        indices_data_ptr[i],
        " bucket id: ",
        global_bucket_id,
        " must fall into the range between:",
        bucket_start,
        " and ",
        bucket_end);
    if (bucket_id_to_cnt.find(global_bucket_id) == bucket_id_to_cnt.end()) {
      bucket_id_to_cnt[global_bucket_id] = 0;
    }
    bucket_id_to_cnt[global_bucket_id] += 1;
  }
  // fill the bucket_tensor with counts
  at::Tensor bucket_tensor =
      at::zeros({bucket_end - bucket_start, 1}, unordered_indices.options());
  auto bucket_tensor_data_ptr = bucket_tensor.data_ptr<int64_t>();
  for (int64_t i = bucket_start; i < bucket_end; ++i) {
    if (bucket_id_to_cnt.find(i) != bucket_id_to_cnt.end()) {
      bucket_tensor_data_ptr[i - bucket_start] = bucket_id_to_cnt[i];
    } else {
      bucket_tensor_data_ptr[i - bucket_start] = 0;
    }
  }

  // calc offset
  std::map<int64_t, int64_t> bucket_id_to_offset;
  int64_t offset = 0;
  for (auto& [bucket_id, cnt] : bucket_id_to_cnt) {
    bucket_id_to_offset[bucket_id] = offset;
    offset += cnt;
  }

  // second loop to parallel set each id into its bucket range
  at::Tensor id_tensor =
      at::empty(unordered_indices.sizes(), unordered_indices.options());
  auto res_data_ptr = id_tensor.data_ptr<int64_t>();

  auto executors = std::make_unique<folly::CPUThreadPoolExecutor>(
      facebook::Proc::getCpuInfo().numCpuCores);

  auto num_executors = executors->numThreads();
  std::vector<folly::Future<folly::Unit>> futures;
  for (int64_t i = 0; i < num_executors; ++i) {
    auto f = folly::via(executors.get()).thenValue([&, i](folly::Unit) {
      for (int64_t j = 0; j < num_indices; j++) {
        auto bucket_id = _get_bucket_id(
            indices_data_ptr[j], hash_mode, bucket_size, total_num_buckets);
        if (bucket_id % num_executors != i) {
          // each bucket will have a specific executor to handle
          continue;
        }
        res_data_ptr[bucket_id_to_offset[bucket_id]] = indices_data_ptr[j];
        bucket_id_to_offset[bucket_id] += 1;
      }
    });
    futures.push_back(std::move(f));
  }
  folly::collect(futures).wait();
  return std::make_tuple(id_tensor, bucket_tensor);
}

} // namespace kv_db_utils
