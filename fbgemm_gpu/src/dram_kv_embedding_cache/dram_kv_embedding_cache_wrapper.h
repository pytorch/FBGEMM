/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include "../ssd_split_embeddings_cache/kv_tensor_wrapper.h"
#include "dram_kv_embedding_cache.h"

namespace {
using DramKVEmbeddingCacheVariant = std::variant<
    std::shared_ptr<kv_mem::DramKVEmbeddingCache<float>>,
    std::shared_ptr<kv_mem::DramKVEmbeddingCache<at::Half>>>;
}

namespace kv_mem {

class DramKVEmbeddingCacheWrapper : public torch::jit::CustomClassHolder {
 public:
  DramKVEmbeddingCacheWrapper(
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t num_shards = 8,
      int64_t num_threads = 32,
      int64_t row_storage_bitwidth = 32,
      int64_t weight_ttl_in_hours = 2) {
    if (row_storage_bitwidth == 16) {
      impl_ = std::make_shared<kv_mem::DramKVEmbeddingCache<at::Half>>(
          max_D,
          uniform_init_lower,
          uniform_init_upper,
          num_shards,
          num_threads,
          row_storage_bitwidth,
          weight_ttl_in_hours);
    } else if (row_storage_bitwidth == 32) {
      impl_ = std::make_shared<kv_mem::DramKVEmbeddingCache<float>>(
          max_D,
          uniform_init_lower,
          uniform_init_upper,
          num_shards,
          num_threads,
          row_storage_bitwidth,
          weight_ttl_in_hours);
    } else {
      throw std::runtime_error("Failed to create recording device");
    }
  }

  void set_cuda(
      at::Tensor indices,
      at::Tensor weights,
      at::Tensor count,
      int64_t timestep,
      bool is_bwd) {
    return impl_->set_cuda(indices, weights, count, timestep);
  }

  void get_cuda(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    return impl_->get_cuda(indices, weights, count);
  }

  void set(at::Tensor indices, at::Tensor weights, at::Tensor count) {
    return impl_->set(indices, weights, count);
  }

  void flush() {
    return impl_->flush();
  }

  void set_range_to_storage(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length) {
    return impl_->set_range_to_storage(weights, start, length);
  }

  void get(
      at::Tensor indices,
      at::Tensor weights,
      at::Tensor count,
      int64_t sleep_ms) {
    return impl_->get(indices, weights, count, sleep_ms);
  }

  void wait_util_filling_work_done() {
    return impl_->wait_util_filling_work_done();
  }

  at::Tensor get_keys_in_range(int64_t start, int64_t end) {
    return impl_->get_keys_in_range(start, end);
  }

 private:
  // friend class EmbeddingRocksDBWrapper;
  friend class ssd::KVTensorWrapper;

  std::shared_ptr<kv_db::EmbeddingKVDB> impl_;
};

} // namespace kv_mem
