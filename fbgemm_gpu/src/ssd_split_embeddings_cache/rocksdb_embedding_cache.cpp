/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "fbcode/deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/rocksdb_embedding_cache.h"
#include <folly/logging/xlog.h>

namespace embedding_cache {

RocksDBEmbeddingCache::RocksDBEmbeddingCache(
    std::unique_ptr<ssd::EmbeddingRocksDB> rocksdb) noexcept
    : rocksdb_(std::move(rocksdb)) {
  XLOG(INFO) << "Initializing RocksDBEmbeddingCache";
}

void RocksDBEmbeddingCache::get(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  rocksdb_->get(indices, weights, count);
}

void RocksDBEmbeddingCache::set(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  rocksdb_->set(indices, weights, count);
}

} // namespace embedding_cache
