/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include <ATen/ATen.h>
#include <string>
#include "fbcode/deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/embedding_cache_interface.h"
#include "fbcode/deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/ssd_table_batched_embeddings.h"

namespace embedding_cache {

class RocksDBEmbeddingCache : public EmbeddingCacheInterface {
 public:
  explicit RocksDBEmbeddingCache(
      std::unique_ptr<ssd::EmbeddingRocksDB> rocksdb) noexcept;

  void get(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override;
  void set(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override;

 private:
  std::unique_ptr<ssd::EmbeddingRocksDB> rocksdb_;
};

} // namespace embedding_cache
