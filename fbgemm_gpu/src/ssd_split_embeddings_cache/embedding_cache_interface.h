/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

namespace embedding_cache {

class EmbeddingCacheInterface {
 public:
  /**
   * @brief getter for the embedding.
   *
   * @param indices The indices of the embeddings to be retrieved.
   * @param weights The weights (placeholders) of the embeddings to update in
   * place.
   * @param count The number of embeddings to be retrieved.
   */
  virtual void get(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) = 0;

  /**
   * @brief setter for the embedding.
   *
   * @param indices The indices of the embeddings to be update.
   * @param weights The weights of the embeddings to update.
   * @param count The number of embeddings to update.
   */
  virtual void set(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) = 0;

  virtual ~EmbeddingCacheInterface() {}
};

} // namespace embedding_cache
