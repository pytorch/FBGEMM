/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/custom_class.h>
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_cache.h"

namespace fbgemm_gpu {

class DramKVEmbeddingInferenceWrapper : public torch::jit::CustomClassHolder {
 public:
  DramKVEmbeddingInferenceWrapper(
      int64_t num_shards = 32,
      double uniform_init_lower = 0.0,
      double uniform_init_upper = 0.0,
      int64_t evict_trigger_mode = 0);

  using SerializedSepcType =
      std::tuple<int64_t, int64_t, int64_t>; // (rows, dime, sparse_type)

  void init(
      const std::vector<SerializedSepcType>& specs,
      const int64_t row_alignment,
      const int64_t scale_bias_size_in_bytes);

  void set_embeddings(const at::Tensor& indices, const at::Tensor& weights);

  at::Tensor get_embeddings(const at::Tensor& indices);

  c10::List<at::Tensor> serialize() const;

  void deserialize(const c10::List<at::Tensor>& states);

 private:
  int64_t num_shards_ = 32;
  double uniform_init_lower_ = 0.0;
  double uniform_init_upper_ = 0.0;
  int64_t evict_trigger_mode_ = 0;

  std::unique_ptr<kv_mem::DramKVEmbeddingCache<uint8_t>> dram_cache_;
  int64_t max_row_bytes_ = 0;
};

} // namespace fbgemm_gpu
