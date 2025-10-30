/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_inference_embedding.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/kv_embedding_inference_interface.h"

namespace fbgemm_gpu {

class DramKVEmbeddingInferenceImpl : public KVEmbeddingInferenceInterface {
 public:
  using SerializedSepcType = KVEmbeddingInferenceInterface::SerializedSepcType;

  DramKVEmbeddingInferenceImpl(
      int64_t num_shards,
      double uniform_init_lower,
      double uniform_init_upper,
      bool disable_random_init);

  void init(
      const std::vector<SerializedSepcType>& specs,
      const int64_t row_alignment,
      const int64_t scale_bias_size_in_bytes,
      const std::optional<at::Tensor>& hash_size_cumsum) override;

  void set_embeddings(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<int64_t> inplace_update_ts_opt = std::nullopt) override;

  at::Tensor get_embeddings(const at::Tensor& indices) override;

  void log_inplace_update_stats() override;

  void trigger_evict(int64_t inplace_update_ts_64b) override;

  void wait_evict_completion() override;

  void transfer_underlying_storage_from(
      std::shared_ptr<KVEmbeddingInferenceInterface> other) override;

  std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>> get_dram_kv()
      override;

  void set_dram_kv(std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>>
                       dram_kv) override;

 private:
  int64_t num_shards_ = 32;
  double uniform_init_lower_ = 0.0;
  double uniform_init_upper_ = 0.0;
  bool disable_random_init_ = false;

  std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>> dram_kv_;
  int64_t max_row_bytes_ = 0;
};

} // namespace fbgemm_gpu
