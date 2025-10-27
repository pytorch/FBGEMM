/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gflags/gflags_declare.h>
#include <torch/custom_class.h>
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_inference_embedding.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/kv_embedding_inference_interface.h"

DECLARE_int64(dram_kv_embedding_num_shards);

namespace fbgemm_gpu {

class DramKVEmbeddingInferenceImpl;

class DramKVEmbeddingInferenceWrapper : public torch::jit::CustomClassHolder {
 public:
  using SerializedSepcType = KVEmbeddingInferenceInterface::SerializedSepcType;

  DramKVEmbeddingInferenceWrapper(
      int64_t num_shards = FLAGS_dram_kv_embedding_num_shards,
      double uniform_init_lower = 0.0,
      double uniform_init_upper = 0.0,
      bool disable_random_init = false);

  void init(
      const std::vector<SerializedSepcType>& specs,
      const int64_t row_alignment,
      const int64_t scale_bias_size_in_bytes,
      const std::optional<at::Tensor>& hash_size_cumsum);

  void set_embeddings(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<int64_t> inplace_update_ts_opt = std::nullopt);

  at::Tensor get_embeddings(const at::Tensor& indices);

  void log_inplace_update_stats();

  void trigger_evict(int64_t inplace_update_ts_64b);

  void wait_evict_completion();

  std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>> get_dram_kv();

  void set_dram_kv(
      std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>> dram_kv);

  void set_impl(std::shared_ptr<KVEmbeddingInferenceInterface> impl);

  std::shared_ptr<KVEmbeddingInferenceInterface> get_impl();

  void transfer_underlying_storage_from(
      const c10::intrusive_ptr<DramKVEmbeddingInferenceWrapper>& other);

  c10::List<at::Tensor> serialize() const;

  void deserialize(const c10::List<at::Tensor>& states);

 private:
  void ensure_impl_initialized();

  std::shared_ptr<KVEmbeddingInferenceInterface> impl_;

  int64_t num_shards_ = 32;
  double uniform_init_lower_ = 0.0;
  double uniform_init_upper_ = 0.0;
  bool disable_random_init_ = false;
};

} // namespace fbgemm_gpu
