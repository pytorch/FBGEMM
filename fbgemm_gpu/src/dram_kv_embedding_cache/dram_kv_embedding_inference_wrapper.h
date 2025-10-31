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
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/kv_inference_embedding_interface.h"

DECLARE_int64(dram_kv_embedding_num_shards);
DECLARE_bool(kv_embedding_async_get_set);

namespace fbgemm_gpu {

class DramKVEmbeddingInferenceWrapper : public torch::jit::CustomClassHolder {
 public:
  DramKVEmbeddingInferenceWrapper(
      int64_t num_shards = FLAGS_dram_kv_embedding_num_shards,
      double uniform_init_lower = 0.0,
      double uniform_init_upper = 0.0,
      bool disable_random_init = false);

  using SerializedSepcType =
      std::tuple<int64_t, int64_t, int64_t>; // (rows, dime, sparse_type)

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

  std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
  get_kv_backend();

  void set_kv_backend(
      std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
          kv_backend);

  c10::List<at::Tensor> serialize() const;

  void deserialize(const c10::List<at::Tensor>& states);

  int64_t get_max_row_bytes() const;

 private:
  int64_t num_shards_ = 32;
  double uniform_init_lower_ = 0.0;
  double uniform_init_upper_ = 0.0;
  bool disable_random_init_ = false;

  std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>> kv_backend_;
  int64_t max_row_bytes_ = 0;
};

} // namespace fbgemm_gpu
