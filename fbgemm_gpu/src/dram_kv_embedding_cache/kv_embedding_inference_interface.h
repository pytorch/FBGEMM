/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/custom_class.h>

namespace kv_mem {
template <typename T>
class DramKVInferenceEmbedding;
}

namespace fbgemm_gpu {

class KVEmbeddingInferenceInterface : public torch::jit::CustomClassHolder {
 public:
  using SerializedSepcType =
      std::tuple<int64_t, int64_t, int64_t>; // (rows, dime, sparse_type)

  ~KVEmbeddingInferenceInterface() override = default;

  virtual void init(
      const std::vector<SerializedSepcType>& specs,
      const int64_t row_alignment,
      const int64_t scale_bias_size_in_bytes,
      const std::optional<at::Tensor>& hash_size_cumsum) = 0;

  virtual void set_embeddings(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<int64_t> inplace_update_ts_opt = std::nullopt) = 0;

  virtual at::Tensor get_embeddings(const at::Tensor& indices) = 0;

  virtual void log_inplace_update_stats() = 0;

  virtual void trigger_evict(int64_t inplace_update_ts_64b) = 0;

  virtual void wait_evict_completion() = 0;

  virtual std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>>
  get_dram_kv() {
    throw std::runtime_error("get_dram_kv() is not implemented");
  };

  virtual void set_dram_kv(
      std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>> /*dram_kv*/) {
    throw std::runtime_error("set_dram_kv() is not implemented");
  };

  virtual void transfer_underlying_storage_from(
      std::shared_ptr<KVEmbeddingInferenceInterface> other) = 0;
};

} // namespace fbgemm_gpu
