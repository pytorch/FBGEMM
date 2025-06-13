/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_inference_wrapper.h"
#include <torch/custom_class.h>
#include "deeplearning/fbgemm/fbgemm_gpu/include/fbgemm_gpu/embedding_common.h" // @manual=//deeplearning/fbgemm/fbgemm_gpu:fbgemm_gpu

namespace fbgemm_gpu {

DramKVEmbeddingInferenceWrapper::DramKVEmbeddingInferenceWrapper() {}

void DramKVEmbeddingInferenceWrapper::init(
    const std::vector<SerializedSepcType>& specs,
    const int64_t row_alignment,
    const int64_t scale_bias_size_in_bytes) {
  int64_t max_D = 0;
  for (auto i = 0; i < specs.size(); ++i) {
    max_D = std::max(max_D, std::get<1>(specs[i]));
  }
  max_row_bytes_ = nbit::padded_row_size_in_bytes(
      static_cast<int32_t>(max_D),
      static_cast<fbgemm_gpu::SparseType>(std::get<2>(specs[0])),
      static_cast<int32_t>(row_alignment),
      static_cast<int32_t>(scale_bias_size_in_bytes));
  dram_cache_ = std::make_unique<kv_mem::DramKVEmbeddingCache<uint8_t>>(
      max_row_bytes_,
      0 /* uniform_init_lower */,
      0 /* uniform_init_upper */,
      0 /* evict_trigger_mode */,
      0 /* trigger_step_intervals */,
      0 /* mem_util_threshold_in_GB */,
      1 /* evict_trigger_strategy */,
      std::nullopt /* counter_thresholds */,
      std::nullopt /* ttls_in_mins */,
      std::nullopt /* counter_decay_rates */,
      std::nullopt /* l2_weight_thresholds */,
      num_shards_ /* num_shards */,
      num_shards_ /* num_threads */,
      8 /* row_storage_bitwidth */);
  cache_initialized_ = true;
  return;
}

void DramKVEmbeddingInferenceWrapper::set_embeddings(
    const at::Tensor& indices,
    const at::Tensor& weights) {
  const auto count = at::tensor({indices.numel()}, at::ScalarType::Long);
  folly::coro::blockingWait(
      dram_cache_->set_kv_db_async(indices, weights, count));
  return;
}

at::Tensor DramKVEmbeddingInferenceWrapper::get_embeddings(
    const at::Tensor& indices) {
  const auto count = at::tensor({indices.numel()}, at::ScalarType::Long);
  auto weights = at::empty(
      {
          indices.numel(),
          max_row_bytes_,
      },
      at::kByte);
  folly::coro::blockingWait(
      dram_cache_->get_kv_db_async(indices, weights, count));
  return weights;
}

void DramKVEmbeddingInferenceWrapper::set_num_shards(const int64_t num_shards) {
  TORCH_CHECK(!cache_initialized_, "Cannot set num_shards after init.")
  num_shards_ = num_shards;
}

} // namespace fbgemm_gpu

static auto dram_kv_embedding_inference_wrapper =
    torch::class_<fbgemm_gpu::DramKVEmbeddingInferenceWrapper>(
        "fbgemm",
        "DramKVEmbeddingInferenceWrapper")
        .def(torch::init<>())
        .def("init", &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::init)
        .def(
            "set_embeddings",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::set_embeddings)
        .def(
            "get_embeddings",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_embeddings)
        .def(
            "set_num_shards",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::set_num_shards)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<
                fbgemm_gpu::DramKVEmbeddingInferenceWrapper>& /* self */)
                -> c10::List<at::Tensor> { return c10::List<at::Tensor>{}; },
            // __setstate__
            [](const c10::List<at::Tensor>& /* states*/) {
              auto ptr = c10::make_intrusive<
                  fbgemm_gpu::DramKVEmbeddingInferenceWrapper>(
                  fbgemm_gpu::DramKVEmbeddingInferenceWrapper());
              return ptr;
            });
