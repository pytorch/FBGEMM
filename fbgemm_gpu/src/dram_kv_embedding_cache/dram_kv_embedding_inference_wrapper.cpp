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

DramKVEmbeddingInferenceWrapper::DramKVEmbeddingInferenceWrapper(
    int64_t num_shards,
    double uniform_init_lower,
    double uniform_init_upper,
    int64_t evict_trigger_mode)
    : num_shards_(num_shards),
      uniform_init_lower_(uniform_init_lower),
      uniform_init_upper_(uniform_init_upper),
      evict_trigger_mode_(evict_trigger_mode) {}

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
      uniform_init_lower_,
      uniform_init_upper_,
      evict_trigger_mode_,
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

c10::List<at::Tensor> DramKVEmbeddingInferenceWrapper::serialize() const {
  c10::List<at::Tensor> results;
  results.push_back(
      torch::tensor({num_shards_, evict_trigger_mode_}, torch::kInt64));
  results.push_back(torch::tensor(
      {uniform_init_lower_, uniform_init_upper_}, torch::kDouble));
  return results;
}

void DramKVEmbeddingInferenceWrapper::deserialize(
    const c10::List<at::Tensor>& states) {
  if (states.empty()) {
    return;
  }
  TORCH_CHECK(states.size() >= 2);

  auto* intPtr = states[0].data_ptr<int64_t>();
  TORCH_CHECK(states[0].numel() >= 2)
  num_shards_ = intPtr[0];
  evict_trigger_mode_ = intPtr[1];

  TORCH_CHECK(states[1].numel() >= 2)
  auto* floatPtr = states[1].data_ptr<double>();
  uniform_init_lower_ = floatPtr[0];
  uniform_init_upper_ = floatPtr[1];
}

} // namespace fbgemm_gpu

static auto dram_kv_embedding_inference_wrapper =
    torch::class_<fbgemm_gpu::DramKVEmbeddingInferenceWrapper>(
        "fbgemm",
        "DramKVEmbeddingInferenceWrapper")
        .def(torch::init<int64_t, double, double, int64_t>())
        .def("init", &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::init)
        .def(
            "set_embeddings",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::set_embeddings)
        .def(
            "get_embeddings",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_embeddings)
        .def(
            "serialize",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::serialize)
        .def(
            "deserialize",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::deserialize)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<
                fbgemm_gpu::DramKVEmbeddingInferenceWrapper>& self)
                -> c10::List<at::Tensor> { return self->serialize(); },
            // __setstate__
            [](const c10::List<at::Tensor>& states) {
              auto ptr = c10::make_intrusive<
                  fbgemm_gpu::DramKVEmbeddingInferenceWrapper>(
                  fbgemm_gpu::DramKVEmbeddingInferenceWrapper());
              ptr->deserialize(states);
              return ptr;
            });
