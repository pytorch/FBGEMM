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
    double uniform_init_upper)
    : num_shards_(num_shards),
      uniform_init_lower_(uniform_init_lower),
      uniform_init_upper_(uniform_init_upper) {}

void DramKVEmbeddingInferenceWrapper::init(
    const std::vector<SerializedSepcType>& specs,
    const int64_t row_alignment,
    const int64_t scale_bias_size_in_bytes,
    const std::optional<at::Tensor>& hash_size_cumsum) {
  LOG(INFO) << "DramKVEmbeddingInferenceWrapper::init() starts";
  int64_t max_D = 0;
  for (auto i = 0; i < specs.size(); ++i) {
    max_D = std::max(max_D, std::get<1>(specs[i]));
  }
  max_row_bytes_ = nbit::padded_row_size_in_bytes(
      static_cast<int32_t>(max_D),
      static_cast<fbgemm_gpu::SparseType>(std::get<2>(specs[0])),
      static_cast<int32_t>(row_alignment),
      static_cast<int32_t>(scale_bias_size_in_bytes));
  LOG(INFO) << "Initialize dram_kv with max_D: " << max_D
            << ", sparse_type: " << std::get<2>(specs[0])
            << ", row_alignment: " << row_alignment
            << ", scale_bias_size_in_bytes: " << scale_bias_size_in_bytes
            << ", max_row_bytes_: " << max_row_bytes_;
  if (dram_kv_ != nullptr) {
    return;
  }
  dram_kv_ = std::make_shared<kv_mem::DramKVInferenceEmbedding<uint8_t>>(
      max_row_bytes_,
      uniform_init_lower_,
      uniform_init_upper_,
      c10::make_intrusive<kv_mem::FeatureEvictConfig>(
          3 /* EvictTriggerMode.MANUAL */,
          4 /* EvictTriggerStrategy::BY_TIMESTAMP_THRESHOLD */,
          0 /* trigger_step_intervals */,
          0 /* mem_util_threshold_in_GB */,
          std::nullopt /* ttls_in_mins */,
          std::nullopt /* counter_thresholds */,
          std::nullopt /* counter_decay_rates */,
          std::nullopt /* feature_score_counter_decay_rates */,
          std::nullopt /* max_training_id_num_per_table */,
          std::nullopt /* target_eviction_percent_per_table */,
          std::nullopt /* l2_weight_thresholds */,
          std::nullopt /* embedding_dims */,
          std::nullopt /* threshold_calculation_bucket_stride */,
          std::nullopt /* threshold_calculation_bucket_num */,
          0 /* interval for insufficient eviction s*/,
          0 /* interval for sufficient eviction s*/,
          0 /* interval_for_feature_statistics_decay_s_*/),
      num_shards_ /* num_shards */,
      num_shards_ /* num_threads */,
      8 /* row_storage_bitwidth */,
      false /* enable_async_update */,
      std::nullopt /* table_dims */,
      hash_size_cumsum);
  return;
}

std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>>
DramKVEmbeddingInferenceWrapper::get_dram_kv() {
  return dram_kv_;
}

void DramKVEmbeddingInferenceWrapper::set_dram_kv(
    std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>> dram_kv) {
  dram_kv_ = std::move(dram_kv);
}

void DramKVEmbeddingInferenceWrapper::set_embeddings(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<int64_t> inplace_update_ts_opt) {
  const auto count = at::tensor({indices.numel()}, at::ScalarType::Long);
  std::optional<uint32_t> inplacee_update_ts = std::nullopt;
  if (inplace_update_ts_opt.has_value()) {
    inplacee_update_ts =
        static_cast<std::uint32_t>(inplace_update_ts_opt.value());
  }
  folly::coro::blockingWait(dram_kv_->inference_set_kv_db_async(
      indices, weights, count, inplacee_update_ts));
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
  folly::coro::blockingWait(dram_kv_->get_kv_db_async(indices, weights, count));
  return weights;
}

void DramKVEmbeddingInferenceWrapper::log_inplace_update_stats() {
  return dram_kv_->log_inplace_update_stats();
}

void DramKVEmbeddingInferenceWrapper::trigger_evict(
    int64_t inplace_update_ts_64b) {
  uint32_t inplace_update_ts_32b =
      static_cast<std::uint32_t>(inplace_update_ts_64b);
  dram_kv_->trigger_feature_evict(inplace_update_ts_32b);
  dram_kv_->resume_ongoing_eviction();
}

void DramKVEmbeddingInferenceWrapper::wait_evict_completion() {
  dram_kv_->wait_until_eviction_done();
}

c10::List<at::Tensor> DramKVEmbeddingInferenceWrapper::serialize() const {
  c10::List<at::Tensor> results;
  results.push_back(torch::tensor({num_shards_}, torch::kInt64));
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
  TORCH_CHECK(states[0].numel() >= 1)
  num_shards_ = intPtr[0];

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
        .def(torch::init<int64_t, double, double>())
        .def("init", &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::init)
        .def(
            "set_embeddings",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::set_embeddings,
            "",
            {
                torch::arg("indices"),
                torch::arg("weights"),
                torch::arg("inplace_update_ts_opt") = std::nullopt,
            })
        .def(
            "get_embeddings",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_embeddings)
        .def(
            "trigger_evict",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::trigger_evict)
        .def(
            "wait_evict_completion",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::wait_evict_completion)
        .def(
            "log_inplace_update_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                log_inplace_update_stats)
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
