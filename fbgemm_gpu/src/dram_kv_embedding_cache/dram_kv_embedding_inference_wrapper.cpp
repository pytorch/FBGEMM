/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_inference_wrapper.h"
#include <gflags/gflags.h>
#include <torch/custom_class.h>
#include "deeplearning/fbgemm/fbgemm_gpu/include/fbgemm_gpu/embedding_common.h" // @manual=//deeplearning/fbgemm/fbgemm_gpu:fbgemm_gpu
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_inference_embedding.h"

DEFINE_int64(
    dram_kv_embedding_num_shards,
    32,
    "Number of shards for DRAM KV inference embedding");
DEFINE_bool(
    kv_embedding_async_get_set,
    true,
    "Whether to use async get/set for DRAM KV inference embedding."
    "This should be true for dram but might be different for other non-Dram backends.");

namespace fbgemm_gpu {

DramKVEmbeddingInferenceWrapper::DramKVEmbeddingInferenceWrapper(
    int64_t num_shards,
    double uniform_init_lower,
    double uniform_init_upper,
    bool disable_random_init)
    : num_shards_(num_shards),
      uniform_init_lower_(uniform_init_lower),
      uniform_init_upper_(uniform_init_upper),
      disable_random_init_(disable_random_init) {
  LOG(INFO)
      << "DramKVEmbeddingInferenceWrapper created with disable_random_init = "
      << disable_random_init_ << ", num_shards = " << num_shards_;
}

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
  if (kv_backend_ != nullptr) {
    return;
  }
  kv_backend_ = std::make_shared<kv_mem::DramKVInferenceEmbedding<uint8_t>>(
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
          std::nullopt /* training_id_eviction_trigger_count */,
          std::nullopt /* training_id_keep_count */,
          std::nullopt /* enable_eviction_for_feature_score_eviction_policy */,
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
      hash_size_cumsum,
      disable_random_init_);
}

int64_t DramKVEmbeddingInferenceWrapper::get_max_row_bytes() const {
  return max_row_bytes_;
}

std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
DramKVEmbeddingInferenceWrapper::get_kv_backend() {
  return kv_backend_;
}

void DramKVEmbeddingInferenceWrapper::set_kv_backend(
    std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
        kv_backend) {
  kv_backend_ = std::move(kv_backend);
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

  if (FLAGS_kv_embedding_async_get_set) {
    folly::coro::blockingWait(kv_backend_->inference_set_kv_db_async(
        indices, weights, count, inplacee_update_ts));
  } else {
    kv_backend_->set_kv_db_sync(indices, weights, count, inplacee_update_ts);
  }
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

  if (FLAGS_kv_embedding_async_get_set) {
    folly::coro::blockingWait(
        kv_backend_->get_kv_db_async(indices, weights, count));
  } else {
    kv_backend_->get_kv_db_sync(indices, weights, count);
  }
  return weights;
}

void DramKVEmbeddingInferenceWrapper::log_inplace_update_stats() {
  kv_backend_->log_inplace_update_stats();
}

void DramKVEmbeddingInferenceWrapper::trigger_evict(
    int64_t inplace_update_ts_64b) {
  uint32_t inplace_update_ts_32b =
      static_cast<std::uint32_t>(inplace_update_ts_64b);
  kv_backend_->trigger_feature_evict(inplace_update_ts_32b);
  kv_backend_->resume_ongoing_eviction();
}

void DramKVEmbeddingInferenceWrapper::wait_evict_completion() {
  kv_backend_->wait_until_eviction_done();
}

c10::List<at::Tensor> DramKVEmbeddingInferenceWrapper::serialize() const {
  c10::List<at::Tensor> results;
  results.push_back(torch::tensor({num_shards_}, torch::kInt64));
  results.push_back(
      torch::tensor(
          {uniform_init_lower_, uniform_init_upper_}, torch::kDouble));
  return results;
}

void DramKVEmbeddingInferenceWrapper::deserialize(
    const c10::List<at::Tensor>& states) {
  if (states.empty()) {
    return;
  }
  TORCH_CHECK(states.size() >= 2);

  const auto* intPtr = states[0].const_data_ptr<int64_t>();
  TORCH_CHECK(states[0].numel() >= 1)
  num_shards_ = intPtr[0];

  TORCH_CHECK(states[1].numel() >= 2)
  const auto* floatPtr = states[1].const_data_ptr<double>();
  uniform_init_lower_ = floatPtr[0];
  uniform_init_upper_ = floatPtr[1];
}

} // namespace fbgemm_gpu

static auto dram_kv_embedding_inference_wrapper =
    torch::class_<fbgemm_gpu::DramKVEmbeddingInferenceWrapper>(
        "fbgemm",
        "DramKVEmbeddingInferenceWrapper")
        .def(torch::init<int64_t, double, double, bool>())
        .def(
            "init",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::init,
            "",
            {
                torch::arg("specs"),
                torch::arg("row_alignment"),
                torch::arg("scale_bias_size_in_bytes"),
                torch::arg("hash_size_cumsum"),
            })
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
