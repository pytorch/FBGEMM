/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_inference_impl.h"
#include "deeplearning/fbgemm/fbgemm_gpu/include/fbgemm_gpu/embedding_common.h" // @manual=//deeplearning/fbgemm/fbgemm_gpu:fbgemm_gpu

namespace fbgemm_gpu {

DramKVEmbeddingInferenceImpl::DramKVEmbeddingInferenceImpl(
    int64_t num_shards,
    double uniform_init_lower,
    double uniform_init_upper,
    bool disable_random_init)
    : num_shards_(num_shards),
      uniform_init_lower_(uniform_init_lower),
      uniform_init_upper_(uniform_init_upper),
      disable_random_init_(disable_random_init) {
  LOG(INFO)
      << "DramKVEmbeddingInferenceImpl created with disable_random_init = "
      << disable_random_init_ << ", num_shards = " << num_shards_;
}

void DramKVEmbeddingInferenceImpl::init(
    const std::vector<SerializedSepcType>& specs,
    const int64_t row_alignment,
    const int64_t scale_bias_size_in_bytes,
    const std::optional<at::Tensor>& hash_size_cumsum) {
  LOG(INFO) << "DramKVEmbeddingInferenceWrapperImpl::init() starts";
  int64_t max_D = 0;
  for (const auto& spec : specs) {
    max_D = std::max(max_D, std::get<1>(spec));
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
          std::nullopt /* training_id_eviction_trigger_count */,
          std::nullopt /* training_id_keep_count */,
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

std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>>
DramKVEmbeddingInferenceImpl::get_dram_kv() {
  return dram_kv_;
}

void DramKVEmbeddingInferenceImpl::set_dram_kv(
    std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>> dram_kv) {
  dram_kv_ = std::move(dram_kv);
}

void DramKVEmbeddingInferenceImpl::set_embeddings(
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
}

at::Tensor DramKVEmbeddingInferenceImpl::get_embeddings(
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

void DramKVEmbeddingInferenceImpl::log_inplace_update_stats() {
  dram_kv_->log_inplace_update_stats();
}

void DramKVEmbeddingInferenceImpl::trigger_evict(
    int64_t inplace_update_ts_64b) {
  uint32_t inplace_update_ts_32b =
      static_cast<std::uint32_t>(inplace_update_ts_64b);
  dram_kv_->trigger_feature_evict(inplace_update_ts_32b);
  dram_kv_->resume_ongoing_eviction();
}

void DramKVEmbeddingInferenceImpl::wait_evict_completion() {
  dram_kv_->wait_until_eviction_done();
}

void DramKVEmbeddingInferenceImpl::transfer_underlying_storage_from(
    std::shared_ptr<KVEmbeddingInferenceInterface> other) {
  LOG(INFO)
      << "DramKVEmbeddingInferenceImpl::transfer_underlying_storage_from() starts";
  auto other_dram =
      std::dynamic_pointer_cast<DramKVEmbeddingInferenceImpl>(other);
  TORCH_CHECK(
      other_dram != nullptr,
      "Cannot transfer underlying storage: source is not a DramKVEmbeddingInferenceImpl");
  this->set_dram_kv(other_dram->get_dram_kv());
}

} // namespace fbgemm_gpu
