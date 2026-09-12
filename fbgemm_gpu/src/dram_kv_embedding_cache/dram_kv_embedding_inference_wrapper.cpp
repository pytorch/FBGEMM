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
namespace {

using SerializedSpec = DramKVEmbeddingInferenceWrapper::SerializedSepcType;

std::pair<int64_t, int64_t> legacy_row_sizes(
    const std::vector<SerializedSpec>& specs,
    const int64_t row_alignment,
    const int64_t scale_bias_size_in_bytes) {
  int64_t max_dim = 0;
  for (const auto& spec : specs) {
    max_dim = std::max(max_dim, std::get<1>(spec));
  }
  const auto row_bytes = nbit::padded_row_size_in_bytes(
      static_cast<int32_t>(max_dim),
      static_cast<fbgemm_gpu::SparseType>(std::get<2>(specs.front())),
      static_cast<int32_t>(row_alignment),
      static_cast<int32_t>(scale_bias_size_in_bytes));
  return {row_bytes, row_bytes};
}

std::pair<int64_t, int64_t> separate_row_sizes(
    const std::vector<SerializedSpec>& specs,
    const int64_t logical_row_alignment,
    const int64_t storage_row_alignment,
    const int64_t scale_bias_size_in_bytes) {
  std::optional<int64_t> expected_storage_row_bytes;
  std::optional<int64_t> expected_lookup_row_bytes;
  for (const auto& spec : specs) {
    const auto dim = std::get<1>(spec);
    const auto sparse_type =
        static_cast<fbgemm_gpu::SparseType>(std::get<2>(spec));
    const auto storage_row_bytes = nbit::padded_row_size_in_bytes(
        static_cast<int32_t>(dim),
        sparse_type,
        static_cast<int32_t>(storage_row_alignment),
        static_cast<int32_t>(scale_bias_size_in_bytes));
    const auto lookup_row_bytes = nbit::padded_row_size_in_bytes(
        static_cast<int32_t>(dim),
        sparse_type,
        static_cast<int32_t>(logical_row_alignment),
        static_cast<int32_t>(scale_bias_size_in_bytes));
    TORCH_CHECK(
        lookup_row_bytes <= storage_row_bytes,
        "Dram KV logical row width must not exceed storage row width");
    if (expected_lookup_row_bytes.has_value()) {
      TORCH_CHECK(
          lookup_row_bytes == expected_lookup_row_bytes.value(),
          "Dram KV embedding requires one logical row width across tables");
      TORCH_CHECK(
          storage_row_bytes == expected_storage_row_bytes.value(),
          "Dram KV embedding requires one storage row width across tables");
    }
    expected_lookup_row_bytes = lookup_row_bytes;
    expected_storage_row_bytes = storage_row_bytes;
  }
  return {
      expected_storage_row_bytes.value(), expected_lookup_row_bytes.value()};
}

} // namespace

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
  init_impl(
      specs,
      row_alignment,
      row_alignment,
      scale_bias_size_in_bytes,
      hash_size_cumsum,
      InitMode::Legacy);
}

void DramKVEmbeddingInferenceWrapper::init_with_row_alignments(
    const std::vector<SerializedSepcType>& specs,
    const int64_t logical_row_alignment,
    const int64_t storage_row_alignment,
    const int64_t scale_bias_size_in_bytes,
    const std::optional<at::Tensor>& hash_size_cumsum) {
  init_impl(
      specs,
      logical_row_alignment,
      storage_row_alignment,
      scale_bias_size_in_bytes,
      hash_size_cumsum,
      InitMode::SeparateAlignments);
}

void DramKVEmbeddingInferenceWrapper::init_impl(
    const std::vector<SerializedSepcType>& specs,
    const int64_t logical_row_alignment,
    const int64_t storage_row_alignment,
    const int64_t scale_bias_size_in_bytes,
    const std::optional<at::Tensor>& hash_size_cumsum,
    const InitMode mode) {
  LOG(INFO) << "DramKVEmbeddingInferenceWrapper::init_impl() starts";
  TORCH_CHECK(!specs.empty(), "Dram KV embedding specs must not be empty");
  TORCH_CHECK(
      logical_row_alignment > 0,
      "Dram KV logical row alignment must be positive");
  TORCH_CHECK(
      storage_row_alignment > 0,
      "Dram KV storage row alignment must be positive");
  const auto [storage_max_row_bytes, lookup_max_row_bytes] =
      mode == InitMode::Legacy
      ? legacy_row_sizes(specs, storage_row_alignment, scale_bias_size_in_bytes)
      : separate_row_sizes(
            specs,
            logical_row_alignment,
            storage_row_alignment,
            scale_bias_size_in_bytes);
  LOG(INFO) << "Initialize dram_kv with logical_row_alignment: "
            << logical_row_alignment
            << ", storage_row_alignment: " << storage_row_alignment
            << ", scale_bias_size_in_bytes: " << scale_bias_size_in_bytes
            << ", storage_max_row_bytes: " << storage_max_row_bytes
            << ", lookup_max_row_bytes: " << lookup_max_row_bytes;
  if (initialized_) {
    TORCH_CHECK(
        storage_max_row_bytes == storage_max_row_bytes_,
        "Cannot change KV storage row width after backend initialization: "
        "existing width=",
        storage_max_row_bytes_,
        ", requested width=",
        storage_max_row_bytes);
    TORCH_CHECK(
        lookup_max_row_bytes == lookup_max_row_bytes_,
        "Cannot change KV lookup row width after backend initialization: "
        "existing width=",
        lookup_max_row_bytes_,
        ", requested width=",
        lookup_max_row_bytes);
  }
  if (backend_storage_row_bytes_.has_value()) {
    TORCH_CHECK(
        storage_max_row_bytes <= backend_storage_row_bytes_.value(),
        "KV storage row width exceeds backend capacity: backend capacity=",
        backend_storage_row_bytes_.value(),
        ", requested width=",
        storage_max_row_bytes);
  }
  storage_max_row_bytes_ = storage_max_row_bytes;
  lookup_max_row_bytes_ = lookup_max_row_bytes;
  initialized_ = true;
  if (kv_backend_ != nullptr) {
    return;
  }
  kv_backend_ = std::make_shared<kv_mem::DramKVInferenceEmbedding<uint8_t>>(
      storage_max_row_bytes_,
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
  uses_dram_backend_ = true;
  backend_storage_row_bytes_ = storage_max_row_bytes_;
}

int64_t DramKVEmbeddingInferenceWrapper::get_max_row_bytes() const {
  return storage_max_row_bytes_;
}

int64_t DramKVEmbeddingInferenceWrapper::get_lookup_row_bytes() const {
  return lookup_max_row_bytes_;
}

std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
DramKVEmbeddingInferenceWrapper::get_kv_backend() {
  return kv_backend_;
}

void DramKVEmbeddingInferenceWrapper::set_kv_backend(
    std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
        kv_backend) {
  TORCH_CHECK(kv_backend != nullptr, "KV backend must not be null");
  const auto backend_storage_row_bytes = kv_backend->get_storage_row_bytes();
  if (initialized_ && backend_storage_row_bytes.has_value()) {
    TORCH_CHECK(
        storage_max_row_bytes_ <= backend_storage_row_bytes.value(),
        "KV storage row width exceeds backend capacity: backend capacity=",
        backend_storage_row_bytes.value(),
        ", requested width=",
        storage_max_row_bytes_);
  }
  kv_backend_ = std::move(kv_backend);
  const auto* dram_backend =
      dynamic_cast<kv_mem::DramKVInferenceEmbedding<uint8_t>*>(
          kv_backend_.get());
  uses_dram_backend_ = dram_backend != nullptr;
  backend_storage_row_bytes_ = backend_storage_row_bytes;
}

void DramKVEmbeddingInferenceWrapper::check_initialized() const {
  TORCH_CHECK(
      initialized_ && kv_backend_ != nullptr,
      "KV embedding cache must be initialized before use");
}

void DramKVEmbeddingInferenceWrapper::set_embeddings(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<int64_t> inplace_update_ts_opt) {
  check_initialized();
  TORCH_CHECK(
      weights.dim() == 2, "Embedding updates must be a two-dimensional tensor");
  const auto backend_row_bytes =
      backend_storage_row_bytes_.value_or(storage_max_row_bytes_);
  TORCH_CHECK(
      weights.size(1) <= storage_max_row_bytes_,
      "Embedding update row is wider than KV storage");
  TORCH_CHECK(
      weights.size(1) <= backend_row_bytes,
      "Embedding update row is wider than backend storage capacity");
  const auto count = at::tensor({indices.numel()}, at::ScalarType::Long);
  std::optional<uint32_t> inplacee_update_ts = std::nullopt;
  if (inplace_update_ts_opt.has_value()) {
    inplacee_update_ts =
        static_cast<std::uint32_t>(inplace_update_ts_opt.value());
  }

  auto backend_weights = weights;
  if (!uses_dram_backend_ && weights.size(1) < backend_row_bytes) {
    // External backends consume physical rows. A shared scratch tensor is
    // unsafe because updates can run concurrently.
    backend_weights =
        at::zeros({weights.size(0), backend_row_bytes}, weights.options());
    backend_weights.slice(1, 0, weights.size(1)).copy_(weights);
  }

  folly::coro::blockingWait(kv_backend_->inference_set_kv_db_async(
      indices, backend_weights, count, inplacee_update_ts));
}

at::Tensor DramKVEmbeddingInferenceWrapper::get_embeddings(
    const at::Tensor& indices) {
  check_initialized();
  const auto count = at::tensor({indices.numel()}, at::ScalarType::Long);
  auto weights = at::empty(
      {
          indices.numel(),
          lookup_max_row_bytes_,
      },
      at::kByte);

  folly::coro::blockingWait(
      kv_backend_->get_kv_db_async(indices, weights, count));
  return weights;
}

void DramKVEmbeddingInferenceWrapper::log_inplace_update_stats() {
  check_initialized();
  kv_backend_->log_inplace_update_stats();
}

std::vector<int64_t>
DramKVEmbeddingInferenceWrapper::get_read_hit_rate_stats() {
  check_initialized();
  return kv_backend_->get_read_hit_rate_stats();
}

void DramKVEmbeddingInferenceWrapper::trigger_evict(
    int64_t inplace_update_ts_64b) {
  check_initialized();
  uint32_t inplace_update_ts_32b =
      static_cast<std::uint32_t>(inplace_update_ts_64b);
  kv_backend_->trigger_feature_evict(inplace_update_ts_32b);
  kv_backend_->resume_ongoing_eviction();
}

void DramKVEmbeddingInferenceWrapper::wait_evict_completion() {
  check_initialized();
  kv_backend_->wait_until_eviction_done();
}

c10::List<at::Tensor> DramKVEmbeddingInferenceWrapper::serialize() const {
  c10::List<at::Tensor> results;
  results.push_back(torch::tensor({num_shards_}, torch::kInt64));
  results.push_back(
      torch::tensor(
          {uniform_init_lower_, uniform_init_upper_}, torch::kDouble));
  results.push_back(
      torch::tensor(
          {static_cast<int64_t>(disable_random_init_)}, torch::kInt64));
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

  // Payloads published before disable_random_init_ was serialized carry only
  // the two entries above; leave the constructor default in place for those.
  if (states.size() >= 3 && states[2].numel() >= 1) {
    disable_random_init_ = states[2].const_data_ptr<int64_t>()[0] != 0;
  }
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
            "init_with_row_alignments",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                init_with_row_alignments,
            "",
            {
                torch::arg("specs"),
                torch::arg("logical_row_alignment"),
                torch::arg("storage_row_alignment"),
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
            "get_max_row_bytes",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_max_row_bytes)
        .def(
            "get_lookup_row_bytes",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_lookup_row_bytes)
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
            "get_read_hit_rate_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_read_hit_rate_stats)
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
