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
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_inference_impl.h"

DEFINE_int64(
    dram_kv_embedding_num_shards,
    32,
    "Number of shards for DRAM KV inference embedding");

namespace fbgemm_gpu {

DramKVEmbeddingInferenceWrapper::DramKVEmbeddingInferenceWrapper(
    int64_t num_shards,
    double uniform_init_lower,
    double uniform_init_upper,
    bool disable_random_init)
    : num_shards_(num_shards),
      uniform_init_lower_(uniform_init_lower),
      uniform_init_upper_(uniform_init_upper),
      disable_random_init_(disable_random_init) {}

void DramKVEmbeddingInferenceWrapper::ensure_impl_initialized() {
  if (impl_ == nullptr) {
    LOG(INFO)
        << "Lazy-initializing DramKVEmbeddingInferenceImpl with num_shards = "
        << num_shards_ << ", uniform_init_lower = " << uniform_init_lower_
        << ", uniform_init_upper = " << uniform_init_upper_
        << ", disable_random_init = " << disable_random_init_;
    impl_ = std::make_shared<DramKVEmbeddingInferenceImpl>(
        num_shards_,
        uniform_init_lower_,
        uniform_init_upper_,
        disable_random_init_);
  }
}

void DramKVEmbeddingInferenceWrapper::init(
    const std::vector<SerializedSepcType>& specs,
    const int64_t row_alignment,
    const int64_t scale_bias_size_in_bytes,
    const std::optional<at::Tensor>& hash_size_cumsum) {
  ensure_impl_initialized();
  impl_->init(specs, row_alignment, scale_bias_size_in_bytes, hash_size_cumsum);
}

std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>>
DramKVEmbeddingInferenceWrapper::get_dram_kv() {
  TORCH_CHECK(impl_ != nullptr, "impl_ is not initialized. Call init first");
  return impl_->get_dram_kv();
}

void DramKVEmbeddingInferenceWrapper::set_dram_kv(
    std::shared_ptr<kv_mem::DramKVInferenceEmbedding<uint8_t>> dram_kv) {
  TORCH_CHECK(impl_ != nullptr, "impl_ is not initialized. Call init first");
  impl_->set_dram_kv(std::move(dram_kv));
}

void DramKVEmbeddingInferenceWrapper::set_impl(
    std::shared_ptr<KVEmbeddingInferenceInterface> impl) {
  impl_ = std::move(impl);
}

std::shared_ptr<KVEmbeddingInferenceInterface>
DramKVEmbeddingInferenceWrapper::get_impl() {
  return impl_;
}

void DramKVEmbeddingInferenceWrapper::transfer_underlying_storage_from(
    const c10::intrusive_ptr<DramKVEmbeddingInferenceWrapper>& other) {
  TORCH_CHECK(impl_ != nullptr, "impl_ is not initialized. Call init first");
  impl_->transfer_underlying_storage_from(other->impl_);
}

void DramKVEmbeddingInferenceWrapper::set_embeddings(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<int64_t> inplace_update_ts_opt) {
  TORCH_CHECK(impl_ != nullptr, "impl_ is not initialized. Call init first");
  impl_->set_embeddings(indices, weights, inplace_update_ts_opt);
}

at::Tensor DramKVEmbeddingInferenceWrapper::get_embeddings(
    const at::Tensor& indices) {
  TORCH_CHECK(impl_ != nullptr, "impl_ is not initialized. Call init first");
  return impl_->get_embeddings(indices);
}

void DramKVEmbeddingInferenceWrapper::log_inplace_update_stats() {
  TORCH_CHECK(impl_ != nullptr, "impl_ is not initialized. Call init first");
  impl_->log_inplace_update_stats();
}

void DramKVEmbeddingInferenceWrapper::trigger_evict(
    int64_t inplace_update_ts_64b) {
  TORCH_CHECK(impl_ != nullptr, "impl_ is not initialized. Call init first");
  impl_->trigger_evict(inplace_update_ts_64b);
}

void DramKVEmbeddingInferenceWrapper::wait_evict_completion() {
  TORCH_CHECK(impl_ != nullptr, "impl_ is not initialized. Call init first");
  impl_->wait_evict_completion();
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
