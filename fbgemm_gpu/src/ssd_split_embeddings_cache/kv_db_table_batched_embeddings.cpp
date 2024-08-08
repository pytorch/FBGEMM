/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kv_db_table_batched_embeddings.h"
#include "torch/csrc/autograd/record_function_ops.h"

namespace kv_db {

void EmbeddingKVDB::get_cuda(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## EmbeddingKVDB::get_cuda ##");
  // take reference to self to avoid lifetime issues.
  auto self = shared_from_this();
  std::function<void()>* functor =
      new std::function<void()>([=]() { self->get(indices, weights, count); });
  AT_CUDA_CHECK(cudaStreamAddCallback(
      at::cuda::getCurrentCUDAStream(),
      kv_db_utils::cuda_callback_func,
      functor,
      0));
  rec->record.end();
}

void EmbeddingKVDB::set_cuda(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    const int64_t timestep) {
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## EmbeddingKVDB::set_cuda ##");
  // take reference to self to avoid lifetime issues.
  auto self = shared_from_this();
  std::function<void()>* functor = new std::function<void()>([=]() {
    self->set(indices, weights, count);
    self->flush_or_compact(timestep);
  });
  AT_CUDA_CHECK(cudaStreamAddCallback(
      at::cuda::getCurrentCUDAStream(),
      kv_db_utils::cuda_callback_func,
      functor,
      0));
  rec->record.end();
}

} // namespace kv_db
