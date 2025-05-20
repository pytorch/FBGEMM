/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <optional>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace fbgemm_gpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch(
    const at::Tensor& routing_scores,
    std::optional<at::Tensor> valid_token_count);

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch_meta(
    const at::Tensor& routing_scores,
    std::optional<at::Tensor> valid_token_count) {
  int T = routing_scores.size(0);
  int E = routing_scores.size(1);
  at::Tensor token_counts_per_expert =
      at::empty({E + 1}, routing_scores.options().dtype(at::kInt));
  at::Tensor expert_indices =
      at::empty({T}, routing_scores.options().dtype(at::kInt));
  at::Tensor token_indices =
      at::empty({T}, routing_scores.options().dtype(at::kInt));
  return {token_counts_per_expert, expert_indices, token_indices};
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.moe");
  m.def(
      "index_shuffling(Tensor routing_scores, Tensor? valid_token_count=None) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("index_shuffling", index_shuffling_torch);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("index_shuffling", index_shuffling_torch_meta);
}

} // namespace fbgemm_gpu
