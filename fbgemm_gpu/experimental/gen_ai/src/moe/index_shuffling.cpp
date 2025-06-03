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

/*
Calculates sorted indices based on routing scores.

Args:
    - `routing_scores`: (T, E) tensor of routing scores.
    - `expert_index_start`: Optional. Start index of the expert to be routed. If
      not passed, it is assumed to be 0.
    - `expert_index_end`: Optional. End index of
      the expert to be routed, exclusive. If not passed, it is assumed to be E.
    - `valid_token_count`: Optional. (1) tensor of valid token count per expert.
      If not passed, it is assumed to be T.

Returns:
    - `token_count_per_expert`: (E + 2) tensor of token count per expert and
      `num_total_tokens`, `num_sorted_tokens` are packed into as the last two
      elements.
  - `expert_indices`: (T) tensor of routed
      expert indices. Only the first `num_sorted_tokens` elements are valid.
  - `token_indices`: (T) tensor of pre-routing token indices. Only the first
      `num_sorted_tokens` elements are valid.
*/
std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch(
    const at::Tensor& routing_scores,
    const std::optional<int64_t>& expert_index_start,
    const std::optional<int64_t>& expert_index_end,
    const std::optional<at::Tensor>& valid_token_count);

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch_meta(
    const at::Tensor& routing_scores,
    const std::optional<int64_t>& expert_index_start,
    const std::optional<int64_t>& expert_index_end,
    const std::optional<at::Tensor>& valid_token_count) {
  int T = routing_scores.size(0);
  int E = routing_scores.size(1);
  at::Tensor token_counts_per_expert =
      at::empty({E + 2}, routing_scores.options().dtype(at::kInt));
  at::Tensor expert_indices =
      at::empty({T}, routing_scores.options().dtype(at::kInt));
  at::Tensor token_indices =
      at::empty({T}, routing_scores.options().dtype(at::kInt));
  return {token_counts_per_expert, expert_indices, token_indices};
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.moe");
  m.def(
      "index_shuffling(Tensor routing_scores,             "
      "                int? expert_index_start=None,      "
      "                int? expert_index_end=None,        "
      "                Tensor? valid_token_count=None) -> "
      "(Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("index_shuffling", index_shuffling_torch);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("index_shuffling", index_shuffling_torch_meta);
}

} // namespace fbgemm_gpu
