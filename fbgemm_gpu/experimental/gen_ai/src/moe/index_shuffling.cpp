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
    const at::Tensor& scores,
    std::optional<at::Tensor> num_valid_tokens);

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch_meta(
    const at::Tensor& scores,
    std::optional<at::Tensor> num_valid_tokens) {
  int T = scores.size(0);
  int E = scores.size(1);
  at::Tensor counts = at::empty({E + 1}, scores.options().dtype(at::kInt));
  at::Tensor expert_indices = at::empty({T}, scores.options().dtype(at::kInt));
  at::Tensor token_indices = at::empty({T}, scores.options().dtype(at::kInt));
  return {counts, expert_indices, token_indices};
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.moe");
  m.def(
      "index_shuffling(Tensor scores, Tensor? num_valid_tokens= None) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("index_shuffling", index_shuffling_torch);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("index_shuffling", index_shuffling_torch_meta);
}

} // namespace fbgemm_gpu
