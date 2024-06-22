/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/permute_multi_embedding_function.h"

namespace fbgemm_gpu {

using Tensor = at::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

std::vector<Tensor> permute_multi_embedding_cpu(
    const at::TensorList& pooled_embs,
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths,
    const bool& reverse_permute) {
  int64_t batch_size = pooled_embs[0].size(0);

  std::vector<Tensor> outputs;
  outputs.reserve(out_lengths.size());
  for (const auto i : c10::irange(out_lengths.size())) {
    outputs.push_back(
        at::empty({batch_size, out_lengths[i]}, pooled_embs[0].options()));
  }

  int64_t in_tensor, out_tensor, in_start, out_start, length, jump;
  const int64_t param = 6;
  for (const auto i : c10::irange(permutes.size() / param)) {
    if (reverse_permute) {
      out_tensor = permutes[i * param];
      in_tensor = permutes[i * param + 1];
      out_start = permutes[i * param + 2];
      in_start = permutes[i * param + 3];
      jump = permutes[i * param + 5];
    } else {
      in_tensor = permutes[i * param];
      out_tensor = permutes[i * param + 1];
      in_start = permutes[i * param + 2];
      out_start = permutes[i * param + 3];
    }
    length = permutes[i * param + 4];
    if (reverse_permute && jump < 0) {
      for (const auto b : c10::irange(batch_size)) {
        for (const auto j : c10::irange(length)) {
          outputs[out_tensor][b][j + out_start] +=
              pooled_embs[in_tensor][b][j + in_start];
        }
      }
    } else {
      for (const auto b : c10::irange(batch_size)) {
        for (const auto j : c10::irange(length)) {
          outputs[out_tensor][b][j + out_start] =
              pooled_embs[in_tensor][b][j + in_start];
        }
      }
    }
  }
  return outputs;
}

std::vector<Tensor> permute_multi_embedding_meta(
    const at::TensorList& pooled_embs,
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths,
    const bool& reverse_permute) {
  int64_t batch_size = pooled_embs[0].size(0);

  std::vector<Tensor> outputs;
  outputs.reserve(out_lengths.size());
  for (const auto i : c10::irange(out_lengths.size())) {
    outputs.push_back(
        at::empty({batch_size, out_lengths[i]}, pooled_embs[0].options()));
  }
  return outputs;
}

std::vector<Tensor> permute_multi_embedding_autograd(
    const at::TensorList& pooled_embs,
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths) {
  return PermuteMultiEmbeddingOp::apply(
      pooled_embs, permutes, in_lengths, out_lengths);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // register the forward function for internal (autograd) usage
  m.def(
      "permute_multi_embedding_function(Tensor[] pooled_embs, int[] permutes, SymInt[] in_lengths, SymInt[] out_lengths, bool reverse=False) -> Tensor[]",
      {PT2_COMPLIANT_TAG});

  // register the main function for external usage
  m.def(
      "permute_multi_embedding(Tensor[] pooled_embs, int[] permutes, SymInt[] in_lengths, SymInt[] out_lengths) -> Tensor[]",
      {PT2_COMPLIANT_TAG});

  // dispatch the forward function to CPU for internal (autograd) usage
  DISPATCH_TO_CPU(
      "permute_multi_embedding_function",
      fbgemm_gpu::permute_multi_embedding_cpu);

  // dispatch the forward function to CPU for internal (autograd) usage
  DISPATCH_TO_META(
      "permute_multi_embedding_function",
      fbgemm_gpu::permute_multi_embedding_meta);

  // dispath the main function to Autograd for external usage
  DISPATCH_TO_AUTOGRAD(
      "permute_multi_embedding", fbgemm_gpu::permute_multi_embedding_autograd);

  // dispath the main function to Autograd for external usage
  DISPATCH_TO_CUDA(
      "permute_multi_embedding", fbgemm_gpu::permute_multi_embedding_autograd);
}
