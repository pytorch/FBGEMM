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
    const Tensor& permutes,
    const std::vector<int64_t>& lengths,
    const Tensor& input_lengths,
    const Tensor& output_lengths,
    const bool& reverse_permute) {
  int64_t num_output_tensors = lengths.size();
  int64_t batch_size = pooled_embs[0].size(0);

  std::vector<Tensor> outputs;
  outputs.reserve(num_output_tensors);
  for (const auto i : c10::irange(num_output_tensors)) {
    outputs.push_back(
        at::empty({batch_size, lengths[i]}, pooled_embs[0].options()));
  }

  int64_t in_tensor, out_tensor, in_start, out_start, length, jump;
  for (const auto i : c10::irange(permutes.size(0))) {
    if (reverse_permute) {
      out_tensor = permutes[i][0].item<int64_t>();
      in_tensor = permutes[i][1].item<int64_t>();
      out_start = permutes[i][2].item<int64_t>();
      in_start = permutes[i][3].item<int64_t>();
      jump = permutes[i][5].item<int64_t>();
    } else {
      in_tensor = permutes[i][0].item<int64_t>();
      out_tensor = permutes[i][1].item<int64_t>();
      in_start = permutes[i][2].item<int64_t>();
      out_start = permutes[i][3].item<int64_t>();
    }
    length = permutes[i][4].item<int64_t>();
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
    const Tensor& permutes,
    const std::vector<int64_t>& lengths,
    const Tensor& input_lengths,
    const Tensor& output_lengths,
    const bool& reverse) {
  int64_t num_output_tensors = lengths.size();
  int64_t batch_size = pooled_embs[0].size(0);

  std::vector<Tensor> output;
  output.reserve(num_output_tensors);
  for (const auto i : c10::irange(num_output_tensors)) {
    output.push_back(
        at::empty({batch_size, lengths[i]}, pooled_embs[0].options()));
  }
  return output;
}

std::vector<Tensor> permute_multi_embedding_autograd(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const std::vector<int64_t>& lengths,
    const Tensor& input_lengths,
    const Tensor& output_lengths) {
  return PermuteMultiEmbeddingOp::apply(
      pooled_embs, permutes, lengths, input_lengths, output_lengths);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // register the forward function for internal (autograd) usage
  m.def(
      "permute_multi_embedding_function(Tensor[] pooled_embs, Tensor permutes, SymInt[] lengths, Tensor in_lengths, Tensor out_lengths, bool reverse=False) -> Tensor[]",
      {PT2_COMPLIANT_TAG});

  // register the main function for external usage
  m.def(
      "permute_multi_embedding(Tensor[] pooled_embs, Tensor permutes, SymInt[] lengths, Tensor in_lengths, Tensor out_lengths) -> Tensor[]",
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
}
