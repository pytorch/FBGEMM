/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/permute_multi_embedding_function.h"
#include <cstdint>
#include <iostream>

namespace fbgemm_gpu {

using Tensor = at::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

variable_list PermuteMultiEmbeddingOp::forward(
    AutogradContext* ctx,
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const std::vector<int64_t>& lengths,
    const Tensor& input_lengths,
    const Tensor& output_lengths) {
  ctx->saved_data["permutes"] = permutes;
  ctx->saved_data["input_lengths"] = input_lengths;
  ctx->saved_data["output_lengths"] = output_lengths;

  std::vector<int64_t> inv_lengths;
  inv_lengths.reserve(pooled_embs.size());
  for (const auto i : c10::irange(pooled_embs.size())) {
    inv_lengths.push_back(pooled_embs[i].size(1));
  }
  ctx->saved_data["inv_lengths"] = inv_lengths;

  /*
    select the correct dispatched (cpu/gpu) forward function
    the cpu/gup function needs to be registered in the dispatcher,
    e.g., DISPATCH_TO_CPU, DISPATCH_TO_CUDA, etc.
  */
  const auto permute_op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_multi_embedding_function", "")
          .typed<decltype(permute_multi_embedding_cpu)>();

  return permute_op.call(
      pooled_embs, permutes, lengths, input_lengths, output_lengths, false);
}

variable_list PermuteMultiEmbeddingOp::backward(
    AutogradContext* ctx,
    variable_list grad_output) {
  const auto permutes = ctx->saved_data["permutes"].toTensor();
  const auto input_lengths = ctx->saved_data["input_lengths"].toTensor();
  const auto output_lengths = ctx->saved_data["output_lengths"].toTensor();
  const auto inv_lengths = ctx->saved_data["inv_lengths"].toIntVector();
  /*
    select the correct dispatched (cpu/gpu) backward function
    the cpu/gup function needs to be registered in the dispatcher,
    e.g., DISPATCH_TO_CPU, DISPATCH_TO_CUDA, etc.
  */
  const auto permute_op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_multi_embedding_function", "")
          .typed<decltype(permute_multi_embedding_cpu)>();
  auto grad_input = permute_op.call(
      grad_output, permutes, inv_lengths, output_lengths, input_lengths, true);
  grad_input.push_back(torch::autograd::Variable()); // permutes
  grad_input.push_back(torch::autograd::Variable()); // lengths
  grad_input.push_back(torch::autograd::Variable()); // input_lengths
  grad_input.push_back(torch::autograd::Variable()); // output_lengths
  return grad_input;
}

} // namespace fbgemm_gpu
