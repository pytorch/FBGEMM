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
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths) {
  ctx->saved_data["permutes"] = permutes;
  ctx->saved_data["in_lengths"] = in_lengths;
  ctx->saved_data["out_lengths"] = out_lengths;

  /*
    select the correct dispatched (cpu/gpu) forward function
    the cpu/gup function needs to be registered in the dispatcher,
    e.g., DISPATCH_TO_CPU, DISPATCH_TO_CUDA, etc.
  */
  const auto permute_op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_multi_embedding_function", "")
          .typed<decltype(permute_multi_embedding_cpu)>();

  return permute_op.call(pooled_embs, permutes, in_lengths, out_lengths, false);
}

variable_list PermuteMultiEmbeddingOp::backward(
    AutogradContext* ctx,
    variable_list grad_output) {
  const auto permutes = ctx->saved_data["permutes"].toIntVector();
  const auto in_lengths = ctx->saved_data["in_lengths"].toIntVector();
  const auto out_lengths = ctx->saved_data["out_lengths"].toIntVector();

  /*
    select the correct dispatched (cpu/gpu) backward function
    the cpu/gup function needs to be registered in the dispatcher,
    e.g., DISPATCH_TO_CPU, DISPATCH_TO_CUDA, etc.
  */
  const auto permute_op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_multi_embedding_function", "")
          .typed<decltype(permute_multi_embedding_cpu)>();
  auto grad_input =
      permute_op.call(grad_output, permutes, out_lengths, in_lengths, true);
  grad_input.push_back(torch::autograd::Variable()); // permutes
  grad_input.push_back(torch::autograd::Variable()); // in_lengths
  grad_input.push_back(torch::autograd::Variable()); // out_lengths
  return grad_input;
}

} // namespace fbgemm_gpu
