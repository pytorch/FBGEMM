/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/fbgemm_gpu/permute_pooled_embedding_function.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {
at::Tensor permute_pooled_embs_cpu(
    const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list);
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

Variable PermutePooledEmbsFunction::forward(
    AutogradContext* ctx,
    const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  ctx->saved_data["offset_dim_list"] = offset_dim_list;
  ctx->saved_data["permute_list"] = permute_list;
  ctx->saved_data["inv_offset_dim_list"] = inv_offset_dim_list;
  ctx->saved_data["inv_permute_list"] = inv_permute_list;
  TORCH_CHECK(
      offset_dim_list.scalar_type() == at::ScalarType::Long,
      "offset_dim_list needs to have long/int64 type");
  TORCH_CHECK(
      permute_list.scalar_type() == at::ScalarType::Long,
      "permute_list needs to have long/int64 type");

  const auto permute_pooled_embs_op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_pooled_embs", "")
          .typed<decltype(permute_pooled_embs_cpu)>();
  return permute_pooled_embs_op.call(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list);
}

variable_list PermutePooledEmbsFunction::backward(
    AutogradContext* ctx,
    variable_list grad_output) {
  const auto& offset_dim_list = ctx->saved_data["offset_dim_list"].toTensor();
  const auto& permute_list = ctx->saved_data["permute_list"].toTensor();
  const auto& inv_offset_dim_list =
      ctx->saved_data["inv_offset_dim_list"].toTensor();
  const auto& inv_permute_list = ctx->saved_data["inv_permute_list"].toTensor();
  variable_list grad_inputs(6);
  static auto permute_pooled_embs_op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_pooled_embs", "")
          .typed<decltype(permute_pooled_embs_cpu)>();
  grad_inputs[0] = permute_pooled_embs_op.call(
      grad_output[0],
      inv_offset_dim_list,
      inv_permute_list,
      offset_dim_list,
      permute_list);
  return grad_inputs;
}

} // namespace fbgemm_gpu
