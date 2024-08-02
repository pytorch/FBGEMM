/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/script.h>

namespace fbgemm_gpu {

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

template <torch::autograd::Variable (*permute_pooled_embs_op)(
    const at::Tensor&, // [B_local][Sum_T_global(D)]
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&)>
class PermutePooledEmbsFunctionSplit
    : public torch::autograd::Function<
          PermutePooledEmbsFunctionSplit<permute_pooled_embs_op>> {
 public:
  static Variable forward(
      AutogradContext* ctx,
      const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
      const at::Tensor& offset_dim_list,
      const at::Tensor& permute_list,
      const at::Tensor& inv_offset_dim_list,
      const at::Tensor& inv_permute_list) {
    at::AutoDispatchBelowADInplaceOrView guard;
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
    return permute_pooled_embs_op(
        pooled_embs,
        offset_dim_list,
        permute_list,
        inv_offset_dim_list,
        inv_permute_list);
  }
  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    const auto& offset_dim_list = ctx->saved_data["offset_dim_list"].toTensor();
    const auto& permute_list = ctx->saved_data["permute_list"].toTensor();
    const auto& inv_offset_dim_list =
        ctx->saved_data["inv_offset_dim_list"].toTensor();
    const auto& inv_permute_list =
        ctx->saved_data["inv_permute_list"].toTensor();
    variable_list grad_inputs(5);
    grad_inputs[0] = permute_pooled_embs_op(
        grad_output[0],
        inv_offset_dim_list,
        inv_permute_list,
        offset_dim_list,
        permute_list);
    return grad_inputs;
  }
};

} // namespace fbgemm_gpu
