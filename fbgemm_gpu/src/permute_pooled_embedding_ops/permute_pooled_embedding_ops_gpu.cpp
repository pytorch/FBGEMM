/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/util/irange.h>
#include <torch/script.h>
#include <vector>

#include "fbgemm_gpu/permute_pooled_embedding_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

///@defgroup permute-pooled-embs-gpu
///@defgroup permute-pooled-embs-cpu

namespace fbgemm_gpu {

///@ingroup permute-pooled-embs-cpu
Tensor permute_pooled_embs_cpu(
    const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  TORCH_CHECK(
      offset_dim_list.scalar_type() == at::ScalarType::Long,
      "offset_dim_list needs to have long/int64 type")
  TORCH_CHECK(
      permute_list.scalar_type() == at::ScalarType::Long,
      "permute_list needs to have long/int64 type")
  auto permute = permute_list.data_ptr<int64_t>();
  const auto n = permute_list.numel();
  std::vector<int64_t> dims;
  dims.reserve(n - 1);
  for (const auto i : c10::irange(1, n)) {
    dims.push_back(offset_dim_list[i].item<int64_t>());
  }
  auto ts = pooled_embs.tensor_split(dims, 1);
  std::vector<Tensor> permuted_ts;
  permuted_ts.reserve(n);
  for (const auto i : c10::irange(n)) {
    permuted_ts.push_back(ts[permute[i]]);
  }
  return at::cat(permuted_ts, 1);
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class PermutePooledEmbsFunction
    : public torch::autograd::Function<PermutePooledEmbsFunction> {
 public:
  static Variable forward(
      AutogradContext* ctx,
      const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
      const Tensor& offset_dim_list,
      const Tensor& permute_list,
      const Tensor& inv_offset_dim_list,
      const Tensor& inv_permute_list,
      const bool& allow_duplicates = false) {
    ctx->saved_data["offset_dim_list"] = offset_dim_list;
    ctx->saved_data["permute_list"] = permute_list;
    ctx->saved_data["inv_offset_dim_list"] = inv_offset_dim_list;
    ctx->saved_data["inv_permute_list"] = inv_permute_list;
    ctx->saved_data["allow_duplicates"] = allow_duplicates;
    TORCH_CHECK(
        offset_dim_list.scalar_type() == at::ScalarType::Long,
        "offset_dim_list needs to have long/int64 type");
    TORCH_CHECK(
        permute_list.scalar_type() == at::ScalarType::Long,
        "permute_list needs to have long/int64 type");

    const auto schema = allow_duplicates
        ? "fbgemm::permute_duplicate_pooled_embs"
        : "fbgemm::permute_pooled_embs";
    const auto permute_pooled_embs_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow(schema, "")
            .typed<decltype(permute_pooled_embs_cpu)>();
    return permute_pooled_embs_op.call(
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
    const auto& allow_duplicates = ctx->saved_data["allow_duplicates"].toBool();
    TORCH_CHECK(
        allow_duplicates == false,
        "permute_pooled_embs does not support allow_duplicates in backward!");
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
};

///@ingroup permute-pooled-embs-gpu
Tensor permute_pooled_embs_auto_grad_gpu(
    const Tensor& pooled_embs,
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return PermutePooledEmbsFunction::apply(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list,
      false);
}

///@ingroup permute-pooled-embs-cpu
Tensor permute_pooled_embs_auto_grad_cpu(
    const Tensor& pooled_embs,
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return PermutePooledEmbsFunction::apply(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list,
      false);
}

///@ingroup permute-pooled-embs-cpu
Tensor permute_pooled_embs_auto_grad(
    const Tensor& pooled_embs,
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return PermutePooledEmbsFunction::apply(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list,
      false);
}

Tensor permute_pooled_embs_meta(
    const Tensor& pooled_embs,
    const Tensor& /* offset_dim_list */,
    const Tensor& /* permute_list */,
    const Tensor& /* inv_offset_dim_list */,
    const Tensor& /* inv_permute_list */) {
  return torch::empty_like(pooled_embs);
}

Tensor permute_pooled_embs_auto_grad_meta(
    const Tensor& pooled_embs,
    const Tensor& /* offset_dim_list */,
    const Tensor& /* permute_list */,
    const Tensor& /* inv_offset_dim_list */,
    const Tensor& /* inv_permute_list */) {
  return torch::empty_like(pooled_embs);
}

///@ingroup permute-duplicate-pooled-embs-gpu
Tensor permute_duplicate_pooled_embs_auto_grad_gpu(
    const Tensor& pooled_embs,
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return PermutePooledEmbsFunction::apply(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list,
      true);
}
} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "permute_pooled_embs(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  DISPATCH_TO_CUDA("permute_pooled_embs", fbgemm_gpu::permute_pooled_embs_gpu);
  DISPATCH_TO_CPU("permute_pooled_embs", fbgemm_gpu::permute_pooled_embs_cpu);
  DISPATCH_TO_META("permute_pooled_embs", fbgemm_gpu::permute_pooled_embs_meta);
  m.def(
      "permute_pooled_embs_auto_grad(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  DISPATCH_TO_AUTOGRAD(
      "permute_pooled_embs_auto_grad",
      fbgemm_gpu::permute_pooled_embs_auto_grad);
  DISPATCH_TO_CPU(
      "permute_pooled_embs_auto_grad",
      fbgemm_gpu::permute_pooled_embs_auto_grad_cpu);
  DISPATCH_TO_CUDA(
      "permute_pooled_embs_auto_grad",
      fbgemm_gpu::permute_pooled_embs_auto_grad_gpu);
  DISPATCH_TO_META(
      "permute_pooled_embs_auto_grad",
      fbgemm_gpu::permute_pooled_embs_auto_grad_meta);
  m.def(
      "permute_duplicate_pooled_embs(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  DISPATCH_TO_CUDA(
      "permute_duplicate_pooled_embs",
      fbgemm_gpu::permute_duplicate_pooled_embs_gpu);
  m.def(
      "permute_duplicate_pooled_embs_auto_grad(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  DISPATCH_TO_CUDA(
      "permute_duplicate_pooled_embs_auto_grad",
      fbgemm_gpu::permute_duplicate_pooled_embs_auto_grad_gpu);
}
