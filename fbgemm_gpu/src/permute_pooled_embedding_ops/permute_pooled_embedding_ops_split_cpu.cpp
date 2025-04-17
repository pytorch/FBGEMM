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

#include "fbgemm_gpu/permute_pooled_embedding_ops_split.h"
#include "fbgemm_gpu/permute_pooled_embs_function_split.h"
#include "fbgemm_gpu/utils/ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

Tensor permute_pooled_embs_split_cpu_impl(
    const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list,
    const bool& allow_duplicates) {
  if (pooled_embs.numel() == 0) {
    return pooled_embs;
  }
  TORCH_CHECK(
      offset_dim_list.scalar_type() == at::ScalarType::Long,
      "offset_dim_list needs to have long/int64 type")
  TORCH_CHECK(
      permute_list.scalar_type() == at::ScalarType::Long,
      "permute_list needs to have long/int64 type")
  auto permute = permute_list.data_ptr<int64_t>();
  const auto n = permute_list.numel();
  const auto dims_size = allow_duplicates ? offset_dim_list.numel() : n;
  std::vector<int64_t> dims;
  dims.reserve(dims_size - 1);
  for (const auto i : c10::irange(1, dims_size)) {
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

Tensor permute_pooled_embs_split_cpu(
    const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return permute_pooled_embs_split_cpu_impl(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list,
      false);
}

Tensor permute_duplicate_pooled_embs_split_cpu(
    const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return permute_pooled_embs_split_cpu_impl(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list,
      true);
}

Tensor permute_pooled_embs_split_dispatch_call(
    const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_pooled_embs_split", "")
          .typed<decltype(fbgemm_gpu::permute_pooled_embs_split_cpu)>();
  return op.call(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list);
}

Tensor permute_duplicate_pooled_embs_split_dispatch_call(
    const Tensor& pooled_embs,
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_duplicate_pooled_embs_split", "")
          .typed<
              decltype(fbgemm_gpu::permute_duplicate_pooled_embs_split_cpu)>();
  return op.call(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list);
}

Tensor permute_pooled_embs_auto_grad_split_cpu(
    const Tensor& pooled_embs,
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return PermutePooledEmbsFunctionSplit<
      permute_pooled_embs_split_dispatch_call>::
      apply(
          pooled_embs,
          offset_dim_list,
          permute_list,
          inv_offset_dim_list,
          inv_permute_list);
}

Tensor permute_duplicate_pooled_embs_auto_grad_split_cpu(
    const Tensor& pooled_embs,
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return PermutePooledEmbsFunctionSplit<
      permute_duplicate_pooled_embs_split_dispatch_call>::
      apply(
          pooled_embs,
          offset_dim_list,
          permute_list,
          inv_offset_dim_list,
          inv_permute_list);
}
} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.sparse_ops");
  m.def(
      "permute_pooled_embs_split(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  DISPATCH_TO_CPU(
      "permute_pooled_embs_split", fbgemm_gpu::permute_pooled_embs_split_cpu);
  m.def(
      "permute_duplicate_pooled_embs_split(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  DISPATCH_TO_CPU(
      "permute_duplicate_pooled_embs_split",
      fbgemm_gpu::permute_duplicate_pooled_embs_split_cpu);
  m.def(
      "permute_pooled_embs_auto_grad_split(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  DISPATCH_TO_CPU(
      "permute_pooled_embs_auto_grad_split",
      fbgemm_gpu::permute_pooled_embs_auto_grad_split_cpu);
  DISPATCH_TO_AUTOGRAD_CPU(
      "permute_pooled_embs_auto_grad_split",
      fbgemm_gpu::permute_pooled_embs_auto_grad_split_cpu);
  m.def(
      "permute_duplicate_pooled_embs_auto_grad_split(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  DISPATCH_TO_CPU(
      "permute_duplicate_pooled_embs_auto_grad_split",
      fbgemm_gpu::permute_duplicate_pooled_embs_auto_grad_split_cpu);
  DISPATCH_TO_AUTOGRAD_CPU(
      "permute_duplicate_pooled_embs_auto_grad_split",
      fbgemm_gpu::permute_duplicate_pooled_embs_auto_grad_split_cpu);
}
