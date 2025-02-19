/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <vector>
#include "fbgemm_gpu/permute_pooled_embedding_ops.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor permute_pooled_embs_cpu_impl(
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

at::Tensor permute_pooled_embs_cpu(
    const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list) {
  return permute_pooled_embs_cpu_impl(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list,
      false);
}

///@ingroup permute-duplicate-pooled-embs-cpu
at::Tensor permute_duplicate_pooled_embs_cpu(
    const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list) {
  return permute_pooled_embs_cpu_impl(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list,
      true);
}

///@ingroup permute-pooled-embs-cpu
at::Tensor permute_pooled_embs_auto_grad(
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
at::Tensor permute_pooled_embs_auto_grad_cpu(
    const Tensor& pooled_embs,
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return permute_pooled_embs_cpu_impl(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list,
      false);
}

///@ingroup permute-duplicate-pooled-embs-cpu
at::Tensor permute_duplicate_pooled_embs_auto_grad_cpu(
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

at::Tensor permute_pooled_embs_meta(
    const Tensor& pooled_embs,
    const Tensor& /* offset_dim_list */,
    const Tensor& /* permute_list */,
    const Tensor& /* inv_offset_dim_list */,
    const Tensor& /* inv_permute_list */) {
  return torch::empty_like(pooled_embs);
}

at::Tensor permute_pooled_embs_auto_grad_meta(
    const Tensor& pooled_embs,
    const Tensor& /* offset_dim_list */,
    const Tensor& /* permute_list */,
    const Tensor& /* inv_offset_dim_list */,
    const Tensor& /* inv_permute_list */) {
  return torch::empty_like(pooled_embs);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "permute_pooled_embs(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "permute_pooled_embs_auto_grad(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "permute_duplicate_pooled_embs(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  m.def(
      "permute_duplicate_pooled_embs_auto_grad(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
}

FBGEMM_OP_DISPATCH(
    CPU,
    "permute_pooled_embs",
    fbgemm_gpu::permute_pooled_embs_cpu);
FBGEMM_OP_DISPATCH(
    CPU,
    "permute_pooled_embs_auto_grad",
    fbgemm_gpu::permute_pooled_embs_auto_grad_cpu);
FBGEMM_OP_DISPATCH(
    CPU,
    "permute_duplicate_pooled_embs",
    fbgemm_gpu::permute_duplicate_pooled_embs_cpu);
FBGEMM_OP_DISPATCH(
    CPU,
    "permute_duplicate_pooled_embs_auto_grad",
    fbgemm_gpu::permute_duplicate_pooled_embs_auto_grad_cpu);

FBGEMM_OP_DISPATCH(
    Meta,
    "permute_pooled_embs",
    fbgemm_gpu::permute_pooled_embs_meta);
FBGEMM_OP_DISPATCH(
    Meta,
    "permute_pooled_embs_auto_grad",
    fbgemm_gpu::permute_pooled_embs_auto_grad_meta);

FBGEMM_OP_DISPATCH(
    Autograd,
    "permute_pooled_embs_auto_grad",
    fbgemm_gpu::permute_pooled_embs_auto_grad);
