/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/util/irange.h>
#include <torch/script.h>
#include <vector>

#include "fbgemm_gpu/permute_pooled_embedding_ops.h"
#include "fbgemm_gpu/permute_pooled_embedding_ops_utils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {
    
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

Tensor permute_pooled_embs_auto_grad_gpu(
    const Tensor& pooled_embs,
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  return PermutePooledEmbsFunction<permute_pooled_embs_gpu>::apply(
      pooled_embs,
      offset_dim_list,
      permute_list,
      inv_offset_dim_list,
      inv_permute_list);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "permute_pooled_embs(Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list, Tensor inv_offset_dim_list, Tensor inv_permute_list) -> Tensor");
  DISPATCH_TO_CUDA("permute_pooled_embs", fbgemm_gpu::permute_pooled_embs_gpu);
  DISPATCH_TO_CUDA(
      "permute_pooled_embs_auto_grad",
      fbgemm_gpu::permute_pooled_embs_auto_grad_gpu);
}
