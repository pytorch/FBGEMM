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

using Tensor = at::Tensor;

namespace fbgemm_gpu {

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
  DISPATCH_TO_CUDA("permute_pooled_embs", fbgemm_gpu::permute_pooled_embs_gpu);
  DISPATCH_TO_CUDA(
      "permute_pooled_embs_auto_grad",
      fbgemm_gpu::permute_pooled_embs_auto_grad_gpu);
  DISPATCH_TO_CUDA(
      "permute_duplicate_pooled_embs",
      fbgemm_gpu::permute_duplicate_pooled_embs_gpu);
  DISPATCH_TO_CUDA(
      "permute_duplicate_pooled_embs_auto_grad",
      fbgemm_gpu::permute_duplicate_pooled_embs_auto_grad_gpu);
}
