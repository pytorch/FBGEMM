/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
///@defgroup permute-pooled-embs-gpu CUDA Permutation Operators

///@defgroup permute-pooled-embs-cpu CPU Permutation Operators

namespace fbgemm_gpu {
///@ingroup permute-pooled-embs-cpu
at::Tensor permute_pooled_embs_split_cpu(
    const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list);

///@ingroup permute-pooled-embs-gpu
at::Tensor permute_pooled_embs_split_gpu(
    const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list);

///@ingroup permute-pooled-embs-cpu
at::Tensor permute_pooled_embs_auto_grad_split_cpu(
    const at::Tensor& pooled_embs,
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list);

///@ingroup permute-pooled-embs-gpu
at::Tensor permute_pooled_embs_auto_grad_split_gpu(
    const at::Tensor& pooled_embs,
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list);
} // namespace fbgemm_gpu
