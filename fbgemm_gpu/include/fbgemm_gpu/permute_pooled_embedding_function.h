/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/csrc/autograd/custom_function.h>

namespace fbgemm_gpu {

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class PermutePooledEmbsFunction
    : public torch::autograd::Function<PermutePooledEmbsFunction> {
 public:
  static Variable forward(
      AutogradContext* ctx,
      const at::Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
      const at::Tensor& offset_dim_list,
      const at::Tensor& permute_list,
      const at::Tensor& inv_offset_dim_list,
      const at::Tensor& inv_permute_list,
      const bool& allow_duplicates = false);

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output);
};

} // namespace fbgemm_gpu
