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

#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/ops_utils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm_gpu {

using Tensor = at::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

using Tensor = at::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

class PermuteMultiEmbeddingOp
    : public torch::autograd::Function<PermuteMultiEmbeddingOp> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      const at::TensorList& pooled_embs,
      const std::vector<int64_t>& permutes,
      const std::vector<int64_t>& in_lengths,
      const std::vector<int64_t>& out_lengths);

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output);
};

std::vector<Tensor> permute_multi_embedding_cpu(
    const at::TensorList& pooled_embs,
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths,
    const bool& reverse_permute);

std::vector<Tensor> permute_multi_embedding_meta(
    const at::TensorList& pooled_embs,
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths,
    const bool& reverse_permute);

std::vector<Tensor> permute_multi_embedding_gpu(
    const at::TensorList& pooled_embs,
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths,
    const bool& reverse_permute);
} // namespace fbgemm_gpu
