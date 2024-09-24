/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/csrc/autograd/custom_function.h>

#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

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
  static constexpr bool is_traceable = true;
  static variable_list forward(
      AutogradContext* ctx,
      const at::TensorList& pooled_embs,
      const Tensor& permutes,
      const Tensor& in_shapes,
      const Tensor& out_shapes,
      const c10::SymIntArrayRef out_lengths);

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output);
};

std::vector<Tensor> permute_multi_embedding_function_cpu(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const c10::IntArrayRef out_lengths,
    const bool& reverse_permute);

std::vector<Tensor> permute_multi_embedding_function_meta(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const c10::SymIntArrayRef out_lengths,
    const bool& reverse_permute);

std::vector<Tensor> permute_multi_embedding_function_gpu(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const c10::IntArrayRef out_lengths,
    const bool& reverse_permute);

std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<int32_t>>
kt_regroup_arguments_impl(
    const std::vector<std::vector<std::string>>& keys,
    const std::vector<std::vector<int64_t>>& lengths,
    const std::vector<std::vector<std::string>>& groups);

enum PermuteParam {
  in_tensor = 0,
  out_tensor = 1,
  in_offset = 2,
  out_offset = 3,
  length = 4,
  next = 5,
  size = 6, // number of elements in PermuteParam excluding this size
};

} // namespace fbgemm_gpu
