/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h> // @manual

#include "fbgemm_gpu/group_gemm_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm_gpu {

namespace {
using Tensor = at::Tensor;

std::vector<Tensor> gemm_grouped_cpu(
    const std::vector<Tensor>& a_group,
    const std::vector<Tensor>& b_group,
    const c10::optional<std::vector<Tensor>>& c_group) {
  std::vector<Tensor> output_group;
  for (int i = 0; i < a_group.size(); i++) {
    Tensor output = at::mm(a_group[i], b_group[i]);
    if (c_group.has_value()) {
      output += c_group.value()[i];
    }
    output_group.push_back(std::move(output));
  }
  return output_group;
}

std::vector<Tensor> group_linear_forward_cpu(
    const std::vector<Tensor>& input_group,
    const std::vector<Tensor>& weight_group,
    const c10::optional<std::vector<Tensor>>& bias_group) {
  auto weight_group_transposed =
      group_linear_forward_helper(input_group, weight_group, bias_group);
  return gemm_grouped_cpu(input_group, weight_group_transposed, bias_group);
}

} // namespace

std::vector<Tensor> group_linear_forward_helper(
    const std::vector<Tensor>& input_group,
    const std::vector<Tensor>& weight_group,
    const c10::optional<std::vector<Tensor>>& bias_group) {
  auto num_groups = input_group.size();
  TORCH_CHECK(num_groups == weight_group.size());
  if (bias_group.has_value()) {
    TORCH_CHECK(num_groups == bias_group.value().size())
  }

  // Transpose weights
  std::vector<Tensor> weight_group_transposed;
  weight_group_transposed.reserve(num_groups);
  for (auto& weight : weight_group) {
    weight_group_transposed.push_back(weight.t());
  }
  TORCH_CHECK(num_groups == weight_group_transposed.size())
  return weight_group_transposed;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // Call it gmm similar to aten bmm op.
  m.def(
      "gmm(Tensor[] a_group, Tensor[] b_group, Tensor[]? c_group=None) -> Tensor[]");
  // Backward is not supported due to the limitation of PyTorch autograd
  // (https://fb.workplace.com/groups/1405155842844877/permalink/6040437302650018/)
  m.def(
      "group_linear_forward(Tensor[] input_group, Tensor[] weight_group, Tensor[]? bias_group=None) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU("gmm", fbgemm_gpu::gemm_grouped_cpu);
  DISPATCH_TO_CPU("group_linear_forward", fbgemm_gpu::group_linear_forward_cpu);
}
