/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

class BatchIndexSelectDim0CPUOp
    : public torch::autograd::Function<BatchIndexSelectDim0CPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& inputs,
      const Tensor& indices,
      const std::vector<int64_t>& input_num_indices,
      const std::vector<int64_t>& input_rows,
      const std::vector<int64_t>& input_columns,
      const bool permute_output_dim_0_1) {
    const int64_t num_inputs = input_num_indices.size();
    ctx->save_for_backward({indices});

    ctx->saved_data["input_numel"] = inputs.numel();
    ctx->saved_data["input_num_indices"] = input_num_indices;
    ctx->saved_data["input_rows"] = input_rows;
    ctx->saved_data["input_columns"] = input_columns;
    ctx->saved_data["permute_output_dim_0_1"] = permute_output_dim_0_1;

    // Early exit
    if (inputs.numel() == 0) {
      return {at::empty({0}, inputs.options())};
    }

    // Compute section sizes for splitting tensors
    std::vector<int64_t> input_numels;
    std::vector<int64_t> indices_numels;
    input_numels.reserve(num_inputs);
    indices_numels.reserve(num_inputs);
    for (auto i = 0; i < num_inputs; i++) {
      input_numels.push_back(input_rows[i] * input_columns[i]);
      indices_numels.push_back(input_num_indices[i]);
    }

    ctx->saved_data["indices_numels"] = indices_numels;

    // Split tensors into vectors
    const auto inputs_ = at::split_with_sizes(inputs, input_numels, 0);
    const auto indices_ = at::split_with_sizes(indices, indices_numels, 0);

    std::vector<Tensor> outputs;
    outputs.reserve(num_inputs);
    for (auto i = 0; i < num_inputs; i++) {
      const auto input = inputs_[i].view({input_rows[i], input_columns[i]});
      const auto index = indices_[i];
      const auto output = at::index_select(input, 0, index);
      if (permute_output_dim_0_1) {
        outputs.push_back(output);
      } else {
        outputs.push_back(output.flatten());
      }
    }

    // permute_output_dim_0_1 = true shape: (batch_size, num_inputs, cols)
    // permute_output_dim_0_1 = false shape: (num_inputs, batch_size cols)
    return {at::concat(outputs, permute_output_dim_0_1 ? 1 : 0).flatten()};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    using torch::autograd::Variable;

    TORCH_CHECK_EQ(grad_outputs.size(), 1);

    const auto grad_output = grad_outputs[0];
    const auto input_numel = ctx->saved_data["input_numel"].toInt();

    // Early exit
    if (input_numel == 0) {
      return {
          at::empty({0}, grad_output.options()),
          Variable(), // indices
          Variable(), // input_num_indices
          Variable(), // input_rows
          Variable(), // input_columns
          Variable() // permute_output_dim_0_1
      };
    }

    const auto saved = ctx->get_saved_variables();
    auto indices = *std::begin(saved);

    const auto input_num_indices =
        ctx->saved_data["input_num_indices"].toIntVector();
    const auto input_rows = ctx->saved_data["input_rows"].toIntVector();
    const auto input_cols = ctx->saved_data["input_columns"].toIntVector();
    const auto permute_output_dim_0_1 =
        ctx->saved_data["permute_output_dim_0_1"].toBool();
    const auto indices_numels = ctx->saved_data["indices_numels"].toIntVector();

    const int64_t num_inputs = input_num_indices.size();

    std::vector<Tensor> grads;
    if (permute_output_dim_0_1) {
      grads = at::split_with_sizes(
          grad_output.view({input_num_indices[0], -1}), input_cols, 1);
    } else {
      std::vector<int64_t> grad_numels;
      grad_numels.reserve(num_inputs);
      for (auto i = 0; i < num_inputs; i++) {
        grad_numels.push_back(input_num_indices[i] * input_cols[i]);
      }
      grads = at::split_with_sizes(grad_output, grad_numels, 0);
    }

    const auto indices_ = at::split_with_sizes(indices, indices_numels, 0);

    std::vector<Tensor> grad_inputs;
    grad_inputs.reserve(num_inputs);
    for (auto i = 0; i < num_inputs; i++) {
      const auto num_indices = input_num_indices[i];
      const auto grad_input =
          at::zeros({input_rows[i], input_cols[i]}, grad_output.options());
      const auto grad =
          permute_output_dim_0_1 ? grads[i] : grads[i].view({num_indices, -1});
      grad_inputs.push_back(
          at::index_add(grad_input, 0, indices_[i], grad).flatten());
    }

    return {
        at::concat(grad_inputs, 0),
        Variable(), // indices
        Variable(), // input_num_indices
        Variable(), // input_rows
        Variable(), // input_columns
        Variable() // permute_output_dim_0_1
    };
  }
};

Tensor batch_index_select_dim0_cpu(
    Tensor inputs,
    Tensor indices,
    std::vector<int64_t> input_num_indices,
    std::vector<int64_t> input_rows,
    std::vector<int64_t> input_columns,
    // Permute dim 0 and 1 of the output tensor
    const bool permute_output_dim_0_1) {
  const int64_t num_inputs = input_num_indices.size();
  TORCH_CHECK(
      num_inputs == static_cast<int64_t>(input_rows.size()),
      "[batch_index_select_dim0] input_rows must have the same length as "
      "input_num_indices.");
  TORCH_CHECK(
      num_inputs == static_cast<int64_t>(input_columns.size()),
      "[batch_index_select_dim0] input_columns must have the same length as "
      "input_num_indices.");

  TORCH_CHECK(
      reinterpret_cast<uint64_t>(inputs.data_ptr()) % 16 == 0,
      "Currently batch_index_select only supports 16-byte align input tensors");

  const auto int_opts = torch::TensorOptions().dtype(torch::kInt64);
  const auto num_cols =
      torch::from_blob(input_columns.data(), {num_inputs}, int_opts);
  const auto input_num_rows =
      torch::from_blob(input_rows.data(), {num_inputs}, int_opts);
  const auto output_num_rows =
      torch::from_blob(input_num_indices.data(), {num_inputs}, int_opts);

  if (num_inputs > 0) {
    TORCH_CHECK(
        torch::all(torch::gt(num_cols, 0)).item<bool>(),
        "[batch_index_select_dim0] All input_columns must be the same.");
    TORCH_CHECK(
        torch::all(torch::gt(input_num_rows, 0)).item<bool>(),
        "[batch_index_select_dim0] All input_rows must be the same.");
    if (permute_output_dim_0_1) {
      // All output rows must be the same
      TORCH_CHECK(input_num_indices[0] > 0);
      TORCH_CHECK(
          torch::all(torch::eq(output_num_rows, input_num_indices[0]))
              .item<bool>(),
          "[batch_index_select_dim0] All input_num_indices must be the same if "
          "permute_output_dim_0_1 is true.");
    } else {
      TORCH_CHECK(
          torch::all(torch::gt(output_num_rows, 0)).item<bool>(),
          "[batch_index_select_dim0] All input_num_indices must be greater than zero.");
    }
  }

  return BatchIndexSelectDim0CPUOp::apply(
      inputs,
      indices,
      input_num_indices,
      input_rows,
      input_columns,
      permute_output_dim_0_1)[0];
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def(
      "batch_index_select_dim0("
      "    Tensor inputs,"
      "    Tensor indices,"
      "    SymInt[] input_num_indices,"
      "    SymInt[] input_rows,"
      "    SymInt[] input_columns,"
      "    bool permute_output_dim_0_1=False) -> Tensor");
  DISPATCH_TO_CPU("batch_index_select_dim0", batch_index_select_dim0_cpu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.impl_abstract_pystub(
      "fbgemm_gpu.sparse_ops",
      "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_py");
  m.def(
      "batch_index_select_dim0("
      "    Tensor inputs,"
      "    Tensor indices,"
      "    SymInt[] input_num_indices,"
      "    SymInt[] input_rows,"
      "    SymInt[] input_columns,"
      "    bool permute_output_dim_0_1=False) -> Tensor");
  DISPATCH_TO_CPU("batch_index_select_dim0", batch_index_select_dim0_cpu);
}
