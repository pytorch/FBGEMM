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

#include <c10/util/ArrayRef.h>
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/ops_utils.h"

#include <memory>
#include <string>

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

namespace {
Tensor tensor_from_vec(const std::vector<int64_t>& vec) {
  auto tensor = at::empty(
      {static_cast<int64_t>(vec.size())},
      at::TensorOptions().dtype(torch::kInt64));
  TORCH_CHECK(tensor.is_contiguous());
  std::memcpy(
      tensor.data_ptr<int64_t>(), vec.data(), sizeof(int64_t) * vec.size());
  return tensor;
};

std::vector<int64_t> vecref_from_tensor(const Tensor& t) {
  TORCH_CHECK(t.is_contiguous());
  const auto numel = static_cast<size_t>(t.numel());
  const auto* ptr = t.data_ptr<int64_t>();
  return std::vector(ptr, ptr + numel);
};

} // namespace

class BatchIndexSelectDim0CPUOp
    : public torch::autograd::Function<BatchIndexSelectDim0CPUOp> {
 public:
  static torch::autograd::variable_list forward_impl(
      const Tensor& inputs,
      const Tensor& indices,
      const c10::SymIntArrayRef _input_num_indices,
      const c10::SymIntArrayRef _input_rows,
      const c10::SymIntArrayRef _input_columns,
      const bool permute_output_dim_0_1) {
    const int64_t num_inputs = _input_num_indices.size();
    TORCH_CHECK(
        num_inputs == static_cast<int64_t>(_input_rows.size()),
        "[batch_index_select_dim0] input_rows must have the same length as "
        "input_num_indices.");
    TORCH_CHECK(
        num_inputs == static_cast<int64_t>(_input_columns.size()),
        "[batch_index_select_dim0] input_columns must have the same length as "
        "input_num_indices.");
    TORCH_CHECK(
        reinterpret_cast<uint64_t>(inputs.data_ptr()) % 16 == 0,
        "Currently batch_index_select only supports 16-byte align input tensors");

    static auto to_vec_int64 =
        [](const c10::SymIntArrayRef& sym_vec) -> std::vector<int64_t> {
      std::vector<int64_t> vec;
      std::transform(
          sym_vec.begin(),
          sym_vec.end(),
          std::back_inserter(vec),
          [](const auto& symint) {
            return symint.guard_int(__FILE__, __LINE__);
          });
      return vec;
    };

    Tensor ret;
    Tensor indices_numels_tensor;
    std::vector<int64_t> input_num_indices;
    std::vector<int64_t> input_rows;
    std::vector<int64_t> input_columns;

    Tensor input_num_indices_tensor;
    Tensor input_columns_tensor;
    Tensor input_rows_tensor;

    // Early exit
    if (inputs.numel() == 0) {
      ret = at::empty({0}, inputs.options());
    } else {
      input_num_indices = to_vec_int64(_input_num_indices);
      input_num_indices_tensor = tensor_from_vec(input_num_indices);
      input_rows = to_vec_int64(_input_rows);
      input_columns = to_vec_int64(_input_columns);
      input_columns_tensor = tensor_from_vec(input_columns);
      input_rows_tensor = tensor_from_vec(input_rows);

      TORCH_CHECK(
          torch::all(torch::gt(input_columns_tensor, 0)).item<bool>(),
          "[batch_index_select_dim0] All input_columns must be the same.");
      TORCH_CHECK(
          torch::all(torch::gt(input_rows_tensor, 0)).item<bool>(),
          "[batch_index_select_dim0] All input_rows must be the same.");

      if (permute_output_dim_0_1) {
        // All output rows must be the same
        TORCH_CHECK(input_num_indices[0] > 0);
        TORCH_CHECK(
            torch::all(
                torch::eq(input_num_indices_tensor, input_num_indices[0]))
                .item<bool>(),
            "[batch_index_select_dim0] All input_num_indices must be the same if "
            "permute_output_dim_0_1 is true.");
      } else {
        TORCH_CHECK(
            torch::all(torch::gt(input_num_indices_tensor, 0)).item<bool>(),
            "[batch_index_select_dim0] All input_num_indices must be greater than zero.");
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
      indices_numels_tensor = tensor_from_vec(indices_numels);

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
      ret = at::concat(outputs, permute_output_dim_0_1 ? 1 : 0).flatten();
    }

    auto saved_data_tensor = tensor_from_vec({inputs.numel()});

    return {
        ret,

        input_num_indices_tensor,
        input_rows_tensor,
        input_columns_tensor,

        indices_numels_tensor,
        saved_data_tensor};
  }

  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& inputs,
      const Tensor& indices,
      const c10::SymIntArrayRef input_num_indices,
      const c10::SymIntArrayRef input_rows,
      const c10::SymIntArrayRef input_columns,
      const bool permute_output_dim_0_1) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto forward_op_impl =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batch_index_select_dim0_forward_cpu_impl", "")
            .typed<decltype(forward_impl)>();

    auto res = forward_op_impl.call(
        inputs,
        indices,
        input_num_indices,
        input_rows,
        input_columns,
        permute_output_dim_0_1);
    ctx->saved_data["permute_output_dim_0_1"] = permute_output_dim_0_1;
    ctx->save_for_backward(
        std::vector<Tensor>{indices, res[1], res[2], res[3], res[4], res[5]});
    res.resize(1);
    return res;
  }

  static Tensor backward_impl(
      const Tensor& grad_output,
      const Tensor& indices,
      const Tensor& indices_numels,
      const Tensor& input_num_indices,
      const Tensor& input_rows,
      const Tensor& input_columns,
      const bool permute_output_dim_0_1,
      const Tensor& saved_tensor) {
    const int64_t input_numel = saved_tensor[0].item<int64_t>();

    // Early exit
    if (input_numel == 0) {
      return at::empty({0}, grad_output.options());
    }
    const int64_t num_inputs = input_num_indices.size(0);

    auto input_num_indices_vec = vecref_from_tensor(input_num_indices);
    auto input_rows_vec = vecref_from_tensor(input_rows);
    auto input_columns_vec = vecref_from_tensor(input_columns);

    std::vector<Tensor> grads;
    if (permute_output_dim_0_1) {
      grads = at::split_with_sizes(
          grad_output.view({input_num_indices_vec[0], -1}),
          input_columns_vec,
          1);
    } else {
      std::vector<int64_t> grad_numels;
      grad_numels.reserve(num_inputs);
      for (auto i = 0; i < num_inputs; i++) {
        grad_numels.push_back(input_num_indices_vec[i] * input_columns_vec[i]);
      }
      grads = at::split_with_sizes(grad_output, grad_numels, 0);
    }

    const auto indices_ =
        at::split_with_sizes(indices, vecref_from_tensor(indices_numels), 0);

    std::vector<Tensor> grad_inputs;
    grad_inputs.reserve(num_inputs);
    for (auto i = 0; i < num_inputs; i++) {
      const auto num_indices = input_num_indices_vec[i];
      const auto grad_input = at::zeros(
          {input_rows_vec[i], input_columns_vec[i]}, grad_output.options());
      const auto grad =
          permute_output_dim_0_1 ? grads[i] : grads[i].view({num_indices, -1});
      grad_inputs.push_back(
          at::index_add(grad_input, 0, indices_[i], grad).flatten());
    }

    return at::concat(grad_inputs, 0);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    using torch::autograd::Variable;
    TORCH_CHECK_EQ(grad_outputs.size(), 1);

    const auto grad_output = grad_outputs[0];
    const auto permute_output_dim_0_1 =
        ctx->saved_data["permute_output_dim_0_1"].toBool();
    const auto saved = ctx->get_saved_variables();

    auto savedItr = std::begin(saved);
    auto indices = *savedItr++;

    auto input_num_indices = *savedItr++;
    auto input_rows = *savedItr++;
    auto input_columns = *savedItr++;

    auto indices_numels = *savedItr++;
    auto saved_tensor = *savedItr++;
    static auto backward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batch_index_select_dim0_backward_cpu_impl", "")
            .typed<decltype(backward_impl)>();
    auto ret = backward_op.call(
        grad_output,
        indices,
        indices_numels,
        input_num_indices,
        input_rows,
        input_columns,
        permute_output_dim_0_1,
        saved_tensor);
    return {
        ret,
        Variable(), // indices
        Variable(), // input_num_indices
        Variable(), // input_rows
        Variable(), // input_columns
        Variable() // permute_output_dim_0_1
    };
  }
};

class BatchIndexSelectDim0TensorCPUOp
    : public torch::autograd::Function<BatchIndexSelectDim0TensorCPUOp> {
 public:
  static torch::autograd::variable_list forward_impl(
      const Tensor& inputs,
      const Tensor& indices,
      const Tensor& input_num_indices,
      const Tensor& input_rows,
      const Tensor& input_columns,
      const bool permute_output_dim_0_1) {
    const int64_t num_inputs = input_num_indices.size(0);
    TORCH_CHECK(
        num_inputs == input_rows.size(0),
        "[batch_index_select_dim0] input_rows must have the same length as "
        "input_num_indices.");
    TORCH_CHECK(
        num_inputs == input_columns.size(0),
        "[batch_index_select_dim0] input_columns must have the same length as "
        "input_num_indices.");
    TORCH_CHECK(
        reinterpret_cast<uint64_t>(inputs.data_ptr()) % 16 == 0,
        "Currently batch_index_select only supports 16-byte align input tensors");

    auto saved_data_tensor = tensor_from_vec({inputs.numel()});

    // Early exit
    if (inputs.numel() == 0) {
      return {at::empty({0}, inputs.options()), saved_data_tensor};
    }

    TORCH_CHECK(
        torch::all(torch::gt(input_columns, 0)).item<bool>(),
        "[batch_index_select_dim0] All input_columns must be the same.");
    TORCH_CHECK(
        torch::all(torch::gt(input_rows, 0)).item<bool>(),
        "[batch_index_select_dim0] All input_rows must be the same.");

    if (permute_output_dim_0_1) {
      // All output rows must be the same
      const auto item0 = input_num_indices[0].item<int64_t>();
      TORCH_CHECK(item0 > 0);
      TORCH_CHECK(
          torch::all(torch::eq(input_num_indices, item0)).item<bool>(),
          "[batch_index_select_dim0] All input_num_indices must be the same if "
          "permute_output_dim_0_1 is true.");
    } else {
      TORCH_CHECK(
          torch::all(torch::gt(input_num_indices, 0)).item<bool>(),
          "[batch_index_select_dim0] All input_num_indices must be greater than zero.");
    }

    const auto input_numels = at::mul(input_rows, input_columns);

    // Split tensors into vectors
    const auto inputs_ =
        at::split_with_sizes(inputs, vecref_from_tensor(input_numels), 0);
    const auto indices_ =
        at::split_with_sizes(indices, vecref_from_tensor(input_num_indices), 0);

    const auto input_rows_vec = vecref_from_tensor(input_rows);
    const auto input_columns_vec = vecref_from_tensor(input_columns);

    std::vector<Tensor> outputs;
    outputs.reserve(num_inputs);
    for (auto i = 0; i < num_inputs; i++) {
      const auto input =
          inputs_[i].view({input_rows_vec[i], input_columns_vec[i]});
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

    return {
        at::concat(outputs, permute_output_dim_0_1 ? 1 : 0).flatten(),
        saved_data_tensor};
  }

  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& inputs,
      const Tensor& indices,
      const Tensor& input_num_indices,
      const Tensor& input_rows,
      const Tensor& input_columns,
      const bool permute_output_dim_0_1) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto forward_op_impl =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batch_index_select_dim0_tensor_forward_cpu_impl", "")
            .typed<decltype(forward_impl)>();

    auto res = forward_op_impl.call(
        inputs,
        indices,
        input_num_indices,
        input_rows,
        input_columns,
        permute_output_dim_0_1);
    ctx->saved_data["permute_output_dim_0_1"] = permute_output_dim_0_1;
    ctx->save_for_backward(std::vector<Tensor>{
        indices, input_num_indices, input_rows, input_columns, res[1]});
    res.resize(1);
    return res;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    using torch::autograd::Variable;
    TORCH_CHECK_EQ(grad_outputs.size(), 1);

    const auto grad_output = grad_outputs[0];
    const auto permute_output_dim_0_1 =
        ctx->saved_data["permute_output_dim_0_1"].toBool();
    const auto saved = ctx->get_saved_variables();

    auto savedItr = std::begin(saved);

    auto indices = *savedItr++;
    auto input_num_indices = *savedItr++;
    auto input_rows = *savedItr++;
    auto input_columns = *savedItr++;
    auto saved_tensor = *savedItr++;

    static auto backward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batch_index_select_dim0_backward_cpu_impl", "")
            .typed<decltype(BatchIndexSelectDim0CPUOp::backward_impl)>();
    auto ret = backward_op.call(
        grad_output,
        indices,
        input_num_indices,
        input_num_indices,
        input_rows,
        input_columns,
        permute_output_dim_0_1,
        saved_tensor);
    return {
        ret,
        Variable(), // indices
        Variable(), // input_num_indices
        Variable(), // input_rows
        Variable(), // input_columns
        Variable() // permute_output_dim_0_1
    };
  }
};

Tensor batch_index_select_dim0_cpu_autograd(
    Tensor inputs,
    Tensor indices,
    const c10::SymIntArrayRef input_num_indices,
    const c10::SymIntArrayRef input_rows,
    const c10::SymIntArrayRef input_columns,
    // Permute dim 0 and 1 of the output tensor
    const bool permute_output_dim_0_1) {
  return BatchIndexSelectDim0CPUOp::apply(
      inputs,
      indices,
      input_num_indices,
      input_rows,
      input_columns,
      permute_output_dim_0_1)[0];
}

Tensor batch_index_select_dim0_tensor_cpu_autograd(
    const Tensor& inputs,
    const Tensor& indices,
    const Tensor& input_num_indices,
    const Tensor& input_rows,
    const Tensor& input_columns,
    // Permute dim 0 and 1 of the output tensor
    const bool permute_output_dim_0_1) {
  return BatchIndexSelectDim0TensorCPUOp::apply(
      inputs,
      indices,
      input_num_indices,
      input_rows,
      input_columns,
      permute_output_dim_0_1)[0];
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
  DISPATCH_TO_CPU(
      "batch_index_select_dim0", batch_index_select_dim0_cpu_autograd);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CPU(
      "batch_index_select_dim0_forward_cpu_impl",
      BatchIndexSelectDim0CPUOp::forward_impl);
  DISPATCH_TO_CPU(
      "batch_index_select_dim0_tensor_forward_cpu_impl",
      BatchIndexSelectDim0TensorCPUOp::forward_impl);

  DISPATCH_TO_CPU(
      "batch_index_select_dim0_backward_cpu_impl",
      BatchIndexSelectDim0CPUOp::backward_impl);

  DISPATCH_TO_AUTOGRAD_CPU(
      "batch_index_select_dim0", batch_index_select_dim0_cpu_autograd);

  DISPATCH_TO_AUTOGRAD_CPU(
      "batch_index_select_dim0_tensor",
      batch_index_select_dim0_tensor_cpu_autograd);
}
