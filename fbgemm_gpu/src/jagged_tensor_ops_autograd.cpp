/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm_gpu {

///@defgroup jagged-tensor-ops-cpu Jagged Tensor Operators
/// The following are Jagged Tensor CPU Operators

using Tensor = at::Tensor;

namespace {

class JaggedToPaddedDenseOp
    : public torch::autograd::Function<JaggedToPaddedDenseOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const std::vector<Tensor>& offsets,
      const std::vector<int64_t>& max_lengths,
      const double padding_value) {
    ctx->save_for_backward(offsets);
    ctx->saved_data["total_L"] = values.size(0);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_forward", "")
            .typed<at::Tensor(
                const Tensor& values,
                const std::vector<Tensor>& offsets,
                const std::vector<int64_t>& max_lengths,
                const double padding_value)>();
    Tensor padded_values = op.call(values, offsets, max_lengths, padding_value);

    return {padded_values};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    int32_t total_L = ctx->saved_data["total_L"].toInt();
    TORCH_CHECK(grad_outputs.size() == 1);

    TORCH_CHECK(total_L >= 0);
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_backward", "")
            .typed<at::Tensor(
                const Tensor& grad_output,
                const std::vector<Tensor>& offsets,
                const int64_t total_L)>();
    auto grad_values = op.call(grad_outputs[0], {offsets}, total_L);

    return {
        grad_values,
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable(), // max_lengths
        torch::autograd::Variable(), // padding_value
    };
  }
};

class JaggedDenseDenseAddJaggedOutputOp
    : public torch::autograd::Function<JaggedDenseDenseAddJaggedOutputOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& x_values,
      const std::vector<Tensor>& offsets,
      const Tensor& dense_0,
      const Tensor& dense_1) {
    ctx->save_for_backward(offsets);
    ctx->saved_data["dense_shape"] = dense_0.sizes();

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::jagged_dense_dense_elementwise_add_jagged_output_forward",
                "")
            .typed<at::Tensor(
                const at::Tensor& x_values,
                const std::vector<at::Tensor>& x_offsets,
                const at::Tensor& y_0,
                const at::Tensor& y_1)>();
    Tensor output = op.call(x_values, offsets, dense_0, dense_1);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    auto dense_shape = ctx->saved_data["dense_shape"].toIntVector();
    TORCH_CHECK(grad_outputs.size() == 1);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_forward", "")
            .typed<at::Tensor(
                const Tensor& values,
                const std::vector<Tensor>& offsets,
                const std::vector<int64_t>& max_lengths,
                const double padding_value)>();
    Tensor dense_values_grad_0 = op.call(
        grad_outputs[0],
        offsets,
        std::vector<int64_t>(dense_shape.begin() + 1, dense_shape.end() - 1),
        /*padding_value=*/0);
    Tensor dense_values_grad_1 = dense_values_grad_0;

    return {
        grad_outputs[0],
        torch::autograd::Variable(), // offsets
        dense_values_grad_0,
        dense_values_grad_1};
  }
};

class JaggedDenseMulOp : public torch::autograd::Function<JaggedDenseMulOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& x_values,
      const std::vector<Tensor>& x_offsets,
      const Tensor& y) {
    std::vector<Tensor> tensors_to_save;
    tensors_to_save.push_back(x_values);
    tensors_to_save.insert(
        tensors_to_save.end(), x_offsets.begin(), x_offsets.end());
    tensors_to_save.push_back(y);
    ctx->save_for_backward(tensors_to_save);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(
                             "fbgemm::jagged_dense_elementwise_mul_forward", "")
                         .typed<at::Tensor(
                             const Tensor& x_values,
                             const std::vector<Tensor>& x_offsets,
                             const Tensor& y)>();
    Tensor output = op.call(x_values, x_offsets, y);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const Tensor x_values = ctx->get_saved_variables().front();
    std::vector<Tensor> x_offsets;
    for (size_t i = 1; i < ctx->get_saved_variables().size() - 1; ++i) {
      x_offsets.push_back(ctx->get_saved_variables()[i]);
    }
    Tensor y = ctx->get_saved_variables().back();
    TORCH_CHECK(grad_outputs.size() == 1);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::jagged_dense_elementwise_mul_backward", "")
            .typed<std::tuple<Tensor, Tensor>(
                const Tensor& grad_output,
                const std::vector<Tensor>& x_offsets,
                const Tensor& y,
                const Tensor& x_values)>();
    auto outputs = op.call(grad_outputs[0], x_offsets, y, x_values);

    return {
        std::get<0>(outputs),
        torch::autograd::Variable(),
        std::get<1>(outputs)};
  }
};

// batched dense vector x jagged 2D tensor multiplication
// dense vector [B H, N]
// jagged tensor [B, N, H D] where N is jagged
class BatchedDenseVecJagged2DMulOp
    : public torch::autograd::Function<BatchedDenseVecJagged2DMulOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& v,
      const Tensor& a_values,
      const Tensor& a_offsets) {
    ctx->save_for_backward({v, a_values, a_offsets});

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batched_dense_vec_jagged_2d_mul_forward", "")
            .typed<Tensor(
                const Tensor& v,
                const Tensor& a_values,
                const Tensor& a_offsets)>();
    Tensor output = op.call(v, a_values, a_offsets);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    const Tensor v = *savedItr++;
    const Tensor a_values = *savedItr++;
    const Tensor a_offsets = *savedItr++;
    TORCH_CHECK(grad_outputs.size() == 1);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batched_dense_vec_jagged_2d_mul_backward", "")
            .typed<std::tuple<Tensor, Tensor>(
                const Tensor& grad_output,
                const Tensor& v,
                const Tensor& a_values,
                const Tensor& a_offsets)>();
    auto outputs = op.call(grad_outputs[0], v, a_values, a_offsets);

    return {
        std::get<0>(outputs),
        std::get<1>(outputs),
        torch::autograd::Variable(), // a_offsets
    };
  }
};

class DenseToJaggedOp : public torch::autograd::Function<DenseToJaggedOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& dense,
      const std::vector<Tensor>& offsets,
      const c10::optional<int64_t>& total_L) {
    ctx->save_for_backward(offsets);

    // dims of dense tensor: <batch, [maxlen0, maxlen1, ...], embedding_dim>
    ctx->saved_data["dense_shape"] = dense.sizes();

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::dense_to_jagged_forward", "")
            .typed<Tensor(
                const Tensor& dense,
                const std::vector<Tensor>& offsets,
                const c10::optional<int64_t>& total_L)>();
    auto output = op.call(dense, offsets, total_L);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    auto dense_shape = ctx->saved_data["dense_shape"].toIntVector();
    TORCH_CHECK(grad_outputs.size() == 1);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_forward", "")
            .typed<Tensor(
                const Tensor& values,
                const std::vector<Tensor>& offsets,
                const std::vector<int64_t>& max_lengths,
                const double padding_value)>();
    auto dense_values_grad = op.call(
        grad_outputs[0],
        offsets,
        std::vector<int64_t>(dense_shape.begin() + 1, dense_shape.end() - 1),
        /*padding_value=*/0);

    TORCH_CHECK(dense_values_grad.sizes() == dense_shape);

    return {
        dense_values_grad,
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable() // total_L
    };
  }
};

class JaggedSoftmaxOp : public torch::autograd::Function<JaggedSoftmaxOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const Tensor& offsets,
      const int64_t max_L) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_softmax_forward", "")
            .typed<Tensor(
                const Tensor& values, const Tensor& offsets, int64_t max_L)>();

    auto output = op.call(values, offsets, max_L);

    ctx->save_for_backward({output, offsets});
    ctx->saved_data["max_L"] = max_L;

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    Tensor output = *savedItr++;
    Tensor offsets = *savedItr++;
    int64_t max_L = ctx->saved_data["max_L"].toInt();
    TORCH_CHECK(grad_outputs.size() == 1);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_softmax_backward", "")
            .typed<Tensor(
                const Tensor& grad_output,
                const Tensor& output,
                const Tensor& offsets,
                int64_t max_L)>();

    auto grad_input = op.call(grad_outputs[0], output, offsets, max_L);

    return {
        grad_input,
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable() // max_L
    };
  }
};

class JaggedJaggedBmmOp : public torch::autograd::Function<JaggedJaggedBmmOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& x_values,
      const Tensor& y_values,
      const Tensor& offsets,
      const int64_t max_L) {
    ctx->save_for_backward({x_values, y_values, offsets});
    ctx->saved_data["max_L"] = max_L;

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_jagged_bmm_forward", "")
            .typed<Tensor(
                const Tensor& x_values,
                const Tensor& y_values,
                const Tensor& offsets,
                int64_t max_L)>();

    auto output = op.call(x_values, y_values, offsets, max_L);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    Tensor x_values = *savedItr++;
    Tensor y_values = *savedItr++;
    Tensor offsets = *savedItr++;
    int64_t max_L = ctx->saved_data["max_L"].toInt();
    TORCH_CHECK(grad_outputs.size() == 1);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_dense_bmm_forward", "")
            .typed<Tensor(
                const Tensor& grad_output,
                const Tensor& offsets,
                const Tensor& y,
                int64_t max_L)>();

    auto grad_input_x =
        op.call(y_values, offsets, at::transpose(grad_outputs[0], 2, 1), max_L);
    auto grad_input_y = op.call(x_values, offsets, grad_outputs[0], max_L);

    return {
        grad_input_x,
        grad_input_y,
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable() // max_L
    };
  }
};

class JaggedDenseBmmOp : public torch::autograd::Function<JaggedDenseBmmOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& x_values,
      const Tensor& x_offsets,
      const Tensor& y,
      const int64_t max_L) {
    ctx->save_for_backward({x_values, x_offsets, y});
    ctx->saved_data["max_L"] = max_L;

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_dense_bmm_forward", "")
            .typed<Tensor(
                const Tensor& x_values,
                const Tensor& x_offsets,
                const Tensor& y,
                int64_t max_L)>();

    auto output = op.call(x_values, x_offsets, y, max_L);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    Tensor x_values = *savedItr++;
    Tensor offsets = *savedItr++;
    Tensor y = *savedItr++;
    int64_t max_L = ctx->saved_data["max_L"].toInt();
    TORCH_CHECK(grad_outputs.size() == 1);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_dense_bmm_forward", "")
            .typed<Tensor(
                const Tensor& grad_output,
                const Tensor& offsets,
                const Tensor& y,
                int64_t max_L)>();

    auto grad_input_x =
        op.call(grad_outputs[0], offsets, at::transpose(y, 2, 1), max_L);

    static auto op2 =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_jagged_bmm_forward", "")
            .typed<Tensor(
                const Tensor& grad_output,
                const Tensor& x_values,
                const Tensor& offsets,
                int64_t max_L)>();

    auto grad_input_y = op2.call(x_values, grad_outputs[0], offsets, max_L);

    return {
        grad_input_x,
        torch::autograd::Variable(), // x_offsets
        grad_input_y,
        torch::autograd::Variable() // max_L
    };
  }
};

} // namespace

///@ingroup jagged-tensor-ops-cpu
Tensor jagged_to_padded_dense(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const std::vector<int64_t>& max_lengths,
    const double padding_value) {
  return JaggedToPaddedDenseOp::apply(
      values, offsets, max_lengths, padding_value)[0];
}

///@ingroup jagged-tensor-ops-cpu
/// Output = x + y where x is jagged, y and output are dense
Tensor jagged_dense_elementwise_add(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  // Construct max_lengths from y
  std::vector<int64_t> max_lengths;
  max_lengths.reserve(x_offsets.size());
  for (int d = 1; d < y.dim() - 1; d++) {
    max_lengths.push_back(y.size(d));
  }
  TORCH_CHECK(max_lengths.size() == x_offsets.size());

  // Convert x to dense (assume padding is 0.0)
  auto xd = JaggedToPaddedDenseOp::apply(
      x_values, x_offsets, max_lengths, /* padding_value */ 0.0)[0];

  auto dense_output = xd + y;
  return dense_output;
}

// output = x + y_0 + y_1 where x is jagged, y_0 and y_1 are dense, and output
// is jagged
std::tuple<Tensor, std::vector<Tensor>>
jagged_dense_dense_elementwise_add_jagged_output(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0,
    const Tensor& y_1) {
  auto sum_values = JaggedDenseDenseAddJaggedOutputOp::apply(
      x_values, x_offsets, y_0, y_1)[0];

  return {sum_values, x_offsets};
}

///@ingroup jagged-tensor-ops-cpu
std::tuple<Tensor, std::vector<Tensor>> jagged_dense_elementwise_mul(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  // Convert to jagged
  auto prod_values = JaggedDenseMulOp::apply(x_values, x_offsets, y)[0];

  return {prod_values, x_offsets};
}

///@ingroup jagged-tensor-ops-cpu
Tensor batched_dense_vec_jagged_2d_mul(
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  return BatchedDenseVecJagged2DMulOp::apply(v, a_values, a_offsets)[0];
}

///@ingroup jagged-tensor-ops-cpu
// output = x + y where x is jagged, y is dense, and output is jagged
std::tuple<Tensor, std::vector<Tensor>> dense_to_jagged(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    const c10::optional<int64_t>& total_L) {
  return {DenseToJaggedOp::apply(dense, offsets, total_L)[0], offsets};
}

///@ingroup jagged-tensor-ops-cpu
/// Output = x + y where x is jagged, y is dense, and output is jagged
std::tuple<Tensor, std::vector<Tensor>>
jagged_dense_elementwise_add_jagged_output(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  // Convert to jagged
  auto jagged_values =
      DenseToJaggedOp::apply(y, x_offsets, c10::optional<int64_t>())[0];

  // Add jagged_values + x_values -> sum_values
  auto sum_values = x_values + jagged_values;

  return {sum_values, x_offsets};
}

///@ingroup jagged-tensor-ops-cpu
Tensor jagged_1d_to_dense(
    Tensor values,
    Tensor offsets,
    int64_t max_L,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  return jagged_to_padded_dense(values, {offsets}, {max_L}, padding_value);
}

///@ingroup jagged-tensor-ops-cpu
Tensor
jagged_2d_to_dense(Tensor values, Tensor offsets, int64_t max_sequence_length) {
  return jagged_to_padded_dense(
      values,
      {offsets},
      {max_sequence_length},
      /*padding_value=*/0);
}

std::tuple<Tensor, Tensor> jagged_softmax(
    const Tensor& values,
    const Tensor& offsets,
    const int64_t max_L) {
  return {JaggedSoftmaxOp::apply(values, offsets, max_L)[0], offsets};
}

Tensor jagged_jagged_bmm(
    const Tensor& x_values,
    const Tensor& y_values,
    const Tensor& offsets,
    const int64_t max_L) {
  return JaggedJaggedBmmOp::apply(x_values, y_values, offsets, max_L)[0];
}

std::tuple<Tensor, Tensor> jagged_dense_bmm(
    const Tensor& x_values,
    const Tensor& x_offsets,
    const Tensor& y,
    const int64_t max_L) {
  return {JaggedDenseBmmOp::apply(x_values, x_offsets, y, max_L)[0], x_offsets};
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, Autograd, m) {
  m.impl(
      "jagged_to_padded_dense", TORCH_FN(fbgemm_gpu::jagged_to_padded_dense));
  m.impl("jagged_2d_to_dense", TORCH_FN(fbgemm_gpu::jagged_2d_to_dense));
  m.impl("jagged_1d_to_dense", TORCH_FN(fbgemm_gpu::jagged_1d_to_dense));
  m.impl(
      "jagged_dense_dense_elementwise_add_jagged_output",
      TORCH_FN(fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output));
  m.impl(
      "jagged_dense_elementwise_mul",
      TORCH_FN(fbgemm_gpu::jagged_dense_elementwise_mul));
  m.impl(
      "batched_dense_vec_jagged_2d_mul",
      TORCH_FN(fbgemm_gpu::batched_dense_vec_jagged_2d_mul));
  m.impl("dense_to_jagged", TORCH_FN(fbgemm_gpu::dense_to_jagged));
  m.impl("jagged_softmax", TORCH_FN(fbgemm_gpu::jagged_softmax));
  m.impl("jagged_jagged_bmm", TORCH_FN(fbgemm_gpu::jagged_jagged_bmm));
  m.impl("jagged_dense_bmm", TORCH_FN(fbgemm_gpu::jagged_dense_bmm));
}
