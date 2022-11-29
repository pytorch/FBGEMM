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

} // namespace

///@ingroup jagged-tensor-ops-cpu
Tensor jagged_to_padded_dense_autograd(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const std::vector<int64_t>& max_lengths,
    const double padding_value) {
  return JaggedToPaddedDenseOp::apply(
      values, offsets, max_lengths, padding_value)[0];
}

///@ingroup jagged-tensor-ops-cpu
/// Output = x + y where x is jagged, y and output are dense
Tensor jagged_dense_elementwise_add_autograd(
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
jagged_dense_dense_elementwise_add_jagged_output_autograd(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0,
    const Tensor& y_1) {
  auto sum_values = JaggedDenseDenseAddJaggedOutputOp::apply(
      x_values, x_offsets, y_0, y_1)[0];

  return {sum_values, x_offsets};
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, Autograd, m) {
  m.impl(
      "jagged_to_padded_dense",
      TORCH_FN(fbgemm_gpu::jagged_to_padded_dense_autograd));
  m.impl(
      "jagged_2d_to_dense", TORCH_FN(fbgemm_gpu::jagged_2d_to_dense_autograd));
  m.impl(
      "jagged_1d_to_dense", TORCH_FN(fbgemm_gpu::jagged_1d_to_dense_autograd));
  m.impl(
      "jagged_dense_dense_elementwise_add_jagged_output",
      TORCH_FN(fbgemm_gpu::
                   jagged_dense_dense_elementwise_add_jagged_output_autograd));
}
