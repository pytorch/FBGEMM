/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm_gpu {

///@defgroup jagged-tensor-ops-cpu Jagged Tensor Operators
/// The following are Jagged Tensor CPU Operators

using Tensor = at::Tensor;

namespace {

///@defgroup jagged-tensor-ops-cpu Jagged Tensor Operators
/// The following are Jagged Tensor CPU Operators

// Ref. http://tensor-compiler.org/kjolstad-oopsla17-tensor-compiler.pdf
template <int NUM_JAGGED_DIM, typename index_t>
inline bool walk_down_tensor_storage_tree_except_last_(
    int& offset,
    const int flattened_jagged_idx,
    const int64_t* jagged_dims,
    const std::vector<at::TensorAccessor<index_t, 1>>& x_offsets) {
  // compute coorindates
  int jagged_coords[NUM_JAGGED_DIM];
  int j_temp = flattened_jagged_idx;
#pragma unroll
  for (int d = NUM_JAGGED_DIM - 2; d >= 0; --d) {
    const int jagged_size = jagged_dims[d + 1];
    jagged_coords[d] = j_temp % jagged_size;
    j_temp /= jagged_size;
  }

  bool is_zero = false;
#pragma unroll
  for (int d = 0; d < NUM_JAGGED_DIM - 1; ++d) {
    const int begin = x_offsets[d][offset];
    const int end = x_offsets[d][offset + 1];
    if (jagged_coords[d] >= end - begin) {
      is_zero = true;
      break;
    }
    offset = begin + jagged_coords[d];
  }
  return is_zero;
}

template <typename index_t>
std::vector<at::TensorAccessor<index_t, 1>> collect_offsets_accessors(
    const std::vector<Tensor>& x_offsets,
    const int outer_dense_size,
    const int num_jagged_dim) {
  // Also check x_offsets are consistent
  int num_lengths_expected = outer_dense_size;
  std::vector<at::TensorAccessor<index_t, 1>> x_offsets_accessors;
  for (int d = 0; d < num_jagged_dim; ++d) {
    TENSOR_ON_CPU(x_offsets[d]);
    x_offsets_accessors.emplace_back(x_offsets[d].accessor<index_t, 1>());
    TORCH_CHECK(
        x_offsets[d].numel() == num_lengths_expected + 1,
        "x_offsets[",
        d,
        "].numel(), ",
        x_offsets[d].numel(),
        " != num_lengths_expected + 1, ",
        num_lengths_expected + 1);
    auto num_lengths = x_offsets_accessors[d][x_offsets[d].numel() - 1];
    num_lengths_expected = num_lengths;
  }

  return x_offsets_accessors;
}

/**
 * @tparam F element wise compute functor
 * @param padding_value instead of zero, can configure the padding
 *                      value for x_values
 *
 * See more details comments at jagged_elementwise_dense_output_kernel in
 * jagged_tensor_ops.cu
 */
template <
    int NUM_JAGGED_DIM,
    bool NO_INNER_DENSE,
    typename index_t,
    typename scalar_t,
    typename F>
void jagged_dense_elementwise_dense_output_kernel_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    F f,
    const scalar_t& padding_value) {
  TENSOR_ON_CPU(x_values);
  TENSOR_ON_CPU(y);
  TENSOR_ON_CPU(output);

  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(NUM_JAGGED_DIM),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != NUM_JAGGED_DIM, ",
      NUM_JAGGED_DIM);

  const int outer_dense_size = y.size(0);
  TORCH_CHECK(
      outer_dense_size == x_offsets[0].numel() - 1,
      "outer_dense_size, ",
      outer_dense_size,
      " != x_offsets[0].numel() - 1, ",
      x_offsets[0].numel() - 1);
  TORCH_CHECK(
      !NO_INNER_DENSE || y.size(-1) == 1, "y.size(-1), ", y.size(-1), " != 1");
  const int inner_dense_size = NO_INNER_DENSE ? 1 : y.size(-1);
  TORCH_CHECK(
      inner_dense_size == x_values.size(-1),
      "inner_dense_size, ",
      inner_dense_size,
      " != x_values.size(-1), ",
      x_values.size(-1));

  if (y.numel() == 0) {
    return;
  }

  const int jagged_folded_size =
      y.numel() / (outer_dense_size * inner_dense_size);
  const int jagged_innermost_size = y.size(-2);

  // Canonicalize y and output to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  Tensor output_reshaped = output.view(y_reshaped.sizes());

  const std::vector<at::TensorAccessor<index_t, 1>> x_offsets_accessors =
      collect_offsets_accessors<index_t>(
          x_offsets, outer_dense_size, NUM_JAGGED_DIM);

  const at::TensorAccessor<scalar_t, 2> x_accessor =
      x_values.accessor<scalar_t, 2>();
  const at::TensorAccessor<scalar_t, 3> y_accessor =
      y_reshaped.accessor<scalar_t, 3>();
  at::TensorAccessor<scalar_t, 3> output_accessor =
      output_reshaped.accessor<scalar_t, 3>();

  for (int oidx = 0; oidx < outer_dense_size; ++oidx) {
    for (int joidx = 0; joidx < jagged_folded_size / jagged_innermost_size;
         ++joidx) {
      int offset_base = oidx;
      const bool is_zero =
          walk_down_tensor_storage_tree_except_last_<NUM_JAGGED_DIM>(
              offset_base, joidx, y.sizes().data(), x_offsets_accessors);

      // As a perf optimization, a separate loop level for the inner-most
      // jagged dimension.
      int jiidx = 0;
      if (!is_zero) {
        const int begin = x_offsets_accessors[NUM_JAGGED_DIM - 1][offset_base];
        const int end =
            x_offsets_accessors[NUM_JAGGED_DIM - 1][offset_base + 1];
        for (; jiidx < std::min(end - begin, jagged_innermost_size); ++jiidx) {
          int jidx = joidx * jagged_innermost_size + jiidx;
          if (NO_INNER_DENSE) {
            output_accessor[oidx][jidx][0] =
                f(x_accessor[begin + jiidx][0], y_accessor[oidx][jidx][0]);
          } else {
            for (int iidx = 0; iidx < inner_dense_size; ++iidx) {
              output_accessor[oidx][jidx][iidx] =
                  f(x_accessor[begin + jiidx][iidx],
                    y_accessor[oidx][jidx][iidx]);
            }
          }
        }
      }
      for (; jiidx < jagged_innermost_size; ++jiidx) {
        int jidx = joidx * jagged_innermost_size + jiidx;
        if (NO_INNER_DENSE) {
          output_accessor[oidx][jidx][0] =
              f(padding_value, y_accessor[oidx][jidx][0]);
        } else {
          for (int iidx = 0; iidx < inner_dense_size; ++iidx) {
            output_accessor[oidx][jidx][iidx] =
                f(padding_value, y_accessor[oidx][jidx][iidx]);
          }
        }
      }
    } // for each joidx
  } // for each oidx
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    F f,
    const scalar_t& padding_value = static_cast<scalar_t>(0)) {
#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                      \
  if (y.size(-1) == 1) {                                            \
    jagged_dense_elementwise_dense_output_kernel_<                  \
        NUM_JAGGED_DIM,                                             \
        true,                                                       \
        index_t>(x_values, x_offsets, y, output, f, padding_value); \
  } else {                                                          \
    jagged_dense_elementwise_dense_output_kernel_<                  \
        NUM_JAGGED_DIM,                                             \
        false,                                                      \
        index_t>(x_values, x_offsets, y, output, f, padding_value); \
  }

  const int num_jagged_dim = y.dim() - 2;
  JAGGED_TENSOR_DISPATCH_DIMS();

#undef INVOKE_KERNEL_WITH_DIM
}

template <typename scalar_t, typename F>
Tensor jagged_dense_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    F f,
    const scalar_t& padding_value = static_cast<scalar_t>(0)) {
  Tensor output = at::empty_like(y);
  jagged_dense_elementwise_dense_output_(
      x_values, x_offsets, y, output, f, padding_value);
  return output;
}

template <
    int NUM_JAGGED_DIM,
    bool NO_INNER_DENSE,
    typename index_t,
    typename scalar_t,
    typename F>
void jagged_dense_elementwise_jagged_output_kernel_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CPU(x_values);
  TENSOR_ON_CPU(y);
  TENSOR_ON_CPU(output_values);

  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(NUM_JAGGED_DIM),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != NUM_JAGGED_DIM, ",
      NUM_JAGGED_DIM);

  const int outer_dense_size = y.size(0);
  TORCH_CHECK(
      outer_dense_size == x_offsets[0].numel() - 1,
      "outer_dense_size, ",
      outer_dense_size,
      " != x_offsets[0].numel() - 1, ",
      x_offsets[0].numel() - 1);
  TORCH_CHECK(
      !NO_INNER_DENSE || y.size(-1) == 1, "y.size(-1), ", y.size(-1), " != 1");
  const int inner_dense_size = NO_INNER_DENSE ? 1 : y.size(-1);
  TORCH_CHECK(
      inner_dense_size == x_values.size(-1),
      "inner_dense_size, ",
      inner_dense_size,
      " != x_values.size(-1), ",
      x_values.size(-1));

  if (y.numel() == 0) {
    return;
  }

  const int jagged_folded_size =
      y.numel() / (outer_dense_size * inner_dense_size);
  const int jagged_innermost_size = y.size(-2);

  // Canonicalize y to 3D, collapsing jagged dimensions.
  Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});

  std::vector<at::TensorAccessor<index_t, 1>> x_offsets_accessors =
      collect_offsets_accessors<index_t>(
          x_offsets, outer_dense_size, NUM_JAGGED_DIM);

  const at::TensorAccessor<scalar_t, 2> x_accessor =
      x_values.accessor<scalar_t, 2>();
  const at::TensorAccessor<scalar_t, 3> y_accessor =
      y_reshaped.accessor<scalar_t, 3>();
  at::TensorAccessor<scalar_t, 2> output_accessor =
      output_values.accessor<scalar_t, 2>();

  for (int oidx = 0; oidx < outer_dense_size; ++oidx) {
    for (int joidx = 0; joidx < jagged_folded_size / jagged_innermost_size;
         ++joidx) {
      int offset_base = oidx;
      bool is_zero = walk_down_tensor_storage_tree_except_last_<NUM_JAGGED_DIM>(
          offset_base, joidx, y.sizes().data(), x_offsets_accessors);

      // As a perf optimization, a separate loop level for the inner-most
      // jagged dimension.
      if (!is_zero) {
        const int begin = x_offsets_accessors[NUM_JAGGED_DIM - 1][offset_base];
        const int end =
            x_offsets_accessors[NUM_JAGGED_DIM - 1][offset_base + 1];
        for (int jiidx = 0;
             jiidx < std::min(end - begin, jagged_innermost_size);
             ++jiidx) {
          int jidx = joidx * jagged_innermost_size + jiidx;
          if (NO_INNER_DENSE) {
            output_accessor[begin + jiidx][0] =
                f(x_accessor[begin + jiidx][0], y_accessor[oidx][jidx][0]);
          } else {
            for (int iidx = 0; iidx < inner_dense_size; ++iidx) {
              output_accessor[begin + jiidx][iidx] =
                  f(x_accessor[begin + jiidx][iidx],
                    y_accessor[oidx][jidx][iidx]);
            }
          }
        }
      }
    } // for each joidx
  } // for each oidx
}

/// @ingroup jagged-tensor-ops-cpu
template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)               \
  if (y.size(-1) == 1) {                                     \
    jagged_dense_elementwise_jagged_output_kernel_<          \
        NUM_JAGGED_DIM,                                      \
        true,                                                \
        index_t,                                             \
        scalar_t>(x_values, x_offsets, y, output_values, f); \
  } else {                                                   \
    jagged_dense_elementwise_jagged_output_kernel_<          \
        NUM_JAGGED_DIM,                                      \
        false,                                               \
        index_t,                                             \
        scalar_t>(x_values, x_offsets, y, output_values, f); \
  }

  int num_jagged_dim = y.dim() - 2;
  JAGGED_TENSOR_DISPATCH_DIMS();

#undef INVOKE_KERNEL_WITH_DIM
}

///@ingroup jagged-tensor-ops-cpu
class JaggedToPaddedDenseCPUOp
    : public torch::autograd::Function<JaggedToPaddedDenseCPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const std::vector<Tensor>& offsets,
      const std::vector<int64_t>& max_lengths,
      const double padding_value) {
    ctx->save_for_backward(offsets);
    ctx->saved_data["total_L"] = values.size(0);

    const size_t num_jagged_dim = offsets.size();
    TORCH_CHECK(
        max_lengths.size() == num_jagged_dim,
        "max_lengths.size(), ",
        max_lengths.size(),
        " != num_jagged_dim, ",
        num_jagged_dim);

    const Tensor values_canonicalized = values.view(
        {values.size(0),
         std::accumulate(
             values.sizes().begin() + 1,
             values.sizes().end(),
             1,
             std::multiplies<size_t>())});
    at::DimVector padded_values_shape({offsets[0].size(0) - 1});
    padded_values_shape.insert(
        padded_values_shape.end(), max_lengths.begin(), max_lengths.end());
    if (values.dim() > 1) {
      padded_values_shape.push_back(values.size(-1));
    }
    Tensor padded_values = at::empty(padded_values_shape, values.options());
    if (values.numel() == 0) {
      // To avoid an error due to values_canonicalized.data_ptr is nullptr.
      padded_values.fill_(padding_value);
      return {padded_values};
    }
    Tensor padded_values_view =
        values.dim() == 1 ? padded_values.unsqueeze(-1) : padded_values;

    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half,
        values.scalar_type(),
        "jagged_to_padded_dense",
        [&] {
          jagged_dense_elementwise_dense_output_<scalar_t>(
              values_canonicalized,
              offsets,
              padded_values_view, // dummy not used in the lambda function
              padded_values_view,
              [](scalar_t x, scalar_t /*unused*/) -> scalar_t { return x; },
              static_cast<scalar_t>(padding_value));
        });

    return {padded_values};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    int32_t total_L = ctx->saved_data["total_L"].toInt();
    TORCH_CHECK(grad_outputs.size() == 1);

    TORCH_CHECK(total_L >= 0);
    auto grad_padded_values = grad_outputs[0];

    int32_t D = grad_padded_values.size(-1);
    // Initialize with zeros so output will be zero for the portion truncated
    // in forward.
    auto grad_values = at::zeros({total_L, D}, grad_padded_values.options());

    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half,
        grad_padded_values.scalar_type(),
        "jagged_2d_to_dense_backward_kernel",
        [&] {
          jagged_dense_elementwise_jagged_output_<scalar_t>(
              grad_values, // dummy not used in the lambda function
              {offsets},
              grad_padded_values,
              grad_values,
              [](scalar_t /*unused*/, scalar_t y) -> scalar_t { return y; });
        });

    return {
        grad_values,
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable(), // max_lengths
        torch::autograd::Variable(), // padding_value
    };
  }
};

///@ingroup jagged-tensor-ops-cpu
Tensor jagged_to_padded_dense(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const std::vector<int64_t>& max_lengths,
    const double padding_value = 0) {
  return JaggedToPaddedDenseCPUOp::apply(
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
  auto xd = JaggedToPaddedDenseCPUOp::apply(
      x_values, x_offsets, max_lengths, /* padding_value */ 0.0)[0];

  auto dense_output = xd + y;
  return dense_output;
}

///@ingroup jagged-tensor-ops-cpu
// TODO: Add option to pass in total_L
class DenseToJaggedCPUOp
    : public torch::autograd::Function<DenseToJaggedCPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& dense,
      const std::vector<Tensor>& offsets,
      const c10::optional<int64_t>& total_L) {
    ctx->save_for_backward(offsets);

    // dims of dense tensor: <batch, [maxlen0, maxlen1, ...], embedding_dim>
    ctx->saved_data["dense_shape"] = dense.sizes();

    // D is the embedding dimension
    auto D = dense.size(-1);

    // If total_L is not given then compute it
    int64_t total_L_computed;
    if (total_L.has_value()) {
      total_L_computed = total_L.value();
    } else {
      total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
    }
    auto values = at::empty({total_L_computed, D}, dense.options());
    auto output = at::zeros({total_L_computed, D}, dense.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::Long,
        values.scalar_type(),
        "jagged_scalars",
        [&] {
          jagged_dense_elementwise_jagged_output_<scalar_t>(
              values,
              offsets,
              dense,
              output,
              [](scalar_t /*unused*/, scalar_t y) -> scalar_t { return y; });
        });

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    auto dense_shape = ctx->saved_data["dense_shape"].toIntVector();
    TORCH_CHECK(grad_outputs.size() == 1);

    Tensor dense_values_grad = jagged_to_padded_dense(
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

///@ingroup jagged-tensor-ops-cpu
// output = x + y where x is jagged, y is dense, and output is jagged
std::tuple<Tensor, std::vector<Tensor>> dense_to_jagged_cpu(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    const c10::optional<int64_t>& total_L) {
  return {DenseToJaggedCPUOp::apply(dense, offsets, total_L)[0], offsets};
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
      DenseToJaggedCPUOp::apply(y, x_offsets, c10::optional<int64_t>())[0];

  // Add jagged_values + x_values -> sum_values
  auto sum_values = x_values + jagged_values;

  return {sum_values, x_offsets};
}

// output = x + y where x is jagged, y is dense, and output is jagged
std::tuple<Tensor, std::vector<Tensor>>
jagged_dense_dense_elementwise_add_jagged_output(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0,
    const Tensor& y_1) {
  // Convert to jagged
  auto jagged_values_0 =
      DenseToJaggedCPUOp::apply(y_0, x_offsets, c10::optional<int64_t>())[0];
  auto jagged_values_1 =
      DenseToJaggedCPUOp::apply(y_1, x_offsets, c10::optional<int64_t>())[0];

  // Add jagged_values + x_values -> sum_values
  auto sum_values = x_values + jagged_values_0 + jagged_values_1;

  return {sum_values, x_offsets};
}

template <
    int NUM_JAGGED_DIM,
    bool NO_INNER_DENSE,
    typename index_t,
    typename scalar_t,
    typename F>
void jagged_jagged_elementwise_dense_output_kernel_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_values,
    const Tensor& output,
    F f,
    const scalar_t& padding_value) {
  TENSOR_ON_CPU(x_values);
  TENSOR_ON_CPU(y_values);
  TENSOR_ON_CPU(output);

  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(NUM_JAGGED_DIM),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != NUM_JAGGED_DIM, ",
      NUM_JAGGED_DIM);

  const int outer_dense_size = output.size(0);
  TORCH_CHECK(
      outer_dense_size == x_offsets[0].numel() - 1,
      "outer_dense_size, ",
      outer_dense_size,
      " != x_offsets[0].numel() - 1, ",
      x_offsets[0].numel() - 1);
  TORCH_CHECK(!NO_INNER_DENSE || output.size(-1) == 1);
  const int inner_dense_size = NO_INNER_DENSE ? 1 : output.size(-1);
  TORCH_CHECK(
      inner_dense_size == x_values.size(-1),
      "inner_dense_size, ",
      inner_dense_size,
      " != x_values.size(-1), ",
      x_values.size(-1));

  if (output.numel() == 0) {
    return;
  }

  const int jagged_folded_size =
      output.numel() / (outer_dense_size * inner_dense_size);
  const int jagged_innermost_size = output.size(-2);

  // Canonicalize output to 3D, collapsing jagged dimensions.
  Tensor output_reshaped = output.view({output.size(0), -1, output.size(-1)});

  std::vector<at::TensorAccessor<index_t, 1>> x_offsets_accessors =
      collect_offsets_accessors<index_t>(
          x_offsets, outer_dense_size, NUM_JAGGED_DIM);

  const at::TensorAccessor<scalar_t, 2> x_accessor =
      x_values.accessor<scalar_t, 2>();
  const at::TensorAccessor<scalar_t, 2> y_accessor =
      y_values.accessor<scalar_t, 2>();
  at::TensorAccessor<scalar_t, 3> output_accessor =
      output_reshaped.accessor<scalar_t, 3>();

  for (int oidx = 0; oidx < outer_dense_size; ++oidx) {
    for (int joidx = 0; joidx < jagged_folded_size / jagged_innermost_size;
         ++joidx) {
      int offset_base = oidx;
      const bool is_zero =
          walk_down_tensor_storage_tree_except_last_<NUM_JAGGED_DIM>(
              offset_base, joidx, output.sizes().data(), x_offsets_accessors);

      // As a perf optimization, a separate loop level for the inner-most
      // jagged dimension.
      int jiidx = 0;
      if (!is_zero) {
        const int begin = x_offsets_accessors[NUM_JAGGED_DIM - 1][offset_base];
        const int end =
            x_offsets_accessors[NUM_JAGGED_DIM - 1][offset_base + 1];
        for (; jiidx < std::min(end - begin, jagged_innermost_size); ++jiidx) {
          int jidx = joidx * jagged_innermost_size + jiidx;
          if (NO_INNER_DENSE) {
            output_accessor[oidx][jidx][0] =
                f(x_accessor[begin + jiidx][0], y_accessor[begin + jiidx][0]);
          } else {
            for (int iidx = 0; iidx < inner_dense_size; ++iidx) {
              output_accessor[oidx][jidx][iidx] =
                  f(x_accessor[begin + jiidx][iidx],
                    y_accessor[begin + jiidx][iidx]);
            }
          }
        }
      }
      for (; jiidx < jagged_innermost_size; ++jiidx) {
        int jidx = joidx * jagged_innermost_size + jiidx;
        if (NO_INNER_DENSE) {
          output_accessor[oidx][jidx][0] = padding_value;
        } else {
          for (int iidx = 0; iidx < inner_dense_size; ++iidx) {
            output_accessor[oidx][jidx][iidx] = padding_value;
          }
        }
      }
    } // for each joidx
  } // for each oidx
}

template <typename scalar_t, typename F>
void jagged_jagged_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_values,
    const Tensor& output,
    F f,
    const scalar_t& padding_value = static_cast<scalar_t>(0)) {
#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                             \
  if (output.size(-1) == 1) {                                              \
    jagged_jagged_elementwise_dense_output_kernel_<                        \
        NUM_JAGGED_DIM,                                                    \
        true,                                                              \
        index_t>(x_values, x_offsets, y_values, output, f, padding_value); \
  } else {                                                                 \
    jagged_jagged_elementwise_dense_output_kernel_<                        \
        NUM_JAGGED_DIM,                                                    \
        false,                                                             \
        index_t>(x_values, x_offsets, y_values, output, f, padding_value); \
  }

  const int num_jagged_dim = output.dim() - 2;
  JAGGED_TENSOR_DISPATCH_DIMS();

#undef INVOKE_KERNEL_WITH_DIM
}

///@addtogroup jagged-tensor-ops-cpu
std::tuple<Tensor, std::vector<Tensor>> jagged_dense_elementwise_mul(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  // Convert to jagged
  auto jagged_values =
      DenseToJaggedCPUOp::apply(y, x_offsets, c10::optional<int64_t>())[0];

  // Multiply x_values * jagged_values -> prod_values
  auto prod_values = x_values * jagged_values;

  return {prod_values, x_offsets};
}

template <typename index_t, typename scalar_t>
void dense_vec_jagged_2d_bmm(
    const at::TensorAccessor<scalar_t, 2>& v,
    const at::TensorAccessor<scalar_t, 2>& a_values,
    const at::TensorAccessor<index_t, 1>& a_offsets,
    at::TensorAccessor<scalar_t, 2> output) {
  const int B = a_offsets.size(0) - 1;
  const int H = v.size(0) / B;
  const int max_L = v.size(1);
  const int D = output.size(1);

  for (int b = 0; b < B; ++b) {
    const int row_start = a_offsets[b];
    const int row_end = a_offsets[b + 1];
    const int length = std::min(row_end - row_start, max_L);
    if (length == 0) {
      for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
          output[b * H + h][d] = 0;
        }
      }
    } else {
      for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
          // use is_cuda=true because acc_type<float, false> = double is too
          // conservative
          at::acc_type<scalar_t, true> acc =
              v[b * H + h][0] * a_values[row_start][h * D + d];
          for (int l = 1; l < length; ++l) {
            acc += v[b * H + h][l] * a_values[row_start + l][h * D + d];
          }
          output[b * H + h][d] = acc;
        }
      }
    } // length > 0
  } // for each b
}

template <typename index_t, typename scalar_t>
void dense_vec_jagged_2d_transposed_bmm(
    const at::TensorAccessor<scalar_t, 2>& v,
    const at::TensorAccessor<scalar_t, 2>& a_values,
    const at::TensorAccessor<index_t, 1>& a_offsets,
    at::TensorAccessor<scalar_t, 2> output) {
  const int B = a_offsets.size(0) - 1;
  const int H = v.size(0) / B;
  const int max_L = output.size(1);
  const int D = v.size(1);

  for (int b = 0; b < B; ++b) {
    const int row_start = a_offsets[b];
    const int row_end = a_offsets[b + 1];
    const int length = std::min(row_end - row_start, max_L);

    if (D == 0) {
      for (int h = 0; h < H; ++h) {
        for (int l = 0; l < max_L; ++l) {
          output[b * H + h][l] = 0;
        }
      }
    } else {
      for (int h = 0; h < H; ++h) {
        int l;
        for (l = 0; l < length; ++l) {
          at::acc_type<scalar_t, true> acc =
              v[b * H + h][0] * a_values[row_start + l][h * D];
          for (int d = 1; d < D; ++d) {
            acc += v[b * H + h][d] * a_values[row_start + l][h * D + d];
          }
          output[b * H + h][l] = acc;
        }
        for (; l < max_L; ++l) {
          output[b * H + h][l] = 0;
        }
      }
    } // D > 0
  } // for each b
}

template <typename index_t, typename scalar_t>
void outer_prod_jagged_2d_output(
    const at::TensorAccessor<scalar_t, 2>& x,
    const at::TensorAccessor<scalar_t, 2>& y,
    const at::TensorAccessor<index_t, 1>& offsets,
    at::TensorAccessor<scalar_t, 2> output_values) {
  const int B = offsets.size(0) - 1;
  const int H = x.size(0) / B;
  const int max_L = x.size(1);
  const int D = y.size(1);

  for (int b = 0; b < B; ++b) {
    const int row_start = offsets[b];
    const int row_end = offsets[b + 1];
    const int length = row_end - row_start;
    for (int h = 0; h < H; ++h) {
      for (int l = 0; l < std::min(length, max_L); ++l) {
        for (int d = 0; d < D; ++d) {
          output_values[row_start + l][h * D + d] =
              x[b * H + h][l] * y[b * H + h][d];
        }
      }
    }
  }
}

// batched dense vector x jagged 2D tensor multiplication
// dense vector [B H, N]
// jagged tensor [B, N, H D] where N is jagged
class BatchedDenseVecJagged2DMulCPUOp
    : public torch::autograd::Function<BatchedDenseVecJagged2DMulCPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& v,
      const Tensor& a_values,
      const Tensor& a_offsets) {
    ctx->save_for_backward({v, a_values, a_offsets});

    TENSOR_ON_CPU(v);
    TENSOR_ON_CPU(a_values);
    TENSOR_ON_CPU(a_offsets);

    const int B = a_offsets.numel() - 1;
    TORCH_CHECK(
        B == 0 || v.size(0) % B == 0,
        "B, ",
        B,
        " doesn't divide v.size(0), ",
        v.size(0));
    const int H = B == 0 ? 1 : v.size(0) / B;
    const int D = a_values.size(-1) / H;
    auto output = at::empty({B * H, D}, v.options());

    if (B > 0 && D > 0) {
      AT_DISPATCH_INDEX_TYPES(
          a_offsets.scalar_type(), "dense_vec_jagged_2d_bmm_kernel_1", [&] {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                a_values.scalar_type(),
                "dense_vec_jagged_2d_bmm_kernel_2",
                [&] {
                  dense_vec_jagged_2d_bmm<index_t, scalar_t>(
                      v.accessor<scalar_t, 2>(),
                      a_values.accessor<scalar_t, 2>(),
                      a_offsets.accessor<index_t, 1>(),
                      output.accessor<scalar_t, 2>());
                });
          });
    }

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

    TENSOR_ON_CPU(grad_outputs[0]);

    Tensor a_values_grad = at::zeros_like(a_values);
    Tensor v_grad = at::empty_like(v);

    const int B = a_offsets.numel() - 1;
    const int D = grad_outputs[0].size(-1);

    if (B > 0 && D > 0) {
      AT_DISPATCH_INDEX_TYPES(
          a_offsets.scalar_type(),
          "dense_vec_jagged_2d_bmm_baackward_kernel_1",
          [&] {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                grad_outputs[0].scalar_type(),
                "dense_vec_jagged_2d_bmm_baackward_kernel_2",
                [&] {
                  dense_vec_jagged_2d_transposed_bmm<index_t, scalar_t>(
                      grad_outputs[0].accessor<scalar_t, 2>(),
                      a_values.accessor<scalar_t, 2>(),
                      a_offsets.accessor<index_t, 1>(),
                      v_grad.accessor<scalar_t, 2>());

                  outer_prod_jagged_2d_output<index_t, scalar_t>(
                      v.accessor<scalar_t, 2>(),
                      grad_outputs[0].accessor<scalar_t, 2>(),
                      a_offsets.accessor<index_t, 1>(),
                      a_values_grad.accessor<scalar_t, 2>());
                });
          });
    } else {
      v_grad.zero_();
    }

    return {
        v_grad,
        a_values_grad,
        torch::autograd::Variable(), // a_offsets
    };
  }
};

///@ingroup jagged-tensor-ops-cpu
Tensor batched_dense_vec_jagged_2d_mul(
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  return BatchedDenseVecJagged2DMulCPUOp::apply(v, a_values, a_offsets)[0];
}

} // namespace

///@ingroup jagged-tensor-ops-cpu
Tensor
jagged_2d_to_dense_forward_cpu(Tensor values, Tensor offsets, int64_t max_L) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  return jagged_to_padded_dense(
      values, {offsets}, {max_L}, /*padding_value=*/0);
}

///@ingroup jagged-tensor-ops-cpu
Tensor jagged_1d_to_dense_cpu(
    Tensor values,
    Tensor offsets,
    int64_t max_L,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  return jagged_to_padded_dense(values, {offsets}, {max_L}, padding_value);
}
} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // (dense, offsets) -> jagged. Offsets output is same as input.
  m.def(
      "dense_to_jagged(Tensor dense, Tensor[] x_offsets, int? total_L=None) -> (Tensor, Tensor[])");
  m.def(
      "jagged_2d_to_dense(Tensor values, Tensor offsets, int max_sequence_length) -> Tensor");
  m.def(
      "jagged_1d_to_dense(Tensor values, Tensor offsets, int max_sequence_length, int padding_value) -> Tensor");
  m.def(
      "stacked_jagged_2d_to_dense_forward(Tensor values, Tensor lengths, int[] offset_per_key, int[] max_lengths_per_key) -> (Tensor[], Tensor[])");
  m.def(
      "stacked_jagged_2d_to_dense_backward(int B, int D, int total_L, Tensor[] grad_padded_values_per_key, Tensor[] offsets_tensor_per_key, int[] offset_per_key) -> Tensor");
  m.def(
      "stacked_jagged_1d_to_dense(Tensor values, Tensor lengths, int[] offset_per_key, int[] max_lengths_per_key, int padding_value) -> Tensor[]");
  m.def(
      "stacked_jagged_2d_to_dense(Tensor values, Tensor lengths, int[] offset_per_key, int[] max_lengths_per_key) -> Tensor[]");
  m.def(
      "jagged_to_padded_dense(Tensor values, Tensor[] offsets, int[] max_lengths, float padding_value = 0) -> Tensor");
  // jagged + dense -> dense
  m.def(
      "jagged_dense_elementwise_add(Tensor x_values, Tensor[] x_offsets, Tensor y) -> Tensor");
  // jagged + dense -> jagged (treat "zeros" in the jagged tensor as unknowns.
  // output offsets is same as x_offsets)
  m.def(
      "jagged_dense_elementwise_add_jagged_output(Tensor x_values, Tensor[] x_offsets, Tensor y) -> (Tensor, Tensor[])");
  m.def(
      "jagged_dense_dense_elementwise_add_jagged_output(Tensor x_values, Tensor[] x_offsets, Tensor y_0, Tensor y_1) -> (Tensor, Tensor[])");
  // jagged * dense -> jagged (its offsets is same as x_offsets)
  m.def(
      "jagged_dense_elementwise_mul(Tensor x_values, Tensor[] x_offsets, Tensor y) -> (Tensor, Tensor[])");
  m.def(
      "batched_dense_vec_jagged_2d_mul(Tensor v, Tensor a_values, Tensor a_offsets) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU(
      "jagged_2d_to_dense", fbgemm_gpu::jagged_2d_to_dense_forward_cpu);
  DISPATCH_TO_CPU("jagged_1d_to_dense", fbgemm_gpu::jagged_1d_to_dense_cpu);
  DISPATCH_TO_CPU("dense_to_jagged", fbgemm_gpu::dense_to_jagged_cpu);
  DISPATCH_TO_CPU("jagged_to_padded_dense", fbgemm_gpu::jagged_to_padded_dense);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_add", fbgemm_gpu::jagged_dense_elementwise_add);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_add_jagged_output",
      fbgemm_gpu::jagged_dense_elementwise_add_jagged_output);
  DISPATCH_TO_CPU(
      "jagged_dense_dense_elementwise_add_jagged_output",
      fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_mul", fbgemm_gpu::jagged_dense_elementwise_mul);
  DISPATCH_TO_CPU(
      "batched_dense_vec_jagged_2d_mul",
      fbgemm_gpu::batched_dense_vec_jagged_2d_mul);
}
