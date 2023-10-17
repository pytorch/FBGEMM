/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "ATen/Parallel.h"

#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/sparse_ops.h"
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
  for (const auto d : c10::irange(num_jagged_dim)) {
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
  for (const auto oidx : c10::irange(outer_dense_size)) {
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
            for (const auto iidx : c10::irange(inner_dense_size)) {
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
          for (const auto iidx : c10::irange(inner_dense_size)) {
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
  for (const auto oidx : c10::irange(outer_dense_size)) {
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
            for (const auto iidx : c10::irange(inner_dense_size)) {
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

at::Tensor jagged_to_padded_dense_forward(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    c10::SymIntArrayRef max_lengths,
    const double padding_value) {
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
  at::SymDimVector padded_values_shape({at::SymInt(offsets[0].size(0) - 1)});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = values.dim() == 1;
  if (!D_folded) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values =
      at::empty_symint(padded_values_shape, values.options());
  if (values.numel() == 0) {
    // To avoid an error due to values_canonicalized.data_ptr is nullptr.
    padded_values.fill_(padding_value);
    return {padded_values};
  }
  Tensor padded_values_view =
      D_folded ? padded_values.unsqueeze(-1) : padded_values;

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
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

  return padded_values;
}

at::Tensor jagged_to_padded_dense_backward(
    const Tensor& grad_output,
    const std::vector<Tensor>& offsets,
    const at::SymInt total_L) {
  auto grad_padded_values = grad_output;

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded =
      static_cast<size_t>(grad_padded_values.dim()) == offsets.size() + 1;
  Tensor grad_padded_values_view =
      D_folded ? grad_padded_values.unsqueeze(-1) : grad_padded_values;
  int32_t D = grad_padded_values_view.size(-1);
  // Initialize with zeros so output will be zero for the portion truncated
  // in forward.
  auto grad_values =
      at::zeros_symint({total_L, D}, grad_padded_values.options());

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_padded_values.scalar_type(),
      "jagged_2d_to_dense_backward_kernel",
      [&] {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            grad_values, // dummy not used in the lambda function
            {offsets},
            grad_padded_values_view,
            grad_values,
            [](scalar_t /*unused*/, scalar_t y) -> scalar_t { return y; });
      });

  return D_folded ? grad_values.squeeze(-1) : grad_values;
}

Tensor dense_to_jagged_forward(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    c10::optional<at::SymInt> total_L) {
  // D is the embedding dimension
  auto D = dense.size(-1);

  // If total_L is not given then compute it
  at::SymInt total_L_computed;
  if (total_L.has_value()) {
    total_L_computed = total_L.value();
  } else {
    total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
  }
  auto values = at::empty_symint({total_L_computed, D}, dense.options());
  auto output = at::zeros_symint({total_L_computed, D}, dense.options());

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
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

  return output;
}

// output = x + y where x is jagged, y is dense, and output is jagged
at::Tensor jagged_dense_dense_elementwise_add_jagged_output_forward(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y_0,
    const at::Tensor& y_1) {
  // Convert to jagged
  auto jagged_values_0 =
      dense_to_jagged_forward(y_0, x_offsets, c10::optional<at::SymInt>());
  auto jagged_values_1 =
      dense_to_jagged_forward(y_1, x_offsets, c10::optional<at::SymInt>());

  // Add jagged_values + x_values -> sum_values
  auto sum_values = x_values + jagged_values_0 + jagged_values_1;

  return sum_values;
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
  for (const auto oidx : c10::irange(outer_dense_size)) {
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
            for (const auto iidx : c10::irange(inner_dense_size)) {
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
          for (const auto iidx : c10::irange(inner_dense_size)) {
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

Tensor jagged_dense_elementwise_mul_forward(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  // Convert to jagged
  auto jagged_values =
      dense_to_jagged_forward(y, x_offsets, c10::optional<at::SymInt>());

  // Multiply x_values * jagged_values -> prod_values
  auto prod_values = x_values * jagged_values;

  return prod_values;
}

std::tuple<Tensor, Tensor> jagged_dense_elementwise_mul_backward(
    const Tensor& grad_output,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& x_values) {
  Tensor x_values_grad = at::zeros_like(grad_output);
  Tensor y_grad = at::zeros_like(y);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x_values.scalar_type(), "jagged_dense_elementwise_mul_backward", [&] {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            grad_output,
            x_offsets,
            y,
            x_values_grad,
            [](scalar_t x, scalar_t y) -> scalar_t { return x * y; });

        jagged_jagged_elementwise_dense_output_<scalar_t>(
            grad_output,
            x_offsets,
            x_values,
            y_grad,
            [](scalar_t x, scalar_t y) -> scalar_t { return x * y; });
      });

  return {x_values_grad, y_grad};
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
  for (const auto b : c10::irange(B)) {
    const int row_start = a_offsets[b];
    const int row_end = a_offsets[b + 1];
    const int length = std::min(row_end - row_start, max_L);
    if (length == 0) {
      for (const auto h : c10::irange(H)) {
        for (const auto d : c10::irange(D)) {
          output[b * H + h][d] = 0;
        }
      }
    } else {
      for (const auto h : c10::irange(H)) {
        for (const auto d : c10::irange(D)) {
          // use is_cuda=true because acc_type<float, false> = double is too
          // conservative
          at::acc_type<scalar_t, true> acc =
              v[b * H + h][0] * a_values[row_start][h * D + d];
          for (const auto l : c10::irange(1, length)) {
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
  for (const auto b : c10::irange(B)) {
    const int row_start = a_offsets[b];
    const int row_end = a_offsets[b + 1];
    const int length = std::min(row_end - row_start, max_L);

    if (D == 0) {
      for (const auto h : c10::irange(H)) {
        for (const auto l : c10::irange(max_L)) {
          output[b * H + h][l] = 0;
        }
      }
    } else {
      for (const auto h : c10::irange(H)) {
        int l;
        for (l = 0; l < length; ++l) {
          at::acc_type<scalar_t, true> acc =
              v[b * H + h][0] * a_values[row_start + l][h * D];
          for (const auto d : c10::irange(1, D)) {
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
  for (const auto b : c10::irange(B)) {
    const int row_start = offsets[b];
    const int row_end = offsets[b + 1];
    const int length = row_end - row_start;
    for (const auto h : c10::irange(H)) {
      for (int l = 0; l < std::min(length, max_L); ++l) {
        for (const auto d : c10::irange(D)) {
          output_values[row_start + l][h * D + d] =
              x[b * H + h][l] * y[b * H + h][d];
        }
      }
    }
  }
}

Tensor batched_dense_vec_jagged_2d_mul_forward(
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
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
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
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
  return output;
}

std::tuple<Tensor, Tensor> batched_dense_vec_jagged_2d_mul_backward(
    const Tensor& grad_output,
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  TENSOR_ON_CPU(grad_output);

  Tensor a_values_grad = at::zeros_like(a_values);
  Tensor v_grad = at::empty_like(v);

  const int B = a_offsets.numel() - 1;
  const int D = grad_output.size(-1);

  if (B > 0 && D > 0) {
    AT_DISPATCH_INDEX_TYPES(
        a_offsets.scalar_type(),
        "dense_vec_jagged_2d_bmm_backward_kernel_1",
        [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              grad_output.scalar_type(),
              "dense_vec_jagged_2d_bmm_backward_kernel_2",
              [&] {
                dense_vec_jagged_2d_transposed_bmm<index_t, scalar_t>(
                    grad_output.accessor<scalar_t, 2>(),
                    a_values.accessor<scalar_t, 2>(),
                    a_offsets.accessor<index_t, 1>(),
                    v_grad.accessor<scalar_t, 2>());

                outer_prod_jagged_2d_output<index_t, scalar_t>(
                    v.accessor<scalar_t, 2>(),
                    grad_output.accessor<scalar_t, 2>(),
                    a_offsets.accessor<index_t, 1>(),
                    a_values_grad.accessor<scalar_t, 2>());
              });
        });
  } else {
    v_grad.zero_();
  }
  return {v_grad, a_values_grad};
}

Tensor jagged_1d_to_truncated_values_cpu(
    Tensor values,
    Tensor lengths,
    int64_t max_truncated_length) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(lengths.dim() == 1);

  const int32_t B = lengths.size(0);
  Tensor truncated_values;
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "jagged_1d_to_truncated_values_cpu_kernel", [&] {
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            values.scalar_type(),
            "copy_values_and_truncate_cpu_kernel",
            [&] {
              const index_t max_length_int =
                  static_cast<index_t>(max_truncated_length);
              const auto lengths_accessor = lengths.accessor<index_t, 1>();
              int32_t num_outputs = 0;
              for (const auto b : c10::irange(B)) {
                num_outputs += std::min(max_length_int, lengths_accessor[b]);
              }
              const auto input_accessor = values.accessor<scalar_t, 1>();
              truncated_values = at::empty({num_outputs}, values.options());
              auto output_accessor = truncated_values.accessor<scalar_t, 1>();
              int64_t input_offset = 0;
              int64_t output_offset = 0;
              for (const auto b : c10::irange(B)) {
                index_t cur_len = std::min(max_length_int, lengths_accessor[b]);
                for (const auto i : c10::irange(cur_len)) {
                  output_accessor[output_offset + i] =
                      input_accessor[input_offset + i];
                }
                output_offset += cur_len;
                input_offset += lengths_accessor[b];
              }
            });
      });

  return truncated_values;
}

} // namespace

std::tuple<Tensor, Tensor> masked_select_jagged_1d(
    const Tensor& values,
    const Tensor& lengths,
    const Tensor& mask) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(lengths.dim() == 1);

  auto values_contiguous = values.expect_contiguous();
  auto lengths_contiguous = lengths.expect_contiguous();
  auto mask_contiguous = mask.expect_contiguous();

  const auto B = lengths.numel();
  Tensor masked_values;
  Tensor masked_lengths = at::empty_like(lengths);

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "mask_select_jagged_1d_kernel1", [&] {
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            values.scalar_type(),
            "mask_select_jagged_1d_kernel2",
            [&] {
              const int32_t num_outputs = mask.sum().item<int32_t>();
              masked_values = at::empty({num_outputs}, values.options());

              const auto values_ptr = values_contiguous->data_ptr<scalar_t>();
              const auto lengths_ptr = lengths_contiguous->data_ptr<index_t>();
              const auto mask_ptr = mask_contiguous->data_ptr<bool>();

              auto masked_values_ptr = masked_values.data_ptr<scalar_t>();
              auto masked_lengths_ptr = masked_lengths.data_ptr<index_t>();

              int64_t input_offset = 0;
              int64_t output_offset = 0;
              for (const auto b : c10::irange(B)) {
                const index_t input_len = lengths_ptr[b];
                index_t output_len = 0;
                for (int i = input_offset; i < input_offset + input_len; ++i) {
                  if (mask_ptr[i]) {
                    masked_values_ptr[output_offset] = values_ptr[i];
                    ++output_offset;
                    ++output_len;
                  }
                }
                input_offset += input_len;
                masked_lengths_ptr[b] = output_len;
              }
            });
      });

  return {masked_values, masked_lengths};
}

Tensor
jagged_2d_to_dense_forward_cpu(Tensor values, Tensor offsets, int64_t max_L) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  return jagged_to_padded_dense_forward(
      values,
      {offsets},
      at::ArrayRef<at::SymInt>({max_L}),
      /*padding_value=*/0);
}

std::vector<Tensor> stacked_jagged_1d_to_dense_cpu(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(lengths.dim() == 2);

  const auto lengths_contig = lengths.contiguous();
  int32_t B = lengths.size(1);
  int32_t T = lengths.size(0);
  auto offsets = at::empty({B + 1}, lengths.options());
  offsets[0].zero_();
  std::vector<Tensor> padded_values_per_key;
  for (const auto t : c10::irange(T)) {
    int64_t max_L = max_lengths_per_key[t];
    AT_DISPATCH_INDEX_TYPES(
        lengths_contig.scalar_type(), "length_to_offset_cpu_kernel", [&] {
          index_t cumsum = 0;
          const auto* input_ptr = &(lengths_contig.data_ptr<index_t>()[t * B]);
          auto* output_ptr = offsets.data_ptr<index_t>() + 1;
          for (const auto i : c10::irange(B)) {
            cumsum += input_ptr[i];
            output_ptr[i] = cumsum;
          }
        });
    padded_values_per_key.push_back(jagged_to_padded_dense(
        values.slice(0, offset_per_key[t], offset_per_key[t + 1]),
        {offsets},
        {max_L},
        padding_value));
  }

  return padded_values_per_key;
}

// stacked ops
std::vector<Tensor> stacked_jagged_2d_to_dense_cpu(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(lengths.dim() == 2);

  const auto lengths_contig = lengths.contiguous();
  int32_t B = lengths.size(1);
  int32_t T = lengths.size(0);
  std::vector<Tensor> padded_values_per_key;
  std::vector<Tensor> offsets_tensor_per_key;
  for (const auto t : c10::irange(T)) {
    int64_t max_L = max_lengths_per_key[t];
    auto offsets = at::empty({B + 1}, lengths.options());
    offsets[0].zero_();
    AT_DISPATCH_INDEX_TYPES(
        lengths_contig.scalar_type(), "length_to_offset_cpu_kernel", [&] {
          index_t cumsum = 0;
          const auto* input_ptr = &(lengths_contig.data_ptr<index_t>()[t * B]);
          auto* output_ptr = offsets.data_ptr<index_t>() + 1;
          for (const auto i : c10::irange(B)) {
            cumsum += input_ptr[i];
            output_ptr[i] = cumsum;
          }
        });
    offsets_tensor_per_key.push_back(offsets);

    padded_values_per_key.push_back(jagged_to_padded_dense(
        values.slice(0, offset_per_key[t], offset_per_key[t + 1]),
        {offsets},
        {max_L},
        padding_value));
  }

  return padded_values_per_key;
}

template <typename index_t, typename offset_t, typename scalar_t>
void jagged_index_select_2d_kernel(
    at::TensorAccessor<scalar_t, 2> output,
    const at::TensorAccessor<scalar_t, 2>& input,
    const at::TensorAccessor<offset_t, 1>& input_offsets,
    const at::TensorAccessor<index_t, 1>& indices,
    const at::TensorAccessor<offset_t, 1>& output_offsets) {
  const auto num_output_rows = output_offsets.size(0);
  const auto num_dense_output_rows = output.size(0);
  const auto num_cols = input.size(1);
  at::parallel_for(
      0, num_dense_output_rows, 0, [&](int64_t start, int64_t end) {
        for (const auto dense_output_offset : c10::irange(start, end)) {
          int index_pos;
          binary_search_range_cpu(
              &index_pos,
              reinterpret_cast<const offset_t*>(&output_offsets[0]),
              static_cast<offset_t>(dense_output_offset),
              num_output_rows);
          const offset_t rel_index = dense_output_offset -
              (index_pos == 0 ? 0 : output_offsets[index_pos - 1]);
          const index_t index = indices[index_pos];
          const offset_t input_offset =
              (index == 0 ? 0 : input_offsets[index - 1]) + rel_index;
          for (const auto i : c10::irange(num_cols)) {
            output[dense_output_offset][i] = input[input_offset][i];
          }
        }
      });
}

/// Copy sequences from input jagged tensor based on indices specified in the
/// indices tensor to output jagged tensor (this function invokes
/// jagged_index_select_2d_kernel)
/// @param values                2D dense value tensor of input jagged tensor
/// @param indices               1D tensor that contains indices to be selected
///                              from input jagged tensor
/// @param input_offsets         1D tensor that contains offsets of input
///                              jagged tensor
/// @param output_offsets        1D tensor that contains offsets of output
///                              jagged tensor
/// @param num_dense_output_rows The total number of rows in the 2D dense value
///                              tensor of output jagged tensor
Tensor jagged_index_select_2d_forward_cpu(
    const Tensor& values,
    const Tensor& indices,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    const int64_t num_dense_output_rows) {
  TORCH_CHECK(
      values.dim() == 2,
      "jagged_index_select_2d_forward_cpu supports only 2D inputs");
  auto num_cols = values.size(1);
  Tensor output =
      at::empty({num_dense_output_rows, num_cols}, values.options());

  if (num_dense_output_rows > 0) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        values.scalar_type(),
        "jagged_index_select_2d_kernel_wrapper_1",
        [&] {
          AT_DISPATCH_INDEX_TYPES(
              indices.scalar_type(),
              "jagged_index_select_2d_kernel_wrapper_2",
              [&] {
                jagged_index_select_2d_kernel(
                    output.accessor<scalar_t, 2>(),
                    values.accessor<scalar_t, 2>(),
                    input_offsets.accessor<int64_t, 1>(),
                    indices.accessor<index_t, 1>(),
                    output_offsets.accessor<int64_t, 1>());
              });
        });
  }

  return output;
}

template <typename index_t, typename offset_t, typename scalar_t>
void jagged_index_add_2d_kernel(
    at::TensorAccessor<scalar_t, 2> output,
    const at::TensorAccessor<scalar_t, 2>& input,
    const at::TensorAccessor<offset_t, 1>& input_offsets,
    const at::TensorAccessor<index_t, 1>& indices,
    const at::TensorAccessor<offset_t, 1>& output_offsets) {
  const auto num_input_rows = input_offsets.size(0);
  const auto num_dense_input_rows = input.size(0);
  const auto num_cols = input.size(1);
  // Allocate one lock per row
  std::atomic_flag* locks = new std::atomic_flag[output.size(0)];
  // Initialize all locks since before c++20 std::atomic_flag is initialized to
  // an unspecified state.
  // https://en.cppreference.com/w/cpp/atomic/atomic_flag/atomic_flag
  for (auto i = 0; i < output.size(0); i++) {
    locks[i].clear();
  }

  at::parallel_for(0, num_dense_input_rows, 0, [&](int64_t start, int64_t end) {
    for (const auto dense_input_offset : c10::irange(start, end)) {
      int index_pos;
      binary_search_range_cpu(
          &index_pos,
          reinterpret_cast<const offset_t*>(&input_offsets[0]),
          static_cast<offset_t>(dense_input_offset),
          num_input_rows);
      const offset_t rel_index = dense_input_offset -
          (index_pos == 0 ? 0 : input_offsets[index_pos - 1]);
      const index_t index = indices[index_pos];
      const offset_t output_offset =
          (index == 0 ? 0 : output_offsets[index - 1]) + rel_index;

      // Spin lock
      auto& lock = locks[output_offset];
      while (lock.test_and_set(std::memory_order_acquire)) {
        // For C++20
#if defined(__cpp_lib_atomic_flag_test)
        while (lock.test(std::memory_order_relaxed))
#endif
          ;
      }
      for (const auto i : c10::irange(num_cols)) {
        output[output_offset][i] += input[dense_input_offset][i];
      }
      // Release lock
      lock.clear(std::memory_order_release);
    }
  });
}

/// Add sequences from input jagged tensor to output jagged tensor based on
/// indices specified in the indices tensor (this function invokes
/// jagged_index_add_2d_kernel)
/// @param values               2D dense value tensor of input jagged tensor
/// @param indices              1D tensor that contains indices to be added in
///                             output jagged tensor
/// @param input_offsets        1D tensor that contains offsets of input
///                             jagged tensor
/// @param output_offsets       1D tensor that contains offsets of output
///                             jagged tensor
/// @param num_dense_input_rows The total number of rows in the 2D dense value
///                             tensor of input jagged tensor
/// @param num_output_rows      The number of sequences in jagged output tensor
Tensor jagged_index_add_2d_forward_cpu(
    const Tensor& values,
    const Tensor& indices,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    const int64_t num_dense_input_rows,
    const int64_t num_output_rows) {
  TORCH_CHECK(
      values.dim() == 2,
      "jagged_index_add_2d_forward_cpu supports only 2D inputs");
  auto num_cols = values.size(1);
  Tensor output = at::zeros({num_output_rows, num_cols}, values.options());
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "jagged_index_add_2d_kernel_wrapper_1",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "jagged_index_add_2d_kernel_wrapper_2", [&] {
              jagged_index_add_2d_kernel(
                  output.accessor<scalar_t, 2>(),
                  values.accessor<scalar_t, 2>(),
                  input_offsets.accessor<int64_t, 1>(),
                  indices.accessor<index_t, 1>(),
                  output_offsets.accessor<int64_t, 1>());
            });
      });
  return output;
}

template <typename index_t, typename scalar_t>
void jagged_softmax_kernel(
    const at::TensorAccessor<scalar_t, 2>& values,
    const at::TensorAccessor<index_t, 1>& offsets,
    at::TensorAccessor<scalar_t, 2> output,
    const int64_t max_L) {
  const int B = offsets.size(0) - 1;
  const int D = values.size(1);
  for (const auto b : c10::irange(B)) {
    const int row_start = offsets[b];
    const int row_end = offsets[b + 1];
    const int length = std::min(row_end - row_start, (int)max_L);

    if (length == 0)
      continue;
    for (const auto d : c10::irange(D)) {
      // use is_cuda=true because acc_type<float, false> = double is too
      // conservative
      scalar_t max_value = values[row_start][d];
      for (const auto l : c10::irange(1, length)) {
        max_value = std::max(max_value, values[row_start + l][d]);
      }
      at::acc_type<scalar_t, true> acc =
          std::exp(values[row_start][d] - max_value);
      for (const auto l : c10::irange(1, length)) {
        acc += std::exp(values[row_start + l][d] - max_value);
      }
      for (const auto l : c10::irange(length)) {
        output[row_start + l][d] =
            std::exp(values[row_start + l][d] - max_value) / acc;
      }
    } // for each d
  } // for each b
}

Tensor jagged_softmax_forward(
    const Tensor& values,
    const Tensor& offsets,
    const int64_t max_L) {
  TENSOR_ON_CPU(values);
  TENSOR_ON_CPU(offsets);
  const int B = offsets.numel() - 1;
  const int D = values.size(1);
  auto output = at::empty_like(values);

  if (B > 0 && D > 0) {
    AT_DISPATCH_INDEX_TYPES(
        offsets.scalar_type(), "jagged_softmax_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              values.scalar_type(),
              "jagged_softmax_kernel_2",
              [&] {
                jagged_softmax_kernel<index_t, scalar_t>(
                    values.accessor<scalar_t, 2>(),
                    offsets.accessor<index_t, 1>(),
                    output.accessor<scalar_t, 2>(),
                    max_L);
              });
        });
  }
  return output;
}

template <typename index_t, typename scalar_t>
void jagged_softmax_backward_kernel(
    const at::TensorAccessor<scalar_t, 2>& grad_output,
    const at::TensorAccessor<scalar_t, 2>& output,
    const at::TensorAccessor<index_t, 1>& offsets,
    at::TensorAccessor<scalar_t, 2> grad_input,
    const int64_t max_L) {
  const int B = offsets.size(0) - 1;
  const int D = grad_output.size(1);
  for (const auto b : c10::irange(B)) {
    const int row_start = offsets[b];
    const int row_end = offsets[b + 1];
    const int length = std::min(row_end - row_start, (int)max_L);
    if (length == 0)
      continue;
    for (const auto d : c10::irange(D)) {
      at::acc_type<scalar_t, true> sum_value =
          grad_output[row_start][d] * output[row_start][d];
      for (const auto l : c10::irange(1, length)) {
        sum_value += grad_output[row_start + l][d] * output[row_start + l][d];
      }
      for (const auto l : c10::irange(length)) {
        grad_input[row_start + l][d] =
            (grad_output[row_start + l][d] - sum_value) *
            output[row_start + l][d];
      }
    }
  }
}

Tensor jagged_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& offsets,
    const int64_t max_L) {
  TENSOR_ON_CPU(grad_output);
  TENSOR_ON_CPU(output);
  TENSOR_ON_CPU(offsets);
  const int B = offsets.numel() - 1;
  const int D = grad_output.size(1);
  auto grad_input = at::empty_like(grad_output);

  if (B > 0 && D > 0) {
    AT_DISPATCH_INDEX_TYPES(
        offsets.scalar_type(), "jagged_backward_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              grad_output.scalar_type(),
              "jagged_softmax_backward_kernel_2",
              [&] {
                jagged_softmax_backward_kernel<index_t, scalar_t>(
                    grad_output.accessor<scalar_t, 2>(),
                    output.accessor<scalar_t, 2>(),
                    offsets.accessor<index_t, 1>(),
                    grad_input.accessor<scalar_t, 2>(),
                    max_L);
              });
        });
  }
  return grad_input;
}

template <typename index_t, typename scalar_t>
void jagged_jagged_bmm_kernel(
    const at::TensorAccessor<scalar_t, 2>& x_values,
    const at::TensorAccessor<scalar_t, 2>& y_values,
    const at::TensorAccessor<index_t, 1>& offsets,
    at::TensorAccessor<scalar_t, 3> output,
    const int64_t max_L) {
  const int B = offsets.size(0) - 1;
  const int M = x_values.size(1);
  const int N = y_values.size(1);
  for (const auto b : c10::irange(B)) {
    const int row_start = offsets[b];
    const int row_end = offsets[b + 1];
    const int length = std::min(row_end - row_start, (int)max_L);
    for (const auto m : c10::irange(M)) {
      for (const auto n : c10::irange(N)) {
        at::acc_type<scalar_t, true> acc = 0;
        for (const auto l : c10::irange(length)) {
          acc += x_values[row_start + l][m] * y_values[row_start + l][n];
        }
        output[b][m][n] = acc;
      }
    }
  } // for each b
}

Tensor jagged_jagged_bmm_forward(
    const Tensor& x_values,
    const Tensor& y_values,
    const Tensor& offsets,
    const int64_t max_L) {
  TENSOR_ON_CPU(x_values);
  TENSOR_ON_CPU(y_values);
  TENSOR_ON_CPU(offsets);
  const int B = offsets.size(0) - 1;
  const int M = x_values.size(-1);
  const int N = y_values.size(-1);
  auto output = at::zeros({B, M, N}, x_values.options());
  if (B > 0 && M > 0 && N > 0) {
    AT_DISPATCH_INDEX_TYPES(
        offsets.scalar_type(), "jagged_jagged_bmm_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              x_values.scalar_type(),
              "jagged_jagged_bmm_kernel_2",
              [&] {
                jagged_jagged_bmm_kernel<index_t, scalar_t>(
                    x_values.accessor<scalar_t, 2>(),
                    y_values.accessor<scalar_t, 2>(),
                    offsets.accessor<index_t, 1>(),
                    output.accessor<scalar_t, 3>(),
                    max_L);
              });
        });
  }

  return output;
}

template <typename index_t, typename scalar_t>
void jagged_dense_bmm_kernel(
    const at::TensorAccessor<scalar_t, 2>& x_values,
    const at::TensorAccessor<index_t, 1>& x_offsets,
    const at::TensorAccessor<scalar_t, 3>& y,
    at::TensorAccessor<scalar_t, 2> output,
    const int64_t max_L) {
  // [sum_B, K] x [B, K, N] -> [B, L, N] -> [sum_B, N]
  const int B = x_offsets.size(0) - 1;
  const int K = x_values.size(1);
  const int N = y.size(2);
  for (const auto b : c10::irange(B)) {
    const int row_start = x_offsets[b];
    const int row_end = x_offsets[b + 1];
    const int length = std::min(row_end - row_start, (int)max_L);
    for (const auto l : c10::irange(length)) {
      for (const auto n : c10::irange(N)) {
        at::acc_type<scalar_t, true> acc = 0;
        for (const auto k : c10::irange(K)) {
          acc += x_values[row_start + l][k] * y[b][k][n];
        }
        output[row_start + l][n] = acc;
      }
    }
  } // for each b
}

Tensor jagged_dense_bmm_forward(
    const Tensor& x_values,
    const Tensor& x_offsets,
    const Tensor& y,
    const int64_t max_L) {
  TENSOR_ON_CPU(x_values);
  TENSOR_ON_CPU(x_offsets);
  TENSOR_ON_CPU(y);
  const int B = x_offsets.size(0) - 1;
  const int M = x_values.size(-1);
  const int N = y.size(-1);
  const int total_L = x_values.size(0);
  auto output = at::zeros({total_L, N}, x_values.options());
  if (B > 0 && M > 0 && N > 0) {
    AT_DISPATCH_INDEX_TYPES(
        x_offsets.scalar_type(), "jagged_dense_bmm_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              x_values.scalar_type(),
              "jagged_dense_bmm_kernel_2",
              [&] {
                jagged_dense_bmm_kernel<index_t, scalar_t>(
                    x_values.accessor<scalar_t, 2>(),
                    x_offsets.accessor<index_t, 1>(),
                    y.accessor<scalar_t, 3>(),
                    output.accessor<scalar_t, 2>(),
                    (int)max_L);
              });
        });
  }

  return output;
}

template <typename scalar_t, typename offset_t>
void jagged_slice_forward_cpu_kernel(
    at::TensorAccessor<scalar_t, 1> output,
    const at::TensorAccessor<offset_t, 1>& output_lengths,
    const at::TensorAccessor<offset_t, 1>& output_offsets,
    const at::TensorAccessor<offset_t, 1>& tgt_start,
    const at::TensorAccessor<scalar_t, 1>& input,
    const at::TensorAccessor<offset_t, 1>& input_lengths,
    const at::TensorAccessor<offset_t, 1>& input_offsets,
    const at::TensorAccessor<offset_t, 1>& src_start,
    const int64_t slice_length) {
  const auto B = output_offsets.size(0);

  // TODO (devashisht) parallelize this loop
  for (const auto row_i : c10::irange(B)) {
    const int64_t output_offset_start = output_offsets[row_i];
    const int64_t input_offset_start = input_offsets[row_i];
    const auto tgt_start_ = tgt_start[row_i];
    const auto src_start_ = src_start[row_i];
    for (auto col_i = 0;
         col_i < slice_length && tgt_start_ + col_i < output_lengths[row_i] &&
         src_start_ + col_i < input_lengths[row_i];
         ++col_i) {
      const int64_t output_offset = output_offset_start + tgt_start_ + col_i;
      const int64_t input_offset = input_offset_start + src_start_ + col_i;
      output[output_offset] = input[input_offset];
    }
  }
}

/// Slice the jagged dim to max length from slice_length,
/// from start point `start`. This is a jagged -> jagged op
/// @param x_values - X values of shape B * J_DIM where J_DIM is
///                   jagged dim
/// @param x_lengths - length along jagged dim
/// @param src_start - start of slice operation from the src tensor
/// @param output_lengths - length of jagged dim for output tensor
/// @param tgt_start - position to start filling in sliced values from source
/// @param num_output_rows - output dense dim
/// @param slice_length - length of jagged dim to slice
/// @param fill_zeros - option exists as an optimization, we can reuse
///                     the same code path for forward & backward. For backward
///                     we need to fill zeros in output tensor but fwd we don't.
Tensor jagged_slice_forward_cpu(
    const Tensor& x_values,
    const Tensor& x_lengths,
    const Tensor& src_start,
    const Tensor& output_lengths,
    const Tensor& tgt_start,
    const int64_t num_output_rows,
    const int64_t slice_length,
    const bool fill_zeros) {
  TENSOR_ON_CPU(x_values);
  TENSOR_ON_CPU(x_lengths);
  TENSOR_NDIM_EQUALS(x_values, 1);
  TENSOR_NDIM_EQUALS(x_lengths, 1);

  auto output_values = fill_zeros
      ? at::zeros({num_output_rows}, x_values.options())
      : at::empty({num_output_rows}, x_values.options());

  auto output_offsets = asynchronous_exclusive_cumsum_cpu(output_lengths);
  auto input_offsets = asynchronous_exclusive_cumsum_cpu(x_lengths);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_values.scalar_type(),
      "jagged_slice_wrapper_1",
      [&] {
        jagged_slice_forward_cpu_kernel<scalar_t>(
            output_values.accessor<scalar_t, 1>(),
            output_lengths.accessor<int64_t, 1>(),
            output_offsets.accessor<int64_t, 1>(),
            tgt_start.accessor<int64_t, 1>(),
            x_values.accessor<scalar_t, 1>(),
            x_lengths.accessor<int64_t, 1>(),
            input_offsets.accessor<int64_t, 1>(),
            src_start.accessor<int64_t, 1>(),
            slice_length);
      });

  return output_values;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // (dense, offsets) -> jagged. Offsets output is same as input.
  // SymInt is a new PyTorch 2.0 feature to support dynamic shape. See more
  // details at https://pytorch.org/get-started/pytorch-2.0/#dynamic-shapes. If
  // you find it doesn't compile, please pull the new PyTorch 2.0 code
  m.def(
      "dense_to_jagged(Tensor dense, Tensor[] x_offsets, SymInt? total_L=None) -> (Tensor, Tensor[])");
  m.def(
      "dense_to_jagged_forward(Tensor dense, Tensor[] x_offsets, SymInt? total_L=None) -> Tensor");
  m.def(
      "jagged_2d_to_dense(Tensor values, Tensor offsets, SymInt max_sequence_length) -> Tensor");
  m.def(
      "jagged_1d_to_dense(Tensor values, Tensor offsets, SymInt max_sequence_length, int padding_value) -> Tensor");
  m.def(
      "stacked_jagged_2d_to_dense_forward(Tensor values, Tensor lengths, int[] offset_per_key, int[] max_lengths_per_key, int padding_value = 0) -> (Tensor[], Tensor[])");
  m.def(
      "stacked_jagged_2d_to_dense_backward(int B, int D, int total_L, Tensor[] grad_padded_values_per_key, Tensor[] offsets_tensor_per_key, int[] offset_per_key) -> Tensor");
  m.def(
      "stacked_jagged_1d_to_dense(Tensor values, Tensor lengths, int[] offset_per_key, int[] max_lengths_per_key, int padding_value) -> Tensor[]");
  m.def(
      "stacked_jagged_2d_to_dense(Tensor values, Tensor lengths, int[] offset_per_key, int[] max_lengths_per_key, int padding_value = 0) -> Tensor[]");
  m.def(
      "jagged_to_padded_dense(Tensor values, Tensor[] offsets, SymInt[] max_lengths, float padding_value = 0) -> Tensor");
  m.def(
      "jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, SymInt[] max_lengths, float padding_value = 0) -> Tensor");
  m.def(
      "jagged_to_padded_dense_backward(Tensor grad_output, Tensor[] offsets, SymInt total_L) -> Tensor");
  // jagged + dense -> dense
  m.def(
      "jagged_dense_elementwise_add(Tensor x_values, Tensor[] x_offsets, Tensor y) -> Tensor");
  // jagged + dense -> jagged (treat "zeros" in the jagged tensor as unknowns.
  // output offsets is same as x_offsets)
  m.def(
      "jagged_dense_elementwise_add_jagged_output(Tensor x_values, Tensor[] x_offsets, Tensor y) -> (Tensor, Tensor[])");
  m.def(
      "jagged_dense_dense_elementwise_add_jagged_output_forward(Tensor x_values, Tensor[] x_offsets, Tensor y_0, Tensor y_1) -> Tensor");
  m.def(
      "jagged_dense_dense_elementwise_add_jagged_output(Tensor x_values, Tensor[] x_offsets, Tensor y_0, Tensor y_1) -> (Tensor, Tensor[])");
  // jagged * dense -> jagged (its offsets is same as x_offsets)
  m.def(
      "jagged_dense_elementwise_mul(Tensor x_values, Tensor[] x_offsets, Tensor y) -> (Tensor, Tensor[])");
  m.def(
      "jagged_dense_elementwise_mul_forward(Tensor x_values, Tensor[] x_offsets, Tensor y) -> Tensor");
  m.def(
      "jagged_dense_elementwise_mul_backward(Tensor grad_output, Tensor[] x_offsets, Tensor y, Tensor x_values) -> (Tensor, Tensor)");
  m.def(
      "batched_dense_vec_jagged_2d_mul(Tensor v, Tensor a_values, Tensor a_offsets) -> Tensor");
  m.def(
      "batched_dense_vec_jagged_2d_mul_forward(Tensor v, Tensor a_values, Tensor a_offsets) -> Tensor");
  m.def(
      "batched_dense_vec_jagged_2d_mul_backward(Tensor grad_output, Tensor v, Tensor a_values, Tensor a_offsets) -> (Tensor, Tensor)");
  m.def(
      "jagged_index_select(Tensor values, Tensor lengths, Tensor indices) -> Tensor[]");
  m.def(
      "jagged_index_select_2d_forward(Tensor values, Tensor indices, Tensor input_offsets, Tensor output_offsets, int num_dense_output_rows) -> Tensor");
  m.def(
      "jagged_index_add_2d_forward(Tensor values, Tensor indices, Tensor input_offsets, Tensor output_offsets, int num_dense_input_rows, int num_output_rows) -> Tensor");
  m.def(
      "jagged_1d_to_truncated_values(Tensor values, Tensor lengths, int max_truncated_length) -> Tensor");
  m.def(
      "masked_select_jagged_1d(Tensor values, Tensor lengths, Tensor mask) -> (Tensor, Tensor)");
  m.def(
      "jagged_softmax(Tensor values, Tensor x_offsets, int max_L) -> (Tensor, Tensor)");
  m.def(
      "jagged_softmax_forward(Tensor values, Tensor x_offsets, int max_L) -> Tensor");
  m.def(
      "jagged_softmax_backward(Tensor grad_output, Tensor output, Tensor x_offsets, int max_L) -> Tensor");
  m.def(
      "jagged_jagged_bmm(Tensor x_values, Tensor y_values, Tensor x_offsets, int max_L) -> Tensor");
  m.def(
      "jagged_jagged_bmm_forward(Tensor x_values, Tensor y_values, Tensor x_offsets, int max_L) -> Tensor");
  m.def(
      "jagged_dense_bmm(Tensor x_values, Tensor x_offsets, Tensor y, int max_L) -> (Tensor, Tensor)");
  m.def(
      "jagged_dense_bmm_forward(Tensor x_values, Tensor x_offsets, Tensor y, int max_L) -> Tensor");
  // jagged -> jagged
  m.def(
      "jagged_slice(Tensor x_values, Tensor x_lengths, Tensor start, int slice_length) -> (Tensor, Tensor)");
  m.def(
      "jagged_slice_forward(Tensor x_values, Tensor x_lengths, Tensor src_start, Tensor output_lengths, Tensor tgt_start, int num_output_rows, int slice_length, bool fill_zeros) -> Tensor");
  m.def(
      "jagged_unique_indices(Tensor hash_size_cumsum, Tensor hash_size_offsets, Tensor offsets, Tensor indices) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "jagged_hash_size_cumsum(Tensor offsets, Tensor indices, int batch_size) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU("jagged_2d_to_dense", fbgemm_gpu::jagged_2d_to_dense);
  DISPATCH_TO_CPU("jagged_1d_to_dense", fbgemm_gpu::jagged_1d_to_dense);
  DISPATCH_TO_CPU("dense_to_jagged", fbgemm_gpu::dense_to_jagged);
  DISPATCH_TO_CPU(
      "dense_to_jagged_forward", fbgemm_gpu::dense_to_jagged_forward);
  DISPATCH_TO_CPU("jagged_to_padded_dense", fbgemm_gpu::jagged_to_padded_dense);
  DISPATCH_TO_CPU(
      "jagged_to_padded_dense_forward",
      fbgemm_gpu::jagged_to_padded_dense_forward);
  DISPATCH_TO_CPU(
      "jagged_to_padded_dense_backward",
      fbgemm_gpu::jagged_to_padded_dense_backward);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_add", fbgemm_gpu::jagged_dense_elementwise_add);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_add_jagged_output",
      fbgemm_gpu::jagged_dense_elementwise_add_jagged_output);
  DISPATCH_TO_CPU(
      "jagged_dense_dense_elementwise_add_jagged_output_forward",
      fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output_forward);
  DISPATCH_TO_CPU(
      "jagged_dense_dense_elementwise_add_jagged_output",
      fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_mul", fbgemm_gpu::jagged_dense_elementwise_mul);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_mul_forward",
      fbgemm_gpu::jagged_dense_elementwise_mul_forward);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_mul_backward",
      fbgemm_gpu::jagged_dense_elementwise_mul_backward);
  DISPATCH_TO_CPU(
      "batched_dense_vec_jagged_2d_mul",
      fbgemm_gpu::batched_dense_vec_jagged_2d_mul);
  DISPATCH_TO_CPU(
      "batched_dense_vec_jagged_2d_mul_forward",
      fbgemm_gpu::batched_dense_vec_jagged_2d_mul_forward);
  DISPATCH_TO_CPU(
      "batched_dense_vec_jagged_2d_mul_backward",
      fbgemm_gpu::batched_dense_vec_jagged_2d_mul_backward);
  DISPATCH_TO_CPU(
      "stacked_jagged_1d_to_dense", fbgemm_gpu::stacked_jagged_1d_to_dense_cpu);
  DISPATCH_TO_CPU(
      "stacked_jagged_2d_to_dense", fbgemm_gpu::stacked_jagged_2d_to_dense_cpu);
  DISPATCH_TO_CPU(
      "jagged_index_select_2d_forward",
      fbgemm_gpu::jagged_index_select_2d_forward_cpu);
  DISPATCH_TO_CPU("jagged_index_select", fbgemm_gpu::jagged_index_select_2d);
  DISPATCH_TO_CPU(
      "jagged_index_add_2d_forward",
      fbgemm_gpu::jagged_index_add_2d_forward_cpu);
  DISPATCH_TO_CPU(
      "jagged_1d_to_truncated_values",
      fbgemm_gpu::jagged_1d_to_truncated_values_cpu);
  DISPATCH_TO_CPU(
      "masked_select_jagged_1d", fbgemm_gpu::masked_select_jagged_1d);
  DISPATCH_TO_CPU("jagged_softmax", fbgemm_gpu::jagged_softmax);
  DISPATCH_TO_CPU("jagged_softmax_forward", fbgemm_gpu::jagged_softmax_forward);
  DISPATCH_TO_CPU(
      "jagged_softmax_backward", fbgemm_gpu::jagged_softmax_backward);
  DISPATCH_TO_CPU("jagged_jagged_bmm", fbgemm_gpu::jagged_jagged_bmm);
  DISPATCH_TO_CPU(
      "jagged_jagged_bmm_forward", fbgemm_gpu::jagged_jagged_bmm_forward);
  DISPATCH_TO_CPU("jagged_dense_bmm", fbgemm_gpu::jagged_dense_bmm);
  DISPATCH_TO_CPU(
      "jagged_dense_bmm_forward", fbgemm_gpu::jagged_dense_bmm_forward);
  DISPATCH_TO_CPU("jagged_slice_forward", fbgemm_gpu::jagged_slice_forward_cpu);
}
