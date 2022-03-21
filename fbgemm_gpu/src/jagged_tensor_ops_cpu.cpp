/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm_gpu {

using Tensor = at::Tensor;

namespace {

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
    TORCH_CHECK(x_offsets[d].numel() == num_lengths_expected + 1);
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

  TORCH_CHECK(x_offsets.size() == static_cast<size_t>(NUM_JAGGED_DIM));

  const int outer_dense_size = y.size(0);
  TORCH_CHECK(outer_dense_size == x_offsets[0].numel() - 1);
  TORCH_CHECK(!NO_INNER_DENSE || y.size(-1) == 1);
  const int inner_dense_size = NO_INNER_DENSE ? 1 : y.size(-1);
  TORCH_CHECK(inner_dense_size == x_values.size(-1));
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
        for (; jiidx < end - begin; ++jiidx) {
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

Tensor jagged_to_padded_dense(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const std::vector<int64_t>& max_lengths,
    const int64_t padding_value = 0) {
  const size_t num_jagged_dim = offsets.size();
  TORCH_CHECK(max_lengths.size() == num_jagged_dim);

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

  return padded_values;
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

Tensor jagged_dense_elementwise_add(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  Tensor output;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x_values.scalar_type(), "jagged_scalars", [&] {
        output = jagged_dense_elementwise_dense_output_<scalar_t>(
            x_values, x_offsets, y, [](scalar_t x, scalar_t y) -> scalar_t {
              return x + y;
            });
      });
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

  TORCH_CHECK(x_offsets.size() == static_cast<size_t>(NUM_JAGGED_DIM));

  const int outer_dense_size = y.size(0);
  TORCH_CHECK(outer_dense_size == x_offsets[0].numel() - 1);
  TORCH_CHECK(!NO_INNER_DENSE || y.size(-1) == 1);
  const int inner_dense_size = NO_INNER_DENSE ? 1 : y.size(-1);
  TORCH_CHECK(inner_dense_size == x_values.size(-1));
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
      int jiidx = 0;
      if (!is_zero) {
        const int begin = x_offsets_accessors[NUM_JAGGED_DIM - 1][offset_base];
        const int end =
            x_offsets_accessors[NUM_JAGGED_DIM - 1][offset_base + 1];
        for (; jiidx < end - begin; ++jiidx) {
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

template <typename scalar_t, typename F>
Tensor jagged_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    F f) {
  Tensor output = at::empty_like(x_values);
  jagged_dense_elementwise_jagged_output_<scalar_t>(
      x_values, x_offsets, y, output, f);
  return output;
}

std::tuple<Tensor, std::vector<Tensor>> jagged_dense_elementwise_mul(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  Tensor output;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x_values.scalar_type(), "jagged_scalars", [&] {
        output = jagged_dense_elementwise_jagged_output_<scalar_t>(
            x_values, x_offsets, y, [](scalar_t x, scalar_t y) -> scalar_t {
              return x * y;
            });
      });
  return {output, x_offsets};
}

} // namespace

Tensor
jagged_2d_to_dense_forward_cpu(Tensor values, Tensor offsets, int64_t max_L) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  return jagged_to_padded_dense(values, {offsets}, {max_L});
}

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
      "jagged_to_padded_dense(Tensor values, Tensor[] offsets, int[] max_lengths, int padding_value = 0) -> Tensor");
  // jagged + dense -> dense
  m.def(
      "jagged_dense_elementwise_add(Tensor x_values, Tensor[] x_offsets, Tensor y) -> Tensor");
  // jagged * dense -> jagged (its offsets is same as x_offsets)
  m.def(
      "jagged_dense_elementwise_mul(Tensor x_values, Tensor[] x_offsets, Tensor y) -> (Tensor, Tensor[])");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU(
      "jagged_2d_to_dense", fbgemm_gpu::jagged_2d_to_dense_forward_cpu);
  DISPATCH_TO_CPU("jagged_1d_to_dense", fbgemm_gpu::jagged_1d_to_dense_cpu);
  DISPATCH_TO_CPU("jagged_to_padded_dense", fbgemm_gpu::jagged_to_padded_dense);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_add", fbgemm_gpu::jagged_dense_elementwise_add);
  DISPATCH_TO_CPU(
      "jagged_dense_elementwise_mul", fbgemm_gpu::jagged_dense_elementwise_mul);
}
