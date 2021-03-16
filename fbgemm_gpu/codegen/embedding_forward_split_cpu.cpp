/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "codegen/embedding_forward_split_cpu.h"

#include <ATen/AccumulateType.h>

using namespace at;

template <typename weights_t, typename ind_weights_t, typename output_t>
void split_embedding_forward_cpu_kernel(
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    Tensor output) {
  int64_t T = D_offsets.numel() - 1;
  TORCH_CHECK(T > 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B > 0);

  const auto D_offsets_data = D_offsets.accessor<int, 1>();
  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  const auto offsets_data = offsets.accessor<int64_t, 1>();
  const auto indices_data = indices.accessor<int64_t, 1>();

  const auto weights_data = weights.accessor<weights_t, 1>();
  // If indice_weights not defined, then this accessor won't be used.
  // The else condition is just to make compiler happy
  const auto indice_weights_data = indice_weights.defined()
      ? indice_weights.accessor<ind_weights_t, 1>()
      : TensorAccessor<ind_weights_t, 1>(nullptr, nullptr, nullptr);

  auto output_data = output.accessor<output_t, 2>();

  at::parallel_for(0, T * B, 0, [&](int64_t tb_begin, int64_t tb_end) {
    int t_begin = tb_begin / B;
    int t_end = (tb_end + B - 1) / B;
    for (int t = t_begin; t < t_end; ++t) {
      const auto D_begin = D_offsets_data[t];
      const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
      const auto table_begin = weights_offsets_data[t];

      int b_begin = (t == t_begin) ? tb_begin % B : 0;
      int b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
      for (int b = b_begin; b < b_end; ++b) {
        const auto pool_begin = offsets_data[t * B + b];
        const auto pool_end = offsets_data[t * B + b + 1];
        const auto L = pool_end - pool_begin;
        const double scale_factor =
            // NOTE: MEAN pooling will not work with indice_weights!
            (pooling_mode == MEAN && !indice_weights.defined() && L > 0)
            ? 1.0 / L
            : 1.0;
        for (auto p = pool_begin; p < pool_end; ++p) {
          const int64_t embedding_begin = table_begin + indices_data[p] * D;
          for (int64_t d = 0; d < D; ++d) {
            output_data[b][D_begin + d] += scale_factor *
                (indice_weights.defined()
                     ? static_cast<output_t>(
                           weights_data[embedding_begin + d]) *
                         static_cast<output_t>(indice_weights_data[p])
                     : static_cast<output_t>(
                           weights_data[embedding_begin + d]));
          }
        }
      } // for each b
    } // for each t
  }); // parallel for
}

Tensor split_embedding_codegen_forward_cpu(
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights) {
  int64_t T = D_offsets.numel() - 1;
  TORCH_CHECK(T > 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B > 0);

  Tensor output;
  if (weights.scalar_type() == at::kHalf) {
    output = zeros({B, total_D}, weights.options().dtype(at::kFloat));
  } else {
    output = zeros({B, total_D}, weights.options());
  }

  // It is assumed that the indice_weights will always be float
  TORCH_CHECK(
      !indice_weights.defined() || indice_weights.scalar_type() != at::kHalf);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.scalar_type(), "split_embedding_cpu_forward", [&]() {
        split_embedding_forward_cpu_kernel<
            scalar_t,
            acc_type<scalar_t, true>,
            acc_type<scalar_t, true>>(
            weights,
            weights_offsets,
            D_offsets,
            total_D,
            indices,
            offsets,
            pooling_mode,
            indice_weights,
            output);
      });

  return output;
}

template <typename weights_t, typename grad_t>
void split_embedding_grad_indice_weights_cpu_kernel(
    Tensor grad_output,
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor indices,
    Tensor offsets,
    Tensor feature_requires_grad,
    Tensor grad_indice_weights) {
  int64_t T = D_offsets.numel() - 1;
  TORCH_CHECK(T > 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B > 0);

  const auto D_offsets_data = D_offsets.accessor<int, 1>();
  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  const auto offsets_data = offsets.accessor<int64_t, 1>();
  const auto indices_data = indices.accessor<int64_t, 1>();

  auto weights_data = weights.accessor<weights_t, 1>();
  auto grad_output_data = grad_output.accessor<grad_t, 2>();
  auto grad_indice_weights_data = grad_indice_weights.accessor<grad_t, 1>();
  for (int64_t t = 0; t < T; ++t) {
    if (feature_requires_grad.defined() &&
        !feature_requires_grad[t].is_nonzero()) {
      // NOTE: skip if the table does not require gradient computation!
      continue;
    }
    const auto D_begin = D_offsets_data[t];
    const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
    const auto table_begin = weights_offsets_data[t];
    at::parallel_for(0, B, 0, [&](int64_t b_begin, int64_t b_end) {
      for (int64_t b = b_begin; b < b_end; ++b) {
        const auto pool_begin = offsets_data[t * B + b];
        const auto pool_end = offsets_data[t * B + b + 1];
        for (auto p = pool_begin; p < pool_end; ++p) {
          const int64_t embedding_begin = table_begin + indices_data[p] * D;
          for (int64_t d = 0; d < D; ++d) {
            grad_indice_weights_data[p] += grad_output_data[b][D_begin + d] *
                weights_data[embedding_begin + d];
          }
        }
      }
    });
  }
}

Tensor split_embedding_codegen_grad_indice_weights_cpu(
    Tensor grad_output,
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor indices,
    Tensor offsets,
    Tensor feature_requires_grad) {
  auto grad_indice_weights =
      zeros_like(indices, indices.options().dtype(grad_output.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.scalar_type(), "split_embedding_grad_indice_weights_cpu", [&]() {
        using weights_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_output.scalar_type(),
            "split_embedding_grad_indice_weights_cpu_inner",
            [&]() {
              using grad_t = scalar_t;

              split_embedding_grad_indice_weights_cpu_kernel<weights_t, grad_t>(
                  grad_output,
                  weights,
                  weights_offsets,
                  D_offsets,
                  indices,
                  offsets,
                  feature_requires_grad,
                  grad_indice_weights);
            });
      });

  return grad_indice_weights;
}
