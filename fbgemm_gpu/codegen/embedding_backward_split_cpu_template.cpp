/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <map>
#include <tuple>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "codegen/embedding_forward_split_cpu.h"

using namespace at;

// The template for exact optimizers
{{ "void" if not dense else "Tensor" }}  split_embedding_backward_codegen_{{ optimizer }}_cpu(
    Tensor grad_output,
    Tensor host_weights,
    {% if not dense %}
    Tensor weights_placements,
    {% endif %}
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    {% if not dense %}
    bool stochastic_rounding,
    {% endif %}
    {{ args.split_function_args | join(", ") }}
) {

  int64_t T = D_offsets.numel() - 1;
  TORCH_CHECK(T > 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B > 0);

  {% if not dense %}
  offsets.contiguous();
  indices.contiguous();
  indice_weights.contiguous();
  {% endif %}

  const auto D_offsets_data = D_offsets.accessor<int, 1>();
  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  const auto offsets_data = offsets.accessor<int64_t, 1>();
  const auto indices_data = indices.accessor<int64_t, 1>();
  const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();
  {% if "momentum1_offsets" in args.split_function_arg_names %}
  const auto momentum1_offsets_data = momentum1_offsets.accessor<int64_t, 1>();
  {% endif %}
  {% if "momentum2_offsets" in args.split_function_arg_names %}
  const auto momentum2_offsets_data = momentum2_offsets.accessor<int64_t, 1>();
  {% endif %}

  int num_tables = 0; // # of physical tables
  int table_to_feature_offset[T + 1];
  table_to_feature_offset[0] = 0;
  for (int feature = 0; feature < T - 1; ++feature) {
    if (hash_size_cumsum_data[feature + 1] != hash_size_cumsum_data[feature]) {
      ++num_tables;
      table_to_feature_offset[num_tables] = feature + 1;
    }
  }
  ++num_tables;
  table_to_feature_offset[num_tables] = T;

  TORCH_CHECK(host_weights.dim() == 1);

  {% if not dense %}
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      host_weights.scalar_type(), "split_embedding_backward_exact_cpu", [&]() {
        {{ args.split_host_accessor_constructors | join("; ") }}

        using grad_t = acc_type<scalar_t, true>;
        ::internal::BatchedHyperCompressedSparseColumn batched_csc;
        ::internal::batched_csr2csc(
            batched_csc,
            num_tables,
            B,
            offsets.data_ptr<int64_t>(),
            indices.data_ptr<int64_t>(),
            indice_weights.defined() ? indice_weights.data_ptr<grad_t>()
                                     : nullptr,
            pooling_mode,
            table_to_feature_offset);
        std::vector<int>& table_ptr = batched_csc.table_ptr;
        std::vector<int>& column_ptr = batched_csc.column_ptr;

        auto grad_output_data = grad_output.accessor<grad_t, 2>();
        auto host_weights_data = host_weights.accessor<scalar_t, 1>();

        const bool has_weights = !batched_csc.weights.empty();

        for (int t = 0; t < num_tables; ++t) {
          int feature_begin = table_to_feature_offset[t];
          const auto D_begin = D_offsets_data[feature_begin];
          const auto D =
              D_offsets_data[feature_begin + 1] - D_offsets_data[feature_begin];
          const auto table_begin = weights_offsets_data[feature_begin];
          grad_t grad_buffer[D];
          for (int c = table_ptr[t]; c < table_ptr[t + 1]; ++c) {
            memset(grad_buffer, 0, D * sizeof(grad_t));
            const int64_t embedding_begin =
                table_begin + batched_csc.column_indices[c] * D;
            for (int r = column_ptr[c]; r < column_ptr[c + 1]; ++r) {
              int f_times_b = batched_csc.row_indices[r];
              int feature = f_times_b / B;
              int b = f_times_b % B;
              int D_offset = D_begin + (feature - feature_begin) * D;
              for (int64_t d = 0; d < D; ++d) {
                grad_buffer[d] += has_weights
                    ? grad_output_data[b][D_offset + d] * batched_csc.weights[r]
                    : grad_output_data[b][D_offset + d];
              }
            }
            {{ split_weight_update_cpu }}
          }
        } // for each table
      });


  return;
  {% endif %}

  {% if dense %}
  // When input is dense enough, avoid sorting and just treat as dense.
  auto grad = zeros_like(host_weights, grad_output.dtype());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      host_weights.scalar_type(), "split_embedding_backward_exact_cpu", [&]() {
        {{ args.split_host_accessor_constructors | join("; ") }}

        using grad_t = acc_type<scalar_t, true>;
        auto grad_data = grad.data_ptr<grad_t>();
        const auto indice_weights_data = indice_weights.defined()
            ?
            // If indice_weights are not defined, then this accessor won't be
            // used
            indice_weights.accessor<grad_t, 1>()
            : grad.accessor<grad_t, 1>(); // this is just to make compiler
                                          // happy

        auto grad_output_data = grad_output.accessor<grad_t, 2>();
        auto host_weights_data = host_weights.accessor<scalar_t, 1>();

        at::parallel_for(0, num_tables, 0, [&](int64_t t_begin, int64_t t_end) {
          for (int64_t t = table_to_feature_offset[t_begin];
               t < table_to_feature_offset[t_end];
               ++t) {
            const auto D_begin = D_offsets_data[t];
            const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
            const auto table_begin = weights_offsets_data[t];
            for (int64_t b = 0; b < B; ++b) {
              const auto pool_begin = offsets_data[t * B + b];
              const auto pool_end = offsets_data[t * B + b + 1];
              const auto L = pool_end - pool_begin;
              const double scale_factor =
                  // NOTE: MEAN pooling will not work with indice_weights!
                  (pooling_mode == MEAN && !indice_weights.defined() && L > 0)
                  ? 1.0 / L
                  : 1.0;
              for (auto p = pool_begin; p < pool_end; ++p) {
                const int64_t embedding_begin =
                    table_begin + indices_data[p] * D;
                for (int64_t d = 0; d < D; ++d) {
                  grad_data[embedding_begin + d] += scale_factor *
                      (indice_weights.defined()
                           ? grad_output_data[b][D_begin + d] *
                               indice_weights_data[p]
                           : grad_output_data[b][D_begin + d]);
                }
              }
            }

          int64_t embedding_end =
              t == T - 1 ? host_weights.numel() : weights_offsets_data[t + 1];
          for (int64_t embedding_begin = table_begin;
               embedding_begin < embedding_end;
               embedding_begin += D) {
            const grad_t* grad_buf = grad_data + embedding_begin;
            {{ split_weight_update_cpu }}
          }
        }
        });
      });

  return grad;
  {% endif %}
}
