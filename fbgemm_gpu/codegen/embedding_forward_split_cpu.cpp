/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "codegen/embedding_forward_split_cpu.h"
#include "fbgemm/FbgemmEmbedding.h"
#include "fbgemm/Types.h"

#include <ATen/AccumulateType.h>

using namespace at;

namespace internal {
// A helper trait to handle that fbgemm doesn't support double precision
template <typename T>
struct double2float {
  using type = T;
};

template <>
struct double2float<double> {
  using type = float;
};

template <typename T>
struct half2float16 {
  using type = T;
};

template <>
struct half2float16<at::Half> {
  using type = fbgemm::float16;
};

} // namespace internal

namespace {
void report_error_(
    int t,
    int B,
    int b_begin,
    int b_end,
    const int64_t* offsets_data,
    const int64_t* indices_data,
    int64_t hash_size) {
  for (int b = b_begin; b < b_end; ++b) {
    const auto pool_begin = offsets_data[t * B + b];
    const auto pool_end = offsets_data[t * B + b + 1];
    for (auto p = pool_begin; p < pool_end; ++p) {
      auto idx = indices_data[p];
      TORCH_CHECK(
          0 <= idx && idx < hash_size,
          "Index ",
          p,
          " is out of bouunds: ",
          idx,
          ", range 0 to ",
          hash_size);
    }
  }
}
} // namespace

template <typename weights_t, typename ind_weights_t, typename output_t>
void split_embedding_forward_cpu_kernel(
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    Tensor hash_size_cumsum,
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

  offsets.contiguous();
  indices.contiguous();
  weights.contiguous();
  if (indice_weights.defined()) {
    indice_weights.contiguous();
  }

  const auto D_offsets_data = D_offsets.accessor<int, 1>();
  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();
  const auto offsets_data = offsets.data_ptr<int64_t>();
  const auto indices_data = indices.data_ptr<int64_t>();

  const auto weights_data = weights.data_ptr<weights_t>();
  // If indice_weights not defined, then this accessor won't be used.
  // The else condition is just to make compiler happy
  const auto indice_weights_data = indice_weights.defined()
      ? indice_weights.data_ptr<ind_weights_t>()
      : nullptr;

  auto output_data = output.data_ptr<output_t>();
  auto output_stride = output.size(1);

  constexpr bool use_fbgemm = std::is_same<weights_t, float>::value ||
      std::is_same<weights_t, at::Half>::value;

  at::parallel_for(0, T * B, 0, [&](int64_t tb_begin, int64_t tb_end) {
    int t_begin = tb_begin / B;
    int t_end = (tb_end + B - 1) / B;
    for (int t = t_begin; t < t_end; ++t) {
      const auto D_begin = D_offsets_data[t];
      const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
      const auto table_begin = weights_offsets_data[t];

      int64_t hash_size;
      int t_temp = t + 1;
      do {
        hash_size = hash_size_cumsum_data[t_temp] - hash_size_cumsum_data[t];
        ++t_temp;
      } while (hash_size == 0);

      int b_begin = (t == t_begin) ? tb_begin % B : 0;
      int b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;

      bool success = true;
      if (use_fbgemm) {
        using fbgemm_weight_t =
            typename ::internal::half2float16<weights_t>::type;
        auto kernel = fbgemm::GenerateEmbeddingSpMDMWithOutputStride<
            fbgemm_weight_t,
            /*IndexType=*/int64_t,
            /*OffsetType=*/int64_t>(
            D,
            indice_weights.defined(),
            pooling_mode == MEAN,
            /*prefetch=*/16,
            /*is_weight_positional=*/false,
            /*use_offsets=*/true,
            output_stride);
        auto offsets_begin_ptr = offsets_data + t * B + b_begin;
        auto indices_size = offsets_data[t * B + b_end] - *offsets_begin_ptr;
        success = kernel(
            b_end - b_begin,
            indices_size,
            hash_size,
            reinterpret_cast<const fbgemm_weight_t*>(
                weights_data + table_begin),
            indices_data + *offsets_begin_ptr,
            offsets_begin_ptr,
            indice_weights.defined()
                ? reinterpret_cast<const typename ::internal::double2float<
                      ind_weights_t>::type*>(
                      indice_weights_data + *offsets_begin_ptr)
                : nullptr,
            reinterpret_cast<
                typename ::internal::double2float<output_t>::type*>(
                output_data + b_begin * output_stride + D_begin));
      } else {
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
            int64_t idx = indices_data[p];
            if (idx < 0 || idx >= hash_size) {
              success = false;
              break;
            }
            const int64_t embedding_begin = table_begin + idx * D;
            for (int64_t d = 0; d < D; ++d) {
              output_data[b * output_stride + D_begin + d] += scale_factor *
                  (indice_weights.defined()
                       ? static_cast<output_t>(
                             weights_data[embedding_begin + d]) *
                           static_cast<output_t>(indice_weights_data[p])
                       : static_cast<output_t>(
                             weights_data[embedding_begin + d]));
            }
          }
          if (!success) {
            break;
          }
        } // for each b
      } // !use_fbgemm

      if (!success) {
        report_error_(
            t, B, b_begin, b_end, offsets_data, indices_data, hash_size);
      } // !success
    } // for each t
  }); // parallel for
}

Tensor split_embedding_codegen_forward_cpu(
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    Tensor hash_size_cumsum,
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
            hash_size_cumsum,
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

namespace internal {

template <typename scalar_t>
void batched_csr2csc(
    BatchedHyperCompressedSparseColumn& batched_csc,
    int num_tables, // number of tables, not number of features
    int B,
    // TODO: use accessor for the following 3 parameters
    const int64_t* batched_csr_offsets,
    const int64_t* batched_csr_indices,
    const scalar_t* batched_csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset) {
  batched_csc.num_tables = num_tables;
  batched_csc.table_ptr.resize(num_tables + 1);
  int64_t nnz = batched_csr_offsets[table_to_feature_offset[num_tables] * B];
  batched_csc.row_indices.resize(nnz);
  if (batched_csr_weights || pooling_mode == MEAN) {
    batched_csc.weights.resize(nnz);
  }

  batched_csc.table_ptr.push_back(0);
  batched_csc.column_ptr.push_back(0);
  int column_ptr_curr = 0;
  for (int t = 0; t < num_tables; ++t) {
    std::unordered_map<int64_t, std::vector<std::pair<int, scalar_t>>>
        non_empty_columns;
    for (int feature = table_to_feature_offset[t];
         feature < table_to_feature_offset[t + 1];
         ++feature) {
      for (int b = 0; b < B; ++b) {
        int64_t pool_begin = batched_csr_offsets[feature * B + b];
        int64_t pool_end = batched_csr_offsets[feature * B + b + 1];
        int64_t L = pool_end - pool_begin;
        // MEAN pooling will not work with indice_weights!
        double scale_factor =
            (pooling_mode == MEAN && !batched_csr_weights && L > 0) ? 1.0 / L
                                                                    : 1.0;
        for (int64_t p = pool_begin; p < pool_end; ++p) {
          non_empty_columns[batched_csr_indices[p]].emplace_back(
              feature * B + b,
              scale_factor *
                  (batched_csr_weights ? batched_csr_weights[p] : 1.0f));
        }
      }
    } // for each feature

    batched_csc.table_ptr[t + 1] =
        batched_csc.table_ptr[t] + non_empty_columns.size();
    batched_csc.column_ptr.reserve(batched_csc.table_ptr[t + 1] + 1);
    batched_csc.column_indices.reserve(batched_csc.table_ptr[t + 1]);
    for (auto const& column : non_empty_columns) {
      batched_csc.column_ptr.push_back(column_ptr_curr + column.second.size());
      batched_csc.column_indices.push_back(column.first);

      for (auto const& non_zero : column.second) {
        batched_csc.row_indices[column_ptr_curr] = non_zero.first;
        if (!batched_csc.weights.empty()) {
          batched_csc.weights[column_ptr_curr] = non_zero.second;
        }
        ++column_ptr_curr;
      }
    }
  } // for each matrix (table)

  assert(column_ptr_curr == nnz);
}

template void batched_csr2csc<float>(
    BatchedHyperCompressedSparseColumn& batched_csc,
    int T,
    int B,
    const int64_t* batched_csr_offsets,
    const int64_t* batched_csr_indices,
    const float* batched_csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset);

template void batched_csr2csc<double>(
    BatchedHyperCompressedSparseColumn& batched_csc,
    int T,
    int B,
    const int64_t* batched_csr_offsets,
    const int64_t* batched_csr_indices,
    const double* batched_csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset);

} // namespace internal
