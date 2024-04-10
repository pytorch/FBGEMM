/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/embedding_forward_split_cpu.h"
#include "fbgemm/FbgemmEmbedding.h"
#include "fbgemm/Types.h"
#include "fbgemm/Utils.h"
#include "fbgemm_gpu/cpu_utils.h"
#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"
#ifdef FBCODE_CAFFE2
#include <libdivide.h>
#include "folly/container/F14Map.h"
#else
#include <omp.h>
#endif

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

template <typename weights_t, typename ind_weights_t, typename output_t>
void split_embedding_forward_cpu_kernel(
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    c10::SymInt total_D,
    Tensor hash_size_cumsum,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    Tensor output) {
  int64_t T = D_offsets.numel() - 1;
  TORCH_CHECK_GT(T, 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK_GE(B, 0);

  TORCH_CHECK(weights.is_contiguous());
  indices = indices.contiguous();
  offsets = offsets.contiguous();
  if (indice_weights.defined()) {
    indice_weights = indice_weights.contiguous();
  }

  const auto D_offsets_data = D_offsets.accessor<int, 1>();
  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  const auto indices_data = indices.data_ptr<int64_t>();
  const auto offsets_data = offsets.data_ptr<int64_t>();
  const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();

  const auto weights_data = weights.data_ptr<weights_t>();
  // If indice_weights not defined, then this accessor won't be used.
  // The else condition is just to make compiler happy
  const auto indice_weights_data = indice_weights.defined()
      ? indice_weights.data_ptr<ind_weights_t>()
      : nullptr;

  auto output_data = output.data_ptr<output_t>();
  auto output_stride = output.size(1);

  constexpr bool use_fbgemm = (std::is_same<weights_t, float>::value ||
                               std::is_same<weights_t, at::Half>::value ||
                               std::is_same<weights_t, uint8_t>::value) &&
      std::is_same<output_t, float>::value &&
      std::is_same<ind_weights_t, float>::value;

  at::parallel_for(0, B, 0, [&](int64_t b_begin, int64_t b_end) {
    for (const auto t : c10::irange(T)) {
      const auto D_begin = D_offsets_data[t];
      const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
      const auto table_begin = weights_offsets_data[t];

      int64_t hash_size;
      int t_temp = t + 1;
      do {
        hash_size = hash_size_cumsum_data[t_temp] - hash_size_cumsum_data[t];
        ++t_temp;
      } while (hash_size == 0);

      bool success = true;
      if (use_fbgemm) {
        using fbgemm_weight_t = typename std::conditional<
            std::is_same<weights_t, at::Half>::value,
            fbgemm::float16,
            weights_t>::type;
        auto kernel = fbgemm::GenerateEmbeddingSpMDMWithStrides<
            fbgemm_weight_t,
            /*IndexType=*/int64_t,
            /*OffsetType=*/int64_t>(
            D,
            indice_weights.defined(),
            static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN,
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
                ? reinterpret_cast<const float*>(
                      indice_weights_data + *offsets_begin_ptr)
                : nullptr,
            reinterpret_cast<float*>(
                output_data + b_begin * output_stride + D_begin));
      } else {
        at::acc_type<output_t, true> output_buf[D];
        for (const auto b : c10::irange(b_begin, b_end)) {
          const auto pool_begin = offsets_data[t * B + b];
          const auto pool_end = offsets_data[t * B + b + 1];
          const auto L = pool_end - pool_begin;
          memset(output_buf, 0, D * sizeof(at::acc_type<output_t, true>));
          for (const auto p : c10::irange(pool_begin, pool_end)) {
            int64_t idx = indices_data[p];
            if (idx < 0 || idx >= hash_size) {
              success = false;
              break;
            }
            const int64_t embedding_begin = table_begin + idx * D;
            for (const auto d : c10::irange(D)) {
              output_buf[d] +=
                  (indice_weights.defined()
                       ? static_cast<at::acc_type<output_t, true>>(
                             weights_data[embedding_begin + d]) *
                           static_cast<at::acc_type<output_t, true>>(
                               indice_weights_data[p])
                       : static_cast<at::acc_type<output_t, true>>(
                             weights_data[embedding_begin + d]));
            }
          }
          const double scale_factor =
              // NOTE: MEAN pooling will not work with indice_weights!
              (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN &&
               !indice_weights.defined() && L > 0)
              ? 1.0 / L
              : 1.0;
          for (const auto d : c10::irange(D)) {
            output_data[b * output_stride + D_begin + d] =
                scale_factor * output_buf[d];
          }
          if (!success) {
            break;
          }
        } // for each b
      } // !use_fbgemm

      if (!success) {
        fbgemm_gpu::report_embedding_error(
            t, B, b_begin, b_end, offsets_data, indices_data, hash_size);
      } // !success
    } // for each t
  }); // parallel for
}

Tensor split_embedding_codegen_forward_cpu(
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    c10::SymInt total_D_,
    Tensor hash_size_cumsum,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t output_dtype) {
  const int64_t total_D = total_D_.guard_int(__FILE__, __LINE__);
  int64_t T = D_offsets.numel() - 1;
  TORCH_CHECK_GT(T, 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK_GE(B, 0);

  Tensor output;
  if (output_dtype == static_cast<int64_t>(SparseType::FP32)) {
    output = at::empty({B, total_D}, weights.options().dtype(at::kFloat));
  } else if (output_dtype == static_cast<int64_t>(SparseType::FP16)) {
    output = at::empty({B, total_D}, weights.options().dtype(at::kHalf));
  } else if (output_dtype == static_cast<int64_t>(SparseType::BF16)) {
    output = at::empty({B, total_D}, weights.options().dtype(at::kBFloat16));
  } else {
    output = at::empty({B, total_D}, weights.options());
  }

  // It is assumed that the indice_weights will always be float
  TORCH_CHECK(
      !indice_weights.defined() || indice_weights.scalar_type() != at::kHalf);
  FBGEMM_DISPATCH_FLOAT_AND_HALF(
      output.scalar_type(), "split_embedding_cpu_forward", [&]() {
        using output_t = scalar_t;
        FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
            weights.scalar_type(), "split_embedding_cpu_forward", [&] {
              using ind_weights_t = std::conditional<
                  std::is_same<scalar_t, double>::value,
                  double,
                  float>::type;
              split_embedding_forward_cpu_kernel<
                  scalar_t,
                  ind_weights_t,
                  output_t>(
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
      });
  return output;
}

Tensor split_embedding_codegen_forward_cpu_meta(
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    c10::SymInt total_D,
    Tensor hash_size_cumsum,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t output_dtype) {
  c10::SymInt T = D_offsets.sym_numel() - 1;
  TORCH_CHECK_GT(T, 0);
  // offsets = [T x B  + 1]
  c10::SymInt B = (offsets.sym_size(0) - 1) / T;
  TORCH_CHECK_GE(B, 0);

  Tensor output;
  if (output_dtype == static_cast<int64_t>(SparseType::FP32)) {
    output =
        at::empty_symint({B, total_D}, weights.options().dtype(at::kFloat));
  } else if (output_dtype == static_cast<int64_t>(SparseType::FP16)) {
    output = at::empty_symint({B, total_D}, weights.options().dtype(at::kHalf));
  } else if (output_dtype == static_cast<int64_t>(SparseType::BF16)) {
    output =
        at::empty_symint({B, total_D}, weights.options().dtype(at::kBFloat16));
  } else {
    output = at::empty_symint({B, total_D}, weights.options());
  }

  // It is assumed that the indice_weights will always be float
  TORCH_CHECK(
      !indice_weights.defined() || indice_weights.scalar_type() != at::kHalf);
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
  TORCH_CHECK_GT(T, 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK_GE(B, 0);

  const auto D_offsets_data = D_offsets.accessor<int, 1>();
  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  const auto offsets_data = offsets.accessor<int64_t, 1>();
  const auto indices_data = indices.accessor<int64_t, 1>();

  const auto weights_data = weights.accessor<weights_t, 1>();
  const auto grad_output_data = grad_output.accessor<grad_t, 2>();
  auto grad_indice_weights_data =
      grad_indice_weights.accessor<at::acc_type<grad_t, true>, 1>();

  at::parallel_for(0, B, 0, [&](int64_t b_begin, int64_t b_end) {
    for (const auto t : c10::irange(T)) {
      if (feature_requires_grad.defined() &&
          !feature_requires_grad[t].is_nonzero()) {
        // NOTE: skip if the table does not require gradient computation!
        continue;
      }
      const auto D_begin = D_offsets_data[t];
      const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
      const auto table_begin = weights_offsets_data[t];
      for (const auto b : c10::irange(b_begin, b_end)) {
        const auto pool_begin = offsets_data[t * B + b];
        const auto pool_end = offsets_data[t * B + b + 1];
        for (const auto p : c10::irange(pool_begin, pool_end)) {
          const int64_t embedding_begin = table_begin + indices_data[p] * D;
          for (const auto d : c10::irange(D)) {
            grad_indice_weights_data[p] +=
                static_cast<at::acc_type<weights_t, true>>(
                    grad_output_data[b][D_begin + d]) *
                weights_data[embedding_begin + d];
          }
        }
      }
    } // for each t
  }); // parallel for
}

Tensor split_embedding_codegen_grad_indice_weights_cpu(
    Tensor grad_output,
    Tensor weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor indices,
    Tensor offsets,
    Tensor feature_requires_grad) {
  auto grad_indice_weights = zeros_like(
      indices,
      indices.options().dtype(
          at::toAccumulateType(grad_output.scalar_type(), true)));
  FBGEMM_DISPATCH_FLOAT_AND_HALF(
      grad_output.scalar_type(),
      "split_embedding_grad_indice_weights_cpu_outer",
      [&] {
        using grad_t = scalar_t;
        FBGEMM_DISPATCH_FLOAT_AND_HALF(
            weights.scalar_type(),
            "split_embedding_grad_indice_weights_cpu",
            [&] {
              using weights_t = scalar_t;
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

namespace {

template <typename scalar_t, bool IS_VALUE_PAIR>
void csr2csc_template_(
    HyperCompressedSparseColumn& csc,
    int B,
    const at::TensorAccessor<int64_t, 1>& csr_offsets,
    const at::TensorAccessor<int64_t, 1>& csr_indices,
    const at::TensorAccessor<scalar_t, 1>& csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset,
    int64_t num_embeddings) {
  csc.num_non_zero_columns = 0;
  int64_t nnz = csr_offsets[table_to_feature_offset[1] * B] -
      csr_offsets[table_to_feature_offset[0] * B];
  if (nnz == 0) {
    return;
  }
  csc.row_indices =
      static_cast<int*>(fbgemm::fbgemmAlignedAlloc(64, nnz * sizeof(int)));
  bool has_weights = csr_weights.data() != nullptr;
  if (IS_VALUE_PAIR) {
    csc.weights = static_cast<float*>(
        fbgemm::fbgemmAlignedAlloc(64, nnz * sizeof(float)));
  }

  int column_ptr_curr = 0;
  bool is_shared_table =
      table_to_feature_offset[1] > table_to_feature_offset[0] + 1;
  auto NS = csr_offsets[table_to_feature_offset[1] * B] -
      csr_offsets[table_to_feature_offset[0] * B];
  int num_non_empty_segments = 0;

  using pair_t = std::pair<int, scalar_t>;
  using value_t = typename std::conditional<IS_VALUE_PAIR, pair_t, int>::type;

  csc.column_segment_ids =
      static_cast<int*>(fbgemm::fbgemmAlignedAlloc(64, nnz * sizeof(int)));
  int* tmpBufKeys =
      static_cast<int*>(fbgemm::fbgemmAlignedAlloc(64, NS * sizeof(int)));
  value_t* tmpBufValues = static_cast<value_t*>(
      fbgemm::fbgemmAlignedAlloc(64, NS * sizeof(value_t)));
  int* tmpBuf1Keys =
      static_cast<int*>(fbgemm::fbgemmAlignedAlloc(64, NS * sizeof(int)));
  value_t* tmpBuf1Values = static_cast<value_t*>(
      fbgemm::fbgemmAlignedAlloc(64, NS * sizeof(value_t)));

  const auto FBo = csr_offsets[table_to_feature_offset[0] * B];
  for (int feature = table_to_feature_offset[0];
       feature < table_to_feature_offset[1];
       ++feature) {
    const auto FBs = (feature - table_to_feature_offset[0]) * B;
#pragma omp parallel for
    for (int b = 0; b < B; ++b) {
      const auto FBb = feature * B + b;
      int64_t pool_begin = csr_offsets[FBb];
      int64_t pool_end = csr_offsets[FBb + 1];
      int64_t L = pool_end - pool_begin;
      // MEAN pooling will not work with indice_weights!
      double scale_factor =
          (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN &&
           !has_weights && L > 0)
          ? 1.0 / L
          : 1.0;
      for (const auto p : c10::irange(pool_begin, pool_end)) {
        tmpBufKeys[p - FBo] = csr_indices[p];
        if (IS_VALUE_PAIR) {
          reinterpret_cast<pair_t*>(tmpBufValues)[p - FBo] = std::make_pair(
              FBs + b, scale_factor * (has_weights ? csr_weights[p] : 1.0f));
        } else {
          reinterpret_cast<int*>(tmpBufValues)[p - FBo] = FBs + b;
        }
      }
    }
  }

  int* sorted_col_row_index_keys;
  value_t* sorted_col_row_index_values;
  std::tie(sorted_col_row_index_keys, sorted_col_row_index_values) =
      fbgemm::radix_sort_parallel(
          tmpBufKeys,
          tmpBufValues,
          tmpBuf1Keys,
          tmpBuf1Values,
          NS,
          num_embeddings);

  int max_thds = omp_get_max_threads();
  int num_uniq[max_thds][64];
  for (const auto i : c10::irange(max_thds)) {
    num_uniq[i][0] = 0;
  }

  int U = 0;
  if (at::get_num_threads() > 1) {
    // This block is not needed for single thread
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      num_uniq[tid][0] = 0;
#pragma omp for schedule(static)
      for (int i = 1; i < NS; i++) {
        if (sorted_col_row_index_keys[i] != sorted_col_row_index_keys[i - 1]) {
          num_uniq[tid][0]++;
        }
      }
    }
    num_uniq[0][0] += 1;
    for (const auto i : c10::irange(1, max_thds)) {
      num_uniq[i][0] += num_uniq[i - 1][0];
    }
    U = num_uniq[max_thds - 1][0];
  }

  csc.column_segment_ptr =
      static_cast<int*>(fbgemm::fbgemmAlignedAlloc(64, (NS + 1) * sizeof(int)));
  csc.column_segment_indices =
      static_cast<int*>(fbgemm::fbgemmAlignedAlloc(64, NS * sizeof(int)));
  csc.column_segment_ptr[0] = 0;
  const pair_t* sorted_col_row_index_values_pair =
      reinterpret_cast<const pair_t*>(sorted_col_row_index_values);
  const int* sorted_col_row_index_values_int =
      reinterpret_cast<const int*>(sorted_col_row_index_values);
  if (IS_VALUE_PAIR) {
    csc.row_indices[0] = sorted_col_row_index_values_pair[0].first % B;
    csc.weights[0] = sorted_col_row_index_values_pair[0].second;
    csc.column_segment_ids[0] = sorted_col_row_index_values_pair[0].first / B;
  } else {
    csc.row_indices[0] = sorted_col_row_index_values_int[0] % B;
    csc.column_segment_ids[0] = sorted_col_row_index_values_int[0] / B;
  }
  csc.column_segment_indices[0] = sorted_col_row_index_keys[0];

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int* tstart =
        (tid == 0 ? csc.column_segment_indices + 1
                  : csc.column_segment_indices + num_uniq[tid - 1][0]);

    int* t_offs =
        (tid == 0 ? csc.column_segment_ptr + 1
                  : csc.column_segment_ptr + num_uniq[tid - 1][0]);

    if (!IS_VALUE_PAIR && !is_shared_table) {
      // For non shared table, no need for computing modulo.
      // As an optimization, pointer swap instead of copying.
#pragma omp master
      std::swap(
          csc.row_indices,
          *reinterpret_cast<int**>(
              sorted_col_row_index_values == tmpBufValues ? &tmpBufValues
                                                          : &tmpBuf1Values));
    } else {
#ifdef FBCODE_CAFFE2
      libdivide::divider<int> divisor(B);
#endif

#pragma omp for schedule(static)
      for (int i = 1; i < NS; ++i) {
        int v = IS_VALUE_PAIR ? sorted_col_row_index_values_pair[i].first
                              : sorted_col_row_index_values_int[i];
#ifdef FBCODE_CAFFE2
        int q = v / divisor;
#else
        int q = v / B;
#endif
        csc.column_segment_ids[i] = q;
        csc.row_indices[i] = v - q * B;
        if (IS_VALUE_PAIR) {
          csc.weights[i] = sorted_col_row_index_values_pair[i].second;
        }
      }
    }

#pragma omp for schedule(static)
    for (int i = 1; i < NS; ++i) {
      if (sorted_col_row_index_keys[i] != sorted_col_row_index_keys[i - 1]) {
        *tstart = sorted_col_row_index_keys[i];
        *t_offs = i;
        tstart++;
        t_offs++;
      }
    }

    if (at::get_num_threads() == 1 && tid == 0) {
      // Special handling of single thread case
      U = t_offs - csc.column_segment_ptr;
    }

  } // omp parallel

  csc.num_non_zero_columns = U;
  csc.column_segment_ptr[U] = NS;
  column_ptr_curr += NS;

  fbgemm::fbgemmAlignedFree(tmpBufKeys);
  fbgemm::fbgemmAlignedFree(tmpBufValues);
  fbgemm::fbgemmAlignedFree(tmpBuf1Keys);
  fbgemm::fbgemmAlignedFree(tmpBuf1Values);

  assert(column_ptr_curr == nnz);
}

#define INSTANTIATE_BATCHED_CSR2CSC(SCALAR_T)             \
  template void csr2csc_template_<SCALAR_T, true>(        \
      HyperCompressedSparseColumn & csc,                  \
      int B,                                              \
      const at::TensorAccessor<int64_t, 1>& csr_offsets,  \
      const at::TensorAccessor<int64_t, 1>& csr_indices,  \
      const at::TensorAccessor<SCALAR_T, 1>& csr_weights, \
      int64_t pooling_mode,                               \
      const int* table_to_feature_offset,                 \
      int64_t num_embeddings);                            \
                                                          \
  template void csr2csc_template_<SCALAR_T, false>(       \
      HyperCompressedSparseColumn & csc,                  \
      int B,                                              \
      const at::TensorAccessor<int64_t, 1>& csr_offsets,  \
      const at::TensorAccessor<int64_t, 1>& csr_indices,  \
      const at::TensorAccessor<SCALAR_T, 1>& csr_weights, \
      int64_t pooling_mode,                               \
      const int* table_to_feature_offset,                 \
      int64_t num_embeddings);

INSTANTIATE_BATCHED_CSR2CSC(float)
INSTANTIATE_BATCHED_CSR2CSC(double)
#undef INSTANTIATE_BATCHED_CSR2CSC

} // namespace

template <typename scalar_t>
void csr2csc(
    HyperCompressedSparseColumn& csc,
    int B,
    const at::TensorAccessor<int64_t, 1>& csr_offsets,
    const at::TensorAccessor<int64_t, 1>& csr_indices,
    const at::TensorAccessor<scalar_t, 1>& csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset,
    int64_t num_embeddings) {
  bool has_weights = csr_weights.data() != nullptr;
  if (has_weights ||
      static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN) {
    csr2csc_template_<scalar_t, /*IS_VALUE_PAIR=*/true>(
        csc,
        B,
        csr_offsets,
        csr_indices,
        csr_weights,
        pooling_mode,
        table_to_feature_offset,
        num_embeddings);
  } else {
    csr2csc_template_<scalar_t, /*IS_VALUE_PAIR=*/false>(
        csc,
        B,
        csr_offsets,
        csr_indices,
        csr_weights,
        pooling_mode,
        table_to_feature_offset,
        num_embeddings);
  }
}

template void csr2csc<float>(
    HyperCompressedSparseColumn& csc,
    int B,
    const at::TensorAccessor<int64_t, 1>& csr_offsets,
    const at::TensorAccessor<int64_t, 1>& csr_indices,
    const at::TensorAccessor<float, 1>& csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset,
    int64_t num_embeddings);

template void csr2csc<double>(
    HyperCompressedSparseColumn& csc,
    int B,
    const at::TensorAccessor<int64_t, 1>& csr_offsets,
    const at::TensorAccessor<int64_t, 1>& csr_indices,
    const at::TensorAccessor<double, 1>& csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset,
    int64_t num_embeddings);

} // namespace internal

namespace {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "split_embedding_codegen_grad_indice_weights_cpu(Tensor grad_output, Tensor weights, Tensor weights_offsets, Tensor D_offsets, Tensor indices, Tensor offsets, Tensor feature_requires_grad) -> Tensor");
  DISPATCH_TO_CPU(
      "split_embedding_codegen_grad_indice_weights_cpu",
      split_embedding_codegen_grad_indice_weights_cpu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "split_embedding_codegen_forward_cpu(Tensor weights, Tensor weights_offsets, Tensor D_offsets, SymInt total_D, Tensor hash_size_cumsum, Tensor indices, Tensor offsets, int pooling_mode, Tensor indice_weights, int output_dtype) -> Tensor");
  DISPATCH_TO_CPU(
      "split_embedding_codegen_forward_cpu",
      split_embedding_codegen_forward_cpu);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl(
      "split_embedding_codegen_forward_cpu",
      &split_embedding_codegen_forward_cpu_meta);
}

} // namespace
