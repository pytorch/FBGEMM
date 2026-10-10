/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/embedding_backward_split_cpu.h"
#include "fbgemm/FbgemmEmbedding.h"
#include "fbgemm/Utils.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/utils/cpu_utils.h"
#ifdef FBCODE_CAFFE2
#include <libdivide.h>
#endif

#include <ATen/Parallel.h>

#include <algorithm>
#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace internal {

namespace {

template <typename index_t, typename scalar_t, bool IS_VALUE_PAIR>
void csr2csc_template_(
    HyperCompressedSparseColumn& csc,
    int B,
    const pta::TensorAccessor<index_t, 1>& csr_offsets,
    const pta::TensorAccessor<index_t, 1>& csr_indices,
    const pta::TensorAccessor<scalar_t, 1>& csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset,
    int64_t num_embeddings) {
  csc.num_non_zero_columns = 0;
  const auto nnz = csr_offsets[(size_t)table_to_feature_offset[1] * B] -
      csr_offsets[(size_t)table_to_feature_offset[0] * B];
  if (nnz == 0) {
    return;
  }
  csc.row_indices = fbgemm::makeAlignedUniquePtr<int>(64, nnz);
  bool has_weights = csr_weights.data() != nullptr;
  if (IS_VALUE_PAIR) {
    csc.weights = fbgemm::makeAlignedUniquePtr<float>(64, nnz);
  }

  [[maybe_unused]] int column_ptr_curr = 0;
  bool is_shared_table =
      table_to_feature_offset[1] > table_to_feature_offset[0] + 1;
  const auto NS = csr_offsets[(size_t)table_to_feature_offset[1] * B] -
      csr_offsets[(size_t)table_to_feature_offset[0] * B];

  using pair_t = std::pair<int, scalar_t>;
  using value_t = std::conditional_t<IS_VALUE_PAIR, pair_t, int>;

  csc.column_segment_ids = fbgemm::makeAlignedUniquePtr<int>(64, nnz);
  auto tmpBufKeys = fbgemm::makeAlignedUniquePtr<int>(64, NS);
  fbgemm::aligned_unique_ptr<value_t> tmpBufValues(
      static_cast<value_t*>(
          fbgemm::fbgemmAlignedAlloc(64, NS * sizeof(value_t))));
  auto tmpBuf1Keys = fbgemm::makeAlignedUniquePtr<int>(64, NS);
  fbgemm::aligned_unique_ptr<value_t> tmpBuf1Values(
      static_cast<value_t*>(
          fbgemm::fbgemmAlignedAlloc(64, NS * sizeof(value_t))));

  const auto FBo = csr_offsets[(size_t)table_to_feature_offset[0] * B];
  for (int feature = table_to_feature_offset[0];
       feature < table_to_feature_offset[1];
       ++feature) {
    const auto FBs = (feature - table_to_feature_offset[0]) * B;
    at::parallel_for(0, B, 0, [&](int64_t b_begin, int64_t b_end) {
      for (int b = b_begin; b < b_end; ++b) {
        const auto FBb = (size_t)feature * B + b;
        const auto pool_begin = csr_offsets[FBb];
        const auto pool_end = csr_offsets[FBb + 1];
        const auto L = pool_end - pool_begin;
        // MEAN pooling will not work with indice_weights!
        double scale_factor =
            (static_cast<fbgemm_gpu::PoolingMode>(pooling_mode) ==
                 fbgemm_gpu::PoolingMode::MEAN &&
             !has_weights && L > 0)
            ? 1.0 / L
            : 1.0;
        for (const auto p : c10::irange(pool_begin, pool_end)) {
          tmpBufKeys[p - FBo] = csr_indices[p];
          if constexpr (IS_VALUE_PAIR) {
            tmpBufValues[p - FBo] = pair_t{
                FBs + b,
                static_cast<scalar_t>(
                    scale_factor *
                    (has_weights ? static_cast<double>(csr_weights[p]) : 1.0))};
          } else {
            tmpBufValues[p - FBo] = FBs + b;
          }
        }
      }
    });
  }

  int* sorted_col_row_index_keys;
  value_t* sorted_col_row_index_values;
  std::tie(sorted_col_row_index_keys, sorted_col_row_index_values) =
      fbgemm::radix_sort_parallel(
          tmpBufKeys.get(),
          tmpBufValues.get(),
          tmpBuf1Keys.get(),
          tmpBuf1Values.get(),
          NS,
          num_embeddings);

  int num_chunks =
      std::min<int>(at::get_num_threads(), std::max<int>(1, NS - 1));
  std::vector<int> chunk_uniq_counts(num_chunks, 0);
  std::vector<int> chunk_uniq_offsets(num_chunks + 1, 0);

  at::parallel_for(
      0, num_chunks, 1, [&](int64_t chunk_begin, int64_t chunk_end) {
        for (int64_t c = chunk_begin; c < chunk_end; ++c) {
          int64_t begin = 1 + c * (NS - 1) / num_chunks;
          int64_t end = 1 + (c + 1) * (NS - 1) / num_chunks;
          int local_count = 0;
          for (int i = begin; i < end; i++) {
            if (sorted_col_row_index_keys[i] !=
                sorted_col_row_index_keys[i - 1]) {
              local_count++;
            }
          }
          chunk_uniq_counts[c] = local_count;
        }
      });

  chunk_uniq_offsets[0] = 0;
  for (int c = 0; c < num_chunks; ++c) {
    chunk_uniq_offsets[c + 1] = chunk_uniq_offsets[c] + chunk_uniq_counts[c];
  }
  int U = chunk_uniq_offsets[num_chunks] + 1;

  csc.column_segment_ptr = fbgemm::makeAlignedUniquePtr<int>(64, NS + 1);
  csc.column_segment_indices = fbgemm::makeAlignedUniquePtr<int>(64, NS);
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

  int* col_seg_indices = csc.column_segment_indices.get();
  int* col_seg_ptr = csc.column_segment_ptr.get();

  if (!IS_VALUE_PAIR && !is_shared_table) {
    // For non shared table, no need for computing modulo.
    // As an optimization, pointer swap instead of copying.
    auto& buf = sorted_col_row_index_values == tmpBufValues.get()
        ? tmpBufValues
        : tmpBuf1Values;
    int* tmp = csc.row_indices.release();
    csc.row_indices.reset(reinterpret_cast<int*>(buf.release()));
    buf.reset(reinterpret_cast<value_t*>(tmp));
  } else {
#ifdef FBCODE_CAFFE2
    libdivide::divider<int> divisor(B);
#endif
    at::parallel_for(1, NS, 0, [&](int64_t begin, int64_t end) {
      for (int i = begin; i < end; ++i) {
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
    });
  }

  at::parallel_for(
      0, num_chunks, 1, [&](int64_t chunk_begin, int64_t chunk_end) {
        for (int64_t c = chunk_begin; c < chunk_end; ++c) {
          int64_t begin = 1 + c * (NS - 1) / num_chunks;
          int64_t end = 1 + (c + 1) * (NS - 1) / num_chunks;

          int* tstart = col_seg_indices + 1 + chunk_uniq_offsets[c];
          int* t_offs = col_seg_ptr + 1 + chunk_uniq_offsets[c];

          for (int i = begin; i < end; ++i) {
            if (sorted_col_row_index_keys[i] !=
                sorted_col_row_index_keys[i - 1]) {
              *tstart = sorted_col_row_index_keys[i];
              *t_offs = i;
              tstart++;
              t_offs++;
            }
          }
        }
      });

  csc.num_non_zero_columns = U;
  csc.column_segment_ptr[U] = NS;
  column_ptr_curr += NS;

  assert(column_ptr_curr == nnz);
}

#define INSTANTIATE_CSR2CSC_TEMPLATE_0(index_t, scalar_t, is_value_pair) \
  template void csr2csc_template_<index_t, scalar_t, is_value_pair>(     \
      HyperCompressedSparseColumn & csc,                                 \
      int B,                                                             \
      const pta::TensorAccessor<index_t, 1>& csr_offsets,                \
      const pta::TensorAccessor<index_t, 1>& csr_indices,                \
      const pta::TensorAccessor<scalar_t, 1>& csr_weights,               \
      int64_t pooling_mode,                                              \
      const int* table_to_feature_offset,                                \
      int64_t num_embeddings);

#define INSTANTIATE_CSR2CSC_TEMPLATE_1(index_t, scalar_t)  \
  INSTANTIATE_CSR2CSC_TEMPLATE_0(index_t, scalar_t, true); \
  INSTANTIATE_CSR2CSC_TEMPLATE_0(index_t, scalar_t, false);

#define INSTANTIATE_CSR2CSC_TEMPLATE_2(index_t)   \
  INSTANTIATE_CSR2CSC_TEMPLATE_1(index_t, float); \
  INSTANTIATE_CSR2CSC_TEMPLATE_1(index_t, double);

INSTANTIATE_CSR2CSC_TEMPLATE_2(int32_t);
INSTANTIATE_CSR2CSC_TEMPLATE_2(int64_t);

#undef INSTANTIATE_CSR2CSC_TEMPLATE_2
#undef INSTANTIATE_CSR2CSC_TEMPLATE_1
#undef INSTANTIATE_CSR2CSC_TEMPLATE_0

} // namespace

template <typename index_t, typename scalar_t>
void csr2csc(
    HyperCompressedSparseColumn& csc,
    int B,
    const pta::TensorAccessor<index_t, 1>& csr_offsets,
    const pta::TensorAccessor<index_t, 1>& csr_indices,
    const pta::TensorAccessor<scalar_t, 1>& csr_weights,
    int64_t pooling_mode,
    const int* table_to_feature_offset,
    int64_t num_embeddings) {
  bool has_weights = csr_weights.data() != nullptr;
  if (has_weights ||
      static_cast<fbgemm_gpu::PoolingMode>(pooling_mode) ==
          fbgemm_gpu::PoolingMode::MEAN) {
    csr2csc_template_<index_t, scalar_t, /*IS_VALUE_PAIR=*/true>(
        csc,
        B,
        csr_offsets,
        csr_indices,
        csr_weights,
        pooling_mode,
        table_to_feature_offset,
        num_embeddings);
  } else {
    csr2csc_template_<index_t, scalar_t, /*IS_VALUE_PAIR=*/false>(
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

#define INSTANTIATE_CSR2CSC_0(index_t, scalar_t)           \
  template void csr2csc<index_t, scalar_t>(                \
      HyperCompressedSparseColumn & csc,                   \
      int B,                                               \
      const pta::TensorAccessor<index_t, 1>& csr_offsets,  \
      const pta::TensorAccessor<index_t, 1>& csr_indices,  \
      const pta::TensorAccessor<scalar_t, 1>& csr_weights, \
      int64_t pooling_mode,                                \
      const int* table_to_feature_offset,                  \
      int64_t num_embeddings);

#define INSTANTIATE_CSR2CSC_1(index_t)   \
  INSTANTIATE_CSR2CSC_0(index_t, float); \
  INSTANTIATE_CSR2CSC_0(index_t, double);

INSTANTIATE_CSR2CSC_1(int32_t);
INSTANTIATE_CSR2CSC_1(int64_t);

#undef INSTANTIATE_CSR2CSC_1
#undef INSTANTIATE_CSR2CSC_0

} // namespace internal
