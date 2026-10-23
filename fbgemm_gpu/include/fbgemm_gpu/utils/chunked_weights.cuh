/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"

namespace fbgemm_gpu {

// Row addressing for TBE weights that may be split into separately allocated
// chunks (see `weights_chunk_size_bytes` in the training TBE).
//
// A non-negative table offset is the usual flat layout: the table starts at
// `weights[table_offset]` and row `idx` is at `idx * D_emb` from there.
//
// A negative table offset is the negated device address of a chunk header:
//   [magic, shift, rows_per_chunk, row_width, num_rows, chunk_0_offset, ...]
// where each chunk offset is relative to the start of the buffer and
// row_width is the number of elements per row. A row's chunk is
// (idx * magic) >> shift, which equals idx / rows_per_chunk for every
// idx < 2^31 without a 64-bit division.
//
// Callers form row pointers as
//   &weights[tbe_weights_table_base(off)] + tbe_weights_row_rel_offset(...)

DEVICE_INLINE int64_t tbe_weights_table_base(const int64_t table_offset) {
  return table_offset >= 0 ? table_offset : 0;
}

DEVICE_INLINE int64_t
tbe_chunk_of_row(const int64_t* header, const int64_t idx) {
  return static_cast<int64_t>(
      (static_cast<uint64_t>(idx) * static_cast<uint64_t>(header[0])) >>
      header[1]);
}

DEVICE_INLINE int64_t tbe_weights_row_rel_offset(
    const int64_t table_offset,
    const int64_t idx,
    const int64_t D_emb) {
  if (table_offset >= 0) {
    return idx * D_emb;
  }
  const auto* header = reinterpret_cast<const int64_t*>(-table_offset);
  const int64_t chunk = tbe_chunk_of_row(header, idx);
  return header[5 + chunk] + (idx - chunk * header[2]) * D_emb;
}

// Optimizer state is indexed relative to its table start, e.g. momentum[idx]
// or &momentum[idx * D]. Returns the offset of a virtual table start that
// makes such indexing land on row idx, for flat and chunked tables alike.
// Each buffer has its own header, since rows_per_chunk depends on row size.
DEVICE_INLINE int64_t
tbe_state_table_offset(const int64_t table_offset, const int64_t idx) {
  if (table_offset >= 0) {
    return table_offset;
  }
  const auto* header = reinterpret_cast<const int64_t*>(-table_offset);
  const int64_t chunk = tbe_chunk_of_row(header, idx);
  return header[5 + chunk] - chunk * header[2] * header[3];
}

// Elements in a chunked table, for bounds checks
DEVICE_INLINE int64_t tbe_chunked_table_numel(const int64_t table_offset) {
  const auto* header = reinterpret_cast<const int64_t*>(-table_offset);
  return header[4] * header[3];
}

// Same as tbe_weights_row_rel_offset, in units of VecT for kernels that
// address rows by vector (row_vecs vectors per row)
template <typename T, typename VecT>
DEVICE_INLINE int64_t tbe_weights_row_rel_vec_offset(
    const int64_t table_offset,
    const int64_t idx,
    const int64_t row_vecs) {
  constexpr int64_t kElemsPerVec = sizeof(VecT) / sizeof(T);
  if (table_offset >= 0) {
    return idx * row_vecs;
  }
  return tbe_weights_row_rel_offset(
             table_offset, idx, row_vecs * kElemsPerVec) /
      kElemsPerVec;
}

} // namespace fbgemm_gpu
