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
//   [rows_shift, rows_mask, rows_per_chunk, row_width, num_rows,
//    chunk_0_offset, ...]
// where each chunk offset is relative to the header's own address, in
// elements, and row_width is the number of elements per row. Headers start on
// 64-byte boundaries so offsets in vector units stay exact. rows_per_chunk is a
// power of two, so row idx is row (idx & rows_mask) of chunk (idx >>
// rows_shift). Every table in a chunked TBE has < 2^31 rows, so this math is
// 32-bit.
//
// Callers form row pointers as
//   tbe_weights_table_ptr(weights, off) + tbe_weights_row_rel_offset(...)

// Chunked TBEs pass a 1-element weights_dev anchor, which is how the forward
// host code picks the kernel instantiations that handle chunked tables without
// an extra op argument. A flat TBE whose DEVICE tables add up to one element
// also matches, which is fine since those instantiations handle flat tables too
inline bool tbe_may_have_chunked_weights(const at::Tensor& dev_weights) {
  return dev_weights.numel() == 1;
}

// Chunked rows are addressed from their header rather than from `weights`, so
// a copy of the 1-element anchor (opcheck, auto-functionalization) still
// reaches the real chunks
template <typename T>
DEVICE_INLINE T* tbe_weights_table_ptr(T* weights, const int64_t table_offset) {
  return table_offset >= 0
      ? weights + table_offset
      : reinterpret_cast<T*>(static_cast<uintptr_t>(-table_offset));
}

DEVICE_INLINE const int64_t* tbe_chunk_header(const int64_t table_offset) {
  return reinterpret_cast<const int64_t*>(-table_offset);
}

// Header that maps every row of a flat table to idx * D_emb, for loops that
// handle flat and chunked tables without branching
static __device__ const int64_t
    kTbeFlatChunkHeader[6] = {31, 0x7fffffff, 0, 0, 0, 0};

DEVICE_INLINE const int64_t* tbe_flat_chunk_header() {
  return kTbeFlatChunkHeader;
}

// Row offset within a chunked table
DEVICE_INLINE int64_t tbe_chunked_row_rel_offset(
    const int64_t* header,
    const int64_t idx,
    const int64_t D_emb) {
  const auto row = static_cast<uint32_t>(idx);
  const uint32_t chunk = row >> static_cast<uint32_t>(__ldg(&header[0]));
  const uint32_t row_in_chunk = row & static_cast<uint32_t>(__ldg(&header[1]));
  return __ldg(&header[5 + chunk]) +
      static_cast<int64_t>(
             static_cast<uint64_t>(row_in_chunk) *
             static_cast<uint32_t>(D_emb));
}

DEVICE_INLINE int64_t tbe_weights_row_rel_offset(
    const int64_t table_offset,
    const int64_t idx,
    const int64_t D_emb) {
  if (table_offset >= 0) {
    return idx * D_emb;
  }
  return tbe_chunked_row_rel_offset(tbe_chunk_header(table_offset), idx, D_emb);
}

// Rows in a chunked table, for bounds checks
DEVICE_INLINE int64_t tbe_chunked_table_rows(const int64_t table_offset) {
  return tbe_chunk_header(table_offset)[4];
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
