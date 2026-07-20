/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>

#include "fbgemm/Utils.h"

// Shared software-prefetch helpers for the TBE CPU kernels (SVE + autovec)

namespace fbgemm {

// Tuning constants (Neoverse-V2 / CG1).
constexpr int64_t CACHE_LINE_SIZE = 64;
constexpr int64_t DEFAULT_L1_PREFETCH_DISTANCE = 16;
constexpr int64_t MAX_TLB_PRIME_INDEX_SIZE = 256;
constexpr int64_t DEFAULT_L2_LARGE_TABLE_MB = 64;

// Enable the farther L2 tier only when its per-row overhead can pay off: the
// row spans multiple cache lines (small-dim rows likely already fit L1) and the
// table is too big to stay cached (> tbe_l2_large_table_bytes(), default 64 MiB
// via FBGEMM_TBE_L2_LARGE_TABLE_MB).
inline bool tbe_use_l2_prefetch(
    int64_t l2_distance,
    int64_t input_stride, // row size in byte
    int64_t data_size // number of rows in the embedding table
) {
  int64_t table_size = data_size * input_stride;
  int64_t threshold_bytes = tbe_l2_large_table_bytes();
  if (threshold_bytes == 0) {
    threshold_bytes = DEFAULT_L2_LARGE_TABLE_MB * (int64_t{1} << 20);
  }
  return l2_distance > 0 && input_stride > CACHE_LINE_SIZE &&
      table_size > threshold_bytes;
}

#ifdef _WIN32
inline void prefetch_row_l1(const uint8_t*, int64_t) {}
inline void prefetch_row_l2(const uint8_t*, int64_t) {}
inline void do_tlb_prime(const void*) {}

#else

inline void prefetch_row_l1(const uint8_t* addr, int64_t input_stride) {
  for (int64_t off = 0; off < input_stride; off += CACHE_LINE_SIZE)
    __builtin_prefetch(addr + off, 0, 3);
}

inline void prefetch_row_l2(const uint8_t* addr, int64_t input_stride) {
  for (int64_t off = 0; off < input_stride; off += CACHE_LINE_SIZE)
    __builtin_prefetch(addr + off, 0, 2);
}

inline void do_tlb_prime([[maybe_unused]] const void* addr) {
#ifdef __aarch64__
  asm volatile("ldrb wzr, [%0]" : : "r"(addr) : "memory");
#endif
}
#endif

template <void (*PrefetchRow)(const uint8_t*, int64_t), typename IndexType>
inline void tbe_prefetch_row(
    const uint8_t* input,
    const IndexType* indices,
    int64_t pos,
    int64_t last_index,
    int64_t input_stride,
    int64_t data_size,
    int64_t prefetch_distance) {
  const IndexType idx = indices[std::min(pos + prefetch_distance, last_index)];
  if (idx >= 0 && idx < data_size) {
    PrefetchRow(input + input_stride * idx, input_stride);
  }
}

} // namespace fbgemm
