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
constexpr int64_t INITIAL_PREFETCH_ROWS = 16;
constexpr int64_t MAX_TLB_PRIME_INDEX_SIZE = 256;
constexpr int64_t L2_LARGE_TABLE_BYTES = 64 * 1024 * 1024; // 64 MiB

#ifdef _WIN32
inline void do_prefetch_l1(const void*, bool, bool) {}
inline void do_tlb_prime(const void*) {}
inline void do_prefetch_l2(const void*) {}

#else
inline void do_prefetch_l1(const void* addr, bool l1_keep, bool write) {
  if (l1_keep) {
    __builtin_prefetch(addr, 0, 3);
  } else if (write) {
    __builtin_prefetch(addr, 1);
  } else {
    __builtin_prefetch(addr, 0, 0);
  }
}

inline void do_tlb_prime([[maybe_unused]] const void* addr) {
#ifdef __aarch64__
  asm volatile("ldrb wzr, [%0]" : : "r"(addr) : "memory");
#endif
}

inline void do_prefetch_l2(const void* addr) {
  __builtin_prefetch(addr, 0, 2);
}
#endif

inline void
prefetch_row_l1(const uint8_t* addr, int64_t stride, bool l1_keep, bool write) {
  do_prefetch_l1(addr, l1_keep, write);
  if (stride > CACHE_LINE_SIZE)
    do_prefetch_l1(addr + CACHE_LINE_SIZE, l1_keep, write);
  if (stride > 2 * CACHE_LINE_SIZE)
    do_prefetch_l1(addr + 2 * CACHE_LINE_SIZE, l1_keep, write);
  if (stride > 3 * CACHE_LINE_SIZE)
    do_prefetch_l1(addr + 3 * CACHE_LINE_SIZE, l1_keep, write);
  if (stride > 4 * CACHE_LINE_SIZE)
    do_prefetch_l1(addr + 4 * CACHE_LINE_SIZE, l1_keep, write);
  for (int64_t off = 5 * CACHE_LINE_SIZE; off < stride; off += CACHE_LINE_SIZE)
    do_prefetch_l1(addr + off, l1_keep, write);
}

inline void prefetch_row_l2(const uint8_t* addr, int64_t stride) {
  do_prefetch_l2(addr);
  if (stride > CACHE_LINE_SIZE)
    do_prefetch_l2(addr + CACHE_LINE_SIZE);
  if (stride > 2 * CACHE_LINE_SIZE)
    do_prefetch_l2(addr + 2 * CACHE_LINE_SIZE);
  if (stride > 3 * CACHE_LINE_SIZE)
    do_prefetch_l2(addr + 3 * CACHE_LINE_SIZE);
  if (stride > 4 * CACHE_LINE_SIZE)
    do_prefetch_l2(addr + 4 * CACHE_LINE_SIZE);
  for (int64_t off = 5 * CACHE_LINE_SIZE; off < stride; off += CACHE_LINE_SIZE)
    do_prefetch_l2(addr + off);
}

struct TbePrefetchPlan {
  int64_t l1_stride;
  int64_t l2_stride;
  bool use_l2;
};

inline TbePrefetchPlan tbe_prefetch_plan(
    int64_t index_size,
    int64_t input_stride,
    int64_t data_size,
    bool use_tuned_prefetch,
    int64_t original_l1_distance) {
  const int64_t l1 = std::min(
      use_tuned_prefetch ? std::min(INITIAL_PREFETCH_ROWS, original_l1_distance)
                         : original_l1_distance,
      index_size);

  const int mult = tbe_tuned_prefetch_l2_multiplier();
  // L2 helps once a row spills past a cache line AND either spans multiple
  // lines or the whole table is too big to stay cached (> 64 MiB).
  const bool use_l2 = use_tuned_prefetch && mult > 0 &&
      input_stride > CACHE_LINE_SIZE &&
      (input_stride > 2 * CACHE_LINE_SIZE ||
       data_size * input_stride > L2_LARGE_TABLE_BYTES);
  const int64_t l2 = use_l2 ? std::min(l1 * mult, index_size) : 0;
  return TbePrefetchPlan{l1, l2, use_l2};
}

} // namespace fbgemm
