/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Work-size-aware thread-count selection for the CPU TBE per-table loop
// (see embedding_forward_quantized_cpu_template.cpp). Two env knobs:
//   - TBE_TABLE_THREADS     : the thread-count cap (default 1 =>
//   single-threaded).
//   - TBE_TABLES_PER_THREAD : work-granularity G (tables/thread, default 16).
// The pure decision `choose_table_threads_impl` is split out so it can be unit
// tested without touching the process environment.

namespace fbgemm_gpu {

// Default work-granularity (tables per thread) used when TBE_TABLES_PER_THREAD
// is unset/invalid. With the clamp formula below, the smallest call that gets a
// 2nd thread is `2 * G` tables, so G=16 puts the threading onset at 32 tables
// -- small few-table lookups stay serial, while large gathers (hundreds of
// tables) thread.
constexpr int kDefaultTbeTablesPerThread = 16;

// Pure, env-free per-call thread count from the work size (`work_units` ~=
// tables). A single formula N = clamp(work_units / G, 1, cap) both GATES and
// GRADES:
//   - cap <= 1 (the default, TBE_TABLE_THREADS unset) => always 1 (serial), so
//   the
//     no-env-var path is identical to single-threaded TBE.
//   - a call with fewer than 2*G tables runs serial (work/G < 2, floored to 1).
//   - larger calls scale up one thread per G tables, up to the cap.
inline int
choose_table_threads_impl(int64_t work_units, int cap, int tables_per_thread) {
  if (cap <= 1 || work_units <= 1) {
    return 1;
  }
  const int64_t g = std::max<int>(1, tables_per_thread);
  const int64_t graded = work_units / g;
  return static_cast<int>(
      std::clamp<int64_t>(graded, int64_t{1}, int64_t{cap}));
}

// Thread-count cap from env TBE_TABLE_THREADS (default 1 => single-threaded).
inline int get_tbe_table_threads() {
  static const int n = []() {
    const char* env = std::getenv("TBE_TABLE_THREADS");
    if (!env || *env == '\0') {
      return 1;
    }
    int val = 0;
    auto [ptr, ec] = std::from_chars(env, env + std::strlen(env), val);
    if (ec != std::errc{} || *ptr != '\0') {
      return 1;
    }
    return std::max<int>(1, val);
  }();
  return n;
}

// Work-granularity from env TBE_TABLES_PER_THREAD (default
// kDefaultTbeTablesPerThread). Set to 1 to thread every call (the old
// unconditional behavior).
inline int get_tbe_tables_per_thread() {
  static const int n = []() {
    const char* env = std::getenv("TBE_TABLES_PER_THREAD");
    if (!env || *env == '\0') {
      return kDefaultTbeTablesPerThread;
    }
    int val = 0;
    auto [ptr, ec] = std::from_chars(env, env + std::strlen(env), val);
    if (ec != std::errc{} || *ptr != '\0') {
      return kDefaultTbeTablesPerThread;
    }
    return std::max<int>(1, val);
  }();
  return n;
}

// Env-driven wrapper used by the kernel: combines the cap and granularity
// knobs.
inline int choose_table_threads(int64_t work_units) {
  return choose_table_threads_impl(
      work_units, get_tbe_table_threads(), get_tbe_tables_per_thread());
}

} // namespace fbgemm_gpu
