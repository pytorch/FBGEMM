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
#include <vector>

namespace fbgemm_gpu {

// Default work-granularity (tables per thread)
constexpr int DEFAULT_TABLES_PER_THREAD = 16;

inline int
calculate_num_threads(int num_tables, int cap, int tables_per_thread) {
  if (cap <= 1 || num_tables <= 1) {
    return 1;
  }
  const int num_threads = num_tables / tables_per_thread;
  return std::clamp<int>(num_threads, 1, cap);
}

inline int get_env_int(const char* name, int default_val) {
  const char* env = std::getenv(name);
  if (!env || *env == '\0') {
    return default_val;
  }
  int val = 0;
  auto [ptr, ec] = std::from_chars(env, env + std::strlen(env), val);
  if (ec != std::errc{} || *ptr != '\0') {
    return default_val;
  }
  return std::max<int>(1, val);
}

// Thread-count cap from env FBGEMM_TBE_MAX_NUM_THREADS
inline int get_tbe_max_num_threads() {
  static const int n = get_env_int("FBGEMM_TBE_MAX_NUM_THREADS", 1);
  return n;
}

// Work-granularity from env FBGEMM_TBE_MIN_TABLES_PER_THREAD
// We are using the number of tables as approximated
// minimal workload per thread (default 16) to avoid
// threading overhead
inline int get_tbe_min_tables_per_thread() {
  static const int n = get_env_int(
      "FBGEMM_TBE_MIN_TABLES_PER_THREAD", DEFAULT_TABLES_PER_THREAD);
  return n;
}

inline int choose_num_threads(int num_tables) {
  return calculate_num_threads(
      num_tables, get_tbe_max_num_threads(), get_tbe_min_tables_per_thread());
}

// ---------------------------------------------------------------------------
// Row-range chunking (NOBAG only)
//
// Parallelising over tables caps the speedup at
// sum(rows per table) / max(rows in one table), because the makespan can never
// drop below the largest single table. Sequence models routinely put nearly all
// of their lookups in one or two features, where that ratio is ~2 no matter how
// many threads are available.
//
// In NOBAG each output row is an independent gather with no accumulation, so a
// table's row range can be split across threads instead, which removes the
// bound. Threads still write disjoint output slices, so results stay bitwise
// identical to the serial path.
// ---------------------------------------------------------------------------

// A contiguous slice [r0, r1) of table `t`'s output rows.
struct RowChunk {
  int t;
  int64_t r0;
  int64_t r1;
};

// Minimum rows per thread before threading is worth the OpenMP overhead.
constexpr int64_t DEFAULT_MIN_ROWS_PER_THREAD = 4096;
// Several chunks per thread lets a dynamic schedule absorb stragglers.
constexpr int64_t DEFAULT_CHUNKS_PER_THREAD = 4;
// Floor on chunk size so tiny tables do not each become their own work item.
constexpr int64_t MIN_CHUNK_ROWS = 1024;

inline int64_t get_tbe_min_rows_per_thread() {
  static const int64_t n = get_env_int(
      "FBGEMM_TBE_MIN_ROWS_PER_THREAD",
      static_cast<int>(DEFAULT_MIN_ROWS_PER_THREAD));
  return n;
}

// Thread count derived from actual work (output rows). calculate_num_threads()
// uses the table count as a work proxy, which misclassifies a handful of tables
// carrying a very large gather as "too small to thread".
inline int choose_num_threads_for_rows(int64_t total_rows) {
  const int cap = get_tbe_max_num_threads();
  if (cap <= 1 || total_rows <= 0) {
    return 1;
  }
  const int64_t n = total_rows / get_tbe_min_rows_per_thread();
  return static_cast<int>(std::clamp<int64_t>(n, 1, cap));
}

// Split every table's row range into chunks of at most `grain` rows. Tables
// with no rows still get one empty chunk so the per-table call (and its error
// reporting) happens exactly as it does on the serial path.
template <typename RowsOf>
inline std::vector<RowChunk> build_row_chunks(
    int num_tables,
    int64_t total_rows,
    int num_threads,
    const RowsOf& rows_of) {
  const int64_t grain = std::max<int64_t>(
      MIN_CHUNK_ROWS,
      total_rows /
          std::max<int64_t>(
              1,
              static_cast<int64_t>(num_threads) * DEFAULT_CHUNKS_PER_THREAD));

  std::vector<RowChunk> chunks;
  chunks.reserve(
      static_cast<size_t>(num_tables) +
      static_cast<size_t>(num_threads) * DEFAULT_CHUNKS_PER_THREAD);
  for (int t = 0; t < num_tables; ++t) {
    const int64_t n = rows_of(t);
    if (n <= 0) {
      chunks.push_back({t, 0, 0});
      continue;
    }
    for (int64_t r = 0; r < n; r += grain) {
      chunks.push_back({t, r, std::min(r + grain, n)});
    }
  }
  return chunks;
}

} // namespace fbgemm_gpu
