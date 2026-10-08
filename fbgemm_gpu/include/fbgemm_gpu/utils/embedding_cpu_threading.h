/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
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

// 1/0, true/false, on/off, yes/no in any case; nullopt otherwise.
inline std::optional<bool> parse_bool_flag(std::string_view s) {
  std::string v(s);
  std::ranges::transform(v, v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (v == "1" || v == "true" || v == "on" || v == "yes") {
    return true;
  }
  if (v == "0" || v == "false" || v == "off" || v == "no") {
    return false;
  }
  return std::nullopt;
}

// get_env_int clamps to >= 1, so it cannot express "off". A boolean gate needs
// its own reader or FBGEMM_TBE_BAG_PARALLELISM=0 would read back as 1 (on).
// Unrecognised values keep the default (logged by the op).
inline bool get_env_bool(const char* name, bool default_val) {
  const char* env = std::getenv(name);
  if (!env || *env == '\0') {
    return default_val;
  }
  return parse_bool_flag(env).value_or(default_val);
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
    const RowsOf& rows_of,
    // Pooled callers pass MIN_CHUNK_BAGS. A bag is L gathers plus a pooling
    // reduction, so the floor that stops work items becoming too small to be
    // worth dispatching differs between the two paths. Defaulted, so the
    // existing NOBAG caller is unchanged.
    int64_t min_chunk = MIN_CHUNK_ROWS) {
  const int64_t grain = std::max<int64_t>(
      min_chunk,
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

// ---------------------------------------------------------------------------
// Bag-range chunking (POOLED path)
//
// The pooled analogue of row chunking above. Parallelising over tables alone
// caps the makespan at the largest single table, and a chunk is then a whole
// table, so a lookup with a big batch cannot use more threads than it has
// tables. Drawing (table, bag-range) pairs from ONE flat work list lets tables
// and bags be balanced together by a single dynamic schedule, in a single
// parallel region per lookup.
//
// Threads write disjoint output slices -- chunk (t, b0, b1) owns rows [b0, b1)
// of columns [D_start, D_start + D) -- and a bag's pooling reduction always
// happens inside one kernel call, so results are bitwise identical to serial.
//
// Everything here is inert unless FBGEMM_TBE_BAG_PARALLELISM is set.
// ---------------------------------------------------------------------------

// Minimum bags per thread before threading beats the fork. Much lower than
// DEFAULT_MIN_ROWS_PER_THREAD because a bag (L gathers + a reduction) is far
// heavier than the single gather a NOBAG row costs.
constexpr int64_t DEFAULT_MIN_BAGS_PER_THREAD = 64;
// Floor on chunk size so a small table does not become its own work item.
constexpr int64_t MIN_CHUNK_BAGS = 8;

// Master switch, off by default: turning it on changes the pooled path from
// table-only threading to flattened (table x bag) threading, which would be a
// behaviour change for every existing FBGEMM_TBE_MAX_NUM_THREADS user.
inline bool tbe_bag_parallelism_enabled() {
  static const bool on = get_env_bool("FBGEMM_TBE_BAG_PARALLELISM", false);
  return on;
}

// FBGEMM_TBE_BAG_PARALLELISM if set but unrecognised, else nullptr. First call
// per process only, so both generated op files log it once between them.
inline const char* take_unrecognized_bag_parallelism_value() {
  static std::atomic<bool> taken{false};
  if (taken.exchange(true)) {
    return nullptr;
  }
  const char* env = std::getenv("FBGEMM_TBE_BAG_PARALLELISM");
  return (env && *env != '\0' && !parse_bool_flag(env)) ? env : nullptr;
}

// Minimum batch size before the flattened path will thread at all; below it the
// lookup runs serially (not table-threaded).
//
// At B==1 the work list degenerates to one chunk per table -- plain table
// parallelism, with the bag dimension contributing nothing. Measured on an ads
// ranking model whose read-only lookups all run at B==1, threading them took
// their p99 from 14.0ms serial to 95.9ms at 64 threads, while the large-batch
// lookups improved over the same range. EPO's own batch splitter has the same
// guard (`threadCount_ <= 1 || B <= 1 -> serial`), which is why it never
// regressed those lookups.
//
// Default 2 = "needs a real batch"; set to 1 to disable the guard for A/B.
inline int64_t get_tbe_min_b_for_bag_parallelism() {
  static const int64_t n = get_env_int("FBGEMM_TBE_BAG_PARALLELISM_MIN_B", 2);
  return n;
}

inline int64_t get_tbe_min_bags_per_thread() {
  static const int64_t n = get_env_int(
      "FBGEMM_TBE_MIN_BAGS_PER_THREAD",
      static_cast<int>(DEFAULT_MIN_BAGS_PER_THREAD));
  return n;
}

// Thread count for one pooled lookup with bag parallelism on: 1 (serial) when
// the batch is below min_b or there is too little work, otherwise one thread
// per min_bags_per_thread bags, clamped to the cap. Pure, so the routing can be
// tested without env vars; choose_bag_threads() supplies the configured values.
inline int calculate_bag_threads(
    int64_t num_tables,
    int64_t batch_size,
    int cap,
    int64_t min_bags_per_thread,
    int64_t min_b) {
  const int64_t total_bags = num_tables * batch_size;
  if (cap <= 1 || batch_size < min_b || total_bags <= 0) {
    return 1;
  }
  const int64_t n = total_bags / std::max<int64_t>(1, min_bags_per_thread);
  return static_cast<int>(std::clamp<int64_t>(n, 1, cap));
}

inline int choose_bag_threads(int64_t num_tables, int64_t batch_size) {
  return calculate_bag_threads(
      num_tables,
      batch_size,
      get_tbe_max_num_threads(),
      get_tbe_min_bags_per_thread(),
      get_tbe_min_b_for_bag_parallelism());
}

} // namespace fbgemm_gpu
