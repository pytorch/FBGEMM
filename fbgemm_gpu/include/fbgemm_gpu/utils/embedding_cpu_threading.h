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
#include <cstdlib>
#include <cstring>

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

} // namespace fbgemm_gpu
