/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "fbgemm_gpu/utils/embedding_cpu_threading.h"

using fbgemm_gpu::choose_table_threads_impl;
using fbgemm_gpu::kDefaultTbeTablesPerThread;

namespace {
constexpr int G = kDefaultTbeTablesPerThread; // 16 by default
} // namespace

// The headline guarantee: with no env var set, TBE_TABLE_THREADS defaults to a
// cap of 1, and the per-call decision is ALWAYS 1 (serial) -- regardless of
// work size or granularity. So the no-env-var path is identical to
// single-threaded TBE.
TEST(EmbeddingCpuThreadingTest, DefaultCapIsAlwaysSerial) {
  const int cap =
      1; // == get_tbe_table_threads() when TBE_TABLE_THREADS is unset
  for (int64_t work :
       {int64_t{0},
        int64_t{1},
        int64_t{2},
        int64_t{13},
        int64_t{32},
        int64_t{358},
        int64_t{100000}}) {
    EXPECT_EQ(choose_table_threads_impl(work, cap, G), 1)
        << "cap=1 must stay serial for work=" << work;
    // And it is independent of the granularity knob.
    EXPECT_EQ(choose_table_threads_impl(work, cap, 1), 1);
    EXPECT_EQ(choose_table_threads_impl(work, cap, 64), 1);
  }
}

// Trivial-work calls never thread, even with a higher cap.
TEST(EmbeddingCpuThreadingTest, TrivialWorkIsSerial) {
  for (int cap : {1, 2, 4, 8}) {
    EXPECT_EQ(choose_table_threads_impl(0, cap, G), 1);
    EXPECT_EQ(choose_table_threads_impl(1, cap, G), 1);
  }
}

// Default granularity (G=16) puts the threading onset at 2*G = 32 tables,
// matching the validated A/B gate: small few-table lookups stay serial; large
// gathers thread.
TEST(EmbeddingCpuThreadingTest, DefaultGuardOnsetAt32) {
  // 2T cap.
  EXPECT_EQ(
      choose_table_threads_impl(7, 2, G), 1); // dpa remote_ro_event lookups
  EXPECT_EQ(choose_table_threads_impl(13, 2, G), 1);
  EXPECT_EQ(choose_table_threads_impl(31, 2, G), 1); // just below onset
  EXPECT_EQ(choose_table_threads_impl(32, 2, G), 2); // onset
  EXPECT_EQ(choose_table_threads_impl(358, 2, G), 2); // dpa remote_ro -> cap
}

// Grading scales one thread per G tables, clamped to the cap.
TEST(EmbeddingCpuThreadingTest, GradesUpToCap) {
  // 4T cap, G=16.
  EXPECT_EQ(choose_table_threads_impl(13, 4, G), 1); // event -> serial
  EXPECT_EQ(choose_table_threads_impl(32, 4, G), 2); // 32/16 = 2
  EXPECT_EQ(choose_table_threads_impl(48, 4, G), 3); // 48/16 = 3
  EXPECT_EQ(choose_table_threads_impl(64, 4, G), 4); // 64/16 = 4 (cap)
  EXPECT_EQ(choose_table_threads_impl(358, 4, G), 4); // clamped to cap
}

// G=1 reproduces the old unconditional behavior: thread every non-trivial call.
TEST(EmbeddingCpuThreadingTest, GranularityOneThreadsEverything) {
  EXPECT_EQ(choose_table_threads_impl(2, 2, 1), 2);
  EXPECT_EQ(choose_table_threads_impl(13, 2, 1), 2);
  EXPECT_EQ(choose_table_threads_impl(13, 4, 1), 4);
}
