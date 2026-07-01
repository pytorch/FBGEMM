/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "fbgemm_gpu/utils/embedding_cpu_threading.h"

using fbgemm_gpu::calculate_num_threads;
using fbgemm_gpu::DEFAULT_TABLES_PER_THREAD;

namespace {
constexpr int G = DEFAULT_TABLES_PER_THREAD; // 16 by default
} // namespace

// The headline guarantee: with no env var set, FBGEMM_TBE_MAX_NUM_THREADS
// defaults to a cap of 1, and the per-call decision is ALWAYS 1 (serial) --
// regardless of work size or granularity. So the no-env-var path is identical
// to single-threaded TBE.
TEST(EmbeddingCpuThreadingTest, DefaultCapIsAlwaysSerial) {
  // cap=1 == get_tbe_max_num_threads() when FBGEMM_TBE_MAX_NUM_THREADS is
  // unset.
  for (int work : {0, 1, 2, 13, 32, 358, 100000}) {
    EXPECT_EQ(calculate_num_threads(work, 1, G), 1)
        << "cap=1 must stay serial for work=" << work;
    // And it is independent of the granularity knob.
    EXPECT_EQ(calculate_num_threads(work, 1, 1), 1);
    EXPECT_EQ(calculate_num_threads(work, 1, 64), 1);
  }
}

// Trivial-work calls never thread, even with a higher cap.
TEST(EmbeddingCpuThreadingTest, TrivialWorkIsSerial) {
  for (int cap : {1, 2, 4, 8}) {
    EXPECT_EQ(calculate_num_threads(0, cap, G), 1);
    EXPECT_EQ(calculate_num_threads(1, cap, G), 1);
  }
}

// Default granularity (G=16) puts the threading onset at 2*G = 32 tables,
// matching the validated A/B gate: small few-table lookups stay serial; large
// gathers thread.
TEST(EmbeddingCpuThreadingTest, DefaultGuardOnsetAt32) {
  // 2T cap.
  EXPECT_EQ(calculate_num_threads(7, 2, G), 1); // dpa remote_ro_event lookups
  EXPECT_EQ(calculate_num_threads(13, 2, G), 1);
  EXPECT_EQ(calculate_num_threads(31, 2, G), 1); // just below onset
  EXPECT_EQ(calculate_num_threads(32, 2, G), 2); // onset
  EXPECT_EQ(calculate_num_threads(358, 2, G), 2); // dpa remote_ro -> cap
}

// Grading scales one thread per G tables, clamped to the cap.
TEST(EmbeddingCpuThreadingTest, GradesUpToCap) {
  // 4T cap, G=16.
  EXPECT_EQ(calculate_num_threads(13, 4, G), 1); // event -> serial
  EXPECT_EQ(calculate_num_threads(32, 4, G), 2); // 32/16 = 2
  EXPECT_EQ(calculate_num_threads(48, 4, G), 3); // 48/16 = 3
  EXPECT_EQ(calculate_num_threads(64, 4, G), 4); // 64/16 = 4 (cap)
  EXPECT_EQ(calculate_num_threads(358, 4, G), 4); // clamped to cap
}

// G=1 reproduces the old unconditional behavior: thread every non-trivial call.
TEST(EmbeddingCpuThreadingTest, GranularityOneThreadsEverything) {
  EXPECT_EQ(calculate_num_threads(2, 2, 1), 2);
  EXPECT_EQ(calculate_num_threads(13, 2, 1), 2);
  EXPECT_EQ(calculate_num_threads(13, 4, 1), 4);
}
