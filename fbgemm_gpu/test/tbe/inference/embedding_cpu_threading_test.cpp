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

// ---------------------------------------------------------------------------
// Row-range chunking (NOBAG)
// ---------------------------------------------------------------------------

namespace {

// Rebuild the table sizes a chunk list covers, so tests can assert the chunks
// tile every table exactly: contiguous, non-overlapping, no gaps.
std::vector<int64_t> covered_rows(
    const std::vector<fbgemm_gpu::RowChunk>& chunks,
    int num_tables) {
  std::vector<int64_t> covered(num_tables, 0);
  std::vector<int64_t> next_expected(num_tables, 0);
  for (const auto& c : chunks) {
    EXPECT_GE(c.t, 0);
    EXPECT_LT(c.t, num_tables);
    EXPECT_LE(c.r0, c.r1) << "chunk must not be inverted";
    EXPECT_EQ(c.r0, next_expected[c.t])
        << "chunks of table " << c.t << " must be contiguous and in order";
    next_expected[c.t] = c.r1;
    covered[c.t] += c.r1 - c.r0;
  }
  return covered;
}

std::vector<fbgemm_gpu::RowChunk> chunk_for(
    const std::vector<int64_t>& rows,
    int num_threads) {
  int64_t total = 0;
  for (auto r : rows) {
    total += r;
  }
  return fbgemm_gpu::build_row_chunks(
      static_cast<int>(rows.size()), total, num_threads, [&](int t) {
        return rows[t];
      });
}

} // namespace

// Every row of every table is covered exactly once, including empty tables
// (which still get one empty chunk so their per-table call still happens).
TEST(EmbeddingRowChunkTest, ChunksTileEveryTableExactly) {
  const std::vector<int64_t> rows = {20000, 0, 500, 1, 39614, 1024};
  for (int threads : {2, 4, 8, 16}) {
    const auto chunks = chunk_for(rows, threads);
    const auto covered = covered_rows(chunks, static_cast<int>(rows.size()));
    for (size_t t = 0; t < rows.size(); ++t) {
      EXPECT_EQ(covered[t], rows[t])
          << "table " << t << " at threads=" << threads;
    }
  }
}

TEST(EmbeddingRowChunkTest, EmptyTableStillGetsOneChunk) {
  const auto chunks = chunk_for({0, 0}, 4);
  ASSERT_EQ(chunks.size(), 2u);
  for (const auto& c : chunks) {
    EXPECT_EQ(c.r0, 0);
    EXPECT_EQ(c.r1, 0);
  }
}

// The point of the feature: a workload where two tables hold ~99% of the rows
// must produce many more work items than tables, so a dynamic schedule can
// actually spread the dominant tables across threads.
TEST(EmbeddingRowChunkTest, SkewedWorkloadSplitsTheDominantTables) {
  // Shape of the measured blue_reels_vdd RO lookup: 14 features, two of them
  // carrying 19,641 rows each and the rest negligible.
  std::vector<int64_t> rows(14, 1);
  rows[0] = 19641;
  rows[1] = 19641;
  const auto chunks = chunk_for(rows, 8);

  int chunks_for_dominant = 0;
  for (const auto& c : chunks) {
    if (c.t == 0) {
      ++chunks_for_dominant;
    }
  }
  EXPECT_GT(chunks_for_dominant, 1)
      << "the dominant table must be split, otherwise the makespan is still "
         "pinned to it and the speedup stays capped at ~2x";
  EXPECT_GT(chunks.size(), rows.size());

  // No chunk may exceed the grain, which is what bounds the makespan.
  const int64_t total = 19641 * 2 + 12;
  const int64_t grain = std::max<int64_t>(
      fbgemm_gpu::MIN_CHUNK_ROWS,
      total / (8 * fbgemm_gpu::DEFAULT_CHUNKS_PER_THREAD));
  for (const auto& c : chunks) {
    EXPECT_LE(c.r1 - c.r0, grain);
  }
}

// Tiny workloads must not be chopped into one chunk per row.
TEST(EmbeddingRowChunkTest, GrainHasAFloor) {
  const auto chunks = chunk_for({4000}, 16);
  EXPECT_EQ(chunks.size(), 4u); // 4000 rows / 1024 floor
}

// Thread count now derives from rows, not table count -- the whole reason the
// old heuristic misjudged a few tables carrying a huge gather.
TEST(EmbeddingRowChunkTest, ThreadCountComesFromRowsNotTables) {
  // cap=1 (the default) is always serial, whatever the row count.
  EXPECT_EQ(fbgemm_gpu::choose_num_threads_for_rows(1'000'000), 1);
}
