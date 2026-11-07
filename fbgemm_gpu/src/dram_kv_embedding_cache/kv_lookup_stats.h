/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <folly/Synchronized.h>
#include <gflags/gflags_declare.h>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

DECLARE_int32(kv_embedding_table_log_lookup_stats_interval_secs);

namespace kv_mem {

class KVLookupStats {
 public:
  using Clock = std::chrono::steady_clock;

  struct Counts {
    uint64_t batches{0};
    uint64_t keys{0};
    uint64_t lookupKeys{0};
    uint64_t uniqueKeys{0};
    uint64_t cacheHits{0};
    uint64_t indexMisses{0};
    uint64_t diskReads{0};
    uint64_t diskBytes{0};

    Counts& operator+=(const Counts& other) {
      batches += other.batches;
      keys += other.keys;
      lookupKeys += other.lookupKeys;
      uniqueKeys += other.uniqueKeys;
      cacheHits += other.cacheHits;
      indexMisses += other.indexMisses;
      diskReads += other.diskReads;
      diskBytes += other.diskBytes;
      return *this;
    }

    Counts operator-(const Counts& other) const {
      return {
          batches - other.batches,
          keys - other.keys,
          lookupKeys - other.lookupKeys,
          uniqueKeys - other.uniqueKeys,
          cacheHits - other.cacheHits,
          indexMisses - other.indexMisses,
          diskReads - other.diskReads,
          diskBytes - other.diskBytes};
    }
  };

  struct Snapshot {
    Counts total;
    Counts interval;
    double windowSecs;
  };

  // A batch contributes once after its workers join. Disabled logging does
  // not acquire the lock or read the clock.
  std::optional<Snapshot> record(
      const Counts& counts,
      std::chrono::seconds interval,
      std::optional<Clock::time_point> at = std::nullopt) {
    if (interval.count() <= 0) {
      return std::nullopt;
    }
    auto state = state_.wlock();
    const auto now = at ? *at : Clock::now();
    state->total += counts;
    if (!state->lastLog) {
      state->lastLog = now;
      return std::nullopt;
    }
    if (now - *state->lastLog < interval) {
      return std::nullopt;
    }
    Snapshot snapshot{
        state->total,
        state->total - state->previous,
        std::chrono::duration<double>(now - *state->lastLog).count()};
    state->previous = state->total;
    state->lastLog = now;
    return snapshot;
  }

  Counts totals() const {
    return state_.rlock()->total;
  }

  static std::string format(const Snapshot& snapshot, bool deduplicated) {
    const auto formatCounts = [deduplicated](
                                  const char* prefix, const Counts& counts) {
      return fmt::format(
          " {}_lookup_batches={} {}_keys={} {}_lookup_keys={} {}_unique_keys={} {}_cache_hits={} {}_index_misses={} {}_disk_reads={} {}_disk_bytes={}",
          prefix,
          counts.batches,
          prefix,
          counts.keys,
          prefix,
          counts.lookupKeys,
          prefix,
          deduplicated ? fmt::format("{}", counts.uniqueKeys) : "NA",
          prefix,
          counts.cacheHits,
          prefix,
          counts.indexMisses,
          prefix,
          counts.diskReads,
          prefix,
          counts.diskBytes);
    };
    return fmt::format(
        " dedup={} interval_secs={:.6f}{}{}",
        deduplicated,
        snapshot.windowSecs,
        formatCounts("total", snapshot.total),
        formatCounts("interval", snapshot.interval));
  }

 private:
  struct State {
    Counts total;
    Counts previous;
    std::optional<Clock::time_point> lastLog;
  };
  folly::Synchronized<State> state_;
};

} // namespace kv_mem
