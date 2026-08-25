/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "src/EmbeddingSpMDMPrefetch.h" // @manual

using namespace fbgemm;

namespace {
// An INT4 D=128 row: narrow enough that the in-flight byte cap never binds.
constexpr int64_t kNarrowStride = 68;
// More indices than any look-ahead tested here, so the clamp never binds.
constexpr int64_t kManyIndices = 1000;
} // namespace

TEST(EmbeddingSpMDMPrefetchTest, TunedDistanceIsHonored) {
  EXPECT_EQ(
      tbe_resolve_l1_prefetch_distance(32, kNarrowStride, kManyIndices), 32);
}

TEST(EmbeddingSpMDMPrefetchTest, UnsetDistanceUsesDefault) {
  EXPECT_EQ(
      tbe_resolve_l1_prefetch_distance(-1, kNarrowStride, kManyIndices),
      DEFAULT_L1_PREFETCH_DISTANCE);
}

TEST(EmbeddingSpMDMPrefetchTest, ZeroDisablesPrefetching) {
  EXPECT_EQ(
      tbe_resolve_l1_prefetch_distance(0, kNarrowStride, kManyIndices), 0);
}

TEST(EmbeddingSpMDMPrefetchTest, DistanceNeverExceedsIndexCount) {
  // Looking further ahead than there are indices would walk off the array.
  EXPECT_EQ(tbe_resolve_l1_prefetch_distance(32, kNarrowStride, 8), 8);
  EXPECT_EQ(tbe_resolve_l1_prefetch_distance(-1, kNarrowStride, 8), 8);
  EXPECT_EQ(tbe_resolve_l1_prefetch_distance(32, kNarrowStride, 0), 0);
}

TEST(EmbeddingSpMDMPrefetchTest, NonPositiveStrideDisablesPrefetching) {
  // A zero stride would otherwise divide by zero while capping the default.
  EXPECT_EQ(tbe_resolve_l1_prefetch_distance(-1, 0, kManyIndices), 0);
  EXPECT_EQ(tbe_resolve_l1_prefetch_distance(32, 0, kManyIndices), 0);
  EXPECT_EQ(tbe_resolve_l1_prefetch_distance(-1, -8, kManyIndices), 0);
}

TEST(EmbeddingSpMDMPrefetchTest, WideRowsCapTheDefaultButNotATunedValue) {
  // The default is capped to roughly 4 KiB of rows in flight, so a 1 KiB row
  // gets 4 of them rather than the default 16.
  EXPECT_EQ(tbe_resolve_l1_prefetch_distance(-1, 1024, kManyIndices), 4);
  // A row wider than the whole budget turns the default off entirely.
  EXPECT_EQ(tbe_resolve_l1_prefetch_distance(-1, 8192, kManyIndices), 0);
  // An explicitly tuned distance opts out of the cap.
  EXPECT_EQ(tbe_resolve_l1_prefetch_distance(32, 8192, kManyIndices), 32);
}
