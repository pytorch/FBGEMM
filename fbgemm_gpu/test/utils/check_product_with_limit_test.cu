/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

#include "fbgemm_gpu/utils/check_product_with_limit.h"
#include "fbgemm_gpu/utils/cuda_block_count.h"

namespace fbgemm_gpu::utils {
namespace {

constexpr uint64_t kMaxLaunchThreads =
    std::numeric_limits<uint32_t>::max() - uint64_t{1};

template <typename F>
void expect_product_failure(
    const char* expression,
    const F& calculate_product) {
  try {
    calculate_product();
    FAIL() << "Expected check_product_with_limit to reject the product";
  } catch (const c10::Error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(expression), std::string::npos) << message;
    EXPECT_NE(message.find("exceeds the supported limit"), std::string::npos)
        << message;
  }
}

TEST(CheckProductWithLimitTest, AcceptsLastValidAndRejectsFirstInvalidProduct) {
  EXPECT_EQ(
      check_product_with_limit(
          "attention launch",
          "B * H * split_k",
          kMaxLaunchThreads,
          kMaxLaunchThreads,
          uint64_t{1},
          uint64_t{1}),
      kMaxLaunchThreads);

  expect_product_failure("B * H * split_k", [] {
    check_product_with_limit(
        "attention launch",
        "B * H * split_k",
        kMaxLaunchThreads,
        kMaxLaunchThreads + 1,
        uint64_t{1},
        uint64_t{1});
  });
}

TEST(CheckProductWithLimitTest, AccountsForPaddedWarpLaunches) {
  constexpr uint64_t kThreadsPerWarp = 32;
  constexpr uint64_t kWarpsPerBlock = 4;
  constexpr uint64_t kThreadsPerBlock = kThreadsPerWarp * kWarpsPerBlock;
  constexpr uint64_t kLastValidBlocks = kMaxLaunchThreads / kThreadsPerBlock;
  constexpr uint64_t kLastValidWarps = kLastValidBlocks * kWarpsPerBlock;

  const auto last_valid_blocks =
      cuda_calc_xblock_count(kLastValidWarps, kWarpsPerBlock);
  EXPECT_EQ(
      check_product_with_limit(
          "KV cache QKV launch",
          "ceil(num_warps / 4) * 32 * 4",
          kMaxLaunchThreads,
          last_valid_blocks,
          kThreadsPerWarp,
          kWarpsPerBlock),
      kLastValidBlocks * kThreadsPerBlock);

  const auto first_invalid_blocks =
      cuda_calc_xblock_count(kLastValidWarps + 1, kWarpsPerBlock);
  expect_product_failure("ceil(num_warps / 4) * 32 * 4", [&] {
    check_product_with_limit(
        "KV cache QKV launch",
        "ceil(num_warps / 4) * 32 * 4",
        kMaxLaunchThreads,
        first_invalid_blocks,
        kThreadsPerWarp,
        kWarpsPerBlock);
  });
}

TEST(CheckProductWithLimitTest, ChecksFullThreadCount) {
  constexpr uint64_t kThreadsPerBlock = 128;
  constexpr uint64_t kLastValidCoordinates =
      kMaxLaunchThreads / kThreadsPerBlock;

  EXPECT_EQ(
      check_product_with_limit(
          "exact-coordinate launch",
          "B * H * split_k * threads_per_block",
          kMaxLaunchThreads,
          kLastValidCoordinates,
          uint64_t{1},
          uint64_t{1},
          kThreadsPerBlock),
      kLastValidCoordinates * kThreadsPerBlock);
  expect_product_failure("B * H * split_k * threads_per_block", [] {
    check_product_with_limit(
        "exact-coordinate launch",
        "B * H * split_k * threads_per_block",
        kMaxLaunchThreads,
        kLastValidCoordinates + 1,
        uint64_t{1},
        uint64_t{1},
        kThreadsPerBlock);
  });
}

TEST(CheckProductWithLimitTest, RejectsNegativeDimensions) {
  EXPECT_THROW(
      check_product_with_limit(
          "KV cache conversion launch", "B * N_H_L", 512, -1, 1),
      c10::Error);
}

TEST(CheckProductWithLimitTest, EnforcesIntrainingPruningTableBoundary) {
  EXPECT_EQ(
      check_product_with_limit(
          "intraining embedding pruning", "num_tables", 1024, 1024),
      uint64_t{1024});
  expect_product_failure("num_tables", [] {
    check_product_with_limit(
        "intraining embedding pruning", "num_tables", 1024, 1025);
  });
}

} // namespace
} // namespace fbgemm_gpu::utils
