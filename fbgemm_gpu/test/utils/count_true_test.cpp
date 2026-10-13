/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <span>

#include "fbgemm_gpu/utils/count_true.h"

namespace fbgemm_gpu::utils {
namespace {

constexpr std::size_t kMaxSize = 200000;
constexpr std::array<std::size_t, 10> kSizes =
    {0, 1, 7, 65534, 65535, 65536, 131069, 131070, 131071, kMaxSize};

using Mask = std::array<bool, kMaxSize>;

void count_true_validate(std::span<const bool> mask) {
  EXPECT_EQ(count_true(mask), std::count(mask.begin(), mask.end(), true))
      << "size = " << mask.size();
}

TEST(CountTrueTest, MatchesStdCount) {
  const auto all_true = std::make_unique<Mask>();
  all_true->fill(true);

  const auto all_false = std::make_unique<Mask>();
  all_false->fill(false);

  std::mt19937 rng(42);
  std::bernoulli_distribution dist(0.5);
  const auto random = std::make_unique<Mask>();
  std::ranges::generate(*random, [&] { return dist(rng); });

  for (const auto n : kSizes) {
    count_true_validate(std::span<const bool>(*all_true).first(n));
    count_true_validate(std::span<const bool>(*all_false).first(n));
    count_true_validate(std::span<const bool>(*random).first(n));
  }
}

} // namespace
} // namespace fbgemm_gpu::utils
