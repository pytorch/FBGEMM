/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>

#include "fbgemm/Utils.h"

TEST(cpu_kernel_test, radix_sort_parallel_test) {
  std::array<int, 8> keys = {1, 2, 4, 5, 4, 3, 2, 9};
  std::array<int, 8> values = {0, 0, 0, 0, 1, 1, 1, 1};

  std::array<int, 8> keys_tmp;
  std::array<int, 8> values_tmp;

  const auto [sorted_keys, sorted_values] = fbgemm::radix_sort_parallel(
      keys.data(),
      values.data(),
      keys_tmp.data(),
      values_tmp.data(),
      keys.size(),
      10);

  std::array<int, 8> expect_keys_tmp = {1, 2, 2, 3, 4, 4, 5, 9};
  std::array<int, 8> expect_values_tmp = {0, 0, 1, 1, 0, 1, 0, 1};
  EXPECT_EQ(sorted_keys, keys_tmp.data());
  EXPECT_EQ(sorted_values, values_tmp.data());
  EXPECT_EQ(keys_tmp, expect_keys_tmp);
  EXPECT_EQ(values_tmp, expect_values_tmp);
}

TEST(cpu_kernel_test, radix_sort_parallel_test_neg_vals) {
  std::array<int64_t, 8> keys = {-4, -3, 0, 1, -2, -1, 3, 2};
  std::array<int64_t, 8> values = {0, 0, 0, 0, 1, 1, 1, 1};

  std::array<int64_t, 8> keys_tmp;
  std::array<int64_t, 8> values_tmp;

  const auto [sorted_keys, sorted_values] = fbgemm::radix_sort_parallel(
      keys.data(),
      values.data(),
      keys_tmp.data(),
      values_tmp.data(),
      keys.size(),
      std::numeric_limits<int64_t>::max(),
      /*maybe_with_neg_vals=*/true);

  std::array<int64_t, 8> expect_keys_tmp = {-4, -3, -2, -1, 0, 1, 2, 3};
  std::array<int64_t, 8> expect_values_tmp = {0, 0, 1, 1, 0, 0, 1, 1};
  if (sorted_keys == keys.data()) { // even number of passes
    EXPECT_EQ(expect_keys_tmp, keys);
    EXPECT_EQ(expect_values_tmp, values);
  } else { // odd number of passes
    EXPECT_EQ(expect_keys_tmp, keys_tmp);
    EXPECT_EQ(expect_values_tmp, values_tmp);
  }
}
