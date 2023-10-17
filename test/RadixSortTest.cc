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

#include "fbgemm/Utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
template <typename T, unsigned N>
void test_template(
    std::array<T, N> keys,
    std::array<T, N> values,
    std::array<T, N> expected_keys,
    std::array<T, N> expected_values,
    T max_val = std::numeric_limits<T>::max(),
    bool may_be_neg = std::is_signed_v<T>) {
  std::array<T, N> keys_tmp;
  std::array<T, N> values_tmp;
  const auto [sorted_keys, sorted_values] = fbgemm::radix_sort_parallel(
      keys.data(),
      values.data(),
      keys_tmp.data(),
      values_tmp.data(),
      keys.size(),
      max_val,
      may_be_neg);
  if (sorted_keys == keys.data()) { // even number of passes
    EXPECT_EQ(expected_keys, keys);
    EXPECT_EQ(expected_values, values);
  } else { // odd number of passes
    EXPECT_EQ(expected_keys, keys_tmp);
    EXPECT_EQ(expected_values, values_tmp);
  }
}

} // anonymous namespace

TEST(cpuKernelTest, radix_sort_parallel_test) {
  test_template<int, 8>(
      {1, 2, 4, 5, 4, 3, 2, 9},
      {0, 0, 0, 0, 1, 1, 1, 1},
      {1, 2, 2, 3, 4, 4, 5, 9},
      {0, 0, 1, 1, 0, 1, 0, 1},
      10,
      false);
}

TEST(cpuKernelTest, radix_sort_parallel_test_neg_vals) {
  test_template<int64_t, 8>(
      {-4, -3, 0, 1, -2, -1, 3, 2},
      {0, 0, 0, 0, 1, 1, 1, 1},
      {-4, -3, -2, -1, 0, 1, 2, 3},
      {0, 0, 1, 1, 0, 0, 1, 1});
}

TEST(cpuKernelTest, raidx_sort_heap_overflow) {
#ifdef _OPENMP
  const auto orig_threads = omp_get_num_threads();
  omp_set_num_threads(1);
#endif
  constexpr auto max = std::numeric_limits<int>::max();
  test_template<int, 8>(
      {-1, max, max, -1, max, -1, -1, -1},
      {1, 2, 3, 4, 5, 6, 7, 8},
      {-1, -1, -1, -1, -1, max, max, max},
      {1, 4, 6, 7, 8, 2, 3, 5});
#ifdef _OPENMP
  omp_set_num_threads(orig_threads);
#endif
}
