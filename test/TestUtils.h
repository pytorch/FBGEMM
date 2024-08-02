/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace fbgemm {

/*
 * @brief Check and validate the buffers for reference and FBGEMM result.
 */
template <typename T>
int compare_validate_buffers(
    const T* ref,
    const T* test,
    int m,
    int n,
    int ld,
    T atol);

/*
 * @brief Check if all entries are zero or not.
 * If any entry is non-zero, return True;
 * otherwise, return False.
 */
template <typename T>
bool check_all_zero_entries(const T* test, int m, int n);

// atol: absolute tolerance. <=0 means do not consider atol.
// rtol: relative tolerance. <=0 means do not consider rtol.
template <typename a_T, typename b_T>
::testing::AssertionResult floatCloseAll(
    const std::vector<a_T>& a,
    const std::vector<b_T>& b,
    const float atol = std::numeric_limits<float>::epsilon(),
    const float rtol = 0);

} // namespace fbgemm
