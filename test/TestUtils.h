/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cpuinfo.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "fbgemm/Utils.h"

namespace fbgemm {

// Groupwise and i8 depthwise conv kernels are JIT-generated x86 asm (AVX2+)
// with no other implementation; their entry points throw elsewhere, so tests
// that reach them must skip or the uncaught throw aborts the whole binary.
// Runtime checks, not #ifdefs: a non-AVX2 x86 host throws too.
inline bool hasGroupwiseKernels() {
#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
  return cpuinfo_initialize() && fbgemmHasAvx2Support();
#else
  return false;
#endif
}

inline bool hasDepthwiseKernels() {
  return hasGroupwiseKernels(); // same JIT support today; separate for intent
}

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
