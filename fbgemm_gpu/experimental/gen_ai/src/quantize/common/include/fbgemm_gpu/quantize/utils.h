/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <climits>
#include <cstdint>

namespace fbgemm_gpu {

constexpr int64_t nextPowerOf2(int64_t num) {
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

inline constexpr int64_t roundUp(int64_t num, int64_t roundUp) {
  return ((num + roundUp - 1) / roundUp) * roundUp;
}

constexpr int64_t
nextPowerOf2OrRoundUp(int64_t num, int64_t roundUpTo, int64_t threshold) {
  if (num <= threshold) {
    return nextPowerOf2(num);
  }
  return roundUp(num, roundUpTo);
}

} // namespace fbgemm_gpu
