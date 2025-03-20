/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace fbgemm {

using float16 = std::uint16_t;
using bfloat16 = std::uint16_t;

inline int64_t round_up(int64_t val, int64_t unit) {
  return (val + unit - 1) / unit * unit;
}

inline int64_t div_up(int64_t val, int64_t unit) {
  return (val + unit - 1) / unit;
}

} // namespace fbgemm
