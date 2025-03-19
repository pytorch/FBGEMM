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

enum class FloatFormat : std::uint8_t {
  DEFAULT, // `float` (aka IEEE754 "single").
  FLOAT16, // float16 (aka IEEE754 "half") passed as `uint16_t`
  BFLOAT16, // bfloat16 passed as `uint16_t`. https://arxiv.org/abs/1905.12322v3
};

inline std::int64_t round_up(std::int64_t val, std::int64_t unit) {
  return (val + unit - 1) / unit * unit;
}

inline std::int64_t div_up(std::int64_t val, std::int64_t unit) {
  return (val + unit - 1) / unit;
}

} // namespace fbgemm
