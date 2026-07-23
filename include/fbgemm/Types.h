/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace fbgemm {

struct float16 {
  uint16_t val;
  bool operator==(const float16&) const = default;
};

struct bfloat16 {
  uint16_t val;
  bool operator==(const bfloat16&) const = default;
};

static_assert(sizeof(float16) == 2);
static_assert(sizeof(bfloat16) == 2);
// float16/bfloat16 must stay layout- and ABI-compatible with uint16_t: the
// reinterpret_cast boundaries, the memcpy/bit_cast data paths, and the metablas
// legacy-ABI shim (src/FbgemmFP16.cc) all rely on it.
static_assert(std::is_standard_layout_v<float16>);
static_assert(std::is_trivially_copyable_v<float16>);
static_assert(alignof(float16) == alignof(uint16_t));
static_assert(std::is_standard_layout_v<bfloat16>);
static_assert(std::is_trivially_copyable_v<bfloat16>);
static_assert(alignof(bfloat16) == alignof(uint16_t));

constexpr int64_t round_up(int64_t val, int64_t unit) {
  return (val + unit - 1) / unit * unit;
}

constexpr int64_t div_up(int64_t val, int64_t unit) {
  return (val + unit - 1) / unit;
}

} // namespace fbgemm
