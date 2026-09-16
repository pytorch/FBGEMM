/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <type_traits>

namespace fbgemm_gpu::utils {

template <typename... Integers>
inline uint64_t check_product_with_limit(
    const char* context,
    const char* expression,
    uint64_t inclusive_limit,
    Integers... values) {
  static_assert(
      (std::is_integral_v<Integers> && ...),
      "check_product_with_limit values must be integral");

  uint64_t product = 1;
  const auto multiply = [&](auto value) {
    using Integer = decltype(value);
    if constexpr (std::is_signed_v<Integer>) {
      TORCH_CHECK(
          value >= 0,
          context,
          ": expression ",
          expression,
          " must be nonnegative, but received ",
          value);
    }

    const auto factor = static_cast<uint64_t>(value);
    TORCH_CHECK(
        factor == 0 || product <= inclusive_limit / factor,
        context,
        ": product of ",
        expression,
        " exceeds the supported limit ",
        inclusive_limit);
    product *= factor;
  };
  (multiply(values), ...);
  return product;
}

} // namespace fbgemm_gpu::utils
