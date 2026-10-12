/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/utils/count_true.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace fbgemm_gpu::utils {

namespace {

template <std::size_t N, typename T>
constexpr auto split_first_chunk(std::span<T> s)
    -> std::optional<std::pair<std::span<T, N>, std::span<T>>> {
  if (s.size() < N) {
    return std::nullopt;
  }
  return std::pair{s.template first<N>(), s.subspan(N)};
}

} // namespace

int64_t count_true(std::span<const bool> mask) {
  constexpr std::size_t kChunkSize = std::numeric_limits<uint16_t>::max();
  // Autovectorizes well. Can be done better if there is ever a need.
  const auto count_chunk = [](auto chunk) {
    uint16_t count = 0;
    for (const bool x : chunk) {
      count += x ? 1 : 0;
    }
    return count;
  };

  int64_t count = 0;
  while (const auto split = split_first_chunk<kChunkSize>(mask)) {
    count += count_chunk(split->first);
    mask = split->second;
  }
  return count + count_chunk(mask);
}

} // namespace fbgemm_gpu::utils
