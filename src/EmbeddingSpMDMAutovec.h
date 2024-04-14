/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __linux__

#include <algorithm>
#include <cstdint>

#include "fbgemm/ConvUtils.h"
#include "fbgemm/FbgemmI8Spmdm.h"
#include "fbgemm/Types.h"

#ifdef _WIN32
#define do_prefetch(...)
#else
#define do_prefetch(...) __builtin_prefetch(__VA_ARGS__)
#endif

namespace fbgemm {

template <
    typename IndexType = std::int64_t,
    typename OffsetType = std::int32_t,
    typename OutType = float>
FBGEMM_API bool EmbeddingSpMDMNBit_autovec(
    const int bit_rate,
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const std::uint8_t* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    const bool normalize_by_lengths,
    OutType* out,
    const bool is_weight_positional = false,
    const bool use_offsets = true,
    std::int64_t output_stride = -1,
    std::int64_t input_stride = -1,
    const bool scale_bias_last = true,
    const bool is_bf16_out = false);

} // namespace fbgemm

#else // #ifdef __linux__

#include "RefImplementations.h"

#define ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF)                      \
  template <typename... Args>                                               \
  inline auto highLevelF(                                                   \
      Args&&... args) -> decltype(lowLevelF(std::forward<Args>(args)...)) { \
    return lowLevelF(std::forward<Args>(args)...);                          \
  }

namespace fbgemm {

ALIAS_TEMPLATE_FUNCTION(EmbeddingSpMDMNBit_autovec, EmbeddingSpMDMNBit_ref)

} // namespace fbgemm

#endif // #ifdef __linux__
