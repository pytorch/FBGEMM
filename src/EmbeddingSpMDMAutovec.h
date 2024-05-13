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
#include "fbgemm/Utils.h"

#ifdef _WIN32
#define do_prefetch(...)
#else
#define do_prefetch(...) __builtin_prefetch(__VA_ARGS__)
#endif

namespace fbgemm {
template <
    typename InType = std::uint8_t,
    typename IndexType = std::int64_t,
    typename OffsetType = std::int32_t,
    typename OutType = float>
FBGEMM_API bool EmbeddingSpMDM_autovec(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    OutType* out,
    bool is_weight_positional = false,
    bool use_offsets = true,
    std::int64_t output_stride = -1,
    std::int64_t input_stride = -1,
    bool scale_bias_last = true,
    bool no_bag = false,
    bool is_bf16_out = false,
    bool is_bf16_in = false);

template <typename IndexType, typename OffsetType, typename OutType>
FBGEMM_API bool EmbeddingSpMDM8Bit_autovec(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const uint8_t* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    const bool normalize_by_lengths,
    OutType* out,
    const bool is_weight_positional /*=false*/,
    const bool use_offsets /*=true*/,
    int64_t output_stride /*=-1*/,
    int64_t input_stride /*=-1*/,
    const bool scale_bias_last /*=true*/,
    const bool no_bag /*=false*/,
    const bool is_bf16_out /*=false*/);

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

template <
    typename InType = float, 
    typename IndexType = std::int64_t, 
    typename OffsetType = std::int32_t>
FBGEMM_API bool EmbeddingSpMDMRowWiseSparse_autovec(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t uncompressed_data_size,
    // const int64_t compressed_data_size,
    const InType* input,
    const IndexType* indices,
    const std::int32_t* compressed_indices_table,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional = false,
    bool use_offsets = true);
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
ALIAS_TEMPLATE_FUNCTION(EmbeddingSpMDM8Bit_autovec, EmbeddingSpMDM_ref)
ALIAS_TEMPLATE_FUNCTION(EmbeddingSpMDM_autovec, EmbeddingSpMDM_ref)
ALIAS_TEMPLATE_FUNCTION(EmbeddingSpMDMRowWiseSparse_autovec, EmbeddingSpMDMRowWiseSparse_ref)

} // namespace fbgemm

#endif // #ifdef __linux__
