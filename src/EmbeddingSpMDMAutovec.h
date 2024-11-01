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

#define FBGEMM_AUTOVEC_AVAILABLE

/// @defgroup tbe-cpu-autovec TBE CPU Autovectorization (FP8/16/32)
///

namespace fbgemm {
/// @ingroup tbe-cpu-autovec
///
/// Autovectorized version of method `EmbeddingSpMDM_ref` for FP32 weight type.
///
/// @tparam InType input data type (`uint8_t` is used)
///
/// @tparam IndexType index data type (`int64_t` is used)
///
/// @tparam OffsetType offset data type (`int32_t` is used)
///
/// @tparam OutType output data type (`float` is used)
///
///  @param block_size Number of elements in a block (`int64_t`)
///  @param output_size Number of elements in output (`int64_t`)
///  @param index_size Number of elements in index (`int64_t`)
///  @param data_size Number of elements in data (`int64_t`)
///  @param input Address of input (`InType*`)
///  @param indices Address of index (`IndexType*`)
///  @param offsets_or_lengths Address of offset (`OffsetType*`)
///  @param weights Weights of sum; optional, can be null for non-weighted sum
///  (`float*`)
///  @param normalize_by_lengths Whether or not to normalize by lengths (`bool`)
///  @param out Address of output (`OutType*`)
///  @param is_weight_positional If `true`, weight is positional; set to `false`
///  for FP32 autovec implementation (`bool`)
///  @param use_offsets If `true`, will use offsets instead of lengths; set to
///  `true` for FP32 autovec implementation (`bool`)
///  @param output_stride If -1, output_stride is same as block_size; set to -1
///  for FP32 autovec implementation (`int64_t`)
///  @param input_stride If -1, input_stride is same as block_size; set to -1
///  for FP32 autovec implementation (`int64_t`)
///  @param scale_bias_last If `true`, scale and bias appear at end of each row;
///  set to `true` for FP32 autovec implementation (`bool`)
///  @param no_bag If `true`, no embedding bag; set to `false` for FP32 autovec
///  implementation (`bool`)
///  @param is_bf16_out If `true`, output is `BFLOAT16` type; set to `false` for
///  FP32 autovec implementation (`bool`)
///  @param is_bf16_in If `true`, input is `BFLOAT16` type; set to `false` for
///  FP32 autovec implementation (`bool`)
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
    const int input_bit_rate,
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
    const bool is_bf16_out = false,
    const bool no_bag = false,
    int output_bit_rate = -1);

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

/// @ingroup tbe-cpu-autovec
///
/// Autovectorized version of method `EmbeddingSpMDM_ref` for FP8 weight type.
///
/// @tparam InType input data type (`uint8_t` is used)
///
/// @tparam IndexType index data type (`int64_t` is used)
///
/// @tparam OffsetType offset data type (`int32_t` is used)
///
/// @tparam OutType output data type (`float` is used)
///
///  @param block_size Number of elements in a block (`int64_t`)
///  @param output_size Number of elements in output (`int64_t`)
///  @param index_size Number of elements in index (`int64_t`)
///  @param data_size Number of elements in data (`int64_t`)
///  @param input Address of input (`InType*`)
///  @param indices Address of index (`IndexType*`)
///  @param offsets_or_lengths Address of offset (`OffsetType*`)
///  @param weights Weights of sum; optional, can be null for non-weighted sum
///  (`float*`)
///  @param normalize_by_lengths Whether or not to normalize by lengths (`bool`)
///  @param out Address of output (`OutType*`)
///  @param is_weight_positional If `true`, weight is positional; set to `false`
///  for FP8 autovec implementation (`bool`)
///  @param use_offsets If `true`, will use offsets instead of lengths; set to
///  `true` for FP8 autovec implementation (`bool`)
///  @param output_stride If -1, output_stride is same as block_size; set to -1
///  for FP8 autovec implementation (`int64_t`)
///  @param exponent_bits Bits to use in exponent
///  @param exponent_bias Bias to use in exponent
///  @param is_bf16_out If `true`, output is `BFLOAT16` type; set to `false` for
///  FP8 autovec implementation (`bool`)
template <
    typename IndexType = std::int64_t,
    typename OffsetType = std::int32_t,
    typename OutType = float>
bool EmbeddingSpMDMFP8_autovec(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const uint8_t* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights,
    bool normalize_by_lengths,
    OutType* out,
    bool is_weight_positional = false,
    bool use_offsets = true,
    int64_t output_stride = -1,
    int64_t input_stride = -1,
    int exponent_bits = 4,
    int exponent_bias = 7,
    bool is_bf16_out = false);
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
ALIAS_TEMPLATE_FUNCTION(
    EmbeddingSpMDMRowWiseSparse_autovec,
    EmbeddingSpMDMRowWiseSparse_ref)
ALIAS_TEMPLATE_FUNCTION(EmbeddingSpMDMFP8_autovec, EmbeddingSpMDMFP8_ref)

} // namespace fbgemm

#endif // #ifdef __linux__
