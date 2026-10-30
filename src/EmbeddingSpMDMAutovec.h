/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include "fbgemm/FbgemmEmbedding.h"

#ifndef DISABLE_FBGEMM_AUTOVEC
#define FBGEMM_AUTOVEC_AVAILABLE
#endif

namespace fbgemm {

template <
    typename InType,
    typename IndexType,
    typename OffsetType,
    typename OutType>
EmbeddingSpMDMKernel<InType, IndexType, OffsetType, OutType>
GenerateEmbeddingSpMDMWithStrides_autovec(
    int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets,
    int64_t output_stride,
    int64_t input_stride,
    bool scale_bias_last,
    bool no_bag);

template <typename IndexType, typename OffsetType, typename OutType>
EmbeddingSpMDMKernel<uint8_t, IndexType, OffsetType, OutType>
GenerateEmbeddingSpMDMNBitWithStrides_autovec(
    int input_bit_rate,
    int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets,
    int64_t output_stride,
    int64_t input_stride,
    bool scale_bias_last,
    bool no_bag,
    int output_bit_rate);

// Compile-time weight selection. The runtime-bool overload remains available
// for callers whose configuration is determined at runtime.
template <bool HasWeight, typename IndexType, typename OffsetType = int32_t,
          typename OutType = float>
auto GenerateEmbeddingSpMDMNBitWithStrides_autovec(
    int input_bit_rate, int64_t block_size, bool normalize_by_lengths,
    int prefetch = 16, bool is_weight_positional = false,
    bool use_offsets = true, int64_t output_stride = -1,
    int64_t input_stride = -1, bool scale_bias_last = true, bool no_bag = false,
    int output_bit_rate = -1) {
  return EmbeddingSpMDMNBitKernel<HasWeight, IndexType, OffsetType, OutType>(
      GenerateEmbeddingSpMDMNBitWithStrides_autovec<IndexType, OffsetType,
                                                    OutType>(
          input_bit_rate, block_size, HasWeight, normalize_by_lengths, prefetch,
          is_weight_positional, use_offsets, output_stride, input_stride,
          scale_bias_last, no_bag, output_bit_rate));
}

template <typename IndexType, typename OffsetType, typename OutType>
EmbeddingSpMDMKernel<uint8_t, IndexType, OffsetType, OutType>
GenerateEmbeddingSpMDMFP8WithStrides_autovec(
    int64_t block_size,
    bool normalize_by_lengths,
    bool is_weight_positional,
    bool use_offsets,
    int64_t output_stride,
    int64_t input_stride,
    int exponent_bits,
    int exponent_bias);

template <typename InType, typename IndexType, typename OffsetType>
typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    InType,
    IndexType,
    OffsetType>::Type
GenerateEmbeddingSpMDMRowWiseSparse_autovec(
    int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets);

template <typename IndexType, typename OffsetType>
typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    uint8_t,
    IndexType,
    OffsetType>::Type
GenerateEmbeddingSpMDMNBitRowWiseSparse_autovec(
    int bit_rate,
    int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets);

} // namespace fbgemm
