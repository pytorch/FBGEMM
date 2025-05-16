/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __linux__

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
typename EmbeddingSpMDMKernelSignature<InType, IndexType, OffsetType, OutType>::
    Type
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
        bool no_bag,
        bool is_bf16_out,
        bool is_bf16_in);

template <typename IndexType, typename OffsetType, typename OutType>
typename EmbeddingSpMDMKernelSignature<
    uint8_t,
    IndexType,
    OffsetType,
    OutType>::Type
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
    bool is_bf16_out,
    bool no_bag,
    int output_bit_rate);

template <typename IndexType, typename OffsetType, typename OutType>
typename EmbeddingSpMDMKernelSignature<
    uint8_t,
    IndexType,
    OffsetType,
    OutType>::Type
GenerateEmbeddingSpMDMFP8WithStrides_autovec(
    int64_t block_size,
    bool normalize_by_lengths,
    bool is_weight_positional,
    bool use_offsets,
    int64_t output_stride,
    int64_t input_stride,
    int exponent_bits,
    int exponent_bias,
    bool is_bf16_out);

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

} // namespace fbgemm

#endif // #ifdef __linux__
