/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>
#include <functional>

#include "fbgemm/FbgemmBuild.h"

namespace fbgemm {

template <typename inType, typename IndexType>
class EmbeddingSpMDMKernelSignature {
 public:
  using Type = std::function<bool(
      std::int64_t output_size,
      std::int64_t index_size,
      std::int64_t data_size,
      const inType* input,
      const IndexType* indices,
      const int* lengths,
      const float* weights, // optional, can be null for non-weighted sum
      float* out)>;
};

template <typename inType, typename IndexType>
FBGEMM_API typename EmbeddingSpMDMKernelSignature<inType, IndexType>::Type
GenerateEmbeddingSpMDM(
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch = 16,
    bool is_weight_positional = false);

template <typename inType = std::uint8_t, typename IndexType = std::int64_t>
FBGEMM_API bool EmbeddingSpMDM(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const inType* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    int prefetch = 16,
    bool is_weight_positional = false);

template <typename IndexType>
FBGEMM_API typename EmbeddingSpMDMKernelSignature<std::uint8_t, IndexType>::Type
GenerateEmbeddingSpMDM4Bit(
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch = 16,
    bool is_weight_positional = false);

template <typename IndexType = std::int64_t>
FBGEMM_API bool EmbeddingSpMDM4Bit(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const std::uint8_t* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    int prefetch = 16,
    bool is_weight_positional = false);

/**
 * @return The number of rows processed. If smaller than num_rows, an error
 *         must have happened at the last row processed.
 */
template <typename IndexType>
FBGEMM_API int SparseAdaGrad(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size, // total number of parameters
    float* w, // input/output parameters
    const float* g, // input gradients
    float* h, // input/output momentums
    const IndexType* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise = false,
    int prefetch = 16);

namespace internal {
// Specialization for block size 1 internally called by GenerateEmbeddingSpMDM
template <typename IndexType = std::int64_t>
FBGEMM_API bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const float* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional = false);

} // namespace internal

} // namespace fbgemm
