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

/**
 * @tparam inType can be float or uint8_t
 * @tparam IndexType can be int32_t or int64_t
 */
template <typename inType, typename IndexType>
FBGEMM_API typename EmbeddingSpMDMKernelSignature<inType, IndexType>::Type
GenerateEmbeddingSpMDM(
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch = 16,
    bool is_weight_positional = false);

/**
 * @tparam IndexType can be int32_t or int64_t
 * @param bit_rate can be 2 or 4
 */
template <typename IndexType>
FBGEMM_API typename EmbeddingSpMDMKernelSignature<std::uint8_t, IndexType>::Type
GenerateEmbeddingSpMDMNBit(
    int bit_rate,
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch = 16,
    bool is_weight_positional = false);

template <typename inType, typename IndexType>
class EmbeddingSpMDMRowWiseSparseKernelSignature {
 public:
  using Type = std::function<bool(
      std::int64_t output_size,
      std::int64_t index_size,
      std::int64_t uncompressed_data_size,
      // TODO: add compressed_data_size and check array bound
      const inType* input,
      const IndexType* indices,
      const int* lengths,
      const float* weights, // optional, can be null for non-weighted sum
      float* out,
      const std::int32_t* compressed_indices_table)>;
};

/**
 * @tparam inType can be float or uint8_t
 * @tparam IndexType can be int32_t or int64_t
 */
template <typename inType, typename IndexType>
FBGEMM_API
    typename EmbeddingSpMDMRowWiseSparseKernelSignature<inType, IndexType>::Type
    GenerateEmbeddingSpMDMRowWiseSparse(
        const std::int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch = 16,
        bool is_weight_positional = false);

/**
 * @tparam IndexType can be int32_t or int64_t
 * @param bit_rate can be 2 or 4
 */
template <typename IndexType>
FBGEMM_API typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    std::uint8_t,
    IndexType>::Type
GenerateEmbeddingSpMDMNBitRowWiseSparse(
    int bit_rate,
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch = 16,
    bool is_weight_positional = false);

/**
 * @return The number of rows processed. If smaller than num_rows, an error
 *         must have happened at the last row processed.
 */
template <typename IndexType>
class SparseAdaGradSignature {
 public:
  using Type = std::function<int(
      int num_rows, // number of rows reading
      std::uint64_t param_size, // total number of parameters
      float* w, // input/output parameters
      const float* g, // input gradients
      float* h, // input/output momentums
      const IndexType* indices, // indices of each row
      float epsilon,
      float lr)>;
};

template <typename IndexType>
FBGEMM_API typename SparseAdaGradSignature<IndexType>::Type
GenerateSparseAdaGrad(
    int block_size, // number of parameters per row
    bool rowwise = false,
    int prefetch = 16);

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

// RowWiseSparseAdaGrad fused with SLS gradient
template <typename IndexType>
class RowWiseSparseAdaGradFusedSignature {
 public:
  using Type = std::function<bool(
      std::int64_t output_size,
      std::int64_t index_size,
      std::int64_t data_size, // number of rows in w
      float* w, // input/output parameters
      const float* g, // input gradients
      float* h, // input/output momentums
      const IndexType* indices, // indices of each row
      const int* lengths,
      float epsilon,
      float lr)>;
};

template <typename IndexType>
FBGEMM_API typename RowWiseSparseAdaGradFusedSignature<IndexType>::Type
GenerateRowWiseSparseAdaGradFused(
    int block_size, // number of parameters per row
    int prefetch = 16);

namespace internal {
// Specialization for block size 1 internally called by GenerateEmbeddingSpMDM
template <typename inType = float, typename IndexType = std::int64_t>
FBGEMM_API bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const inType* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional = false);

} // namespace internal

} // namespace fbgemm
