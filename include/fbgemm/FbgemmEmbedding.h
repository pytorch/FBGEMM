/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>

namespace fbgemm {
template <typename inType = std::uint8_t, typename IndexType = std::int64_t>
FBGEMM_API bool EmbeddingSpMDM(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const inType* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    int prefetch = 16,
    bool IS_WEIGHT_POSITIONAL = false);

template <typename IndexType>
FBGEMM_API int SparseAdaGrad(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const IndexType* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise = false,
    int prefetch = 16);

} // namespace fbgemm
