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
} // namespace fbgemm
