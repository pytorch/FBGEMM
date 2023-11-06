/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace fbgemm {

enum EmbeddingSpMDMCornerCase {
  NONE,
  EMPTY_INDICES,
  OUT_OF_BOUND_INDICES,
  UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM,
};

enum EmbeddingSpMDMWeightChoice {
  UNWEIGHTED,
  WEIGHTED,
  POSITIONAL_WEIGHTED,
};

enum EmbeddingSpMDMDtypeChoice {
  FLOAT,
  FLOAT16,
  BFLOAT16,
};

using EmbeddingSpMDMInputDtypeChoice = EmbeddingSpMDMDtypeChoice;
using EmbeddingSpMDMOutputDtypeChoice = EmbeddingSpMDMDtypeChoice;

/**
 * @return lengths_sum
 */
int GenerateLengthsIndicesWeights(
    std::vector<std::int64_t>& lengths,
    std::vector<std::int32_t>& lengths_32,
    std::vector<std::int64_t>& offsets,
    std::vector<std::int32_t>& offsets_32,
    std::vector<std::int64_t>& indices,
    std::vector<std::int32_t>& indices_32,
    std::vector<float>& weights,
    int batch_size,
    int num_rows,
    int average_len,
    EmbeddingSpMDMCornerCase corner_case);

/**
 * @return num_compressed_rows
 */
int CreateMappingTableForRowWiseSparsity(
    std::vector<std::int32_t>& mapping_table,
    int num_rows,
    float sparsity);

}; // namespace fbgemm
