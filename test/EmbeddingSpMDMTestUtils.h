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

#include "fbgemm/Utils.h"

namespace fbgemm {

enum EmbeddingSpMDMKernelChoice {
  // Use the default dispatch (asmjit on x86, autovec on ARM)
  DISPATCH_DEFAULT,
  // Force the autovec implementation
  DISPATCH_AUTOVEC,
};

// RAII guard to temporarily override kernel dispatch settings.
// Forces the dispatch to use autovec when choice == DISPATCH_AUTOVEC,
// and restores the original settings on destruction.
class ScopedKernelOverride {
 public:
  explicit ScopedKernelOverride(EmbeddingSpMDMKernelChoice choice)
      : prev_autovec_forced_(is_autovec_forced()),
        prev_asmjit_disabled_(is_asmjit_disabled()) {
    if (choice == DISPATCH_AUTOVEC) {
      set_autovec_forced(true);
      set_asmjit_disabled(true);
    }
  }
  ~ScopedKernelOverride() {
    set_autovec_forced(prev_autovec_forced_);
    set_asmjit_disabled(prev_asmjit_disabled_);
  }
  ScopedKernelOverride(const ScopedKernelOverride&) = delete;
  ScopedKernelOverride& operator=(const ScopedKernelOverride&) = delete;

 private:
  bool prev_autovec_forced_;
  bool prev_asmjit_disabled_;
};

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
  QINT8,
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

} // namespace fbgemm
