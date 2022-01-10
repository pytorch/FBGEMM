/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>

namespace {

// Keep in sync with split_embedding_configs.py:SparseType
enum class SparseType : uint8_t {
  FP32 = 0,
  FP16 = 1,
  INT8 = 2,
  INT4 = 3,
  INT2 = 4,
  BF16 = 5,
  INVALID = 6,
};

enum class PoolingMode : uint8_t { SUM = 0, MEAN = 1, NONE = 2 };

// Keep in sync with EmbeddingLocation in split_table_batched_embeddings_ops.py
enum class PlacementType : uint8_t {
  DEVICE = 0,
  MANAGED = 1,
  MANAGED_CACHING = 2,
  HOST = 3,
};

enum class BoundsCheckMode : uint8_t {
  FATAL = 0,
  WARNING = 1,
  IGNORE = 2,
};

} // namespace
