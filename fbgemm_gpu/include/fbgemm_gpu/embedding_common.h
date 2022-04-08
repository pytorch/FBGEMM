/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
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

at::ScalarType getScalarType(SparseType dtype) {
  switch (dtype) {
    case SparseType::FP32:
      return at::kFloat;
    case SparseType::FP16:
      return at::kHalf;
    case SparseType::INT8:
      return at::kByte;
    case SparseType::BF16:
      return at::kBFloat16;
    default:
      return at::ScalarType::Undefined;
  }
};

SparseType getSparseType(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return SparseType::FP32;
    case at::kHalf:
      return SparseType::FP16;
    case at::kByte:
      return SparseType::INT8;
    case at::kBFloat16:
      return SparseType::BF16;
    default:
      return SparseType::INVALID;
  }
};

} // namespace

namespace nbit {

C10_HOST_DEVICE C10_ALWAYS_INLINE uint32_t round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b) * b;
}

C10_HOST_DEVICE C10_ALWAYS_INLINE uint32_t
div_round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b);
}

C10_HOST_DEVICE C10_ALWAYS_INLINE int32_t
unpadded_row_size_in_bytes(int32_t dim, SparseType weight_ty) {
  if (weight_ty == SparseType::FP32) {
    return dim * 4;
  }
  if (weight_ty == SparseType::FP16) {
    return dim * 2;
  }
  if (weight_ty == SparseType::INT8) {
    return dim + 4;
  }
  if (weight_ty == SparseType::INT4) {
    return dim / 2 + 4;
  }
  if (weight_ty == SparseType::INT2) {
    return dim / 4 + 4;
  }
  return 0;
}

C10_HOST_DEVICE C10_ALWAYS_INLINE int32_t padded_row_size_in_bytes(
    int32_t dim,
    SparseType weight_ty,
    int32_t row_alignment) {
  auto r = unpadded_row_size_in_bytes(dim, weight_ty);
  return round_up(r, row_alignment);
}

} // namespace nbit
