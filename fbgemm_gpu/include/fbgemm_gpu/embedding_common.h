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
  FP8 = 6,
  INVALID = 7,
};

enum class PoolingMode : uint8_t { SUM = 0, MEAN = 1, NONE = 2 };

// Keep in sync with EmbeddingLocation in split_table_batched_embeddings_ops.py
enum class PlacementType : uint8_t {
  DEVICE = 0,
  MANAGED = 1,
  MANAGED_CACHING = 2,
  HOST = 3,
  HOST_FAR = 4,
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
    case SparseType::INT4:
      return at::kQUInt4x2;
    case SparseType::INT2:
      return at::kQUInt2x4;
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
    case at::kChar:
    case at::kQUInt8:
    case at::kQInt8:
      return SparseType::INT8;
    case at::kBFloat16:
      return SparseType::BF16;
    case at::kQUInt4x2:
      return SparseType::INT4;
    case at::kQUInt2x4:
      return SparseType::INT2;
    default:
      return SparseType::INVALID;
  }
};

using int_nbit_split_embedding_codegen_forward_far_cpu_type = at::Tensor (*)(
    at::Tensor /*dev_weights*/,
    at::Tensor /*uvm_weights*/,
    at::Tensor /*weights_placements*/,
    at::Tensor /*weights_offsets*/,
    at::Tensor /*weights_tys*/,
    at::Tensor /*D_offsets*/,
    int64_t /*total_D*/,
    int64_t /* max_int2_D */,
    int64_t /* max_int4_D */,
    int64_t /* max_int8_D */,
    int64_t /* max_float16_D */,
    int64_t /* max_float32_D */,
    at::Tensor /*indices*/,
    at::Tensor /*offsets*/,
    int64_t /*pooling_mode*/,
    c10::optional<at::Tensor> /*per_sample_weights*/,
    int64_t /*output_dtype*/,
    c10::optional<at::Tensor> /*lxu_cache_weights*/,
    c10::optional<at::Tensor> /*lxu_cache_locations*/,
    c10::optional<int64_t> /*row_alignment*/,
    c10::optional<int64_t> /*max_float8_D*/,
    c10::optional<int64_t> /*fp8_exponent_bits*/,
    c10::optional<int64_t> /*fp8_exponent_bias*/);

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
  if (weight_ty == SparseType::FP8) {
    return dim;
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
