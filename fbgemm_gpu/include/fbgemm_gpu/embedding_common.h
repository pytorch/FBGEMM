/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <cstdint>
#ifdef USE_ROCM
#include <string_view>

#include <ATen/cuda/CUDAContext.h>
#endif

namespace fbgemm_gpu {

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
  MX4 = 8,
  NFP8 = 9,

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

// Resolves the native FP8 (e4m3) scalar type for the current runtime device.
//
// The FP8 encoding is hardware-specific: gfx942 (MI300) and gfx90a use the
// "fnuz" encoding, while gfx950 and CUDA use the OCP "fn" encoding. Because a
// ROCm build can be a fat binary spanning multiple archs, this cannot be a
// compile-time decision on the host: it must query the device actually in use
// at runtime so the host-allocated tensor dtype matches what device kernels
// (whose format is selected per-arch at device-compile time) read and write.
inline at::ScalarType getNFP8ScalarType() {
#ifdef USE_ROCM
  if (at::cuda::is_available()) {
    const std::string_view arch{
        at::cuda::getCurrentDeviceProperties()->gcnArchName};
    // fnuz archs: gfx942 and gfx90a.
    if (arch.find("gfx94") != std::string_view::npos ||
        arch.find("gfx90a") != std::string_view::npos) {
      return at::kFloat8_e4m3fnuz;
    }
  }
#endif
  return at::kFloat8_e4m3fn;
}

inline at::ScalarType getScalarType(SparseType dtype) {
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
    case SparseType::NFP8:
      return getNFP8ScalarType();
    default:
      return at::ScalarType::Undefined;
  }
};

inline SparseType getSparseType(at::ScalarType dtype) {
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
    case at::kFloat8_e4m3fn:
      return SparseType::NFP8;
    case at::kFloat8_e4m3fnuz:
      return SparseType::NFP8;
    default:
      return SparseType::INVALID;
  }
};

} // namespace fbgemm_gpu

namespace nbit {

C10_HOST_DEVICE C10_ALWAYS_INLINE uint64_t round_up(uint64_t a, uint64_t b) {
  return ((a + b - 1) / b) * b;
}

C10_HOST_DEVICE C10_ALWAYS_INLINE uint32_t
div_round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b);
}

C10_HOST_DEVICE C10_ALWAYS_INLINE int32_t unpadded_row_size_in_bytes(
    int32_t dim,
    fbgemm_gpu::SparseType weight_ty,
    const int32_t scale_bias_bytes = 4) {
  if (weight_ty == fbgemm_gpu::SparseType::FP32) {
    return dim * 4;
  }
  if (weight_ty == fbgemm_gpu::SparseType::FP16) {
    return dim * 2;
  }
  if (weight_ty == fbgemm_gpu::SparseType::FP8) {
    return dim;
  }
  if (weight_ty == fbgemm_gpu::SparseType::NFP8) {
    return dim;
  }
  if (weight_ty == fbgemm_gpu::SparseType::INT8) {
    return dim + scale_bias_bytes;
  }
  if (weight_ty == fbgemm_gpu::SparseType::INT4) {
    return dim / 2 + scale_bias_bytes;
  }
  if (weight_ty == fbgemm_gpu::SparseType::INT2) {
    return dim / 4 + scale_bias_bytes;
  }
  return 0;
}

C10_HOST_DEVICE C10_ALWAYS_INLINE int32_t padded_row_size_in_bytes(
    int32_t dim,
    fbgemm_gpu::SparseType weight_ty,
    const int32_t row_alignment,
    const int32_t scale_bias_bytes = 4) {
  auto r = unpadded_row_size_in_bytes(dim, weight_ty, scale_bias_bytes);
  return static_cast<int32_t>(
      round_up(static_cast<uint64_t>(r), static_cast<uint64_t>(row_alignment)));
}

} // namespace nbit
