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
#include <ATen/detail/CUDAHooksInterface.h>
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

// ---------------------------------------------------------------------------
// NFP8 (FP8 e4m3) dtype flow on ROCm -- read this first
//
//   1. Allocation     getNFP8ScalarType() (C++) and nfp8_dtype() (Python) label
//                     the weight buffer per-arch: fnuz on gfx94x/gfx90a, OCP fn
//                     on gfx950 and CUDA. This is the label Python sees, and it
//                     is never changed.
//   2. Host boundary  relabel_nfp8_for_dispatch() takes a *view* labeled fnuz
//                     immediately before the tensor enters a kernel, because
//                     only the fnuz emb_t variant is instantiated. Metadata
//                     only -- no copy, no conversion, no kernel launch.
//   3. Dispatch       dispatch_emb_cache_types() selects that single fnuz
//                     instantiation.
//   4. Device         loads/stores reinterpret through the __nv_fp8_e4m3 alias
//                     (utils/float.cuh), which the HIP headers bind to the
//                     arch's physical encoding per device-compile pass. The c10
//                     label is discarded here, so a gfx950 kernel does OCP fn
//                     math no matter which label step 2 applied.
//
// INVARIANT: every host entry point that hands NFP8 embedding weights to a TBE
// kernel must call relabel_nfp8_for_dispatch() first, and must not let the
// relabeled view escape back to Python. Forget the first half and the tensor is
// rejected by TensorAccessorBuilder or data_ptr<emb_t>(); forget the second and
// callers observe the wrong dtype.
// ---------------------------------------------------------------------------

// Resolves the native FP8 (e4m3) scalar type for the current runtime device.
//
// The FP8 encoding is hardware-specific: the gfx94x family (gfx940/941/942,
// MI300) and gfx90a use the "fnuz" encoding, while gfx950 and CUDA use the OCP
// "fn" encoding. Because a ROCm build can be a fat binary spanning multiple
// archs, this cannot be a compile-time decision on the host: it must query the
// device actually in use at runtime so the host-allocated tensor dtype matches
// what device kernels (whose format is selected per-arch at device-compile
// time) read and write.
inline at::ScalarType getNFP8ScalarType() {
#ifdef USE_ROCM
  // fnuz archs: the gfx94x family (gfx940/941/942, MI300) and gfx90a. The
  // substring match mirrors split_embedding_configs.py:_nfp8_is_fnuz; keep the
  // two in sync. Query goes through the ATen-cpu CUDA hooks so this header is
  // safe to compile into the CPU/meta libraries (no ATen/cuda/CUDAContext.h).
  const auto& cuda_hooks = at::detail::getCUDAHooks();
  if (cuda_hooks.hasCUDA() && cuda_hooks.isGPUArch({"gfx94", "gfx90a"})) {
    return at::kFloat8_e4m3fnuz;
  }
#endif
  return at::kFloat8_e4m3fn;
}

// Relabels an NFP8 tensor's scalar type to fnuz for kernel dispatch.
// Step 2 of the flow documented above.
//
// IMPORTANT: this does NOT change the FP8 encoding and does NOT force fnuz
// numerics. It only rewrites the host tensor's dtype tag, on a view, so that it
// matches the emb_t template parameter of the single instantiated kernel. The
// physical encoding stays arch-bound in device code (step 4), so a gfx950
// kernel does OCP fn math regardless of the label applied here.
//
// It is needed because emb_t must EQUAL the tensor's scalar type:
// TensorAccessorBuilder::checkTensorConstraints and, un-patchably,
// at::TensorBase::data_ptr<T>() both enforce that. Without this the gfx950
// tensor (labeled fn by getNFP8ScalarType) cannot be passed to the fnuz kernel.
//
// Cost: metadata-only, ~400 ns and independent of tensor size.
//
// Exception worth knowing: the ROCm optimized backward converts emb_t through
// the c10 FP8 types instead of the alias, which IS label-bound (exponent bias 8
// vs 7), so step 4 does not apply there. FP8 is excluded from that path on the
// host, pinned by a static_assert in
// codegen/training/backward/embedding_backward_split_template.cu.
inline at::Tensor relabel_nfp8_for_dispatch(const at::Tensor& tensor) {
#ifdef USE_ROCM
  if (tensor.defined() && tensor.scalar_type() == at::kFloat8_e4m3fn) {
    return tensor.view(at::kFloat8_e4m3fnuz);
  }
#endif
  return tensor;
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
