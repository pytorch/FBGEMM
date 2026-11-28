/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/Types.h"
#include "fbgemm/detail/PlatformSpecificKernelsInterface.h"

namespace fbgemm {

/**
 * @ Transform all entries in a matrix from fp32 to bfloat16: reference
 * implementation.
 *
 */
FBGEMM_API void
FloatToBfloat16_ref(const float* src, bfloat16* dst, size_t size);

/**
 * @ Transform all entries in a matrix from bfloat16 to fp32: reference
 * implementation.
 *
 */
FBGEMM_API void
Bfloat16ToFloat_ref(const bfloat16* src, float* dst, size_t size);

/**
 * @ Transform all entries in a matrix from fp32 to bfloat16: simd
 * implementation.
 *
 */
inline void FloatToBfloat16_simd(const float* src, bfloat16* dst, size_t size) {
  platformKernels().Float32ToBFloat16(src, dst, size);
}

/**
 * @ Transform all entries in a matrix from bfloat16 to fp32: simd
 * implementation.
 *
 */
inline void Bfloat16ToFloat_simd(const bfloat16* src, float* dst, size_t size) {
  platformKernels().BFloat16ToFloat32(src, dst, size);
}

/**
 * @ Transform all entries in a matrix from fp32 to float16: reference
 * implementation.
 *
 * @param do_clip if true we saturate to fp16 min and max instead of generating
 *                infinities.
 */
FBGEMM_API void FloatToFloat16_ref(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip = false);

/**
 * @ Transform all entries in a matrix from float16 to fp32: reference
 * implementation.
 *
 */
FBGEMM_API void Float16ToFloat_ref(const float16* src, float* dst, size_t size);

/**
 * @ Transform all entries in a matrix from fp32 to float16: simd
 * implementation.
 *
 */
inline void FloatToFloat16_simd(const float* src, float16* dst, size_t size) {
  platformKernels().Float32ToFloat16(src, dst, size);
}

/**
 * @ Transform all entries in a matrix from fp32 to float16: simd
 * implementation.
 *
 * @param do_clip if true we saturate to fp16 min and max instead of generating
 *                infinities.
 */
inline void
FloatToFloat16_simd(const float* src, float16* dst, size_t size, bool do_clip) {
  if (do_clip) {
    platformKernels().Float32ToFloat16Clipping(src, dst, size);
  } else {
    platformKernels().Float32ToFloat16(src, dst, size);
  }
}

/**
 * @ Transform all entries in a matrix from float16 to fp32: simd
 * implementation.
 *
 */
inline void Float16ToFloat_simd(const float16* src, float* dst, size_t size) {
  platformKernels().Float16ToFloat32(src, dst, size);
}

#if !defined(__aarch64__)
/**
 * @brief Same as FloatToFloat16_simd, kept for existing callers.
 *
 */
inline void FloatToFloat16_avx2(const float* src, float16* dst, size_t size) {
  FloatToFloat16_simd(src, dst, size);
}

/**
 * @brief Same as FloatToFloat16_simd, kept for existing callers.
 *
 * @param do_clip if true we saturate to fp16 min and max instead of generating
 *                infinities.
 */
inline void
FloatToFloat16_avx2(const float* src, float16* dst, size_t size, bool do_clip) {
  FloatToFloat16_simd(src, dst, size, do_clip);
}

/**
 * @brief Same as Float16ToFloat_simd, kept for existing callers.
 *
 */
inline void Float16ToFloat_avx2(const float16* src, float* dst, size_t size) {
  Float16ToFloat_simd(src, dst, size);
}
#endif

/**
 * @brief Transform all entries in a matrix from fp32 to float16 and back to
 * fp32.
 */
FBGEMM_API void RoundToFloat16(
    const float* input,
    float* output,
    size_t size,
    bool clamp = false,
    bool clamp_denorms = false);

/**
 * @brief Quantize float32 to float8. The code is a copy of float_to_hfp8() in
 * fbgemm_gpu/quantize_ops_utils.h
 */
FBGEMM_API void FloatToFloat8_ref(
    float input,
    uint8_t* output,
    int exponent_bits,
    int exponent_bias);

/**
 * @brief Dequantize float8 to float32. The code is a copy of hf8_to_float() in
 * fbgemm_gpu/quantize_ops_utils.h
 */
FBGEMM_API void Float8ToFloat_ref(
    uint8_t input,
    float* output,
    int exponent_bits,
    int exponent_bias);

} // namespace fbgemm
