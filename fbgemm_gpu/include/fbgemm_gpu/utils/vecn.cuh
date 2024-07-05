/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/float.cuh"
#include "fbgemm_gpu/utils/types.h"

namespace fbgemm_gpu {

enum class PrimitiveType : uint8_t { FP = 0, INT = 1, BF = 2 };

__forceinline__ __device__ half16
dequantize_permuted_int2(uint32_t packedVals, __half2 shift_scale) {
  half16 res;
  uint32_t v = packedVals;
  // See comment below, this is a minor variation. Check N1600402.
  res.vals[0] = hmul_short2(v & 0x00030003, __float2half(32768));
  res.vals[1] = hmul_short2(v & 0x000C000C, __float2half(32768));
  res.vals[2] = hmul_short2(v & 0x00300030, __float2half(32768));
  res.vals[3] = hmul_short2(v & 0x00C000C0, __float2half(32768));
  v >>= 8;
  res.vals[4] = hmul_short2(v & 0x00030003, __float2half(32768));
  res.vals[5] = hmul_short2(v & 0x000C000C, __float2half(32768));
  res.vals[6] = hmul_short2(v & 0x00300030, __float2half(32768));
  res.vals[7] = hmul_short2(v & 0x00C000C0, __float2half(32768));

  // ~5% perf gain is observed with the explicit type conversions using
  // __float2half on Nvidia A100 GPUs (https://fburl.com/diff/ss8372zw) using
  // NVCC 11.0. Additionally, HIP compiler requires these explicit type
  // conversions.
  half shift_scale_x = __low2half(shift_scale);
  half shift_scale_y = __high2half(shift_scale);

  res.vals[0] = hfma2(
      res.vals[0],
      __half2(
          hmul(shift_scale_x, __float2half(512)),
          hmul(shift_scale_x, __float2half(512))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[1] = hfma2(
      res.vals[1],
      __half2(
          hmul(shift_scale_x, __float2half(128)),
          hmul(shift_scale_x, __float2half(128))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[2] = hfma2(
      res.vals[2],
      __half2(
          hmul(shift_scale_x, __float2half(32)),
          hmul(shift_scale_x, __float2half(32))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[3] = hfma2(
      res.vals[3],
      __half2(
          hmul(shift_scale_x, __float2half(8)),
          hmul(shift_scale_x, __float2half(8))),
      __half2(shift_scale_y, shift_scale_y));

  res.vals[4] = hfma2(
      res.vals[4],
      __half2(
          hmul(shift_scale_x, __float2half(512)),
          hmul(shift_scale_x, __float2half(512))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[5] = hfma2(
      res.vals[5],
      __half2(
          hmul(shift_scale_x, __float2half(128)),
          hmul(shift_scale_x, __float2half(128))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[6] = hfma2(
      res.vals[6],
      __half2(
          hmul(shift_scale_x, __float2half(32)),
          hmul(shift_scale_x, __float2half(32))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[7] = hfma2(
      res.vals[7],
      __half2(
          hmul(shift_scale_x, __float2half(8)),
          hmul(shift_scale_x, __float2half(8))),
      __half2(shift_scale_y, shift_scale_y));
  return res;
}

__forceinline__ __device__ half8
dequantize_permuted_int4(uint32_t packedVals, __half2 shift_scale) {
  half8 res;
  uint32_t v = packedVals;
  // What's going on here, you might ask? We extra out 4-bit pairs of integers
  // as 2xuint16 packed into an int32 via the mask operation, and then we
  // convert them to half precision values. As these are all integers in [0,
  // 15], we can actually just interpret the 4-bit integer values as
  // half-precision values. We multiply by 4096 x 4096 to go from the 4-bit
  // representation to the equivalent fp16 value, or alternatively 32768 * 512
  // (or 32 when we have shifted the 4-bit value up). See e.g.
  // https://gist.github.com/ajtulloch/021254a291a95966bc509db4e34ffeff for a
  // NumPy implementation. We do this dance because: a) doing bitwise operations
  // on each 4-bit value is expensive on the ALU, and 4-bit to half is expensive
  // on the XU. b) doing a 256-entry shared memory LUT on 8-bit pairs is
  // expensive on SMEM throughput. Credit to @jhj.
  res.vals[0] = hmul_short2(v & 0x000F000F, __float2half(32768));
  res.vals[1] = hmul_short2(v & 0x00F000F0, __float2half(32768));
  v >>= 8;
  res.vals[2] = hmul_short2(v & 0x000F000F, __float2half(32768));
  res.vals[3] = hmul_short2(v & 0x00F000F0, __float2half(32768));

  // ~5% perf gain is observed with the explicit type conversions using
  // __float2half on Nvidia A100 GPUs (https://fburl.com/diff/ss8372zw) using
  // NVCC 11.0. Additionally, HIP compiler requires these explicit type
  // conversions.
  half shift_scale_x = __low2half(shift_scale);
  half shift_scale_y = __high2half(shift_scale);

  res.vals[0] = hfma2(
      res.vals[0],
      __half2(
          hmul(shift_scale_x, __float2half(512)),
          hmul(shift_scale_x, __float2half(512))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[1] = hfma2(
      res.vals[1],
      __half2(
          hmul(shift_scale_x, __float2half(32)),
          hmul(shift_scale_x, __float2half(32))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[2] = hfma2(
      res.vals[2],
      __half2(
          hmul(shift_scale_x, __float2half(512)),
          hmul(shift_scale_x, __float2half(512))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[3] = hfma2(
      res.vals[3],
      __half2(
          hmul(shift_scale_x, __float2half(32)),
          hmul(shift_scale_x, __float2half(32))),
      __half2(shift_scale_y, shift_scale_y));
  return res;
}

__forceinline__ __device__ half4
dequantize_permuted_int8(uint32_t packedVals, __half2 shift_scale) {
  half4 res;
  uint32_t v = packedVals;
  // See comment above, this is a minor variation.
  res.vals[0] = hmul_short2(v & 0x00FF00FF, __float2half(32768));
  v >>= 8;
  res.vals[1] = hmul_short2(v & 0x00FF00FF, __float2half(32768));

  half shift_scale_x = __low2half(shift_scale);
  half shift_scale_y = __high2half(shift_scale);

  res.vals[0] = hfma2(
      res.vals[0],
      __half2(
          hmul(shift_scale_x, __float2half(512)),
          hmul(shift_scale_x, __float2half(512))),
      __half2(shift_scale_y, shift_scale_y));
  res.vals[1] = hfma2(
      res.vals[1],
      __half2(
          hmul(shift_scale_x, __float2half(512)),
          hmul(shift_scale_x, __float2half(512))),
      __half2(shift_scale_y, shift_scale_y));
  return res;
}

__forceinline__ __device__ float4
dequantize_packed_hfp8(uint32_t vals, int exp_bits, int exp_bias) {
  union fint128 {
    uint32_t I[4];
    uint64_t L[2];
    float4 F;
  } res, sign;

  union b32 {
    uint32_t I;
    uint8_t S[4];
  } v;

  v.I = vals;

  fint32 multiplier;
  multiplier.I = (127 + (127 - exp_bias)) << 23;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    sign.I[i] = v.S[i] & 0x80;
    res.I[i] = v.S[i] & 0x7F;
  }

  // Shift sign and mantissa bits
  // (Shift 64 bits instead of 8 bits in the above loop)
  sign.L[0] <<= 24;
  sign.L[1] <<= 24;
  res.L[0] <<= (16 + exp_bits);
  res.L[1] <<= (16 + exp_bits);

  // Obtain FP32
  res.F.x *= multiplier.F;
  res.F.y *= multiplier.F;
  res.F.z *= multiplier.F;
  res.F.w *= multiplier.F;

  // Compute sign
  res.L[0] |= sign.L[0];
  res.L[1] |= sign.L[1];

  return res.F;
}

__forceinline__ __device__ float accumulate_fp32(float acc, float vals) {
  acc += vals;
  return acc;
}

__forceinline__ __device__ float
accumulate_weighted_fp32(float acc, float vals, float weight) {
  return fmaf(vals, weight, acc);
}

__forceinline__ __device__ float2 accumulate_fp16(float2 acc, __half2 vals) {
  float2 v = __half22float2(vals);
  acc.x += v.x;
  acc.y += v.y;
  return acc;
}

__forceinline__ __device__ float2
accumulate_weighted_fp16(float2 acc, __half2 vals, float weight) {
  float2 v = __half22float2(vals);
  acc.x = fmaf(v.x, weight, acc.x);
  acc.y = fmaf(v.y, weight, acc.y);
  return acc;
}

__forceinline__ __device__ float4 accumulate_packed_hfp8(
    float4 acc,
    uint32_t packedVals,
    int exp_bits,
    int exp_bias) {
  float4 res = dequantize_packed_hfp8(packedVals, exp_bits, exp_bias);
  acc.x += res.x;
  acc.y += res.y;
  acc.z += res.z;
  acc.w += res.w;
  return acc;
}

__forceinline__ __device__ float4 accumulate_weighted_packed_hfp8(
    float4 acc,
    uint32_t packedVals,
    int exp_bits,
    int exp_bias,
    float weight) {
  float4 res = dequantize_packed_hfp8(packedVals, exp_bits, exp_bias);
  acc.x = fmaf(res.x, weight, acc.x);
  acc.y = fmaf(res.y, weight, acc.y);
  acc.z = fmaf(res.z, weight, acc.z);
  acc.w = fmaf(res.w, weight, acc.w);
  return acc;
}

__forceinline__ __device__ float4
accumulate_packed_int8(float4 acc, uint32_t packedVals, __half2 shift_scale) {
  half4 res = dequantize_permuted_int8(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);

  // Twiddle after permutations.
  acc.x += v0.x;
  acc.y += v1.x;
  acc.z += v0.y;
  acc.w += v1.y;
  return acc;
}

__forceinline__ __device__ float4 accumulate_weighted_packed_int8(
    float4 acc,
    uint32_t packedVals,
    __half2 shift_scale,
    float weight) {
  half4 res = dequantize_permuted_int8(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);

  // Twiddle after permutations.
  acc.x = fmaf(v0.x, weight, acc.x);
  acc.y = fmaf(v1.x, weight, acc.y);
  acc.z = fmaf(v0.y, weight, acc.z);
  acc.w = fmaf(v1.y, weight, acc.w);
  return acc;
}

__forceinline__ __device__ float8
accumulate_packed_int4(float8 acc, uint32_t packedVals, __half2 shift_scale) {
  half8 res = dequantize_permuted_int4(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);
  float2 v2 = __half22float2(res.vals[2]);
  float2 v3 = __half22float2(res.vals[3]);

  // Twiddle after permutations.
  acc.vals[0].x += v0.x;
  acc.vals[0].y += v1.x;
  acc.vals[0].z += v2.x;
  acc.vals[0].w += v3.x;
  acc.vals[1].x += v0.y;
  acc.vals[1].y += v1.y;
  acc.vals[1].z += v2.y;
  acc.vals[1].w += v3.y;
  return acc;
}

__forceinline__ __device__ float8 accumulate_weighted_packed_int4(
    float8 acc,
    uint32_t packedVals,
    __half2 shift_scale,
    float weight) {
  half8 res = dequantize_permuted_int4(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);
  float2 v2 = __half22float2(res.vals[2]);
  float2 v3 = __half22float2(res.vals[3]);

  // Twiddle after permutations.
  acc.vals[0].x = fmaf(v0.x, weight, acc.vals[0].x);
  acc.vals[0].y = fmaf(v1.x, weight, acc.vals[0].y);
  acc.vals[0].z = fmaf(v2.x, weight, acc.vals[0].z);
  acc.vals[0].w = fmaf(v3.x, weight, acc.vals[0].w);
  acc.vals[1].x = fmaf(v0.y, weight, acc.vals[1].x);
  acc.vals[1].y = fmaf(v1.y, weight, acc.vals[1].y);
  acc.vals[1].z = fmaf(v2.y, weight, acc.vals[1].z);
  acc.vals[1].w = fmaf(v3.y, weight, acc.vals[1].w);
  return acc;
}

__forceinline__ __device__ float_16
accumulate_packed_int2(float_16 acc, uint32_t packedVals, __half2 shift_scale) {
  half16 res = dequantize_permuted_int2(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);
  float2 v2 = __half22float2(res.vals[2]);
  float2 v3 = __half22float2(res.vals[3]);
  float2 v4 = __half22float2(res.vals[4]);
  float2 v5 = __half22float2(res.vals[5]);
  float2 v6 = __half22float2(res.vals[6]);
  float2 v7 = __half22float2(res.vals[7]);

  // Twiddle after permutations.
  acc.vals[0].vals[0].x += v0.x;
  acc.vals[0].vals[0].y += v1.x;
  acc.vals[0].vals[0].z += v2.x;
  acc.vals[0].vals[0].w += v3.x;

  acc.vals[0].vals[1].x += v4.x;
  acc.vals[0].vals[1].y += v5.x;
  acc.vals[0].vals[1].z += v6.x;
  acc.vals[0].vals[1].w += v7.x;

  acc.vals[1].vals[0].x += v0.y;
  acc.vals[1].vals[0].y += v1.y;
  acc.vals[1].vals[0].z += v2.y;
  acc.vals[1].vals[0].w += v3.y;

  acc.vals[1].vals[1].x += v4.y;
  acc.vals[1].vals[1].y += v5.y;
  acc.vals[1].vals[1].z += v6.y;
  acc.vals[1].vals[1].w += v7.y;

  return acc;
}

__forceinline__ __device__ float_16 accumulate_weighted_packed_int2(
    float_16 acc,
    uint32_t packedVals,
    __half2 shift_scale,
    float weight) {
  half16 res = dequantize_permuted_int2(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0]);
  float2 v1 = __half22float2(res.vals[1]);
  float2 v2 = __half22float2(res.vals[2]);
  float2 v3 = __half22float2(res.vals[3]);
  float2 v4 = __half22float2(res.vals[4]);
  float2 v5 = __half22float2(res.vals[5]);
  float2 v6 = __half22float2(res.vals[6]);
  float2 v7 = __half22float2(res.vals[7]);

  // Twiddle after permutations.
  acc.vals[0].vals[0].x = fmaf(v0.x, weight, acc.vals[0].vals[0].x);
  acc.vals[0].vals[0].y = fmaf(v1.x, weight, acc.vals[0].vals[0].y);
  acc.vals[0].vals[0].z = fmaf(v2.x, weight, acc.vals[0].vals[0].z);
  acc.vals[0].vals[0].w = fmaf(v3.x, weight, acc.vals[0].vals[0].w);

  acc.vals[0].vals[1].x = fmaf(v4.x, weight, acc.vals[0].vals[1].x);
  acc.vals[0].vals[1].y = fmaf(v5.x, weight, acc.vals[0].vals[1].y);
  acc.vals[0].vals[1].z = fmaf(v6.x, weight, acc.vals[0].vals[1].z);
  acc.vals[0].vals[1].w = fmaf(v7.x, weight, acc.vals[0].vals[1].w);

  acc.vals[1].vals[0].x = fmaf(v0.y, weight, acc.vals[1].vals[0].x);
  acc.vals[1].vals[0].y = fmaf(v1.y, weight, acc.vals[1].vals[0].y);
  acc.vals[1].vals[0].z = fmaf(v2.y, weight, acc.vals[1].vals[0].z);
  acc.vals[1].vals[0].w = fmaf(v3.y, weight, acc.vals[1].vals[0].w);

  acc.vals[1].vals[1].x = fmaf(v4.y, weight, acc.vals[1].vals[1].x);
  acc.vals[1].vals[1].y = fmaf(v5.y, weight, acc.vals[1].vals[1].y);
  acc.vals[1].vals[1].z = fmaf(v6.y, weight, acc.vals[1].vals[1].z);
  acc.vals[1].vals[1].w = fmaf(v7.y, weight, acc.vals[1].vals[1].w);

  return acc;
}

// Customized N-element vector data types (with element type float for
// accumulation type).
template <int N, PrimitiveType>
struct VecNT {};

template <>
struct VecNT<1, PrimitiveType::FP> {
  float acc;

  DEVICE_INLINE VecNT() {
    acc = 0;
  }

  DEVICE_INLINE VecNT(float a) {
    acc = a;
  }

  DEVICE_INLINE void store(float* output_ptr, int num_valid_outputs = 1) {
    *output_ptr = acc;
  }

  DEVICE_INLINE void store(at::Half* output_ptr, int num_valid_outputs = 1) {
    __half val = to_half(acc);
    *reinterpret_cast<__half*>(output_ptr) = val;
  }

#if defined(USE_ROCM) ||                                  \
    !(((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
       (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
  DEVICE_INLINE void store(
      at::BFloat16* output_ptr,
      const int num_valid_outputs = 1) {
    __nv_bfloat16 val = to_bfloat16(acc);
    *reinterpret_cast<__nv_bfloat16*>(output_ptr) = val;
  }
#endif

  DEVICE_INLINE void store(uint8_t* output_ptr, int num_valid_outputs = 1) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(uint8_t* output_ptr, float2 qparams, int num_valid_outputs = 1) {
    const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
    output_ptr[0] = lrintf((acc - qparams.y) * inv_scale);
  }

  DEVICE_INLINE void
  store(float* output_ptr, float2 qparams, int num_valid_outputs = 1) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::Half* output_ptr, float2 qparams, int num_valid_outputs = 1) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::BFloat16* output_ptr, float2 qparams, int num_valid_outputs = 1) {
    CUDA_KERNEL_ASSERT(false);
  }

  // acc <- acc + a * b
  DEVICE_INLINE void fma(float a, float b) {
    acc = accumulate_weighted_fp32(acc, a, b);
  }

  // acc <- acc + a
  DEVICE_INLINE void add(float a) {
    acc = accumulate_fp32(acc, a);
  }

  // acc <- acc * a
  DEVICE_INLINE void mul(float a) {
    acc = acc * a;
  }
};

template <>
struct VecNT<2, PrimitiveType::FP> {
  float2 acc;

  DEVICE_INLINE VecNT() {
    acc = make_zero_float2();
  }

  DEVICE_INLINE VecNT(half2 a) {
    acc = __half22float2(a);
  }

  DEVICE_INLINE void store(float* output_ptr, int num_valid_outputs = 2) {
    // num_valid_outputs can be any integer for half.
    if (uintptr_t(output_ptr) % 8 == 0 && num_valid_outputs == 2) {
      *reinterpret_cast<float2*>(output_ptr) = acc;
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] = *(&acc.x + i);
        }
      }
    }
  }

  DEVICE_INLINE void store(at::Half* output_ptr, int num_valid_outputs = 2) {
    half2 val = to_half2(acc);
    // num_valid_outputs can be any integer for half.
    if (uintptr_t(output_ptr) % 4 == 0 && num_valid_outputs == 2) {
      *reinterpret_cast<half2*>(output_ptr) = val;
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] = *reinterpret_cast<const at::Half*>(&val.x + i);
        }
      }
    }
  }

#if defined(USE_ROCM) ||                                  \
    !(((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
       (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
  DEVICE_INLINE void store(
      at::BFloat16* output_ptr,
      const int num_valid_outputs = 2) {
    __nv_bfloat162 val = to_bfloat16_2(acc);
    // num_valid_outputs can be any integer for half.
    if (uintptr_t(output_ptr) % 4 == 0 && num_valid_outputs == 2) {
      *reinterpret_cast<__nv_bfloat162*>(output_ptr) = val;
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] = *reinterpret_cast<const at::BFloat16*>(&val.x + i);
        }
      }
    }
  }
#endif

  DEVICE_INLINE void store(uint8_t* output_ptr, int num_valid_outputs = 2) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(uint8_t* output_ptr, float2 qparams, int num_valid_outputs = 2) {
    const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (i < num_valid_outputs) {
        output_ptr[i] = lrintf(((&acc.x)[i] - qparams.y) * inv_scale);
      }
    }
  }

  DEVICE_INLINE void
  store(float* output_ptr, float2 qparams, int num_valid_outputs = 2) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::Half* output_ptr, float2 qparams, int num_valid_outputs = 2) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::BFloat16* output_ptr, float2 qparams, int num_valid_outputs = 2) {
    CUDA_KERNEL_ASSERT(false);
  }

  // acc <- acc + a * b
  DEVICE_INLINE void fma(half2 a, float b) {
    acc = accumulate_weighted_fp16(acc, a, b);
  }

  // acc <- acc + a
  DEVICE_INLINE void add(half2 a) {
    acc = accumulate_fp16(acc, a);
  }

  // acc <- acc * a
  DEVICE_INLINE void mul(float a) {
    acc.x *= a;
    acc.y *= a;
  }
};

template <>
struct VecNT<4, PrimitiveType::FP> {
  float4 acc;

  DEVICE_INLINE VecNT() {
    acc = make_zero_float4();
  }

  DEVICE_INLINE VecNT(uint32_t v, const int exp_bits, const int exp_bias) {
    acc = make_zero_float4();
    acc = accumulate_packed_hfp8(acc, v, exp_bits, exp_bias);
  }

  DEVICE_INLINE void store(float* output_ptr, int num_valid_outputs = 4) {
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    // Since byte granule is guaranteed, num_valid_outputs can be any integer
    // for int8.
    if (aligned_16b && num_valid_outputs == 4) {
      *reinterpret_cast<uint4*>(output_ptr) =
          *reinterpret_cast<const uint4*>(&acc);
    } else if (aligned_8b && num_valid_outputs >= 2) {
      *reinterpret_cast<uint2*>(output_ptr) =
          *reinterpret_cast<const uint2*>(&(acc.x));
      if (num_valid_outputs == 4) {
        *reinterpret_cast<uint2*>(output_ptr + 2) =
            *reinterpret_cast<const uint2*>(&(acc.x) + 2);
      } else if (num_valid_outputs == 3) {
        *(output_ptr + 2) = *(&(acc.x) + 2);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] = *(&(acc.x) + i);
        }
      }
    }
  }

  DEVICE_INLINE void store(at::Half* output_ptr, int num_valid_outputs = 4) {
    half4 val = to_half4(acc);
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    // Since byte granule is guaranteed, num_valid_outputs can be any integer
    // for int8.
    if (aligned_8b && num_valid_outputs == 4) {
      *reinterpret_cast<uint2*>(output_ptr) =
          *reinterpret_cast<const uint2*>(&val);
    } else if (aligned_4b && num_valid_outputs >= 2) {
      *reinterpret_cast<uint*>(output_ptr) =
          *reinterpret_cast<const uint*>(&(val.vals[0].x));
      if (num_valid_outputs == 4) {
        *reinterpret_cast<uint*>(output_ptr + 2) =
            *reinterpret_cast<const uint*>(&(val.vals[0].x) + 2);
      } else if (num_valid_outputs == 3) {
        *(output_ptr + 2) =
            *reinterpret_cast<const at::Half*>(&(val.vals[0].x) + 2);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] =
              *reinterpret_cast<const at::Half*>(&(val.vals[0].x) + i);
        }
      }
    }
  }

#if defined(USE_ROCM) ||                                  \
    !(((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
       (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
  DEVICE_INLINE void store(
      at::BFloat16* output_ptr,
      const int num_valid_outputs = 4) {
    bfloat16_4 val = to_bfloat16_4(acc);
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    // Since byte granule is guaranteed, num_valid_outputs can be any integer
    // for int8.
    if (aligned_8b && num_valid_outputs == 4) {
      *reinterpret_cast<uint2*>(output_ptr) =
          *reinterpret_cast<const uint2*>(&val);
    } else if (aligned_4b && num_valid_outputs >= 2) {
      *reinterpret_cast<uint*>(output_ptr) =
          *reinterpret_cast<const uint*>(&(val.vals[0].x));
      if (num_valid_outputs == 4) {
        *reinterpret_cast<uint*>(output_ptr + 2) =
            *reinterpret_cast<const uint*>(&(val.vals[0].x) + 2);
      } else if (num_valid_outputs == 3) {
        *(output_ptr + 2) =
            *reinterpret_cast<const at::BFloat16*>(&(val.vals[0].x) + 2);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] =
              *reinterpret_cast<const at::BFloat16*>(&(val.vals[0].x) + i);
        }
      }
    }
  }
#endif

  DEVICE_INLINE void store(uint8_t* output_ptr, int num_valid_outputs = 4) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(uint8_t* output_ptr, float2 qparams, int num_valid_outputs = 4) {
    const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (i < num_valid_outputs) {
        output_ptr[i] = lrintf(((&(acc.x))[i] - qparams.y) * inv_scale);
      }
    }
  }

  DEVICE_INLINE void
  store(float* output_ptr, float2 qparams, int num_valid_outputs = 4) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::Half* output_ptr, float2 qparams, int num_valid_outputs = 4) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::BFloat16* output_ptr, float2 qparams, int num_valid_outputs = 4) {
    CUDA_KERNEL_ASSERT(false);
  }

  // acc <- acc + a * b
  DEVICE_INLINE void fma(uint32_t v, int exp_bits, int exp_bias, float b) {
    acc = accumulate_weighted_packed_hfp8(acc, v, exp_bits, exp_bias, b);
  }

  // acc <- acc + a
  DEVICE_INLINE void add(uint32_t v, int exp_bits, int exp_bias) {
    acc = accumulate_packed_hfp8(acc, v, exp_bits, exp_bias);
  }

  // acc <- acc * a
  DEVICE_INLINE void mul(float a) {
    acc.x *= a;
    acc.y *= a;
    acc.z *= a;
    acc.w *= a;
  }
};

template <>
struct VecNT<4, PrimitiveType::INT> {
  float4 acc;

  DEVICE_INLINE VecNT() {
    acc = make_zero_float4();
  }

  DEVICE_INLINE VecNT(uint32_t v, half2 shift_scale) {
    acc = make_zero_float4();
    acc = accumulate_packed_int8(acc, v, shift_scale);
  }

  DEVICE_INLINE void store(float* output_ptr, int num_valid_outputs = 4) {
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    // Since byte granule is guaranteed, num_valid_outputs can be any integer
    // for int8.
    if (aligned_16b && num_valid_outputs == 4) {
      *reinterpret_cast<uint4*>(output_ptr) =
          *reinterpret_cast<const uint4*>(&acc);
    } else if (aligned_8b && num_valid_outputs >= 2) {
      *reinterpret_cast<uint2*>(output_ptr) =
          *reinterpret_cast<const uint2*>(&(acc.x));
      if (num_valid_outputs == 4) {
        *reinterpret_cast<uint2*>(output_ptr + 2) =
            *reinterpret_cast<const uint2*>(&(acc.x) + 2);
      } else if (num_valid_outputs == 3) {
        *(output_ptr + 2) = *(&(acc.x) + 2);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] = *(&(acc.x) + i);
        }
      }
    }
  }

  DEVICE_INLINE void store(at::Half* output_ptr, int num_valid_outputs = 4) {
    half4 val = to_half4(acc);
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    // Since byte granule is guaranteed, num_valid_outputs can be any integer
    // for int8.
    if (aligned_8b && num_valid_outputs == 4) {
      *reinterpret_cast<uint2*>(output_ptr) =
          *reinterpret_cast<const uint2*>(&val);
    } else if (aligned_4b && num_valid_outputs >= 2) {
      *reinterpret_cast<uint*>(output_ptr) =
          *reinterpret_cast<const uint*>(&(val.vals[0].x));
      if (num_valid_outputs == 4) {
        *reinterpret_cast<uint*>(output_ptr + 2) =
            *reinterpret_cast<const uint*>(&(val.vals[0].x) + 2);
      } else if (num_valid_outputs == 3) {
        *(output_ptr + 2) =
            *reinterpret_cast<const at::Half*>(&(val.vals[0].x) + 2);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] =
              *reinterpret_cast<const at::Half*>(&(val.vals[0].x) + i);
        }
      }
    }
  }

#if defined(USE_ROCM) ||                                  \
    !(((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
       (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
  DEVICE_INLINE void store(
      at::BFloat16* output_ptr,
      const int num_valid_outputs = 4) {
    bfloat16_4 val = to_bfloat16_4(acc);
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    // Since byte granule is guaranteed, num_valid_outputs can be any integer
    // for int8.
    if (aligned_8b && num_valid_outputs == 4) {
      *reinterpret_cast<uint2*>(output_ptr) =
          *reinterpret_cast<const uint2*>(&val);
    } else if (aligned_4b && num_valid_outputs >= 2) {
      *reinterpret_cast<uint*>(output_ptr) =
          *reinterpret_cast<const uint*>(&(val.vals[0].x));
      if (num_valid_outputs == 4) {
        *reinterpret_cast<uint*>(output_ptr + 2) =
            *reinterpret_cast<const uint*>(&(val.vals[0].x) + 2);
      } else if (num_valid_outputs == 3) {
        *(output_ptr + 2) =
            *reinterpret_cast<const at::BFloat16*>(&(val.vals[0].x) + 2);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] =
              *reinterpret_cast<const at::BFloat16*>(&(val.vals[0].x) + i);
        }
      }
    }
  }
#endif

  DEVICE_INLINE void store(uint8_t* output_ptr, int num_valid_outputs = 4) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(uint8_t* output_ptr, float2 qparams, int num_valid_outputs = 4) {
    const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (i < num_valid_outputs) {
        output_ptr[i] = lrintf(((&(acc.x))[i] - qparams.y) * inv_scale);
      }
    }
  }

  DEVICE_INLINE void
  store(float* output_ptr, float2 qparams, int num_valid_outputs = 4) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::Half* output_ptr, float2 qparams, int num_valid_outputs = 4) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::BFloat16* output_ptr, float2 qparams, int num_valid_outputs = 4) {
    CUDA_KERNEL_ASSERT(false);
  }

  // acc <- acc + a * b
  DEVICE_INLINE void fma(uint32_t v, half2 shift_scale, float b) {
    acc = accumulate_weighted_packed_int8(acc, v, shift_scale, b);
  }

  // acc <- acc + a
  DEVICE_INLINE void add(uint32_t v, half2 shift_scale) {
    acc = accumulate_packed_int8(acc, v, shift_scale);
  }

  // acc <- acc * a
  DEVICE_INLINE void mul(float a) {
    acc.x *= a;
    acc.y *= a;
    acc.z *= a;
    acc.w *= a;
  }
};

template <>
struct VecNT<8, PrimitiveType::INT> {
  float8 acc;

  DEVICE_INLINE VecNT() {
    acc = make_zero_float8();
  }

  DEVICE_INLINE VecNT(uint32_t v, half2 shift_scale) {
    acc = make_zero_float8();
    acc = accumulate_packed_int4(acc, v, shift_scale);
  }

  DEVICE_INLINE void store(float* output_ptr, int num_valid_outputs = 8) {
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    // Since byte granule is guaranteed, num_valid_outputs is multiple of 2 for
    // int4.
    if (aligned_16b && num_valid_outputs >= 4) { // 128 bit cache line
      *reinterpret_cast<uint4*>(output_ptr) =
          *reinterpret_cast<const uint4*>(&(acc.vals[0]));
      if (num_valid_outputs == 8) {
        *reinterpret_cast<uint4*>(output_ptr + 4) =
            *reinterpret_cast<const uint4*>(&(acc.vals[1]));
      } else if (num_valid_outputs == 6) {
        *reinterpret_cast<uint2*>(output_ptr + 4) =
            *reinterpret_cast<const uint2*>(&(acc.vals[1]));
      }
    } else if (aligned_8b) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<uint2*>(output_ptr + i) =
              *reinterpret_cast<const uint2*>(&(acc.vals[0].x) + i);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] = *(&(acc.vals[0].x) + i);
        }
      }
    }
  }

  DEVICE_INLINE void store(at::Half* output_ptr, int num_valid_outputs = 8) {
    half8 val = to_half8(acc);
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    // Since byte granule is guaranteed, num_valid_outputs is multiple of 2 for
    // int4.
    if (aligned_16b && num_valid_outputs == 8) {
      *reinterpret_cast<uint4*>(output_ptr) =
          *reinterpret_cast<const uint4*>(&val);
    } else if (aligned_8b && num_valid_outputs >= 4) {
      *reinterpret_cast<uint2*>(output_ptr) =
          *reinterpret_cast<const uint2*>(&(val.vals[0].x));
      if (num_valid_outputs == 8) {
        *reinterpret_cast<uint2*>(output_ptr + 4) =
            *reinterpret_cast<const uint2*>(&(val.vals[0].x) + 4);
      } else if (num_valid_outputs == 6) {
        *reinterpret_cast<uint*>(output_ptr + 4) =
            *reinterpret_cast<const uint*>(&(val.vals[0].x) + 4);
      }
    } else if (aligned_4b) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<uint*>(output_ptr + i) =
              *reinterpret_cast<const uint*>(&(val.vals[0].x) + i);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] =
              *reinterpret_cast<const at::Half*>(&(val.vals[0].x) + i);
        }
      }
    }
  }

#if defined(USE_ROCM) ||                                  \
    !(((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
       (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
  DEVICE_INLINE void store(
      at::BFloat16* output_ptr,
      const int num_valid_outputs = 8) {
    bfloat16_8 val = to_bfloat16_8(acc);
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    // Since byte granule is guaranteed, num_valid_outputs is multiple of 2 for
    // int4.
    if (aligned_16b && num_valid_outputs == 8) {
      *reinterpret_cast<uint4*>(output_ptr) =
          *reinterpret_cast<const uint4*>(&val);
    } else if (aligned_8b && num_valid_outputs >= 4) {
      *reinterpret_cast<uint2*>(output_ptr) =
          *reinterpret_cast<const uint2*>(&(val.vals[0].x));
      if (num_valid_outputs == 8) {
        *reinterpret_cast<uint2*>(output_ptr + 4) =
            *reinterpret_cast<const uint2*>(&(val.vals[0].x) + 4);
      } else if (num_valid_outputs == 6) {
        *reinterpret_cast<uint*>(output_ptr + 4) =
            *reinterpret_cast<const uint*>(&(val.vals[0].x) + 4);
      }
    } else if (aligned_4b) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<uint*>(output_ptr + i) =
              *reinterpret_cast<const uint*>(&(val.vals[0].x) + i);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] =
              *reinterpret_cast<const at::BFloat16*>(&(val.vals[0].x) + i);
        }
      }
    }
  }
#endif

  DEVICE_INLINE void store(uint8_t* output_ptr, int num_valid_outputs = 8) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(uint8_t* output_ptr, float2 qparams, int num_valid_outputs = 8) {
    const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i < num_valid_outputs) {
        output_ptr[i] = lrintf(
            (reinterpret_cast<const float*>(&(acc.vals[0].x))[i] - qparams.y) *
            inv_scale);
      }
    }
  }

  DEVICE_INLINE void
  store(float* output_ptr, float2 qparams, int num_valid_outputs = 8) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::Half* output_ptr, float2 qparams, int num_valid_outputs = 8) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::BFloat16* output_ptr, float2 qparams, int num_valid_outputs = 8) {
    CUDA_KERNEL_ASSERT(false);
  }

  // acc <- acc + a * b
  DEVICE_INLINE void fma(uint32_t v, half2 shift_scale, float b) {
    acc = accumulate_weighted_packed_int4(acc, v, shift_scale, b);
  }

  // acc <- acc + a
  DEVICE_INLINE void add(uint32_t v, half2 shift_scale) {
    acc = accumulate_packed_int4(acc, v, shift_scale);
  }

  // acc <- acc * a
  DEVICE_INLINE void mul(float a) {
    acc.vals[0].x *= a;
    acc.vals[0].y *= a;
    acc.vals[0].z *= a;
    acc.vals[0].w *= a;
    acc.vals[1].x *= a;
    acc.vals[1].y *= a;
    acc.vals[1].z *= a;
    acc.vals[1].w *= a;
  }
};

template <>
struct VecNT<16, PrimitiveType::INT> {
  float_16 acc;

  DEVICE_INLINE VecNT() {
    acc = make_zero_float_16();
  }

  DEVICE_INLINE VecNT(uint32_t v, half2 shift_scale) {
    acc = make_zero_float_16();
    acc = accumulate_packed_int2(acc, v, shift_scale);
  }

  DEVICE_INLINE void store(float* output_ptr, int num_valid_outputs = 16) {
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    if (aligned_16b) { // 128 bit cache line
#pragma unroll
      for (int i = 0; i < 16; i += 4) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<uint4*>(output_ptr + i) =
              *reinterpret_cast<const uint4*>(&(acc.vals[0].vals[0]) + i);
        }
      }
    } else if (aligned_8b) {
#pragma unroll
      for (int i = 0; i < 16; i += 2) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<uint2*>(output_ptr + i) =
              *reinterpret_cast<const uint2*>(&(acc.vals[0].vals[0]) + i);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<float*>(output_ptr + i) =
              *reinterpret_cast<const float*>(&(acc.vals[0].vals[0]) + i);
        }
      }
    }
  }

  DEVICE_INLINE void store(at::Half* output_ptr, int num_valid_outputs = 16) {
    half16 val = to_half16(acc);
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    // Since byte granule is guaranteed, num_valid_outputs is multiple of 4 for
    // int2.
    if (aligned_16b && num_valid_outputs >= 8) {
      *reinterpret_cast<uint4*>(output_ptr) =
          *reinterpret_cast<const uint4*>(&(val.vals[0].x));
      if (num_valid_outputs == 16) {
        *reinterpret_cast<uint4*>(output_ptr + 8) =
            *reinterpret_cast<const uint4*>(&(val.vals[0].x) + 8);
      } else if (num_valid_outputs == 12) {
        *reinterpret_cast<uint2*>(output_ptr + 8) =
            *reinterpret_cast<const uint2*>(&(val.vals[0].x) + 8);
      }
    } else if (aligned_8b) {
#pragma unroll
      for (int i = 0; i < 16; i += 4) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<uint2*>(output_ptr + i) =
              *reinterpret_cast<const uint2*>(&(val.vals[0].x) + i);
        }
      }
    } else if (aligned_4b) {
#pragma unroll
      for (int i = 0; i < 16; i += 2) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<uint*>(output_ptr + i) =
              *reinterpret_cast<const uint*>(&(val.vals[0].x) + i);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] =
              *reinterpret_cast<const at::Half*>(&(val.vals[0].x) + i);
        }
      }
    }
  }

#if defined(USE_ROCM) ||                                  \
    !(((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
       (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
  DEVICE_INLINE void store(
      at::BFloat16* output_ptr,
      const int num_valid_outputs = 16) {
    bfloat16_16 val = to_bfloat16_16(acc);
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    // Since byte granule is guaranteed, num_valid_outputs is multiple of 4 for
    // int2.
    if (aligned_16b && num_valid_outputs >= 8) {
      *reinterpret_cast<uint4*>(output_ptr) =
          *reinterpret_cast<const uint4*>(&(val.vals[0].x));
      if (num_valid_outputs == 16) {
        *reinterpret_cast<uint4*>(output_ptr + 8) =
            *reinterpret_cast<const uint4*>(&(val.vals[0].x) + 8);
      } else if (num_valid_outputs == 12) {
        *reinterpret_cast<uint2*>(output_ptr + 8) =
            *reinterpret_cast<const uint2*>(&(val.vals[0].x) + 8);
      }
    } else if (aligned_8b) {
#pragma unroll
      for (int i = 0; i < 16; i += 4) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<uint2*>(output_ptr + i) =
              *reinterpret_cast<const uint2*>(&(val.vals[0].x) + i);
        }
      }
    } else if (aligned_4b) {
#pragma unroll
      for (int i = 0; i < 16; i += 2) {
        if (i < num_valid_outputs) {
          *reinterpret_cast<uint*>(output_ptr + i) =
              *reinterpret_cast<const uint*>(&(val.vals[0].x) + i);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (i < num_valid_outputs) {
          output_ptr[i] =
              *reinterpret_cast<const at::BFloat16*>(&(val.vals[0].x) + i);
        }
      }
    }
  }
#endif

  DEVICE_INLINE void store(uint8_t* output_ptr, int num_valid_outputs = 16) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(uint8_t* output_ptr, float2 qparams, int num_valid_outputs = 16) {
    const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (i < num_valid_outputs) {
        output_ptr[i] = lrintf(
            (reinterpret_cast<const float*>(&(acc.vals[0].vals[0]))[i] -
             qparams.y) *
            inv_scale);
      }
    }
  }

  DEVICE_INLINE void
  store(float* output_ptr, float2 qparams, int num_valid_outputs = 16) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::Half* output_ptr, float2 qparams, int num_valid_outputs = 16) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  store(at::BFloat16* output_ptr, float2 qparams, int num_valid_outputs = 16) {
    CUDA_KERNEL_ASSERT(false);
  }

  // acc <- acc + a * b
  DEVICE_INLINE void fma(uint32_t v, half2 shift_scale, float b) {
    acc = accumulate_weighted_packed_int2(acc, v, shift_scale, b);
  }

  // acc <- acc + a
  DEVICE_INLINE void add(uint32_t v, half2 shift_scale) {
    acc = accumulate_packed_int2(acc, v, shift_scale);
  }

  // acc <- acc * a
  DEVICE_INLINE void mul(float a) {
    acc.vals[0].vals[0].x *= a;
    acc.vals[0].vals[0].y *= a;
    acc.vals[0].vals[0].z *= a;
    acc.vals[0].vals[0].w *= a;
    acc.vals[0].vals[1].x *= a;
    acc.vals[0].vals[1].y *= a;
    acc.vals[0].vals[1].z *= a;
    acc.vals[0].vals[1].w *= a;

    acc.vals[1].vals[0].x *= a;
    acc.vals[1].vals[0].y *= a;
    acc.vals[1].vals[0].z *= a;
    acc.vals[1].vals[0].w *= a;
    acc.vals[1].vals[1].x *= a;
    acc.vals[1].vals[1].y *= a;
    acc.vals[1].vals[1].z *= a;
    acc.vals[1].vals[1].w *= a;
  }
};

} // namespace fbgemm_gpu
