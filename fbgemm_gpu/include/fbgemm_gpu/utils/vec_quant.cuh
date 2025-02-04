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

#if !(                                                  \
    defined(USE_ROCM) ||                                \
    ((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#elif (defined(USE_ROCM))
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#endif

#if CUDART_VERSION >= 12000
#include <cuda_fp8.h>
#elif (defined(USE_ROCM) && ROCM_VERSION >= 60200)
#include <hip/hip_fp8.h>
#endif

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
#endif

namespace fbgemm_gpu {

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

#ifdef __HIP_PLATFORM_AMD__
// #if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
constexpr int32_t kThreadsPerWarp = 64;
constexpr int32_t kWarpsPerBlock = 16;
// #endif
#else
constexpr int32_t kThreadsPerWarp = 32;
constexpr int32_t kWarpsPerBlock = 32;
#endif

constexpr int32_t D_H = 128;

#ifdef __HIP_PLATFORM_AMD__

using __nv_bfloat16 = hip_bfloat16;

static __host__ __device__ float __bfloat162float(__nv_bfloat16 f) {
  // float output;
  // https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/hip__bfloat16_8h_source.html
  return float(f);
}

static __host__ __device__ __nv_bfloat162
__floats2bfloat162_rn(float x, float y) {
  __nv_bfloat162 output;
  output.x = __float2bfloat16_rn(x);
  output.y = __float2bfloat16_rn(y);
  return output;
}

#endif

struct __align__(16) bf16x8 {
  __nv_bfloat162 vals[4];
};

struct __align__(16) fx4 {
  float x;
  float y;
  float z;
  float w;
  __host__ __device__ fx4() {
    x = 0;
    y = 0;
    z = 0;
    w = 0;
  }
};

struct __align__(8) bfx4 {
  __nv_bfloat162 vals[2];
};

struct __align__(16) bfx8 {
  __nv_bfloat162 vals[4];
};
#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
DEVICE_INLINE bfx4 dequantize_packed_fp8(uint32_t vs, __half2 shift_scale_0);
#endif
DEVICE_INLINE bfx4 dequantize_packed_int4(uint16_t vs, __half2 shift_scale_0);
DEVICE_INLINE bfx8 dequantize_packed_int4(
    uint32_t v,
    __half2 shift_scale_0,
    __half2 shift_scale_1);

DEVICE_INLINE float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#elif defined(USE_ROCM)
  float2 f_val;
  f_val.x = __bfloat162float(val.x);
  f_val.y = __bfloat162float(val.y);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}

#define CALL_INT4_KERNEL_WITH_KV_GROUPWISE_QUANT_CHECK(NAME, NUM_GROUPS, ...)                                    \
  switch (NUM_GROUPS) {                                                                                          \
    case 1:                                                                                                      \
      NAME(1, __VA_ARGS__);                                                                                      \
      break;                                                                                                     \
    case 2:                                                                                                      \
      NAME(2, __VA_ARGS__);                                                                                      \
      break;                                                                                                     \
    case 4:                                                                                                      \
      NAME(4, __VA_ARGS__);                                                                                      \
      break;                                                                                                     \
    case 8:                                                                                                      \
      NAME(8, __VA_ARGS__);                                                                                      \
      break;                                                                                                     \
    case 16:                                                                                                     \
      TORCH_CHECK(                                                                                               \
          false,                                                                                                 \
          "With head dim = 128 we're almost even with int8 at this point. Are you sure about this? Num groups:", \
          NUM_GROUPS);                                                                                           \
      break;                                                                                                     \
    default:                                                                                                     \
      TORCH_CHECK(false, "Unsupported number of groups: ", NUM_GROUPS);                                          \
  }

DEVICE_INLINE float bfx4_dot(bfx4 a, bfx4 b) {
  // float2 acc = {0, 0};
  // __nv_bfloat162 acc;
  // acc.x = static_cast<int>(0);
  // acc.y = static_cast<int>(0);
  // TODO: need to be performed in float32?
  auto a0 = bf1622float2(a.vals[0]);
  auto a1 = bf1622float2(a.vals[1]);
  auto b0 = bf1622float2(b.vals[0]);
  auto b1 = bf1622float2(b.vals[1]);
  return a0.x * b0.x + a0.y * b0.y + a1.x * b1.x + a1.y * b1.y;

  // acc = __hfma2(a.vals[0], b.vals[0], acc);
  // acc = __hfma2(a.vals[1], b.vals[1], acc);
  // auto r = bf1622float2(acc);
  // return r.x + r.y;
}

DEVICE_INLINE fx4 bfx4_scale_acc(fx4 acc, bfx4 a, float b) {
  auto axy = bf1622float2(a.vals[0]);
  auto azw = bf1622float2(a.vals[1]);
  acc.x += axy.x * b;
  acc.y += axy.y * b;
  acc.z += azw.x * b;
  acc.w += azw.y * b;
  return acc;
}

DEVICE_INLINE fx4 fx4_acc(fx4 a, fx4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}
DEVICE_INLINE float fx4_dot(fx4 a, fx4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

DEVICE_INLINE fx4 fx4_scale(fx4 a, float scale) {
  a.x *= scale;
  a.y *= scale;
  a.z *= scale;
  a.w *= scale;
  return a;
}

DEVICE_INLINE bfx4 fx4_to_bfx4(fx4 a) {
  bfx4 r;
  r.vals[0] = __floats2bfloat162_rn(a.x, a.y);
  r.vals[1] = __floats2bfloat162_rn(a.z, a.w);
  return r;
}

#define FINAL_MASK 0xffffffff

template <typename T>
DEVICE_INLINE T shfl_xor(
    unsigned shfl_sync_mask,
    const T val,
    int laneMask,
    int width = kThreadsPerWarp) {
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < 9000
  return __shfl_xor(val, laneMask, width);
#else
  return __shfl_xor_sync(shfl_sync_mask, val, laneMask, width);
#endif
}

template <typename T>
DEVICE_INLINE T warpReduceSum(T val, uint32_t warp_mask = FINAL_MASK) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += shfl_xor(warp_mask, val, mask, 32);
  return val;
}

template <typename T>
DEVICE_INLINE T warpReduceMax(T val, uint32_t warp_mask = FINAL_MASK) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, shfl_xor(warp_mask, val, mask, 32));
  return val;
}

struct __align__(8) halfx4 {
  __half2 vals[2];
};

struct __align__(16) halfx8 {
  __half2 vals[4];
};

DEVICE_INLINE bfx4 dequantize_packed_int4(uint16_t vs, __half2 shift_scale_0) {
  uint32_t v = vs;
  // move 2nd byte to 3rd byte, so our bits are in 0x00FF00FF positions.
  v = (v & 0xFF) | ((v & 0xFF00) << 8);

  halfx4 res;
  res.vals[0] = hmul_short2(v & 0x000F000F, __float2half(32768));
  res.vals[1] = hmul_short2(v & 0x00F000F0, __float2half(32768));

  // ~5% perf gain is observed with the explicit type conversions using
  // __float2half on Nvidia A100 GPUs (https://fburl.com/diff/ss8372zw) using
  // NVCC 11.0. Additionally, HIP compiler requires these explicit type
  // conversions.
  half shift_scale_0_x = __low2half(shift_scale_0);
  half shift_scale_0_y = __high2half(shift_scale_0);

  // now, dequantize
  auto shifts = __half2(shift_scale_0_y, shift_scale_0_y);
  auto scales_lower = __half2(
      __hmul(shift_scale_0_x, __float2half(512)),
      __hmul(shift_scale_0_x, __float2half(512)));
  auto scales_upper = __half2(
      __hmul(shift_scale_0_x, __float2half(32)),
      __hmul(shift_scale_0_x, __float2half(32)));

  auto r0 = __half22float2(__hfma2(res.vals[0], scales_lower, shifts));
  auto r1 = __half22float2(__hfma2(res.vals[1], scales_upper, shifts));

  bfx4 result;
  result.vals[0] = __floats2bfloat162_rn(r0.x, r1.x);
  result.vals[1] = __floats2bfloat162_rn(r0.y, r1.y);
  return result;
}

DEVICE_INLINE bfx8 dequantize_packed_int4(
    uint32_t v,
    __half2 shift_scale_0,
    __half2 shift_scale_1) {
  halfx8 res;
  res.vals[0] = hmul_short2(v & 0x000F000F, __float2half(32768));
  res.vals[1] = hmul_short2(v & 0x00F000F0, __float2half(32768));
  v >>= 8;
  res.vals[2] = hmul_short2(v & 0x000F000F, __float2half(32768));
  res.vals[3] = hmul_short2(v & 0x00F000F0, __float2half(32768));

  half shift_scale_0_x = __low2half(shift_scale_0);
  half shift_scale_0_y = __high2half(shift_scale_0);
  half shift_scale_1_x = __low2half(shift_scale_1);
  half shift_scale_1_y = __high2half(shift_scale_1);

  // now, dequantize
  auto shifts = __half2(shift_scale_0_y, shift_scale_1_y);
  auto scales_lower = __half2(
      __hmul(shift_scale_0_x, __float2half(512)),
      __hmul(shift_scale_1_x, __float2half(512)));
  auto scales_upper = __half2(
      __hmul(shift_scale_0_x, __float2half(32)),
      __hmul(shift_scale_1_x, __float2half(32)));

  auto r0 = __half22float2(__hfma2(res.vals[0], scales_lower, shifts));
  auto r1 = __half22float2(__hfma2(res.vals[1], scales_upper, shifts));
  auto r2 = __half22float2(__hfma2(res.vals[2], scales_lower, shifts));
  auto r3 = __half22float2(__hfma2(res.vals[3], scales_upper, shifts));

  bfx8 result;
  result.vals[0] = __floats2bfloat162_rn(r0.x, r1.x);
  result.vals[1] = __floats2bfloat162_rn(r2.x, r3.x);
  result.vals[2] = __floats2bfloat162_rn(r0.y, r1.y);
  result.vals[3] = __floats2bfloat162_rn(r2.y, r3.y);
  return result;
}

__forceinline__ __device__ bfx8
dequantize_permuted_int4(uint32_t packedVals, __half2 shift_scale) {
  halfx8 res;
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

  // now, dequantize
  auto shifts = __half2(shift_scale_y, shift_scale_y);
  auto scales_lower_temp = __hmul(shift_scale_x, __float2half(512));
  auto scales_lower = __half2(scales_lower_temp, scales_lower_temp);
  auto scales_upper_temp = __hmul(shift_scale_x, __float2half(32));
  auto scales_upper = __half2(scales_upper_temp, scales_upper_temp);

  auto r0 = __half22float2(__hfma2(res.vals[0], scales_lower, shifts));
  auto r1 = __half22float2(__hfma2(res.vals[1], scales_upper, shifts));
  auto r2 = __half22float2(__hfma2(res.vals[2], scales_lower, shifts));
  auto r3 = __half22float2(__hfma2(res.vals[3], scales_upper, shifts));

  bfx8 result;
  result.vals[0] = __floats2bfloat162_rn(r0.x, r1.x);
  result.vals[1] = __floats2bfloat162_rn(r2.x, r3.x);
  result.vals[2] = __floats2bfloat162_rn(r0.y, r1.y);
  result.vals[3] = __floats2bfloat162_rn(r2.y, r3.y);

  return result;
}

enum class CacheLogicalDtype { BF16, FP8, INT4 };

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
DEVICE_INLINE bfx8 dequantize_packed_fp8_symmetric(
    uint64_t v, // Vq1 Vq0 Kq1 Kq0
    float scale_0, // k scale
    float scale_1) { // v scale
  uint32_t k_ = v & 0xFFFFFFFF; // 32 LSB
  __nv_fp8_e4m3* fp8_k = reinterpret_cast<__nv_fp8_e4m3*>(&k_);
  v >>= 32;
  uint32_t v_ = v & 0xFFFFFFFF;
  __nv_fp8_e4m3* fp8_v = reinterpret_cast<__nv_fp8_e4m3*>(&v_);

  // now, dequantize
  auto r0 = make_float2(float(fp8_k[0]) * scale_0, float(fp8_k[1]) * scale_0);
  auto r1 = make_float2(float(fp8_k[2]) * scale_0, float(fp8_k[3]) * scale_0);
  auto r2 = make_float2(float(fp8_v[0]) * scale_1, float(fp8_v[1]) * scale_1);
  auto r3 = make_float2(float(fp8_v[2]) * scale_1, float(fp8_v[3]) * scale_1);

  bfx8 result;
  result.vals[0] = __floats2bfloat162_rn(r0.x, r0.y); // (k0_dq, k1_dq)
  result.vals[1] = __floats2bfloat162_rn(r1.x, r1.y);
  result.vals[2] = __floats2bfloat162_rn(r2.x, r2.y); // (v0_dq, v1_dq)
  result.vals[3] = __floats2bfloat162_rn(r3.x, r3.y);
  return result;
}
DEVICE_INLINE bfx4 dequantize_packed_fp8(uint32_t vs, __half2 shift_scale_0) {
  uint32_t v = vs;
  __nv_fp8_e4m3* fp8_k = reinterpret_cast<__nv_fp8_e4m3*>(&v); // 4 element

  auto shift_0 = __half2float(__high2half(shift_scale_0));
  auto scale_0 = __half2float(__low2half(shift_scale_0));

  // now, dequantize
  auto r0 = make_float2(
      float(fp8_k[0]) * scale_0 + shift_0, float(fp8_k[1]) * scale_0 + shift_0);
  auto r1 = make_float2(
      float(fp8_k[2]) * scale_0 + shift_0, float(fp8_k[3]) * scale_0 + shift_0);

  bfx4 result;
  result.vals[0] = __floats2bfloat162_rn(r0.x, r0.y);
  result.vals[1] = __floats2bfloat162_rn(r1.x, r1.y);
  return result;
}
DEVICE_INLINE bfx8 dequantize_packed_fp8(
    uint64_t v, // Vq1 Vq0 Kq1 Kq0
    __half2 shift_scale_k,
    __half2 shift_scale_v) {
  uint32_t k_ = v & 0xFFFFFFFF; // 32 LSB
  __nv_fp8_e4m3* fp8_k = reinterpret_cast<__nv_fp8_e4m3*>(&k_);
  v >>= 32;
  uint32_t v_ = v & 0xFFFFFFFF;
  __nv_fp8_e4m3* fp8_v = reinterpret_cast<__nv_fp8_e4m3*>(&v_);

  auto shift_0 = __half2float(__high2half(shift_scale_k));
  auto scale_0 = __half2float(__low2half(shift_scale_k));
  auto shift_1 = __half2float(__high2half(shift_scale_v));
  auto scale_1 = __half2float(__low2half(shift_scale_v));

  // now, dequantize
  auto r0 = make_float2(
      float(fp8_k[0]) * scale_0 + shift_0, float(fp8_k[1]) * scale_0 + shift_0);
  auto r1 = make_float2(
      float(fp8_k[2]) * scale_0 + shift_0, float(fp8_k[3]) * scale_0 + shift_0);
  auto r2 = make_float2(
      float(fp8_v[0]) * scale_1 + shift_1, float(fp8_v[1]) * scale_1 + shift_1);
  auto r3 = make_float2(
      float(fp8_v[2]) * scale_1 + shift_1, float(fp8_v[3]) * scale_1 + shift_1);

  bfx8 result;
  result.vals[0] = __floats2bfloat162_rn(r0.x, r0.y); // (k0_dq, k1_dq)
  result.vals[1] = __floats2bfloat162_rn(r1.x, r1.y);
  result.vals[2] = __floats2bfloat162_rn(r2.x, r2.y); // (v0_dq, v1_dq)
  result.vals[3] = __floats2bfloat162_rn(r3.x, r3.y);
  return result;
}
#endif

} // namespace fbgemm_gpu
