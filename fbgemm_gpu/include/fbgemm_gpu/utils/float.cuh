/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#if !(                                                  \
    defined(USE_ROCM) ||                                \
    ((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#elif (defined(USE_ROCM))
#include <hip/hip_bfloat16.h>
#endif
#include <cuda_fp16.h>
#include "fbgemm_gpu/utils/cuda_prelude.cuh"

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// Floating Type Definitions
////////////////////////////////////////////////////////////////////////////////

// Customized Half4 data types with two half2 (64-bit in total)
struct Half4 {
  half2 a;
  half2 b;

  __device__ inline void store(at::Half* p) {
#ifdef USE_ROCM
    p[0] = __low2half(a);
    p[1] = __high2half(a);
    p[2] = __low2half(b);
    p[3] = __high2half(b);
#elif CUDA_VERSION >= 9000

#ifndef __HALF2_TO_UI
// cuda_fp16.hpp doesn't export this
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif

    asm("st.v2.u32 [%0], {%1, %2};"
        :
        : "l"(p), "r"(__HALF2_TO_UI(a)), "r"(__HALF2_TO_UI(b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(a.x), "r"(b.x));
#endif
  }
};

struct __align__(32) float8 {
  __host__ __device__ float8() {}
  float4 vals[2];
};

// float_16 refers to the struct with 16 fp32 elements.
struct __align__(64) float_16 {
  __host__ __device__ float_16() {}
  float8 vals[2];
};

struct __align__(8) half4 {
  __host__ __device__ half4() {}
  half2 vals[2];
};

struct __align__(16) half8 {
  __host__ __device__ half8() {}
  half2 vals[4];
};

struct __align__(32) half16 {
  __host__ __device__ half16() {}
  half2 vals[8];
};

#ifdef USE_ROCM

using __nv_bfloat16 = hip_bfloat16;

typedef struct __align__(4) {
  uint16_t x;
  uint16_t y;
}
__nv_bfloat162_raw;

struct __align__(4) __nv_bfloat162 {
  __nv_bfloat16 x;
  __nv_bfloat16 y;
};

#endif

#if defined(USE_ROCM) ||                                  \
    !(((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
       (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))

struct __align__(8) bfloat16_4 {
  __host__ __device__ bfloat16_4() {}
  __nv_bfloat162 vals[2];
};

struct __align__(16) bfloat16_8 {
  __host__ __device__ bfloat16_8() {}
  __nv_bfloat162 vals[4];
};

struct __align__(32) bfloat16_16 {
  __host__ __device__ bfloat16_16() {}
  __nv_bfloat162 vals[8];
};

#endif

////////////////////////////////////////////////////////////////////////////////
// Floating Type Initializers
////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE float2 make_zero_float2() {
  return make_float2(0, 0);
}

DEVICE_INLINE float4 make_zero_float4() {
  return make_float4(0, 0, 0, 0);
}

DEVICE_INLINE float8 make_zero_float8() {
  float8 t;
  t.vals[0] = make_float4(0, 0, 0, 0);
  t.vals[1] = make_float4(0, 0, 0, 0);
  return t;
}

DEVICE_INLINE float_16 make_zero_float_16() {
  float_16 t;
  t.vals[0] = make_zero_float8();
  t.vals[1] = make_zero_float8();
  return t;
}

////////////////////////////////////////////////////////////////////////////////
// Floating Type Conversions
////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE __half to_half(float v) {
  return __float2half_rn(v);
}

DEVICE_INLINE __half2 to_half2(float2 v) {
  return __float22half2_rn(v);
}

DEVICE_INLINE half4 to_half4(float4 v) {
  half4 t;
  t.vals[0] = __float22half2_rn(make_float2(v.x, v.y));
  t.vals[1] = __float22half2_rn(make_float2(v.z, v.w));
  return t;
}

DEVICE_INLINE half8 to_half8(float8 v) {
  half8 t;
  t.vals[0] = __float22half2_rn(make_float2(v.vals[0].x, v.vals[0].y));
  t.vals[1] = __float22half2_rn(make_float2(v.vals[0].z, v.vals[0].w));
  t.vals[2] = __float22half2_rn(make_float2(v.vals[1].x, v.vals[1].y));
  t.vals[3] = __float22half2_rn(make_float2(v.vals[1].z, v.vals[1].w));
  return t;
}

DEVICE_INLINE half16 to_half16(float_16 v) {
  half16 t;
  t.vals[0] =
      __float22half2_rn(make_float2(v.vals[0].vals[0].x, v.vals[0].vals[0].y));
  t.vals[1] =
      __float22half2_rn(make_float2(v.vals[0].vals[0].z, v.vals[0].vals[0].w));
  t.vals[2] =
      __float22half2_rn(make_float2(v.vals[0].vals[1].x, v.vals[0].vals[1].y));
  t.vals[3] =
      __float22half2_rn(make_float2(v.vals[0].vals[1].z, v.vals[0].vals[1].w));

  t.vals[4] =
      __float22half2_rn(make_float2(v.vals[1].vals[0].x, v.vals[1].vals[0].y));
  t.vals[5] =
      __float22half2_rn(make_float2(v.vals[1].vals[0].z, v.vals[1].vals[0].w));
  t.vals[6] =
      __float22half2_rn(make_float2(v.vals[1].vals[1].x, v.vals[1].vals[1].y));
  t.vals[7] =
      __float22half2_rn(make_float2(v.vals[1].vals[1].z, v.vals[1].vals[1].w));
  return t;
}

// Override __bfloat162float to accept at::BFloat16
static DEVICE_INLINE float __bfloat162float(const at::BFloat16 input) {
#ifdef USE_ROCM
  return float(*reinterpret_cast<const __nv_bfloat16*>(&input));
#else
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&input));
#endif
}

// Helper functions for converting data to float
static DEVICE_INLINE float to_float(const float input) {
  return input;
}

static DEVICE_INLINE float to_float(const at::Half input) {
  return __half2float(input);
}

static DEVICE_INLINE float to_float(const at::BFloat16 input) {
  return __bfloat162float(input);
}

#ifdef USE_ROCM

// The descriptions of __float2bfloat16 and __float2bfloat16_rn are identical
// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT16__MISC.html#group__CUDA__MATH____BFLOAT16__MISC
static __host__ __device__ __nv_bfloat16 __float2bfloat16(float f) {
  __nv_bfloat16 output;
  return output.round_to_bfloat16(f);
}

static __host__ __device__ __nv_bfloat16 __float2bfloat16_rn(float f) {
  __nv_bfloat16 output;
  return output.round_to_bfloat16(f);
}

#endif

#if defined(USE_ROCM) ||                                  \
    !(((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
       (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))

DEVICE_INLINE __nv_bfloat16 to_bfloat16(float v) {
  return __float2bfloat16(v);
}

DEVICE_INLINE __nv_bfloat162 to_bfloat16_2(float2 v) {
#if __CUDA_ARCH__ >= 800
  return __float22bfloat162_rn(v);
#else
  union {
    __nv_bfloat162 raw;
    struct {
      __nv_bfloat16 x;
      __nv_bfloat16 y;
    } split;
  } t;
  t.split.x = __float2bfloat16_rn(v.x);
  t.split.y = __float2bfloat16_rn(v.y);
  return t.raw;
#endif
}

DEVICE_INLINE bfloat16_4 to_bfloat16_4(float4 v) {
  bfloat16_4 t;
  t.vals[0] = to_bfloat16_2(make_float2(v.x, v.y));
  t.vals[1] = to_bfloat16_2(make_float2(v.z, v.w));
  return t;
}

DEVICE_INLINE bfloat16_8 to_bfloat16_8(float8 v) {
  bfloat16_8 t;
  t.vals[0] = to_bfloat16_2(make_float2(v.vals[0].x, v.vals[0].y));
  t.vals[1] = to_bfloat16_2(make_float2(v.vals[0].z, v.vals[0].w));
  t.vals[2] = to_bfloat16_2(make_float2(v.vals[1].x, v.vals[1].y));
  t.vals[3] = to_bfloat16_2(make_float2(v.vals[1].z, v.vals[1].w));
  return t;
}

DEVICE_INLINE bfloat16_16 to_bfloat16_16(float_16 v) {
  bfloat16_16 t;
  t.vals[0] =
      to_bfloat16_2(make_float2(v.vals[0].vals[0].x, v.vals[0].vals[0].y));
  t.vals[1] =
      to_bfloat16_2(make_float2(v.vals[0].vals[0].z, v.vals[0].vals[0].w));
  t.vals[2] =
      to_bfloat16_2(make_float2(v.vals[0].vals[1].x, v.vals[0].vals[1].y));
  t.vals[3] =
      to_bfloat16_2(make_float2(v.vals[0].vals[1].z, v.vals[0].vals[1].w));

  t.vals[4] =
      to_bfloat16_2(make_float2(v.vals[1].vals[0].x, v.vals[1].vals[0].y));
  t.vals[5] =
      to_bfloat16_2(make_float2(v.vals[1].vals[0].z, v.vals[1].vals[0].w));
  t.vals[6] =
      to_bfloat16_2(make_float2(v.vals[1].vals[1].x, v.vals[1].vals[1].y));
  t.vals[7] =
      to_bfloat16_2(make_float2(v.vals[1].vals[1].z, v.vals[1].vals[1].w));
  return t;
}

#endif

////////////////////////////////////////////////////////////////////////////////
// Floating Type Arithmetic Opeations
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ __half2
hfma2(const __half2 a, const __half2 b, const __half2 c) {
#if (__CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610) || defined(USE_ROCM)
  return __hfma2(a, b, c);
#else
  float2 fa, fb, fc;
  fa = __half22float2(a);
  fb = __half22float2(b);
  fc = __half22float2(c);
  fc.x = fa.x * fb.x + fc.x;
  fc.y = fa.y * fb.y + fc.y;
  return __float22half2_rn(fc);
#endif
}

__forceinline__ __device__ half hmul(half a, half b) {
#if (__CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610) || defined(USE_ROCM)
  return __hmul(a, b);
#else
  return __float2half(__half2float(a) * __half2float(b));
#endif
}

// Reinterpret a  pair of uint16_t (packed into a uint32_t) as half2, and
// multiply by rhs.
__device__ __forceinline__ __half2 hmul_short2(uint32_t lhs, __half rhs) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
#ifndef __HALF2_TO_UI
// cuda_fp16.hpp
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif
#ifndef __HALF2_TO_CUI
// cuda_fp16.hpp
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int*>(&(var)))
#endif
  __half2 ret;
  __half2 rhsp = make_half2(rhs, rhs);
  asm("mul.f16x2 %0, %1, %2;"
      : "=r"(__HALF2_TO_UI(ret))
      : "r"(__HALF2_TO_CUI(lhs)), "r"(__HALF2_TO_CUI(rhsp)));
  return ret;
#else
#ifndef __HALF2_TO_UI
// cuda_fp16.hpp
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif
  __half2 lhs_h2;
  __HALF2_TO_UI(lhs_h2) = lhs;
  float2 fx = __half22float2(lhs_h2);
  float2 fy = __half22float2(make_half2(rhs, rhs));
  float2 fr;
  fr.x = fx.x * fy.x;
  fr.y = fx.y * fy.y;
  return __float22half2_rn(fr);
#endif
}

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

DEVICE_INLINE float float1_max(float val) {
  return val;
}

DEVICE_INLINE float float1_min(float val) {
  return val;
}

DEVICE_INLINE float float2_max(float2 val) {
  float max_val = val.x;
  max_val = max(max_val, val.y);
  return max_val;
}

DEVICE_INLINE float float2_min(float2 val) {
  float min_val = val.x;
  min_val = min(min_val, val.y);
  return min_val;
}

DEVICE_INLINE float float4_max(float4 val) {
  float max_val = val.x;
  max_val = max(max_val, val.y);
  max_val = max(max_val, val.z);
  max_val = max(max_val, val.w);
  return max_val;
}

DEVICE_INLINE float float4_min(float4 val) {
  float min_val = val.x;
  min_val = min(min_val, val.y);
  min_val = min(min_val, val.z);
  min_val = min(min_val, val.w);
  return min_val;
}

DEVICE_INLINE float float8_max(float8 val) {
  float max_val0 = float4_max(val.vals[0]);
  float max_val1 = float4_max(val.vals[1]);
  return max(max_val0, max_val1);
}

DEVICE_INLINE float float8_min(float8 val) {
  float min_val0 = float4_min(val.vals[0]);
  float min_val1 = float4_min(val.vals[1]);
  return min(min_val0, min_val1);
}

DEVICE_INLINE float float16_max(float_16 val) {
  float max_val0 = float8_max(val.vals[0]);
  float max_val1 = float8_max(val.vals[1]);
  return max(max_val0, max_val1);
}

DEVICE_INLINE float float16_min(float_16 val) {
  float min_val0 = float8_min(val.vals[0]);
  float min_val1 = float8_min(val.vals[1]);
  return min(min_val0, min_val1);
}

#undef min
#undef max

} // namespace fbgemm_gpu
