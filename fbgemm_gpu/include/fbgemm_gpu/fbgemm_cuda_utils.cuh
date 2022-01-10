/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace fbgemm_gpu {

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

// Warp size
#ifdef __HIP_PLATFORM_HCC__
static constexpr int32_t kWarpSize = 64;
#else
static constexpr int32_t kWarpSize = 32;
#endif
// Max thread num in one thread block
static constexpr int32_t kMaxThreads = 1024;
static constexpr float kQParamEps = 1e-8f;

/* For rowwise int8 quantization, two quantization parameters (qparams)
will be stored at the end of each row in FP32 formats, appending a total of
8 bytes to each row.
*/
static constexpr float kINT8QparamsBytes = 8;

// Customized Half4 data types with two half2 (64-bit in total)
struct Half4 {
  half2 a;
  half2 b;

  __device__ inline void store(at::Half* p) {
#ifdef __HIP_PLATFORM_HCC__
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

// Customized 4-element vector data types (with element type Half, float, or
// double).
template <typename T>
struct Vec4T {};

template <>
struct Vec4T<float> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const float* p) {
    acc = *((const float4*)p);
  }

  DEVICE_INLINE Vec4T(const double* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE Vec4T(const at::Half* p) {
#ifdef __HIP_PLATFORM_HCC__
    acc.x = __half2float(p[0]);
    acc.y = __half2float(p[1]);
    acc.z = __half2float(p[2]);
    acc.w = __half2float(p[3]);
#else
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
        : "l"(p));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(out.a.x), "=r"(out.b.x)
        : "l"(p));
#endif

    float2 a = __half22float2(out.a);
    float2 b = __half22float2(out.b);

    acc.x = a.x;
    acc.y = a.y;
    acc.z = b.x;
    acc.w = b.y;
#endif
  }

  DEVICE_INLINE Vec4T(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE void store(float* p) {
    *((float4*)p) = acc;
  }

  DEVICE_INLINE void store(float4* p) {
    *p = acc;
  }

  DEVICE_INLINE void store(at::Half* p) {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    float2 b;
    b.x = acc.z;
    b.y = acc.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
    out.store(p);
  }

  DEVICE_INLINE void store(at::BFloat16* p) {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE void store(double* p) {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE void store(uint8_t* p) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE static void copy(const float* src, float* dst) {
    *((float4*)dst) = *((const float4*)src);
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(Vec4T<float> a, float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  // this <- this + a
  DEVICE_INLINE void add_(Vec4T<float> a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this * scale
  DEVICE_INLINE void mul_(float scale) {
    acc.x *= scale;
    acc.y *= scale;
    acc.z *= scale;
    acc.w *= scale;
  }
};

template <>
struct Vec4T<at::Half> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const at::Half* p) {
#ifdef __HIP_PLATFORM_HCC__
    acc.x = __half2float(p[0]);
    acc.y = __half2float(p[1]);
    acc.z = __half2float(p[2]);
    acc.w = __half2float(p[3]);
#else
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
        : "l"(p));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(out.a.x), "=r"(out.b.x)
        : "l"(p));
#endif

    float2 a = __half22float2(out.a);
    float2 b = __half22float2(out.b);

    acc.x = a.x;
    acc.y = a.y;
    acc.z = b.x;
    acc.w = b.y;
#endif
  }

  DEVICE_INLINE Vec4T(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE Vec4T(const float* p) {
    acc = *((const float4*)p);
  }

  DEVICE_INLINE Vec4T(const double* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE void store(at::Half* p) {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    float2 b;
    b.x = acc.z;
    b.y = acc.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
    out.store(p);
  }

  DEVICE_INLINE void store(at::BFloat16* p) {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE void store(float* p) {
    *((float4*)p) = acc;
  }

  DEVICE_INLINE void store(double* p) {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE void store(uint8_t* p) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE static void copy(const at::Half* src, at::Half* dst) {
#ifdef __HIP_PLATFORM_HCC__
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
#else
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
        : "l"(src));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(out.a.x), "=r"(out.b.x)
        : "l"(src));
#endif
#if CUDA_VERSION >= 9000
    asm("st.v2.u32 [%0], {%1, %2};"
        :
        : "l"(dst), "r"(__HALF2_TO_UI(out.a)), "r"(__HALF2_TO_UI(out.b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(dst), "r"(out.a.x), "r"(out.b.x));
#endif
#endif
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(Vec4T<at::Half> a, float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  DEVICE_INLINE void fma_(Vec4T<float> a, float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  // this <- this + a
  DEVICE_INLINE void add_(Vec4T<float> a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this + a
  DEVICE_INLINE void add_(Vec4T<at::Half> a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this * scale
  DEVICE_INLINE void mul_(float scale) {
    acc.x *= scale;
    acc.y *= scale;
    acc.z *= scale;
    acc.w *= scale;
  }
};

template <>
struct Vec4T<at::BFloat16> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE Vec4T(const at::Half* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE Vec4T(const float* p) {
    acc = *((const float4*)p);
  }

  DEVICE_INLINE Vec4T(const double* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE void store(at::Half* p) {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    float2 b;
    b.x = acc.z;
    b.y = acc.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
    out.store(p);
  }

  DEVICE_INLINE void store(at::BFloat16* p) {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE void store(float* p) {
    *((float4*)p) = acc;
  }

  DEVICE_INLINE void store(double* p) {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE void store(uint8_t* p) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE static void copy(const at::BFloat16* src, at::BFloat16* dst) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(Vec4T<at::Half> a, float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  DEVICE_INLINE void fma_(Vec4T<float> a, float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  // this <- this + a
  DEVICE_INLINE void add_(Vec4T<float> a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this + a
  DEVICE_INLINE void add_(Vec4T<at::Half> a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this * scale
  DEVICE_INLINE void mul_(float scale) {
    acc.x *= scale;
    acc.y *= scale;
    acc.z *= scale;
    acc.w *= scale;
  }
};

template <>
struct Vec4T<double> {
  double4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const at::Half* p) {
#ifdef __HIP_PLATFORM_HCC__
    acc.x = __half2float(p[0]);
    acc.y = __half2float(p[1]);
    acc.z = __half2float(p[2]);
    acc.w = __half2float(p[3]);
#else
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
        : "l"(p));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(out.a.x), "=r"(out.b.x)
        : "l"(p));
#endif

    float2 a = __half22float2(out.a);
    float2 b = __half22float2(out.b);

    acc.x = a.x;
    acc.y = a.y;
    acc.z = b.x;
    acc.w = b.y;
#endif
  }

  DEVICE_INLINE Vec4T(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE Vec4T(const float* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE Vec4T(const double* p) {
    acc = *((const double4*)p);
  }

  DEVICE_INLINE void store(double* p) {
    *((double4*)p) = acc;
  }

  DEVICE_INLINE void store(float* p) {
    float4* f4 = (float4*)p;
    f4->x = acc.x;
    f4->y = acc.y;
    f4->z = acc.z;
    f4->w = acc.w;
  }

  DEVICE_INLINE void store(at::Half* p) {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    float2 b;
    b.x = acc.z;
    b.y = acc.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
    out.store(p);
  }

  DEVICE_INLINE void store(at::BFloat16* p) {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE static void copy(const double* src, double* dst) {
    *((double4*)dst) = *((const double4*)src);
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(Vec4T<double> a, double b) {
    acc.x = __fma_rn(a.acc.x, b, acc.x);
    acc.y = __fma_rn(a.acc.y, b, acc.y);
    acc.z = __fma_rn(a.acc.z, b, acc.z);
    acc.w = __fma_rn(a.acc.w, b, acc.w);
  }

  // this <- this + a
  DEVICE_INLINE void add_(Vec4T<double> a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this * scale
  DEVICE_INLINE void mul_(float scale) {
    acc.x *= scale;
    acc.y *= scale;
    acc.z *= scale;
    acc.w *= scale;
  }
};

template <typename scalar_t>
DEVICE_INLINE Vec4T<scalar_t> vec4_acc(
    Vec4T<scalar_t> lhs,
    Vec4T<scalar_t> rhs) {
  Vec4T<scalar_t> s;
  s.acc.x = lhs.acc.x + rhs.acc.x;
  s.acc.y = lhs.acc.y + rhs.acc.y;
  s.acc.z = lhs.acc.z + rhs.acc.z;
  s.acc.w = lhs.acc.w + rhs.acc.w;
  return s;
}

template <typename T>
DEVICE_INLINE T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if defined(__HIP_PLATFORM_HCC__) || CUDA_VERSION < 9000
  return __shfl_xor(val, laneMask, width);
#else
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#endif
}

template <typename T>
DEVICE_INLINE T shfl_sync(const T val, int srcLane = 0, int width = kWarpSize) {
#if defined(__HIP_PLATFORM_HCC__) || CUDA_VERSION < 9000
  return __shfl(val, srcLane, width);
#else
  return __shfl_sync(0xffffffff, val, srcLane, width);
#endif
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warpReduceAllSum(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val += shfl_xor(val, mask);
  }
  return val;
}

// Correct for cases where x is not subnormal.
static DEVICE_INLINE __half
stochastic_rounding_scalar(float x, uint32_t random_value) {
  uint32_t w_int = __float_as_uint(x);
  unsigned assmebles = (w_int & 0xff800000) | (random_value >> 19);
  unsigned subtract = (w_int & 0xff800000);
  float assmeble_float = __uint_as_float(assmebles) - __uint_as_float(subtract);
  return __float2half_rz(x + assmeble_float);
}

static DEVICE_INLINE uint8_t
stochastic_rounding_scalar_uint8(float x, uint32_t random_bits) {
  typedef union {
    uint32_t I;
    float F;
  } fint32;

  fint32 noise;
  noise.F = 1;
  noise.I = (noise.I & 0x7F800000) | (random_bits & 0x007FFFFF);
  // noise.F in [1, 2]
  noise.F = noise.F - 1.5;
  // noise.F in [-0.5, 0.5]
  return lrintf(x + noise.F);
}

// This is a simple xorshift* RNG with 64 bits of state (vs 384 bits of state
// for curandStatePhilox4_32_10)
struct StochasticRoundingRNGState {
  uint64_t a;
};

// From https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h
__host__ DEVICE_INLINE uint64_t splitmix64_stateless(uint64_t index) {
  uint64_t z = (index + UINT64_C(0x9E3779B97F4A7C15));
  z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
  return z ^ (z >> 31);
}

DEVICE_INLINE void stochastic_rounding_init(
    uint64_t s0,
    uint64_t s1,
    StochasticRoundingRNGState* state) {
  state->a = splitmix64_stateless(s0) ^ splitmix64_stateless(s1);
  // Ensure we never have a zero state (insanely low probability, but still...).
  if (state->a == 0) {
    state->a = 1;
  }
}

// See https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf and
// https://en.wikipedia.org/wiki/Xorshift#xorshift*
DEVICE_INLINE uint4
stochastic_rounding_rand4(StochasticRoundingRNGState* state) {
  uint4 random_bits;
  uint64_t x = state->a; /* The state must be seeded with a nonzero value. */
  x ^= x >> 12; // a
  x ^= x << 25; // b
  x ^= x >> 27; // c
  random_bits.x = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
  x ^= x >> 12; // a
  x ^= x << 25; // b
  x ^= x >> 27; // c
  random_bits.y = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
  x ^= x >> 12; // a
  x ^= x << 25; // b
  x ^= x >> 27; // c
  random_bits.z = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
  x ^= x >> 12; // a
  x ^= x << 25; // b
  x ^= x >> 27; // c
  random_bits.w = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
  state->a = x;
  return random_bits;
}

template <typename dst_t, typename src_t>
DEVICE_INLINE void stochastic_rounding_vector(
    dst_t* output,
    Vec4T<src_t> value,
    StochasticRoundingRNGState& state,
    float2 /* not used */) {
  value.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    at::Half* output,
    Vec4T<at::Half> value,
    StochasticRoundingRNGState& state,
    float2 /* not used */) {
  uint4 random_bits = stochastic_rounding_rand4(&state);
  Half4 v;
  v.a = __halves2half2(
      stochastic_rounding_scalar(value.acc.x, random_bits.x),
      stochastic_rounding_scalar(value.acc.y, random_bits.y));
  v.b = __halves2half2(
      stochastic_rounding_scalar(value.acc.z, random_bits.z),
      stochastic_rounding_scalar(value.acc.w, random_bits.w));
  v.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    at::Half* output,
    Vec4T<float> value,
    StochasticRoundingRNGState& state,
    float2 /* not used */) {
  uint4 random_bits = stochastic_rounding_rand4(&state);
  Half4 v;
  v.a = __halves2half2(
      stochastic_rounding_scalar(value.acc.x, random_bits.x),
      stochastic_rounding_scalar(value.acc.y, random_bits.y));
  v.b = __halves2half2(
      stochastic_rounding_scalar(value.acc.z, random_bits.z),
      stochastic_rounding_scalar(value.acc.w, random_bits.w));
  v.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    uint8_t* output,
    Vec4T<float> value,
    StochasticRoundingRNGState& state,
    float2 qparams) {
  uint4 random_bits = stochastic_rounding_rand4(&state);
  float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = stochastic_rounding_scalar_uint8(
      (value.acc.x - qparams.y) * inv_scale, random_bits.x);
  output[1] = stochastic_rounding_scalar_uint8(
      (value.acc.y - qparams.y) * inv_scale, random_bits.y);
  output[2] = stochastic_rounding_scalar_uint8(
      (value.acc.z - qparams.y) * inv_scale, random_bits.z);
  output[3] = stochastic_rounding_scalar_uint8(
      (value.acc.w - qparams.y) * inv_scale, random_bits.w);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    uint8_t* output,
    Vec4T<at::Half> value,
    StochasticRoundingRNGState& state,
    float2 qparams) {
  uint4 random_bits = stochastic_rounding_rand4(&state);
  float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = stochastic_rounding_scalar_uint8(
      (value.acc.x - qparams.y) * inv_scale, random_bits.x);
  output[1] = stochastic_rounding_scalar_uint8(
      (value.acc.y - qparams.y) * inv_scale, random_bits.y);
  output[2] = stochastic_rounding_scalar_uint8(
      (value.acc.z - qparams.y) * inv_scale, random_bits.z);
  output[3] = stochastic_rounding_scalar_uint8(
      (value.acc.w - qparams.y) * inv_scale, random_bits.w);
}

// begin nearest rounding and store implementations
template <typename dst_t, typename src_t>
DEVICE_INLINE void nearest_rounding_vector(
    dst_t* output,
    Vec4T<src_t> value,
    float2 /* not used */) {
  value.store(output);
}

template <>
DEVICE_INLINE void
nearest_rounding_vector(uint8_t* output, Vec4T<float> value, float2 qparams) {
  float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = lrintf((value.acc.x - qparams.y) * inv_scale);
  output[1] = lrintf((value.acc.y - qparams.y) * inv_scale);
  output[2] = lrintf((value.acc.z - qparams.y) * inv_scale);
  output[3] = lrintf((value.acc.w - qparams.y) * inv_scale);
}

template <>
DEVICE_INLINE void nearest_rounding_vector(
    uint8_t* output,
    Vec4T<at::Half> value,
    float2 qparams) {
  float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = lrintf((value.acc.x - qparams.y) * inv_scale);
  output[1] = lrintf((value.acc.y - qparams.y) * inv_scale);
  output[2] = lrintf((value.acc.z - qparams.y) * inv_scale);
  output[3] = lrintf((value.acc.w - qparams.y) * inv_scale);
}

template <>
DEVICE_INLINE void
nearest_rounding_vector(uint8_t* output, Vec4T<double> value, float2 qparams) {
  CUDA_KERNEL_ASSERT(false);
}

template <typename dst_t, typename src_t>
DEVICE_INLINE void quantize_store(
    dst_t* output,
    Vec4T<src_t> value,
    StochasticRoundingRNGState* state,
    float2 qparams) {
  if (!state) {
    nearest_rounding_vector<dst_t, src_t>(output, value, qparams);
  } else {
    stochastic_rounding_vector<dst_t, src_t>(output, value, *state, qparams);
  }
}

template <typename dst_t, typename src_t>
DEVICE_INLINE Vec4T<dst_t> dequantize_load(src_t* value, float2 /* unused */) {
  return Vec4T<dst_t>(value);
}

template <>
DEVICE_INLINE Vec4T<float> dequantize_load(uint8_t* value, float2 qparams) {
  Vec4T<float> out;
  out.acc.x = value[0] * qparams.x + qparams.y;
  out.acc.y = value[1] * qparams.x + qparams.y;
  out.acc.z = value[2] * qparams.x + qparams.y;
  out.acc.w = value[3] * qparams.x + qparams.y;
  return out;
}

template <>
DEVICE_INLINE Vec4T<at::Half> dequantize_load(uint8_t* value, float2 qparams) {
  Vec4T<at::Half> out;
  out.acc.x = value[0] * qparams.x + qparams.y;
  out.acc.y = value[1] * qparams.x + qparams.y;
  out.acc.z = value[2] * qparams.x + qparams.y;
  out.acc.w = value[3] * qparams.x + qparams.y;
  return out;
}

template <typename emb_t>
DEVICE_INLINE float2 load_qparams_from_row(emb_t* qparam_ptr) {
  float2 qparams;
  float* qparams_fp_ptr = reinterpret_cast<float*>(qparam_ptr);
  qparams.x = qparams_fp_ptr[0];
  qparams.y = qparams_fp_ptr[1];
  return qparams;
}

template <typename emb_t>
DEVICE_INLINE void store_qparams_to_row(emb_t* ptr, float2 qparams) {
  CUDA_KERNEL_ASSERT(false); // Only int8 embeddding should call this
}

template <>
DEVICE_INLINE void store_qparams_to_row(uint8_t* ptr, float2 qparams) {
  float* qparam_ptr = reinterpret_cast<float*>(ptr);
  qparam_ptr[0] = qparams.x;
  qparam_ptr[1] = qparams.y;
}

template <typename emb_t, typename cache_t, typename dst_t>
// TODO: pass in dimension info and calculate qparams for rowwise integer
// quantization
struct WeightRow {
  DEVICE_INLINE WeightRow(
      emb_t* row,
      cache_t* cache_row,
      int dim,
      StochasticRoundingRNGState* stoc_rounding_state)
      : row_(row),
        cache_row_(cache_row),
        dim_(dim),
        stoc_rounding_state_(stoc_rounding_state) {}
  emb_t* row_;
  cache_t* cache_row_;
  int dim_;
  StochasticRoundingRNGState* stoc_rounding_state_;

  DEVICE_INLINE void set_stoc_state(
      StochasticRoundingRNGState* stoc_rounding_state) {
    stoc_rounding_state_ = stoc_rounding_state;
  }

  // load from cache if resident; else load from embedding
  DEVICE_INLINE Vec4T<dst_t> load(int32_t d, float2 qparams) {
    if (cache_row_) {
      return dequantize_load<dst_t, cache_t>(cache_row_ + d, qparams);
    } else {
      return dequantize_load<dst_t, emb_t>(row_ + d, qparams);
    }
  }

  // write back weight (high precision) to cache if resident; else write to
  // embedding assume dst_t is higher precision than cache_t and emb_t
  DEVICE_INLINE void store(Vec4T<dst_t> v, int32_t d, float2 qparams) {
    if (cache_row_) {
      quantize_store(cache_row_ + d, v, stoc_rounding_state_, qparams);
    } else {
      quantize_store(row_ + d, v, stoc_rounding_state_, qparams);
    }
  }

  // evict cached row into embedding row (high prec -> low prec)
  DEVICE_INLINE void evict(Vec4T<dst_t> v, int32_t d, float2 qparams) {
    quantize_store(row_ + d, v, stoc_rounding_state_, qparams);
  }

  DEVICE_INLINE void store_qparams(float2 qparams) {
    store_qparams_to_row(row_ + dim_, qparams);
  }

  DEVICE_INLINE float2 load_qparams() {
    return load_qparams_from_row<emb_t>(row_ + dim_);
  }
};

__host__ DEVICE_INLINE int32_t div_round_up(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}

__host__ DEVICE_INLINE int32_t round_down(int32_t a, int32_t b) {
  return a / b * b;
}

// Shared memory with template supports.
// See https://leimao.github.io/blog/CUDA-Shared-Memory-Templated-Kernel/
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<int64_t> {
  __device__ int64_t* getPointer() {
    extern __shared__ int64_t s_int64_t[];
    return s_int64_t;
  }
};

template <>
struct SharedMemory<int32_t> {
  __device__ int32_t* getPointer() {
    extern __shared__ int32_t s_int32_t[];
    return s_int32_t;
  }
};

template <>
struct SharedMemory<float> {
  __device__ float* getPointer() {
    extern __shared__ float s_float_t[];
    return s_float_t;
  }
};

template <>
struct SharedMemory<double> {
  __device__ double* getPointer() {
    extern __shared__ double s_double_t[];
    return s_double_t;
  }
};

template <>
struct SharedMemory<Vec4T<at::acc_type<float, true>>> {
  __device__ Vec4T<at::acc_type<float, true>>* getPointer() {
    extern __shared__ Vec4T<at::acc_type<float, true>> s_acc_float_vec_t[];
    return s_acc_float_vec_t;
  }
};

template <>
struct SharedMemory<Vec4T<at::acc_type<double, true>>> {
  __device__ Vec4T<at::acc_type<double, true>>* getPointer() {
    extern __shared__ Vec4T<at::acc_type<double, true>> s_acc_double_vec_t[];
    return s_acc_double_vec_t;
  }
};

// Return if the address is aligned to the type (mainly for Vec4T).
template <class T>
DEVICE_INLINE bool is_aligned(const void* ptr) {
  auto iptr = reinterpret_cast<uintptr_t>(ptr);
  return !(iptr % alignof(T));
}

template <typename scalar_t>
__device__ float2 thrust_find_qparams(scalar_t* input_row, int D) {
  float2 qparams;

  scalar_t scalar_minimum = *(input_row++);
  scalar_t scalar_maximum = scalar_minimum;

  while (--D > 0) {
    scalar_t next = *(input_row++);
    scalar_minimum = (scalar_minimum <= next) ? scalar_minimum : next;
    scalar_maximum = (scalar_maximum >= next) ? scalar_maximum : next;
  }
  float minimum_element = scalar_minimum;
  float maximum_element = scalar_maximum;

  float range = maximum_element - minimum_element;
  qparams.x = range / 255.0f;
  qparams.y = minimum_element;
  return qparams;
}

template <typename scalar_t>
__device__ float2
thrust_find_qparams(fbgemm_gpu::Vec4T<scalar_t>* input_row, int D) {
  // TODO: replace uses in backward kernels with warp find qparams
  float2 qparams;
  float min_val = vec4_min(input_row[0]);
  float max_val = vec4_max(input_row[0]);
  for (int i = 0; i < D / 4; ++i) {
    min_val = min(min_val, vec4_min(input_row[i]));
    max_val = max(max_val, vec4_max(input_row[i]));
  }
  qparams.x = (max_val - min_val) / 255.0f;
  qparams.y = min_val;
  return qparams;
}

template <typename scalar_t>
DEVICE_INLINE scalar_t vec4_min(fbgemm_gpu::Vec4T<scalar_t> vec4) {
  scalar_t min_val = vec4.acc.x;
  min_val = min(vec4.acc.y, min_val);
  min_val = min(vec4.acc.z, min_val);
  min_val = min(vec4.acc.w, min_val);
  return min_val;
}

template <typename scalar_t>
DEVICE_INLINE scalar_t vec4_max(fbgemm_gpu::Vec4T<scalar_t> vec4) {
  scalar_t max_val = vec4.acc.x;
  max_val = max(vec4.acc.y, max_val);
  max_val = max(vec4.acc.z, max_val);
  max_val = max(vec4.acc.w, max_val);
  return max_val;
}

// Min a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warp_reduce_min(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val = std::min(val, shfl_xor(val, mask));
  }
  return val;
}

// Max a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val = std::max(val, shfl_xor(val, mask));
  }
  return val;
}

template <typename scalar_t>
__device__ float2 warp_find_qparams(scalar_t local_min, scalar_t local_max) {
  float2 qparams;
  local_min = warp_reduce_min<scalar_t>(local_min);
  local_max = warp_reduce_max<scalar_t>(local_max);
  if (threadIdx.x == 0) {
    qparams.x = (local_max - local_min) / 255.0f;
    qparams.y = local_min;
  }
  qparams.x = shfl_sync(qparams.x, 0);
  qparams.y = shfl_sync(qparams.y, 0);
  return qparams;
}

struct __align__(32) float8 {
  __host__ __device__ float8() {}
  float4 vals[2];
};

struct __align__(8) half4 {
  __host__ __device__ half4() {}
  half2 vals[2];
};

struct __align__(16) half8 {
  __host__ __device__ half8() {}
  half2 vals[4];
};

DEVICE_INLINE half8 to_half8(float8 v) {
  half8 t;
  t.vals[0] = __float22half2_rn(make_float2(v.vals[0].x, v.vals[0].y));
  t.vals[1] = __float22half2_rn(make_float2(v.vals[0].z, v.vals[0].w));
  t.vals[2] = __float22half2_rn(make_float2(v.vals[1].x, v.vals[1].y));
  t.vals[3] = __float22half2_rn(make_float2(v.vals[1].z, v.vals[1].w));
  return t;
}

DEVICE_INLINE half4 to_half4(float4 v) {
  half4 t;
  t.vals[0] = __float22half2_rn(make_float2(v.x, v.y));
  t.vals[1] = __float22half2_rn(make_float2(v.z, v.w));
  return t;
}

DEVICE_INLINE __half2 to_half2(float2 v) {
  return __float22half2_rn(v);
}

DEVICE_INLINE __half to_half(float v) {
  return __float2half_rn(v);
}

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

__forceinline__ __device__ __half2
hfma2(const __half2 a, const __half2 b, const __half2 c) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
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
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
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

// Customized N-element vector data types (with element type float for
// accumulation type).
template <int N>
struct VecNT {};

template <>
struct VecNT<1> {
  float acc;

  DEVICE_INLINE VecNT() {
    acc = 0;
  }

  DEVICE_INLINE VecNT(float a) {
    acc = a;
  }

  DEVICE_INLINE void store(float* output_ptr) {
    *output_ptr = acc;
  }

  DEVICE_INLINE void store(at::Half* output_ptr) {
    __half val = to_half(acc);
    *reinterpret_cast<__half*>(output_ptr) = val;
  }

  DEVICE_INLINE void store(uint8_t* output_ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(uint8_t* output_ptr, float2 qparams) {
    float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
    output_ptr[0] = lrintf((acc - qparams.y) * inv_scale);
  }

  DEVICE_INLINE void store(float* output_ptr, float2 qparams) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(at::Half* output_ptr, float2 qparams) {
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
struct VecNT<2> {
  float2 acc;

  DEVICE_INLINE VecNT() {
    acc = make_zero_float2();
  }

  DEVICE_INLINE VecNT(half2 a) {
    acc = __half22float2(a);
  }

  DEVICE_INLINE void store(float* output_ptr) {
    *reinterpret_cast<int2*>(output_ptr) = *reinterpret_cast<const int2*>(&acc);
  }

  DEVICE_INLINE void store(at::Half* output_ptr) {
    half2 val = to_half2(acc);
    *reinterpret_cast<int1*>(output_ptr) = *reinterpret_cast<const int1*>(&val);
  }

  DEVICE_INLINE void store(uint8_t* output_ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(uint8_t* output_ptr, float2 qparams) {
    float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
    output_ptr[0] = lrintf((acc.x - qparams.y) * inv_scale);
    output_ptr[1] = lrintf((acc.y - qparams.y) * inv_scale);
  }

  DEVICE_INLINE void store(float* output_ptr, float2 qparams) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(at::Half* output_ptr, float2 qparams) {
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
struct VecNT<4> {
  float4 acc;

  DEVICE_INLINE VecNT() {
    acc = make_zero_float4();
  }

  DEVICE_INLINE VecNT(uint32_t v, half2 shift_scale) {
    acc = make_zero_float4();
    acc = accumulate_packed_int8(acc, v, shift_scale);
  }

  DEVICE_INLINE void store(float* output_ptr) {
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    if (aligned_16b) {
      *reinterpret_cast<int4*>(output_ptr) =
          *reinterpret_cast<const int4*>(&acc);
    } else if (aligned_8b) {
      auto v = *reinterpret_cast<const int4*>(&acc);
      *reinterpret_cast<int2*>(output_ptr + 0) = make_int2(v.x, v.y);
      *reinterpret_cast<int2*>(output_ptr + 2) = make_int2(v.z, v.w);
    } else {
      *(output_ptr + 0) = acc.x;
      *(output_ptr + 1) = acc.y;
      *(output_ptr + 2) = acc.z;
      *(output_ptr + 3) = acc.w;
    }
  }

  DEVICE_INLINE void store(at::Half* output_ptr) {
    half4 val = to_half4(acc);
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    if (aligned_8b) {
      *reinterpret_cast<int2*>(output_ptr) =
          *reinterpret_cast<const int2*>(&val);
    } else if (aligned_4b) {
      auto v = *reinterpret_cast<const int2*>(&val);
      *reinterpret_cast<int*>(output_ptr + 0) = v.x;
      *reinterpret_cast<int*>(output_ptr + 2) = v.y;
    } else {
      *(output_ptr + 0) = __low2half(val.vals[0]);
      *(output_ptr + 1) = __high2half(val.vals[0]);
      *(output_ptr + 2) = __low2half(val.vals[1]);
      *(output_ptr + 3) = __high2half(val.vals[1]);
    }
  }

  DEVICE_INLINE void store(uint8_t* output_ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(uint8_t* output_ptr, float2 qparams) {
    float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
    output_ptr[0] = lrintf((acc.x - qparams.y) * inv_scale);
    output_ptr[1] = lrintf((acc.y - qparams.y) * inv_scale);
    output_ptr[2] = lrintf((acc.z - qparams.y) * inv_scale);
    output_ptr[3] = lrintf((acc.w - qparams.y) * inv_scale);
  }

  DEVICE_INLINE void store(float* output_ptr, float2 qparams) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(at::Half* output_ptr, float2 qparams) {
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
struct VecNT<8> {
  float8 acc;

  DEVICE_INLINE VecNT() {
    acc = make_zero_float8();
  }

  DEVICE_INLINE VecNT(uint32_t v, half2 shift_scale) {
    acc = make_zero_float8();
    acc = accumulate_packed_int4(acc, v, shift_scale);
  }

  DEVICE_INLINE void store(float* output_ptr) {
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    if (aligned_16b) { // 128 bit cache line
      *reinterpret_cast<int4*>(output_ptr) =
          *reinterpret_cast<const int4*>(&(acc.vals[0]));
      *reinterpret_cast<int4*>(output_ptr + 4) =
          *reinterpret_cast<const int4*>(&(acc.vals[1]));
    } else if (aligned_8b) {
      auto v0 = *reinterpret_cast<const int4*>(&(acc.vals[0]));
      auto v1 = *reinterpret_cast<const int4*>(&(acc.vals[1]));
      *reinterpret_cast<int2*>(output_ptr + 0) = make_int2(v0.x, v0.y);
      *reinterpret_cast<int2*>(output_ptr + 2) = make_int2(v0.z, v0.w);
      *reinterpret_cast<int2*>(output_ptr + 4) = make_int2(v1.x, v1.y);
      *reinterpret_cast<int2*>(output_ptr + 6) = make_int2(v1.z, v1.w);
    } else {
      *(output_ptr + 0) = acc.vals[0].x;
      *(output_ptr + 1) = acc.vals[0].y;
      *(output_ptr + 2) = acc.vals[0].z;
      *(output_ptr + 3) = acc.vals[0].w;
      *(output_ptr + 4) = acc.vals[1].x;
      *(output_ptr + 5) = acc.vals[1].y;
      *(output_ptr + 6) = acc.vals[1].z;
      *(output_ptr + 7) = acc.vals[1].w;
    }
  }

  DEVICE_INLINE void store(at::Half* output_ptr) {
    half8 val = to_half8(acc);
    bool aligned_16b = intptr_t(output_ptr) % 16 == 0;
    bool aligned_8b = intptr_t(output_ptr) % 8 == 0;
    bool aligned_4b = intptr_t(output_ptr) % 4 == 0;
    if (aligned_16b) {
      *reinterpret_cast<int4*>(output_ptr) =
          *reinterpret_cast<const int4*>(&val);
    } else if (aligned_8b) {
      auto v = *reinterpret_cast<const int4*>(&val);
      *reinterpret_cast<int2*>(output_ptr) = make_int2(v.x, v.y);
      *reinterpret_cast<int2*>(output_ptr + 4) = make_int2(v.z, v.w);
    } else if (aligned_4b) {
      auto v = *reinterpret_cast<const int4*>(&val);
      *reinterpret_cast<int*>(output_ptr + 0) = v.x;
      *reinterpret_cast<int*>(output_ptr + 2) = v.y;
      *reinterpret_cast<int*>(output_ptr + 4) = v.z;
      *reinterpret_cast<int*>(output_ptr + 6) = v.w;
    } else {
      *(output_ptr + 0) = __low2half(val.vals[0]);
      *(output_ptr + 1) = __high2half(val.vals[0]);
      *(output_ptr + 2) = __low2half(val.vals[1]);
      *(output_ptr + 3) = __high2half(val.vals[1]);
      *(output_ptr + 4) = __low2half(val.vals[2]);
      *(output_ptr + 5) = __high2half(val.vals[2]);
      *(output_ptr + 6) = __low2half(val.vals[3]);
      *(output_ptr + 7) = __high2half(val.vals[3]);
    }
  }

  DEVICE_INLINE void store(uint8_t* output_ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(uint8_t* output_ptr, float2 qparams) {
    float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
    output_ptr[0] = lrintf((acc.vals[0].x - qparams.y) * inv_scale);
    output_ptr[1] = lrintf((acc.vals[0].y - qparams.y) * inv_scale);
    output_ptr[2] = lrintf((acc.vals[0].z - qparams.y) * inv_scale);
    output_ptr[3] = lrintf((acc.vals[0].w - qparams.y) * inv_scale);
    output_ptr[4] = lrintf((acc.vals[1].x - qparams.y) * inv_scale);
    output_ptr[5] = lrintf((acc.vals[1].y - qparams.y) * inv_scale);
    output_ptr[6] = lrintf((acc.vals[1].z - qparams.y) * inv_scale);
    output_ptr[7] = lrintf((acc.vals[1].w - qparams.y) * inv_scale);
  }

  DEVICE_INLINE void store(float* output_ptr, float2 qparams) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(at::Half* output_ptr, float2 qparams) {
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

#undef min
#undef max

} // namespace fbgemm_gpu
