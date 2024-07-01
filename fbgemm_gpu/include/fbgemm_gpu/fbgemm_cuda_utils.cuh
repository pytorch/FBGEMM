/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/float.cuh"

namespace {

using fint32 = union fint32 {
  uint32_t I;
  float F;
};

} // namespace

namespace fbgemm_gpu {

enum class PrimitiveType : uint8_t { FP = 0, INT = 1, BF = 2 };

// Customized 4-element vector data types (with element type Half, or float).
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
    load(p);
  }

  DEVICE_INLINE Vec4T(const at::Half* p) {
    load(p);
  }

  DEVICE_INLINE Vec4T(const at::BFloat16* p) {
    load(p);
  }

  DEVICE_INLINE void load(const float* p) {
    acc = *((const float4*)p);
  }

  DEVICE_INLINE void load(const at::Half* p) {
#ifdef USE_ROCM
    union U {
      half2 h[2];
      uint2 ui;
    } tmp_out;

    // uint2 = 2 uints = 8 bytes
    tmp_out.ui = *reinterpret_cast<uint2 const*>(p);

    float2 a = __half22float2(tmp_out.h[0]);
    float2 b = __half22float2(tmp_out.h[1]);

    acc.x = a.x;
    acc.y = a.y;
    acc.z = b.x;
    acc.w = b.y;
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

  DEVICE_INLINE void load(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE void load(const uint8_t* p) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(float* p) const {
    *((float4*)p) = acc;
  }

  DEVICE_INLINE void store(float4* p) const {
    *p = acc;
  }

  DEVICE_INLINE void store(at::Half* p) const {
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

  DEVICE_INLINE void store(at::BFloat16* p) const {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE void store(uint8_t* p) const {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE static void copy(const float* src, float* dst) {
    *((float4*)dst) = *((const float4*)src);
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(const Vec4T<float>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec4T<float>& a) {
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

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec4T<float>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
    acc.z *= a.acc.z;
    acc.w *= a.acc.w;
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
    load(p);
  }

  DEVICE_INLINE Vec4T(const at::BFloat16* p) {
    load(p);
  }

  DEVICE_INLINE Vec4T(const float* p) {
    load(p);
  }

  DEVICE_INLINE void load(const at::Half* p) {
#ifdef USE_ROCM
    union U {
      half2 h[2];
      uint2 ui;
    } tmp_out;

    // uint2 = 2 uints = 8 bytes
    tmp_out.ui = *reinterpret_cast<uint2 const*>(p);

    float2 a = __half22float2(tmp_out.h[0]);
    float2 b = __half22float2(tmp_out.h[1]);

    acc.x = a.x;
    acc.y = a.y;
    acc.z = b.x;
    acc.w = b.y;
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

  DEVICE_INLINE void load(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE void load(const float* p) {
    acc = *((const float4*)p);
  }

  DEVICE_INLINE void load(const uint8_t* p) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(at::Half* p) const {
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

  DEVICE_INLINE void store(at::BFloat16* p) const {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE void store(float* p) const {
    *((float4*)p) = acc;
  }

  DEVICE_INLINE void store(uint8_t* p) const {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE static void copy(const at::Half* src, at::Half* dst) {
#ifdef USE_ROCM
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
  DEVICE_INLINE void fma_(const Vec4T<at::Half>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  DEVICE_INLINE void fma_(const Vec4T<float>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec4T<float>& a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec4T<at::Half>& a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec4T<float>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
    acc.z *= a.acc.z;
    acc.w *= a.acc.w;
  }

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec4T<at::Half>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
    acc.z *= a.acc.z;
    acc.w *= a.acc.w;
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
    load(p);
  }

  DEVICE_INLINE Vec4T(const at::Half* p) {
    load(p);
  }

  DEVICE_INLINE Vec4T(const float* p) {
    load(p);
  }

  DEVICE_INLINE void load(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE void load(const at::Half* p) {
#ifdef USE_ROCM
    union U {
      half2 h[2];
      uint2 ui;
    } tmp_out;

    // uint2 = 2 uints = 8 bytes
    tmp_out.ui = *reinterpret_cast<uint2 const*>(p);

    float2 a = __half22float2(tmp_out.h[0]);
    float2 b = __half22float2(tmp_out.h[1]);

    acc.x = a.x;
    acc.y = a.y;
    acc.z = b.x;
    acc.w = b.y;
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

  DEVICE_INLINE void load(const float* p) {
    acc = *((const float4*)p);
  }

  DEVICE_INLINE void load(const uint8_t* p) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(at::Half* p) const {
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

  DEVICE_INLINE void store(at::BFloat16* p) const {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE void store(float* p) const {
    *((float4*)p) = acc;
  }

  DEVICE_INLINE void store(uint8_t* p) const {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE static void copy(const at::BFloat16* src, at::BFloat16* dst) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(const Vec4T<at::Half>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  DEVICE_INLINE void fma_(const Vec4T<float>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec4T<float>& a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec4T<at::Half>& a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
    acc.z += a.acc.z;
    acc.w += a.acc.w;
  }

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec4T<float>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
    acc.z *= a.acc.z;
    acc.w *= a.acc.w;
  }

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec4T<at::Half>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
    acc.z *= a.acc.z;
    acc.w *= a.acc.w;
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
    const Vec4T<scalar_t>& lhs,
    const Vec4T<scalar_t>& rhs) {
  Vec4T<scalar_t> s;
  s.acc.x = lhs.acc.x + rhs.acc.x;
  s.acc.y = lhs.acc.y + rhs.acc.y;
  s.acc.z = lhs.acc.z + rhs.acc.z;
  s.acc.w = lhs.acc.w + rhs.acc.w;
  return s;
}

// A wrapper for Vec4T with acc_type
template <typename T>
using Vec4TAcc = Vec4T<at::acc_type<T, true>>;

// Correct for cases where x is not subnormal.
static DEVICE_INLINE __half
stochastic_rounding_scalar(float x, uint32_t random_value) {
  uint32_t w_int = __float_as_uint(x);
  unsigned assembles = (w_int & 0xff800000) | (random_value >> 19);
  unsigned subtract = (w_int & 0xff800000);
  float assemble_float = __uint_as_float(assembles) - __uint_as_float(subtract);
  return __float2half_rz(x + assemble_float);
}

static DEVICE_INLINE uint8_t
stochastic_rounding_scalar_uint8(float x, uint32_t random_bits) {
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
    const Vec4T<src_t>& value,
    StochasticRoundingRNGState& state,
    const float2 /* not used */) {
  value.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    at::Half* output,
    const Vec4T<at::Half>& value,
    StochasticRoundingRNGState& state,
    const float2 /* not used */) {
  const uint4 random_bits = stochastic_rounding_rand4(&state);
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
    const Vec4T<float>& value,
    StochasticRoundingRNGState& state,
    const float2 /* not used */) {
  const uint4 random_bits = stochastic_rounding_rand4(&state);
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
    const Vec4T<float>& value,
    StochasticRoundingRNGState& state,
    const float2 qparams) {
  const uint4 random_bits = stochastic_rounding_rand4(&state);
  const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
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
    const Vec4T<at::Half>& value,
    StochasticRoundingRNGState& state,
    const float2 qparams) {
  const uint4 random_bits = stochastic_rounding_rand4(&state);
  const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
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
    const Vec4T<src_t>& value,
    const float2 /* not used */) {
  value.store(output);
}

template <>
DEVICE_INLINE void nearest_rounding_vector(
    uint8_t* output,
    const Vec4T<float>& value,
    const float2 qparams) {
  const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = lrintf((value.acc.x - qparams.y) * inv_scale);
  output[1] = lrintf((value.acc.y - qparams.y) * inv_scale);
  output[2] = lrintf((value.acc.z - qparams.y) * inv_scale);
  output[3] = lrintf((value.acc.w - qparams.y) * inv_scale);
}

template <>
DEVICE_INLINE void nearest_rounding_vector(
    uint8_t* output,
    const Vec4T<at::Half>& value,
    const float2 qparams) {
  const float inv_scale = 255.0f / (qparams.x * 255.0f + kQParamEps);
  output[0] = lrintf((value.acc.x - qparams.y) * inv_scale);
  output[1] = lrintf((value.acc.y - qparams.y) * inv_scale);
  output[2] = lrintf((value.acc.z - qparams.y) * inv_scale);
  output[3] = lrintf((value.acc.w - qparams.y) * inv_scale);
}

template <typename dst_t, typename src_t>
DEVICE_INLINE void quantize_store(
    dst_t* output,
    const Vec4T<src_t>& value,
    StochasticRoundingRNGState* state,
    const float2 qparams) {
  if (!state) {
    nearest_rounding_vector<dst_t, src_t>(output, value, qparams);
  } else {
    stochastic_rounding_vector<dst_t, src_t>(output, value, *state, qparams);
  }
}

template <typename dst_t, typename src_t>
DEVICE_INLINE Vec4T<dst_t> dequantize_load(
    const src_t* value,
    const float2 /* unused */) {
  return Vec4T<dst_t>(value);
}

template <>
DEVICE_INLINE Vec4T<float> dequantize_load(
    const uint8_t* value,
    const float2 qparams) {
  Vec4T<float> out;
  out.acc.x = value[0] * qparams.x + qparams.y;
  out.acc.y = value[1] * qparams.x + qparams.y;
  out.acc.z = value[2] * qparams.x + qparams.y;
  out.acc.w = value[3] * qparams.x + qparams.y;
  return out;
}

template <>
DEVICE_INLINE Vec4T<at::Half> dequantize_load(
    const uint8_t* value,
    const float2 qparams) {
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
  auto ptr_as_uint = reinterpret_cast<uintptr_t>(ptr);
  if (ptr_as_uint % 8 == 0) {
    *reinterpret_cast<float2*>(ptr) = qparams;
  } else if (ptr_as_uint % 4 == 0) {
    auto* ptr_float = reinterpret_cast<float*>(ptr);
    auto* qparam_ptr = reinterpret_cast<const float*>(&qparams.x);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ptr_float[i] = qparam_ptr[i];
    }
  } else if (ptr_as_uint % 2 == 0) {
    auto* ptr_16bit = reinterpret_cast<uint16_t*>(ptr);
    auto* qparam_ptr = reinterpret_cast<const uint16_t*>(&qparams.x);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ptr_16bit[i] = qparam_ptr[i];
    }
  } else {
    auto* qparam_ptr = reinterpret_cast<const uint8_t*>(&qparams.x);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      ptr[i] = qparam_ptr[i];
    }
  }
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
DEVICE_INLINE float2 warp_find_qparams(scalar_t local_min, scalar_t local_max) {
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

template <typename emb_t, typename cache_t, typename dst_t>
// TODO: pass in dimension info and calculate qparams for rowwise integer
// quantization
struct WeightRow {
  // Constructor for no stochastic rounding
  DEVICE_INLINE WeightRow(emb_t* row, cache_t* cache_row, int dim)
      : row_(row),
        cache_row_(cache_row),
        dim_(dim),
        stoc_rounding_state_(nullptr) {}

  // Constructor for stochastic rounding
  DEVICE_INLINE WeightRow(
      emb_t* row,
      cache_t* cache_row,
      int dim,
      StochasticRoundingRNGState* stoc_rounding_state,
      const at::PhiloxCudaState* stochastic_rounding_philox_args,
      const uint64_t salt_value)
      : row_(row), cache_row_(cache_row), dim_(dim) {
    // Set the internal stoc_rounding_state_
    stoc_rounding_state_ = stoc_rounding_state;

    if constexpr (!std::is_same_v<emb_t, float>) {
      if (stoc_rounding_state != nullptr) {
        const auto stochastic_rounding_seeds =
            at::cuda::philox::unpack(*stochastic_rounding_philox_args);

        stochastic_rounding_init(
            std::get<0>(stochastic_rounding_seeds) ^
                std::get<1>(stochastic_rounding_seeds),
            // The salt value should be different for every *run* and every
            // *thread*.
            salt_value,
            stoc_rounding_state);
      }
    }
  }

  emb_t* row_;
  cache_t* cache_row_;
  int dim_;
  StochasticRoundingRNGState* stoc_rounding_state_;

  // Load from cache if resident; else load from embedding
  DEVICE_INLINE Vec4T<dst_t> load(const int32_t d, const float2 qparams) const {
    if (cache_row_) {
      return dequantize_load<dst_t, cache_t>(cache_row_ + d, qparams);
    } else {
      return dequantize_load<dst_t, emb_t>(row_ + d, qparams);
    }
  }

  // Write back weight (high precision) to cache if resident; else write to
  // embedding assume dst_t is higher precision than cache_t and emb_t
  DEVICE_INLINE void
  store(const Vec4T<dst_t>& v, const int32_t d, const float2 qparams) {
    if (cache_row_) {
      quantize_store(cache_row_ + d, v, stoc_rounding_state_, qparams);
    } else {
      quantize_store(row_ + d, v, stoc_rounding_state_, qparams);
    }
  }

  // Copy vector from src_vec to dst_vec (both are float)
  DEVICE_INLINE void same_type_vector_copy(
      float* dst_vec,
      const float* src_vec) {
    *reinterpret_cast<float4*>(dst_vec) =
        *reinterpret_cast<const float4*>(src_vec);
  }

  // Copy vector from src_vec to dst_vec (both are at::Half)
  DEVICE_INLINE void same_type_vector_copy(
      at::Half* dst_vec,
      const at::Half* src_vec) {
    *reinterpret_cast<float2*>(dst_vec) =
        *reinterpret_cast<const float2*>(src_vec);
  }

  // Evict cached row into embedding row (high prec -> low prec)
  DEVICE_INLINE void evict_cache(const int32_t d, const float2 qparams) {
    if constexpr (std::is_same_v<emb_t, cache_t>) {
      // No conversion required when emb_t and cache_t are the same type
      same_type_vector_copy(
          reinterpret_cast<cache_t*>(row_ + d),
          reinterpret_cast<const cache_t*>(cache_row_ + d));
    } else {
      // Does 2-step conversion: cache_t -> FP32 -> weight_t
      const auto cache_slice = load(d, qparams);
      quantize_store(row_ + d, cache_slice, stoc_rounding_state_, qparams);
    }
  }

  DEVICE_INLINE void store_qparams(const float2 qparams) {
    store_qparams_to_row(row_ + dim_, qparams);
  }

  DEVICE_INLINE float2 load_qparams() const {
    if constexpr (std::is_same_v<emb_t, uint8_t>) {
      return load_qparams_from_row<emb_t>(row_ + dim_);
    } else {
      return make_float2(0.0f, 0.0f);
    }
  }

  DEVICE_INLINE void warp_copy_to_cache(
      cache_t* dst_row,
      const int32_t dim_length,
      const int32_t num_lanes,
      const int32_t lane_id) {
    if constexpr (std::is_same_v<emb_t, cache_t>) {
      // No conversion required when emb_t and cache_t are the same type
      for (int32_t d = lane_id * 4; d < dim_length; d += num_lanes * 4) {
        same_type_vector_copy(
            dst_row + d, reinterpret_cast<const cache_t*>(row_ + d));
      }
    } else {
      // Load quantization params from embedding row
      const auto qparams = load_qparams();

      // Copy over for each warp-sized slice of Vec4's
      // Does 2-step conversion: weight_t -> FP32 -> cache_t
      for (int32_t d = lane_id * 4; d < dim_length; d += num_lanes * 4) {
        const auto slice = load(d, qparams);
        quantize_store(dst_row + d, slice, stoc_rounding_state_, qparams);
      }
    }
  }

  DEVICE_INLINE void warp_evict_cache(
      const int32_t dim_length,
      const int32_t num_lanes,
      const int32_t lane_id) {
    float2 qparams;

    if constexpr (std::is_same_v<emb_t, uint8_t>) {
      auto local_min = std::numeric_limits<at::acc_type<cache_t, true>>::max();
      auto local_max =
          std::numeric_limits<at::acc_type<cache_t, true>>::lowest();

      // Compute the qparams from the cache row (not embedding row) weights
      for (int32_t d = lane_id; d * 4 < dim_length; d += num_lanes) {
        const auto cache_slice = load(d * 4, qparams); // qparams not used
        local_max = max(local_max, vec4_max(cache_slice));
        local_min = min(local_min, vec4_min(cache_slice));
      }

      // Compute the max and min across the warps
      qparams = warp_find_qparams(local_min, local_max);

      if (lane_id == 0) {
        // Store the qparams into the embedding row
        store_qparams(qparams);
      }
    }

    for (int32_t d = lane_id * 4; d < dim_length; d += num_lanes * 4) {
      // Evict the slice into the embedding row
      evict_cache(d, qparams);
    }
  }
};

template <typename emb_t, typename cache_t, typename dst_t, bool uses_cache>
struct WeightRowAccessor {
  const emb_t* row_;
  const cache_t* cache_row_;
  const int dim_;

  DEVICE_INLINE
  WeightRowAccessor(const emb_t* row, const cache_t* cache_row, const int dim)
      : row_(row), cache_row_(cache_row), dim_(dim) {}

  DEVICE_INLINE Vec4T<dst_t> load(const int32_t d, const float2 qparams) const {
    if constexpr (uses_cache) {
      return dequantize_load<dst_t, cache_t>(cache_row_ + d, qparams);
    } else {
      return dequantize_load<dst_t, emb_t>(row_ + d, qparams);
    }
  }

  DEVICE_INLINE float2 load_qparams() const {
    if constexpr (std::is_same_v<emb_t, uint8_t>) {
      return load_qparams_from_row<emb_t>(row_ + dim_);
    } else {
      return make_float2(0.0f, 0.0f);
    }
  }
};

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
struct SharedMemory<Vec4TAcc<float>> {
  __device__ Vec4TAcc<float>* getPointer() {
    extern __shared__ Vec4TAcc<float> s_acc_float_vec_t[];
    return s_acc_float_vec_t;
  }
};

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
DEVICE_INLINE scalar_t vec4_min(const fbgemm_gpu::Vec4T<scalar_t>& vec4) {
  scalar_t min_val = vec4.acc.x;
  min_val = min(vec4.acc.y, min_val);
  min_val = min(vec4.acc.z, min_val);
  min_val = min(vec4.acc.w, min_val);
  return min_val;
}

template <typename scalar_t>
DEVICE_INLINE scalar_t vec4_max(const fbgemm_gpu::Vec4T<scalar_t>& vec4) {
  scalar_t max_val = vec4.acc.x;
  max_val = max(vec4.acc.y, max_val);
  max_val = max(vec4.acc.z, max_val);
  max_val = max(vec4.acc.w, max_val);
  return max_val;
}

// Helper functions for storing float in quantized storage
static DEVICE_INLINE void quantize_float_store(
    at::BFloat16* output,
    const float input) {
  *reinterpret_cast<__nv_bfloat16*>(output) = __float2bfloat16(input);
}

static DEVICE_INLINE void quantize_float_store(
    at::Half* output,
    const float input) {
  *output = __float2half(input);
}

static DEVICE_INLINE void quantize_float_store(
    float* output,
    const float input) {
  *output = input;
}

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
