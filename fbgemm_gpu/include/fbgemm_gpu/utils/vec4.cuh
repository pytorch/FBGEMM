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
#include "fbgemm_gpu/utils/float.cuh"

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// Vec4T Base
////////////////////////////////////////////////////////////////////////////////

// Customized 4-element vector data types (with element type Half, or float).
template <typename T>
struct Vec4BaseT {
  float4 acc;

  DEVICE_INLINE Vec4BaseT() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE T vmin() const {
    T min_val = acc.x;
    min_val = min(acc.y, min_val);
    min_val = min(acc.z, min_val);
    min_val = min(acc.w, min_val);
    return min_val;
  }

  DEVICE_INLINE T vmax() const {
    T max_val = acc.x;
    max_val = max(acc.y, max_val);
    max_val = max(acc.z, max_val);
    max_val = max(acc.w, max_val);
    return max_val;
  }
};

template <typename T>
struct Vec4T {};

// A wrapper for Vec4T with acc_type
template <typename T>
using Vec4TAcc = Vec4T<at::acc_type<T, true>>;

////////////////////////////////////////////////////////////////////////////////
// Vec4T<float>
////////////////////////////////////////////////////////////////////////////////

template <>
struct Vec4T<float> : public Vec4BaseT<float> {
  DEVICE_INLINE Vec4T() {}

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

////////////////////////////////////////////////////////////////////////////////
// Vec4T<at::Half>
////////////////////////////////////////////////////////////////////////////////

template <>
struct Vec4T<at::Half> : public Vec4BaseT<at::Half> {
  DEVICE_INLINE Vec4T() {}

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

////////////////////////////////////////////////////////////////////////////////
// Vec4T<at::BFloat16>
////////////////////////////////////////////////////////////////////////////////

template <>
struct Vec4T<at::BFloat16> : public Vec4BaseT<at::BFloat16> {
  DEVICE_INLINE Vec4T() {}

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

////////////////////////////////////////////////////////////////////////////////
// Vec4T Ops
////////////////////////////////////////////////////////////////////////////////

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

} // namespace fbgemm_gpu
