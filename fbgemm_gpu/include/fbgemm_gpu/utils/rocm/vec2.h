/*******************************************************************************
 * Copyright (c) 2016 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 ******************************************************************************/

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <ATen/ATen.h>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/float.cuh"
#include "fbgemm_gpu/utils/rocm/half2.h"
#include "fbgemm_gpu/utils/types.h"

namespace fbgemm_gpu::rocm {

////////////////////////////////////////////////////////////////////////////////
// Vec2T Base
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Vec2BaseT {
  float2 acc;

  DEVICE_INLINE Vec2BaseT() {
    acc.x = 0;
    acc.y = 0;
  }

  DEVICE_INLINE T vmin() const {
    T min_val = min(acc.x, acc.y);

    return min_val;
  }

  DEVICE_INLINE T vmax() const {
    T max_val = max(acc.x, acc.y);

    return max_val;
  }
};

template <typename T>
struct Vec2T {};

// A wrapper for Vec4T with acc_type
template <typename T>
using Vec2TAcc = Vec2T<at::acc_type<T, true>>;

////////////////////////////////////////////////////////////////////////////////
// Vec2T<float>
////////////////////////////////////////////////////////////////////////////////

template <>
struct Vec2T<float> : public Vec2BaseT<float> {
  DEVICE_INLINE Vec2T() {}

  DEVICE_INLINE Vec2T(const float* p) {
    load(p);
  }

  DEVICE_INLINE Vec2T(const at::Half* p) {
    load(p);
  }

  DEVICE_INLINE Vec2T(const at::BFloat16* p) {
    load(p);
  }

  DEVICE_INLINE void load(const float* p) {
    acc = *((const float2*)p);
  }

  DEVICE_INLINE void load(const at::Half* p) {
    union U {
      half2 h;
      uint32_t ui;
    } tmp_out;

    // 4 bytes
    tmp_out.ui = *reinterpret_cast<uint32_t const*>(p);

    float2 a = __half22float2(tmp_out.h);

    acc.x = a.x;
    acc.y = a.y;
  }

  DEVICE_INLINE void load(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
  }

  DEVICE_INLINE void load(const uint8_t* p) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(float* p) const {
    *((float2*)p) = acc;
  }

  DEVICE_INLINE void store(float2* p) const {
    *p = acc;
  }

  DEVICE_INLINE void store(at::Half* p) const {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    Half2 out;
    out.a = __float22half2_rn(a);
    out.store(p);
  }

  DEVICE_INLINE void store(at::BFloat16* p) const {
    p[0] = acc.x;
    p[1] = acc.y;
  }

  DEVICE_INLINE void store(uint8_t* p) const {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE static void copy(const float* src, float* dst) {
    *((float2*)dst) = *((const float2*)src);
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(const Vec2T<float>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec2T<float>& a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
  }

  // this <- this * scale
  DEVICE_INLINE void mul_(float scale) {
    acc.x *= scale;
    acc.y *= scale;
  }

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec2T<float>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Vec2T<at::Half>
////////////////////////////////////////////////////////////////////////////////

template <>
struct Vec2T<at::Half> : public Vec2BaseT<at::Half> {
  DEVICE_INLINE Vec2T() {}

  DEVICE_INLINE Vec2T(const at::Half* p) {
    load(p);
  }

  DEVICE_INLINE Vec2T(const at::BFloat16* p) {
    load(p);
  }

  DEVICE_INLINE Vec2T(const float* p) {
    load(p);
  }

  DEVICE_INLINE void load(const at::Half* p) {
    union U {
      half2 h;
      uint32_t ui;
    } tmp_out;

    // uint2 = 2 uints = 8 bytes
    tmp_out.ui = *reinterpret_cast<uint32_t const*>(p);

    float2 a = __half22float2(tmp_out.h);

    acc.x = a.x;
    acc.y = a.y;
  }

  DEVICE_INLINE void load(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
  }

  DEVICE_INLINE void load(const float* p) {
    acc = *((const float2*)p);
  }

  DEVICE_INLINE void load(const uint8_t* p) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(at::Half* p) const {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    Half2 out;
    out.a = __float22half2_rn(a);
    out.store(p);
  }

  DEVICE_INLINE void store(at::BFloat16* p) const {
    p[0] = acc.x;
    p[1] = acc.y;
  }

  DEVICE_INLINE void store(float* p) const {
    *((float2*)p) = acc;
  }

  DEVICE_INLINE void store(uint8_t* p) const {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE static void copy(const at::Half* src, at::Half* dst) {
    dst[0] = src[0];
    dst[1] = src[1];
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(const Vec2T<at::Half>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
  }

  DEVICE_INLINE void fma_(const Vec2T<float>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec2T<float>& a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec2T<at::Half>& a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
  }

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec2T<float>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
  }

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec2T<at::Half>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
  }

  // this <- this * scale
  DEVICE_INLINE void mul_(float scale) {
    acc.x *= scale;
    acc.y *= scale;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Vec2T<at::BFloat16>
////////////////////////////////////////////////////////////////////////////////

template <>
struct Vec2T<at::BFloat16> : public Vec2BaseT<at::BFloat16> {
  DEVICE_INLINE Vec2T() {}

  DEVICE_INLINE Vec2T(const at::BFloat16* p) {
    load(p);
  }

  DEVICE_INLINE Vec2T(const at::Half* p) {
    load(p);
  }

  DEVICE_INLINE Vec2T(const float* p) {
    load(p);
  }

  DEVICE_INLINE void load(const at::BFloat16* p) {
    acc.x = p[0];
    acc.y = p[1];
  }

  DEVICE_INLINE void load(const at::Half* p) {
    union U {
      half2 h;
      uint32_t ui;
    } tmp_out;

    // 4 bytes
    tmp_out.ui = *reinterpret_cast<uint32_t const*>(p);

    float2 a = __half22float2(tmp_out.h);

    acc.x = a.x;
    acc.y = a.y;
  }

  DEVICE_INLINE void load(const float* p) {
    acc = *((const float2*)p);
  }

  DEVICE_INLINE void load(const uint8_t* p) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void store(at::Half* p) const {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    Half2 out;
    out.a = __float22half2_rn(a);
    out.store(p);
  }

  DEVICE_INLINE void store(at::BFloat16* p) const {
    p[0] = acc.x;
    p[1] = acc.y;
  }

  DEVICE_INLINE void store(float* p) const {
    *((float2*)p) = acc;
  }

  DEVICE_INLINE void store(uint8_t* p) const {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE static void copy(const at::BFloat16* src, at::BFloat16* dst) {
    dst[0] = src[0];
    dst[1] = src[1];
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(const Vec2T<at::Half>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
  }

  DEVICE_INLINE void fma_(const Vec2T<float>& a, const float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec2T<float>& a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
  }

  // this <- this + a
  DEVICE_INLINE void add_(const Vec2T<at::Half>& a) {
    acc.x += a.acc.x;
    acc.y += a.acc.y;
  }

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec2T<float>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
  }

  // this <- this element-wise mul a
  DEVICE_INLINE void element_wise_mul_(const Vec2T<at::Half>& a) {
    acc.x *= a.acc.x;
    acc.y *= a.acc.y;
  }

  // this <- this * scale
  DEVICE_INLINE void mul_(float scale) {
    acc.x *= scale;
    acc.y *= scale;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Vec2T Ops
////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
DEVICE_INLINE Vec2T<scalar_t> vec2_acc(
    const Vec2T<scalar_t>& lhs,
    const Vec2T<scalar_t>& rhs) {
  Vec2T<scalar_t> s;
  s.acc.x = lhs.acc.x + rhs.acc.x;
  s.acc.y = lhs.acc.y + rhs.acc.y;

  return s;
}

} // namespace fbgemm_gpu::rocm
