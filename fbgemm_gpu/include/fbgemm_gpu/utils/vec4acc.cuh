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

namespace fbgemm_gpu {

// Vec4AccT is a vector data type for representing four floats.
// Vec4AccT provides many vector arithmetic operators (mainly the
// operators that CUDA vector primitives do not provide).  Each
// operator handles operand data type conversion implicitly.
struct Vec4AccT {
  float acc[4];

  DEVICE_INLINE Vec4AccT() {
    reset();
  }

  DEVICE_INLINE void reset() {
    memset(acc, 0, sizeof(float) * 4);
  }

  DEVICE_INLINE void add_(const float* vals) {
    acc[0] += vals[0];
    acc[1] += vals[1];
    acc[2] += vals[2];
    acc[3] += vals[3];
  }

  DEVICE_INLINE void add_(const half2* vals_h) {
    float2 vals_f[2];
    vals_f[0] = __half22float2(vals_h[0]);
    vals_f[1] = __half22float2(vals_h[1]);
    const float* vals = reinterpret_cast<const float*>(&vals_f);
    this->add_(vals);
  }

  DEVICE_INLINE void fma_(const float* vals, const float weight) {
    acc[0] = __fmaf_rn(vals[0], weight, acc[0]);
    acc[1] = __fmaf_rn(vals[1], weight, acc[1]);
    acc[2] = __fmaf_rn(vals[2], weight, acc[2]);
    acc[3] = __fmaf_rn(vals[3], weight, acc[3]);
  }

  DEVICE_INLINE void fma_(const half* vals, const float weight) {
    acc[0] = __fmaf_rn(__half2float(vals[0]), weight, acc[0]);
    acc[1] = __fmaf_rn(__half2float(vals[1]), weight, acc[1]);
    acc[2] = __fmaf_rn(__half2float(vals[2]), weight, acc[2]);
    acc[3] = __fmaf_rn(__half2float(vals[3]), weight, acc[3]);
  }

  DEVICE_INLINE void store_(const float4* src, float4* dst) {
    *dst = *src;
  }

  DEVICE_INLINE void store_(const float4* src, float2* dst) {
    const float2* vals = reinterpret_cast<const float2*>(src);
    half2 vals_h[2];
    vals_h[0] = __float22half2_rn(vals[0]);
    vals_h[1] = __float22half2_rn(vals[1]);
    *dst = *reinterpret_cast<float2*>(vals_h);
  }

  DEVICE_INLINE void store(float4* ptr) {
    this->store_(reinterpret_cast<float4*>(acc), ptr);
  }

  // Store to half
  DEVICE_INLINE void store(float2* ptr) {
    this->store_(reinterpret_cast<const float4*>(acc), ptr);
  }

  DEVICE_INLINE void store(uint8_t* ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void add(const float4* ptr) {
    const float4 loaded_vals_ = *ptr;
    const float* vals = reinterpret_cast<const float*>(&loaded_vals_);
    this->add_(vals);
  }

  DEVICE_INLINE void fma(const float4* ptr, const float weight) {
    const float4 loaded_vals_ = *ptr;
    const float* vals = reinterpret_cast<const float*>(&loaded_vals_);
    this->fma_(vals, weight);
  }

  DEVICE_INLINE void add(const float2* ptr) {
    const float2 loaded_vals_ = *ptr;
    const half2* vals_h = reinterpret_cast<const half2*>(&loaded_vals_);
    this->add_(vals_h);
  }

  DEVICE_INLINE void fma(const float2* ptr, const float weight) {
    const float2 loaded_vals_ = *ptr;
    const half* vals = reinterpret_cast<const half*>(&loaded_vals_);
    this->fma_(vals, weight);
  }

  DEVICE_INLINE void add(const uint8_t* ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void fma(const uint8_t* ptr, const float weight) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void div(uint32_t denom) {
    acc[0] /= denom;
    acc[1] /= denom;
    acc[2] /= denom;
    acc[3] /= denom;
  }
};

template <uint32_t STEP, typename input_t>
struct Vec4StepT : Vec4AccT {};

template <uint32_t STEP>
struct Vec4StepT<STEP, float> : Vec4AccT {
  float4 loaded_vals[STEP];

  DEVICE_INLINE void load(const float4* ptr, const uint32_t idx) {
    loaded_vals[idx] = *ptr;
  }

  DEVICE_INLINE void sum() {
#pragma unroll
    for (uint32_t j = 0; j < STEP; ++j) {
      const float* vals = reinterpret_cast<const float*>(&loaded_vals[j]);
      this->add_(vals);
    }
  }

  DEVICE_INLINE void weighted_sum(
      const float* const weights,
      const uint32_t idx_shift,
      const uint32_t idx_scale) {
#pragma unroll
    for (uint32_t j = 0; j < STEP; ++j) {
      const float weight = weights[j * idx_scale + idx_shift];
      const float* vals = reinterpret_cast<const float*>(&loaded_vals[j]);
      this->fma_(vals, weight);
    }
  }

  DEVICE_INLINE void index_add(uint32_t idx) {
    const float* vals = reinterpret_cast<const float*>(&loaded_vals[idx]);
    this->add_(vals);
  }

  DEVICE_INLINE void index_fma(uint32_t idx, const float weight) {
    const float* vals = reinterpret_cast<const float*>(&loaded_vals[idx]);
    this->fma_(vals, weight);
  }

  // Convert and store from loaded_vals
  DEVICE_INLINE void index_store(uint32_t idx, float4* ptr) {
    this->store_(reinterpret_cast<const float4*>(&loaded_vals[idx]), ptr);
  }

  DEVICE_INLINE void index_store(uint32_t idx, float2* ptr) {
    this->store_(&loaded_vals[idx], ptr);
  }

  DEVICE_INLINE void index_store(uint32_t idx, uint8_t* ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  index_weighted_store(uint32_t idx, float4* ptr, const float weight) {
    const float* vals = reinterpret_cast<const float*>(&loaded_vals[idx]);
    float* ptr_f = reinterpret_cast<float*>(ptr);
    ptr_f[0] = __fmul_rn(vals[0], weight);
    ptr_f[1] = __fmul_rn(vals[1], weight);
    ptr_f[2] = __fmul_rn(vals[2], weight);
    ptr_f[3] = __fmul_rn(vals[3], weight);
  }

  DEVICE_INLINE void
  index_weighted_store(uint32_t idx, float2* ptr, const float weight) {
    const float* vals = reinterpret_cast<const float*>(&loaded_vals[idx]);
    float vals_f[4];
    vals_f[0] = __fmul_rn(vals[0], weight);
    vals_f[1] = __fmul_rn(vals[1], weight);
    vals_f[2] = __fmul_rn(vals[2], weight);
    vals_f[3] = __fmul_rn(vals[3], weight);
    this->store_(reinterpret_cast<float4*>(vals_f), ptr);
  }

  DEVICE_INLINE void
  index_weighted_store(uint32_t idx, uint8_t* ptr, const float weight) {
    CUDA_KERNEL_ASSERT(false);
  }
};

template <uint32_t STEP>
struct Vec4StepT<STEP, at::Half> : Vec4AccT {
  float2 loaded_vals[STEP];

  DEVICE_INLINE void load(const float2* ptr, const uint32_t idx) {
    loaded_vals[idx] = *ptr;
  }

  DEVICE_INLINE void sum() {
#if defined(OPTIMIZE_INNER_LOOP)
#pragma unroll
    for (uint32_t j = 0; j < STEP; j += 2) {
      // If we add an fp16 register to and fp32 accumulator, the following
      // happens in assembly:
      // 1. R32 = R16 + 0 , i.e. the fp16 register is converted to fp32 through
      // an add.
      // 2. Accum += R32
      // This prevents the compiler from using vector addition.
      //
      // As an optimization, we can sum two fp16 inputs together and add their
      // sum to the accumulator:
      const half2* vals_A = reinterpret_cast<const half2*>(&loaded_vals[j]);
      const half2* vals_B = reinterpret_cast<const half2*>(&loaded_vals[j + 1]);
      float2 local_sum[2];
      local_sum[0] = __half22float2(vals_A[0] + vals_B[0]);
      local_sum[1] = __half22float2(vals_A[1] + vals_B[1]);
      // Note that there is some potential precision loss here because the
      // addition above is done in fp16.
      // TODO: There is a SASS instruction HADD2.F32 that adds 2 fp16s and
      // outputs fp32. Check if there is a corresponding intrinsics or PTX
      // instruction to be used here.
      const float* vals = reinterpret_cast<const float*>(&local_sum);
      this->add_(vals);
    }
#else
#pragma unroll
    for (uint32_t j = 0; j < STEP; ++j) {
      const half2* vals_h = reinterpret_cast<const half2*>(&loaded_vals[j]);
      this->add_(vals_h);
    }
#endif
  }

  DEVICE_INLINE void weighted_sum(
      const float* const weights,
      const uint32_t idx_shift,
      const uint32_t idx_scale) {
#pragma unroll
    for (uint32_t j = 0; j < STEP; ++j) {
      const float weight = weights[j * idx_scale + idx_shift];
      const half* vals = reinterpret_cast<const half*>(&loaded_vals[j]);
      this->fma_(vals, weight);
    }
  }

  DEVICE_INLINE void index_add(uint32_t idx) {
    const half2* vals_h = reinterpret_cast<const half2*>(&loaded_vals[idx]);
    this->add_(vals_h);
  }

  DEVICE_INLINE void index_fma(uint32_t idx, const float weight) {
    const half* vals = reinterpret_cast<const half*>(&loaded_vals[idx]);
    this->fma_(vals, weight);
  }

  // Convert and store from loaded_vals
  DEVICE_INLINE void index_store(uint32_t idx, float4* ptr) {
    const half2* vals_h = reinterpret_cast<const half2*>(&loaded_vals[idx]);
    float2 vals_f[2];
    vals_f[0] = __half22float2(vals_h[0]);
    vals_f[1] = __half22float2(vals_h[1]);
    this->store_(reinterpret_cast<const float4*>(vals_f), ptr);
  }

  DEVICE_INLINE void index_store(uint32_t idx, float2* ptr) {
    *ptr = *reinterpret_cast<float2*>(&loaded_vals[idx]);
  }

  DEVICE_INLINE void index_store(uint32_t idx, uint8_t* ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  index_weighted_store(uint32_t idx, float4* ptr, const float weight) {
    const half* vals = reinterpret_cast<const half*>(&loaded_vals[idx]);
    float* ptr_f = reinterpret_cast<float*>(ptr);
    ptr_f[0] = __fmul_rn(__half2float(vals[0]), weight);
    ptr_f[1] = __fmul_rn(__half2float(vals[1]), weight);
    ptr_f[2] = __fmul_rn(__half2float(vals[2]), weight);
    ptr_f[3] = __fmul_rn(__half2float(vals[3]), weight);
  }

  DEVICE_INLINE void
  index_weighted_store(uint32_t idx, float2* ptr, const float weight) {
    const float* vals = reinterpret_cast<const float*>(&loaded_vals[idx]);
    float vals_f[4];
    vals_f[0] = __fmul_rn(vals[0], weight);
    vals_f[1] = __fmul_rn(vals[1], weight);
    vals_f[2] = __fmul_rn(vals[2], weight);
    vals_f[3] = __fmul_rn(vals[3], weight);
    this->store_(reinterpret_cast<float4*>(vals_f), ptr);
  }

  DEVICE_INLINE void
  index_weighted_store(uint32_t idx, uint8_t* ptr, const float weight) {
    CUDA_KERNEL_ASSERT(false);
  }
};

template <uint32_t STEP>
struct Vec4StepT<STEP, uint8_t> : Vec4AccT {
  DEVICE_INLINE Vec4StepT() {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void load(const uint8_t* ptr, const uint32_t idx) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void sum() {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void weighted_sum(
      const float* const weights,
      const uint32_t idx_shift,
      const uint32_t idx_scale) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void index_add(uint32_t idx) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void index_fma(uint32_t idx, const float weight) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void index_store(uint32_t idx, float4* ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void index_store(uint32_t idx, float2* ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void index_store(uint32_t idx, uint8_t* ptr) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  index_weighted_store(uint32_t idx, float4* ptr, const float weight) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  index_weighted_store(uint32_t idx, float2* ptr, const float weight) {
    CUDA_KERNEL_ASSERT(false);
  }

  DEVICE_INLINE void
  index_weighted_store(uint32_t idx, uint8_t* ptr, const float weight) {
    CUDA_KERNEL_ASSERT(false);
  }
};

} // namespace fbgemm_gpu
