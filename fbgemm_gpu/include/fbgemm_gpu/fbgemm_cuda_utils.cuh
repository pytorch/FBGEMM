/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace fbgemm_gpu {

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

// Warp size
static constexpr int32_t kWarpSize = 32;
// Max thread num in one thread block
static constexpr int32_t kMaxThreads = 1024;

// Pooling Mode: currently SUM and MEAN pooling are supported
enum PoolingMode { SUM, MEAN };

// Customized Half4 data types with two half2 (64-bit in total)
struct Half4 {
  half2 a;
  half2 b;

  __device__ inline void store(at::Half* p) {
#if CUDA_VERSION >= 9000

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

  DEVICE_INLINE void store(double* p) {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
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

  DEVICE_INLINE void store(float* p) {
    *((float4*)p) = acc;
  }

  DEVICE_INLINE void store(double* p) {
    p[0] = acc.x;
    p[1] = acc.y;
    p[2] = acc.z;
    p[3] = acc.w;
  }

  DEVICE_INLINE static void copy(const at::Half* src, at::Half* dst) {
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
};

template <typename T>
DEVICE_INLINE T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
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
    StochasticRoundingRNGState& state) {
  value.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    at::Half* output,
    Vec4T<at::Half> value,
    StochasticRoundingRNGState& state) {
  uint4 random_bits = stochastic_rounding_rand4(&state);
  Half4 v;
  v.a.x = stochastic_rounding_scalar(value.acc.x, random_bits.x);
  v.a.y = stochastic_rounding_scalar(value.acc.y, random_bits.y);
  v.b.x = stochastic_rounding_scalar(value.acc.z, random_bits.z);
  v.b.y = stochastic_rounding_scalar(value.acc.w, random_bits.w);
  v.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    at::Half* output,
    Vec4T<float> value,
    StochasticRoundingRNGState& state) {
  uint4 random_bits = stochastic_rounding_rand4(&state);
  Half4 v;
  v.a.x = stochastic_rounding_scalar(value.acc.x, random_bits.x);
  v.a.y = stochastic_rounding_scalar(value.acc.y, random_bits.y);
  v.b.x = stochastic_rounding_scalar(value.acc.z, random_bits.z);
  v.b.y = stochastic_rounding_scalar(value.acc.w, random_bits.w);
  v.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    uint8_t* output,
    Vec4T<double> value,
    StochasticRoundingRNGState& state) {
  CUDA_KERNEL_ASSERT(false); // not yet implemented
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    uint8_t* output,
    Vec4T<float> value,
    StochasticRoundingRNGState& state) {
  CUDA_KERNEL_ASSERT(false); // not yet implemented
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(
    uint8_t* output,
    Vec4T<at::Half> value,
    StochasticRoundingRNGState& state) {
  CUDA_KERNEL_ASSERT(false); // not yet implemented
}

// begin nearest rounding and store implementations
template <typename dst_t, typename src_t>
DEVICE_INLINE void nearest_rounding_vector(dst_t* output, Vec4T<src_t> value) {
  value.store(output);
}

template <>
DEVICE_INLINE void nearest_rounding_vector(
    uint8_t* output,
    Vec4T<double> value) {
  CUDA_KERNEL_ASSERT(false); // not yet implemented
}

template <>
DEVICE_INLINE void nearest_rounding_vector(
    uint8_t* output,
    Vec4T<float> value) {
  CUDA_KERNEL_ASSERT(false); // not yet implemented
}

template <>
DEVICE_INLINE void nearest_rounding_vector(
    uint8_t* output,
    Vec4T<at::Half> value) {
  CUDA_KERNEL_ASSERT(false); // not yet implemented
}

template <typename dst_t, typename src_t>
DEVICE_INLINE void quantize_store(
    dst_t* output,
    Vec4T<src_t> value,
    StochasticRoundingRNGState* state) {
  if (!state) {
    nearest_rounding_vector(output, value);
  } else {
    stochastic_rounding_vector(output, value, *state);
  }
}

template <typename dst_t, typename src_t>
DEVICE_INLINE Vec4T<dst_t> dequantize_load(src_t* value) {
  return Vec4T<dst_t>(value);
}

template <>
DEVICE_INLINE Vec4T<double> dequantize_load(uint8_t* value) {
  CUDA_KERNEL_ASSERT(false); // not yet implemented
  return Vec4T<double>();
}

template <>
DEVICE_INLINE Vec4T<float> dequantize_load(uint8_t* value) {
  CUDA_KERNEL_ASSERT(false); // not yet implemented
  return Vec4T<float>();
}

template <>
DEVICE_INLINE Vec4T<at::Half> dequantize_load(uint8_t* value) {
  CUDA_KERNEL_ASSERT(false); // not yet implemented
  return Vec4T<at::Half>();
}

template <typename emb_t, typename cache_t, typename dst_t>
// TODO: pass in dimension info and calculate qparams for rowwise integer
// quantization
struct WeightRow {
  DEVICE_INLINE WeightRow(
      emb_t* row,
      cache_t* cache_row,
      StochasticRoundingRNGState* stoc_rounding_state)
      : row_(row),
        cache_row_(cache_row),
        stoc_rounding_state_(stoc_rounding_state) {}
  emb_t* row_;
  cache_t* cache_row_;
  StochasticRoundingRNGState* stoc_rounding_state_;

  // load from cache if resident; else load from embedding
  DEVICE_INLINE Vec4T<dst_t> load(int32_t d) {
    if (cache_row_) {
      return dequantize_load<dst_t, cache_t>(cache_row_ + d);
    } else {
      return dequantize_load<dst_t, emb_t>(row_ + d);
    }
  }

  // write back weight (high precision) to cache if resident; else write to
  // embedding assume dst_t is higher precision than cache_t and emb_t
  DEVICE_INLINE void store(Vec4T<dst_t> v, int32_t d) {
    if (cache_row_) {
      quantize_store(cache_row_ + d, v, stoc_rounding_state_);
    } else {
      quantize_store(row_ + d, v, stoc_rounding_state_);
    }
  }

  // evict cached row into embedding row (high prec -> low prec)
  DEVICE_INLINE void evict(Vec4T<dst_t> v, int32_t d) {
    quantize_store(row_ + d, v, stoc_rounding_state_);
  }
};

__host__ DEVICE_INLINE int32_t div_round_up(int32_t a, int32_t b) {
  return (a + b - 1) / b;
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
  auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
  return !(iptr % alignof(T));
}

} // namespace fbgemm_gpu
