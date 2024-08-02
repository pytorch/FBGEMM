/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <curand_kernel.h>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/float.cuh"
#include "fbgemm_gpu/utils/types.h"
#include "fbgemm_gpu/utils/vec4.cuh"

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// Stochastic Rounding
////////////////////////////////////////////////////////////////////////////////

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

} // namespace fbgemm_gpu
