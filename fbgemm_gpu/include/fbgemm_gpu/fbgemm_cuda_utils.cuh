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

////////////////////////////////////////////////////////////////////////////////
// Qparams
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
// Weight Row
////////////////////////////////////////////////////////////////////////////////

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
        local_max = max(local_max, cache_slice.vmax());
        local_min = min(local_min, cache_slice.vmin());
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

////////////////////////////////////////////////////////////////////////////////
// Shared Memory
////////////////////////////////////////////////////////////////////////////////

// A wrapper for Vec4T with acc_type
template <typename T>
using Vec4TAcc = Vec4T<at::acc_type<T, true>>;

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

////////////////////////////////////////////////////////////////////////////////
// Find QParams
////////////////////////////////////////////////////////////////////////////////

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
  float min_val = input_row[0].vmin();
  float max_val = input_row[0].vmax();
  for (int i = 0; i < D / 4; ++i) {
    min_val = min(min_val, input_row[i].vmin());
    max_val = max(max_val, input_row[i].vmax());
  }
  qparams.x = (max_val - min_val) / 255.0f;
  qparams.y = min_val;
  return qparams;
}

} // namespace fbgemm_gpu
