// @lint-ignore-every LICENSELINT

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Aparch 2.0 License
// FP4 quantization routines in this file follow Aparch 2.0 license.
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include <algorithm>

#include "fbgemm_gpu/utils/cuda_block_count.h"
#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/stochastic_rounding.cuh"

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

#ifndef USE_ROCM
#include <mma.h>
#endif
#include <cub/cub.cuh>

#include <torch/torch.h>

#if CUDART_VERSION >= 12000
#include <cuda_fp8.h>
#elif (defined(USE_ROCM) && ROCM_VERSION >= 60200)
#include <hip/hip_fp8.h>
#endif

#include <fbgemm_gpu/utils/vec_quant.cuh>

/// @defgroup FP8/INT8 quantized FC Operators
///

namespace fbgemm_gpu {

// Each block handles a single batch and head

// Each warp handles separate D dimension.

// Load Q into registers in all warps.
// Split T across warps in a block
// Compute S[MAX_T] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
// Use shared reduction to compute max and compute softmax on shared memory.

// Split T across warps in a block

// each warp compute sum(t_subset) P[t] * V[t_subset, d]
// outputs are of size float[D]

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3_fnuz;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
using __nv_fp8_e5m2 = __hip_fp8_e5m2_fnuz;
#define torch_fp8_e4m3 at::kFloat8_e4m3fnuz
#define torch_fp8_e5m2 at::kFloat8_e5m2fnuz
#else
#define torch_fp8_e4m3 at::kFloat8_e4m3fn
#define torch_fp8_e5m2 at::kFloat8_e5m2
#endif

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include <torch/all.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(cmd)                      \
  do {                                       \
    cudaError_t e = cmd;                     \
    if (e != cudaSuccess) {                  \
      printf(                                \
          "Failed: Cuda error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          cudaGetErrorString(e));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
}; // keep for generality

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

#define ELTS_PER_THREAD 8

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;
#endif

struct __align__(8) i8x8 {
  int8_t vals[8];
};

__global__ void per_tensor_quantize_i8_kernel(
    at::PackedTensorAccessor64<at::BFloat16, 1, at::RestrictPtrTraits> X,
    at::PackedTensorAccessor64<int8_t, 1, at::RestrictPtrTraits> XQ,
    at::BFloat16* scale_device,
    float inv_scale) {
  auto N = X.size(0);
  if (scale_device) {
    auto scale = *reinterpret_cast<const __nv_bfloat16*>(scale_device);
    inv_scale = 1.0 / (__bfloat162float(scale) + 1.0e-8);
  }

  for (auto i = threadIdx.x * 8 + blockIdx.x * blockDim.x * 8; i < N;
       i += 8 * blockDim.x * gridDim.x) {
    bf16x8 src;
    *reinterpret_cast<uint4*>(&src) = *reinterpret_cast<uint4*>(&X[i]);

    auto x1_0 = __bfloat162float(src.vals[0].x);
    auto x1_1 = __bfloat162float(src.vals[0].y);
    auto x1_2 = __bfloat162float(src.vals[1].x);
    auto x1_3 = __bfloat162float(src.vals[1].y);
    auto x1_4 = __bfloat162float(src.vals[2].x);
    auto x1_5 = __bfloat162float(src.vals[2].y);
    auto x1_6 = __bfloat162float(src.vals[3].x);
    auto x1_7 = __bfloat162float(src.vals[3].y);

    auto y_0 = x1_0 * inv_scale;
    auto y_1 = x1_1 * inv_scale;
    auto y_2 = x1_2 * inv_scale;
    auto y_3 = x1_3 * inv_scale;
    auto y_4 = x1_4 * inv_scale;
    auto y_5 = x1_5 * inv_scale;
    auto y_6 = x1_6 * inv_scale;
    auto y_7 = x1_7 * inv_scale;

    y_0 = fmaxf(-128.0, fminf(y_0, 127));
    y_1 = fmaxf(-128.0, fminf(y_1, 127));
    y_2 = fmaxf(-128.0, fminf(y_2, 127));
    y_3 = fmaxf(-128.0, fminf(y_3, 127));
    y_4 = fmaxf(-128.0, fminf(y_4, 127));
    y_5 = fmaxf(-128.0, fminf(y_5, 127));
    y_6 = fmaxf(-128.0, fminf(y_6, 127));
    y_7 = fmaxf(-128.0, fminf(y_7, 127));

    i8x8 dst;
    dst.vals[0] = __float2int_rn(y_0);
    dst.vals[1] = __float2int_rn(y_1);
    dst.vals[2] = __float2int_rn(y_2);
    dst.vals[3] = __float2int_rn(y_3);
    dst.vals[4] = __float2int_rn(y_4);
    dst.vals[5] = __float2int_rn(y_5);
    dst.vals[6] = __float2int_rn(y_6);
    dst.vals[7] = __float2int_rn(y_7);
    *reinterpret_cast<uint2*>(&XQ[i]) = *reinterpret_cast<uint2*>(&dst);
  }
}

/// @ingroup int8 per tensor quantization op
///
/// Apply int8 tensor-wise quantization on the input tensor X, with the scale
///
/// @param X The input tensor
/// @param scale The scaling factor
///
/// @return int8 tensor-wise quantized tensor
at::Tensor per_tensor_quantize_i8(at::Tensor X, double scale) {
  CUDA_DEVICE_GUARD(X);
  TORCH_CHECK(scale != 0.0);
  float inv_scale = 1.0 / (scale + 1.0e-8);
  constexpr int32_t kThreadsPerBlock = 1024;
  auto XQ = at::empty({X.numel()}, X.options().dtype(at::kChar));
  dim3 threads = kThreadsPerBlock;
  dim3 blocks =
      cuda_calc_block_count(div_round_up(X.numel(), 8), kThreadsPerBlock);
  per_tensor_quantize_i8_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      X.packed_accessor64<at::BFloat16, 1, at::RestrictPtrTraits>(),
      XQ.packed_accessor64<int8_t, 1, at::RestrictPtrTraits>(),
      nullptr,
      inv_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return XQ;
}

std::tuple<at::Tensor, at::Tensor> per_tensor_dynamic_quantize_i8(
    at::Tensor X) {
  // TORCH_CHECK(scale != 0.0);
  CUDA_DEVICE_GUARD(X);
  constexpr int32_t kThreadsPerBlock = 1024;
  auto XQ = at::empty({X.numel()}, X.options().dtype(at::kChar));

  auto scale = at::norm(X, std::numeric_limits<double>::infinity())
                   .div_(127.0)
                   .to(X.dtype());

  dim3 threads = kThreadsPerBlock;
  dim3 blocks =
      cuda_calc_block_count(div_round_up(X.numel(), 8), kThreadsPerBlock);

  per_tensor_quantize_i8_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      X.packed_accessor64<at::BFloat16, 1, at::RestrictPtrTraits>(),
      XQ.packed_accessor64<int8_t, 1, at::RestrictPtrTraits>(),
      scale.data_ptr<at::BFloat16>(),
      0.0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {XQ, scale};
}

DEVICE_INLINE float __sigmoid(float a) {
  return 1.0 / (1.0 + __expf(-a));
}

__global__ void silu_mul_quantize_i8_kernel(
    at::PackedTensorAccessor64<at::BFloat16, 2, at::RestrictPtrTraits>
        X1, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<at::BFloat16, 2, at::RestrictPtrTraits>
        X2, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<int8_t, 2, at::RestrictPtrTraits>
        Y, // [B][MAX_T][N_KVH][D_H]
    float inv_scale) {
  auto b = blockIdx.x;
  auto N = X1.size(1);
  for (auto i = threadIdx.x * 8; i < N; i += 8 * blockDim.x) {
    bf16x8 src1;
    *reinterpret_cast<uint4*>(&src1) = *reinterpret_cast<uint4*>(&X1[b][i]);
    bf16x8 src2;
    *reinterpret_cast<uint4*>(&src2) = *reinterpret_cast<uint4*>(&X2[b][i]);

    auto x1_0 = __bfloat162float(src1.vals[0].x);
    auto x1_1 = __bfloat162float(src1.vals[0].y);
    auto x1_2 = __bfloat162float(src1.vals[1].x);
    auto x1_3 = __bfloat162float(src1.vals[1].y);
    auto x1_4 = __bfloat162float(src1.vals[2].x);
    auto x1_5 = __bfloat162float(src1.vals[2].y);
    auto x1_6 = __bfloat162float(src1.vals[3].x);
    auto x1_7 = __bfloat162float(src1.vals[3].y);

    auto x2_0 = __bfloat162float(src2.vals[0].x);
    auto x2_1 = __bfloat162float(src2.vals[0].y);
    auto x2_2 = __bfloat162float(src2.vals[1].x);
    auto x2_3 = __bfloat162float(src2.vals[1].y);
    auto x2_4 = __bfloat162float(src2.vals[2].x);
    auto x2_5 = __bfloat162float(src2.vals[2].y);
    auto x2_6 = __bfloat162float(src2.vals[3].x);
    auto x2_7 = __bfloat162float(src2.vals[3].y);

    auto y_0 = x1_0 * __sigmoid(x1_0) * x2_0 * inv_scale;
    auto y_1 = x1_1 * __sigmoid(x1_1) * x2_1 * inv_scale;
    auto y_2 = x1_2 * __sigmoid(x1_2) * x2_2 * inv_scale;
    auto y_3 = x1_3 * __sigmoid(x1_3) * x2_3 * inv_scale;
    auto y_4 = x1_4 * __sigmoid(x1_4) * x2_4 * inv_scale;
    auto y_5 = x1_5 * __sigmoid(x1_5) * x2_5 * inv_scale;
    auto y_6 = x1_6 * __sigmoid(x1_6) * x2_6 * inv_scale;
    auto y_7 = x1_7 * __sigmoid(x1_7) * x2_7 * inv_scale;

    y_0 = fmaxf(-128.0, fminf(y_0, 127));
    y_1 = fmaxf(-128.0, fminf(y_1, 127));
    y_2 = fmaxf(-128.0, fminf(y_2, 127));
    y_3 = fmaxf(-128.0, fminf(y_3, 127));
    y_4 = fmaxf(-128.0, fminf(y_4, 127));
    y_5 = fmaxf(-128.0, fminf(y_5, 127));
    y_6 = fmaxf(-128.0, fminf(y_6, 127));
    y_7 = fmaxf(-128.0, fminf(y_7, 127));

    i8x8 dst;
    dst.vals[0] = __float2int_rn(y_0);
    dst.vals[1] = __float2int_rn(y_1);
    dst.vals[2] = __float2int_rn(y_2);
    dst.vals[3] = __float2int_rn(y_3);
    dst.vals[4] = __float2int_rn(y_4);
    dst.vals[5] = __float2int_rn(y_5);
    dst.vals[6] = __float2int_rn(y_6);
    dst.vals[7] = __float2int_rn(y_7);

    *reinterpret_cast<uint2*>(&Y[b][i]) = *reinterpret_cast<uint2*>(&dst);
  }
}

at::Tensor silu_mul_quantize_i8(at::Tensor X1, at::Tensor X2, double scale) {
  float inv_scale = 1.0 / (scale + 1.0e-8);
  auto Y = at::empty(X1.sizes(), X1.options().dtype(at::kChar));
  TORCH_CHECK(X1.size(1) % 8 == 0);
  constexpr int32_t kThreadsPerBlock = 1024;
  dim3 threads = std::min<int32_t>(kThreadsPerBlock, X1.size(1) / 8);
  dim3 blocks = X1.size(0);
  silu_mul_quantize_i8_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      X1.packed_accessor64<at::BFloat16, 2, at::RestrictPtrTraits>(),
      X2.packed_accessor64<at::BFloat16, 2, at::RestrictPtrTraits>(),
      Y.packed_accessor64<int8_t, 2, at::RestrictPtrTraits>(),
      inv_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Y;
}

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
class FP8_E4M3_MAX {
 public:
#ifndef USE_ROCM
  static constexpr float value = 448.0;
#else
  static constexpr float value = 240.0;
#endif
};
class FP8_E5M2_MAX {
 public:
  static constexpr float value = 57344.0;
};
constexpr int CTA_SIZE = 256;

template <bool QUANTIZE>
__inline__ __device__ float scale(float a, float b) {
  return QUANTIZE ? a / b : a * b;
}

// Correct for cases where x is not subnormal.
// Ref:
// https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/include/fbgemm_gpu/utils/stochastic_rounding.cuh
// https://fb.quip.com/zNWAAmYREHhz : AssembleFloat algorithm
DEVICE_INLINE float stochastic_rounding_scalar_fp8(
    float x,
    uint32_t random_value) {
  uint32_t w_int = __float_as_uint(x);

  // Adjust the subnormal values to normal values if x is subnormal (<pow(2,-6))
  // if x_sign - 127 <= -6, then x_adjust = x + pow(2, -6)
  uint32_t x_exp = (w_int << 1) >> 24;
  int32_t x_exp_int = static_cast<int32_t>(x_exp);
  x_exp_int = x_exp_int - 127;
  uint32_t x_sign = w_int & 0x80000000;
  constexpr uint32_t shift_bits_unsigned = ((127 - 6) << 23);
  uint32_t shift_bits = shift_bits_unsigned | x_sign;
  float shift_value = __uint_as_float(shift_bits);
  // subnormal exp of FP8 E4M3
  constexpr int32_t subnormal_exp = -6;
  if (x_exp_int < subnormal_exp) {
    x = x + shift_value;
  }

  // 1-bit sign + 8-bit exponent
  constexpr uint32_t sign_exp_mask = ((1U << 9) - 1) << (32 - 9);
  unsigned subtract = (w_int & sign_exp_mask);
  // FP32 has 23-bit mantissa, fp8 e4m3 has 3-bit mantissa
  // 12 = 32 - (23 - 3)
  unsigned assmebles = subtract | (random_value >> 12);
  float assemble_float = __uint_as_float(assmebles) - __uint_as_float(subtract);
  // truncate the last 20 bit (keep 3 bit mantissa) to simulate round to zero
  uint32_t ret_int = __float_as_uint(x + assemble_float);
  const uint32_t sign_exp_3bit_mantissa_mask = ((1U << 12) - 1) << (32 - 12);
  float ret = __uint_as_float(ret_int & sign_exp_3bit_mantissa_mask);

  // Before the return, subtract pow(2,-6) from the SR'ed subnormal values
  // SR(x) = SR(x + pow(2,-6)) - pow(2, -6)
  if (x_exp_int < subnormal_exp) {
    ret = ret - shift_value;
  }
  return ret;
}

template <bool QUANTIZE, typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrix(
    T_OUT* const output,
    T_S const* const input_scale,
    T_IN const* const input,
    const int64_t numel,
    const int64_t lda) {
  for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel;
       i += (size_t)blockDim.x * gridDim.x) {
    output[i] = T_OUT(scale<QUANTIZE>(
        static_cast<float>(input[i]), static_cast<float>(input_scale[0])));
  }
}

template <bool QUANTIZE, typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrix(
    T_OUT* const output,
    T_S const* const input_scale,
    T_IN const* const input,
    const int64_t numel,
    const int64_t lda,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  auto stoc_rounding_state = StochasticRoundingRNGState(
      stochastic_rounding_philox_args, threadIdx.x + blockIdx.x * blockDim.x);
  auto input_scal = static_cast<float>(input_scale[0]);

  auto vec_output = reinterpret_cast<__nv_fp8x4_e4m3*>(&output[0]);
  auto vec_input = reinterpret_cast<const bfx4*>(&input[0]);
  for (int32_t d = (threadIdx.x + blockIdx.x * blockDim.x); d * 4 < numel;
       d += (size_t)blockDim.x * gridDim.x) {
    const auto random_bits = stoc_rounding_state.rand4();
    bfx4 v_in = vec_input[d];
    float4 v_float;
    v_float.x = stochastic_rounding_scalar_fp8(
        scale<QUANTIZE>(static_cast<float>(v_in.vals[0].x), input_scal),
        random_bits.x);
    v_float.y = stochastic_rounding_scalar_fp8(
        scale<QUANTIZE>(static_cast<float>(v_in.vals[0].y), input_scal),
        random_bits.y);
    v_float.z = stochastic_rounding_scalar_fp8(
        scale<QUANTIZE>(static_cast<float>(v_in.vals[1].x), input_scal),
        random_bits.z);
    v_float.w = stochastic_rounding_scalar_fp8(
        scale<QUANTIZE>(static_cast<float>(v_in.vals[1].y), input_scal),
        random_bits.w);
    vec_output[d] = __nv_fp8x4_e4m3(v_float);
  }
}

template <bool QUANTIZE, typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrixRowwise(
    T_OUT* const output,
    T_S const* const input_scale,
    T_IN const* const input,
    const int64_t numel,
    const int64_t lda,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  auto stoc_rounding_state = StochasticRoundingRNGState(
      stochastic_rounding_philox_args, threadIdx.x + blockIdx.x * blockDim.x);
  auto input_scal = static_cast<float>(input_scale[0]);

  auto vec_output = reinterpret_cast<__nv_fp8x4_e4m3*>(&output[0]);
  auto vec_input = reinterpret_cast<const bfx4*>(&input[0]);
  auto vec_scale = reinterpret_cast<const float4*>(&input_scale[0]);
  for (int32_t d = (threadIdx.x + blockIdx.x * blockDim.x); d * 4 < numel;
       d += (size_t)blockDim.x * gridDim.x) {
    const auto random_bits = stoc_rounding_state.rand4();
    bfx4 v_in = vec_input[d];
    float4 v_float;
    float4 v_scale = vec_scale[d / lda];
    v_float.x = stochastic_rounding_scalar_fp8(
        scale<QUANTIZE>(
            static_cast<float>(v_in.vals[0].x), static_cast<float>(v_scale.x)),
        random_bits.x);
    v_float.y = stochastic_rounding_scalar_fp8(
        scale<QUANTIZE>(
            static_cast<float>(v_in.vals[0].y), static_cast<float>(v_scale.y)),
        random_bits.y);
    v_float.z = stochastic_rounding_scalar_fp8(
        scale<QUANTIZE>(
            static_cast<float>(v_in.vals[1].x), static_cast<float>(v_scale.z)),
        random_bits.z);
    v_float.w = stochastic_rounding_scalar_fp8(
        scale<QUANTIZE>(
            static_cast<float>(v_in.vals[1].y), static_cast<float>(v_scale.w)),
        random_bits.w);
    vec_output[d] = __nv_fp8x4_e4m3(v_float);
  }
}

template <bool QUANTIZE, typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrixRowwise(
    T_OUT* const output,
    T_S const* const input_scale,
    T_IN const* const input,
    const int64_t numel,
    const int64_t lda) {
  for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel;
       i += (size_t)blockDim.x * gridDim.x) {
    output[i] = T_OUT(scale<QUANTIZE>(
        static_cast<float>(input[i]),
        static_cast<float>(input_scale[i / lda])));
  }
}

template <bool QUANTIZE, typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrixColwise(
    T_OUT* const output,
    T_S const* const input_scale,
    T_IN const* const input,
    const int64_t numel,
    const int64_t lda) {
  for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel;
       i += (size_t)blockDim.x * gridDim.x) {
    output[i] = T_OUT(scale<QUANTIZE>(
        static_cast<float>(input[i]),
        static_cast<float>(input_scale[i % lda])));
  }
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeQuantizeMatrix(
    T_OUT* const output,
    T_S const* const input_scale,
    T_IN const* const input,
    const int64_t numel,
    const int64_t lda,
    bool stochastic_rounding,
    const cudaStream_t stream) {
  constexpr dim3 grid(1024);
  const dim3 block(CTA_SIZE);
  if (stochastic_rounding) {
    at::PhiloxCudaState rng_engine_inputs;
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    std::lock_guard<std::mutex> lock(gen.mutex());
    rng_engine_inputs =
        at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(4);
    scaleMatrix<true><<<grid, block, 0, stream>>>(
        output, input_scale, input, numel, lda, rng_engine_inputs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    scaleMatrix<true>
        <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeQuantizeMatrixRowwise(
    T_OUT* const output,
    T_S const* const input_scale,
    T_IN const* const input,
    const int64_t numel,
    const int64_t lda,
    bool stochastic_rounding,
    const cudaStream_t stream) {
  constexpr dim3 grid(1024);
  const dim3 block(CTA_SIZE);

  if (stochastic_rounding) {
    at::PhiloxCudaState rng_engine_inputs;
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    std::lock_guard<std::mutex> lock(gen.mutex());
    rng_engine_inputs =
        at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(4);

    scaleMatrixRowwise<true><<<grid, block, 0, stream>>>(
        output, input_scale, input, numel, lda, rng_engine_inputs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

  } else {
    scaleMatrixRowwise<true>
        <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeQuantizeMatrixColwise(
    T_OUT* const output,
    T_S const* const input_scale,
    T_IN const* const input,
    const int64_t numel,
    const int64_t lda,
    const cudaStream_t stream) {
  constexpr dim3 grid(1024);
  const dim3 block(CTA_SIZE);
  scaleMatrixColwise<true>
      <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  const uint32_t lane = threadIdx.x & 0x1f; // in-warp idx
  const uint32_t wid = threadIdx.x >> 5; // warp idx

  val = warpReduceMax(val);

  if (lane == 0) {
    shared[wid] = val;
  }

  __syncthreads();

  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

__device__ float atomicMaxExtd(float* address, float val) {
  assert(val >= 0);
  unsigned int* address_as_u = reinterpret_cast<unsigned int*>(address);
  unsigned int old = atomicMax(address_as_u, __float_as_uint(val));
  return __uint_as_float(old);
}

template <typename T_S, typename T_W>
__global__ void computeFP8QuantizeScale(
    T_S* const quant_ptr,
    const T_W* const weights,
    const int64_t size,
    const int64_t n,
    const int64_t total_elements_per_slice,
    const int64_t* bs,
    const float* scale_ub) {
  float max = 0.f;
  int64_t numel_scale = size;
  if (total_elements_per_slice != -1 && bs != nullptr) {
    numel_scale = size / total_elements_per_slice * (*bs);
  }
  constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX::value * 512.f);
  for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel_scale;
       i += (size_t)gridDim.x * blockDim.x) {
    auto val = fabs(static_cast<float>(weights[i]));
    max = max > val ? max : val;
  }
  max = blockReduceMax<float>(max);
  if (threadIdx.x == 0) {
    auto bounded_max = max;
    if (scale_ub != nullptr) {
      bounded_max = std::min(max, *scale_ub);
    }
    const auto scale =
        (T_S)std::max(bounded_max / FP8_E4M3_MAX::value, min_scaling_factor);
    atomicMaxExtd(quant_ptr, scale);
  }
}

template <typename T_S, typename T_W>
__global__ void computeFP8QuantizeScaleColwise(
    T_S* quant_ptr,
    const T_W* weights,
    const int64_t size,
    const int64_t n) {
  constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX::value * 512.f);

  for (int64_t col = threadIdx.x; col < n; col += blockDim.x) {
    float max = 0.f;
    for (int64_t i = col + n * blockIdx.x; i < size; i += gridDim.x * n) {
      auto val = fabs(static_cast<float>(weights[i]));
      max = max > val ? max : val;
    }
    auto const scale =
        (T_S)std::max(max / FP8_E4M3_MAX::value, min_scaling_factor);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr (std::is_same_v<T_S, float>) {
      atomicMaxExtd(quant_ptr + col, scale);
    } else {
      auto const address_u64 = reinterpret_cast<uint64_t>(quant_ptr + col);
      if ((col == 0 && address_u64 % 4 != 0) ||
          (col == n - 1 && address_u64 % 4 == 0))
        atomicMaxExtd(quant_ptr + col, scale);
      else
        atomicMaxExtdV2(quant_ptr + col, scale);
    }
#else // Vector atomics require __CUDA_ARCH__ >= 900
    atomicMaxExtd(quant_ptr + col, scale);
#endif
  }
}

template <typename T_S, typename T_IN>
void invokeComputeScale(
    T_S* const quant_ptr,
    const T_IN* const input,
    const int64_t numel,
    const int64_t lda,
    const int64_t total_elements_per_slice,
    const int64_t* bs,
    const float* scale_ub,
    const cudaStream_t stream) {
  constexpr dim3 block(1024);
  constexpr dim3 grid(1024);
  int64_t numel_scale = numel;
  C10_CUDA_CHECK(cudaMemsetAsync(quant_ptr, 0, sizeof(T_S), stream));
  computeFP8QuantizeScale<<<grid, block, 0, stream>>>(
      quant_ptr,
      input,
      numel_scale,
      lda,
      total_elements_per_slice,
      bs,
      scale_ub);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor get_fp8_per_tensor_scale(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub) // scale upper bound
{
  CUDA_DEVICE_GUARD(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Invalid dim. The dim of input should be greater than or equal to 2");
  auto _st = input.scalar_type();
  TORCH_CHECK(_st == torch::kBFloat16, "Invalid datatype. input must be BF16");

  int out_size = input.numel() == 0 ? 0 : 1;

  at::Tensor scale = torch::empty(
      {out_size},
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));

  // Handle case where input is empty.
  if (input.numel() == 0) {
    return scale;
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  invokeComputeScale(
      reinterpret_cast<float*>(scale.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      input.numel(),
      input.size(-1),
      input.size(0),
      bs.has_value() ? reinterpret_cast<int64_t*>(bs.value().data_ptr())
                     : nullptr,
      scale_ub.has_value()
          ? reinterpret_cast<float*>(scale_ub.value().data_ptr())
          : nullptr,
      stream);

  return scale;
}

at::Tensor quantize_fp8_per_tensor_fixed_scale(
    at::Tensor input,
    at::Tensor scale,
    std::optional<at::Tensor> bs, // batch size
    bool stochastic_rounding) {
  CUDA_DEVICE_GUARD(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Invalid dim. The dim of input should be greater than or equal to 2");
  auto _st = input.scalar_type();
  TORCH_CHECK(_st == torch::kBFloat16, "Invalid datatype. input must be BF16");

  std::vector<long int> quantized_input_shape;
  quantized_input_shape.reserve(input.dim());
  for (int i = 0; i < input.dim(); i++) {
    quantized_input_shape.push_back(input.size(i));
  }

  at::Tensor quantized_input = torch::empty(
      quantized_input_shape,
      torch::dtype(torch_fp8_e4m3)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));

  // When input is empty, return empty scale as well.
  if (input.numel() == 0) {
    return quantized_input;
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  invokeQuantizeMatrix(
      reinterpret_cast<__nv_fp8_e4m3*>(quantized_input.data_ptr()),
      reinterpret_cast<float*>(scale.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      input.numel(),
      input.size(-1),
      stochastic_rounding,
      stream);

  return quantized_input;
}

// TODO: Extend to support other data types for other
// usecases/models when needed
std::vector<at::Tensor> quantize_fp8_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub, // scale upperbound
    bool stochastic_rounding) // stochastic rounding
{
  CUDA_DEVICE_GUARD(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Invalid dim. The dim of input should be greater than or equal to 2");
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16 ||
          input.scalar_type() == torch::kFloat,
      "Invalid datatype. input must be BF16 or FP32");
  TORCH_CHECK(
      !stochastic_rounding || input.size(-1) % 4 == 0,
      "input row dim must be 4's multiple when stochastic_rounding is True");
  std::vector<long int> quantized_input_shape;
  quantized_input_shape.reserve(input.dim());
  for (int i = 0; i < input.dim(); i++) {
    quantized_input_shape.push_back(input.size(i));
  }
  std::vector<long int> scale_shape = {};
  input = input.cuda();
  at::Tensor quantized_input = torch::empty(
      quantized_input_shape,
      torch::dtype(torch_fp8_e4m3)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));
  at::Tensor scales = torch::empty(
      scale_shape,
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));
  // When input is empty, return empty tensors.
  if (input.numel() == 0) {
    return std::vector<at::Tensor>{quantized_input, scales};
  }
  auto* const quantized_input_ptr =
      reinterpret_cast<__nv_fp8_e4m3*>(quantized_input.data_ptr());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (bs.has_value()) {
    int64_t total_elements_per_slice = quantized_input_shape[0];
    for (int i = 1; i < input.dim() - 1; i++) {
      total_elements_per_slice =
          total_elements_per_slice * quantized_input_shape[i];
    }
    invokeComputeScale(
        reinterpret_cast<float*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        input.numel(),
        input.size(-1),
        total_elements_per_slice,
        reinterpret_cast<int64_t*>(bs.value().data_ptr()),
        scale_ub.has_value()
            ? reinterpret_cast<float*>(scale_ub.value().data_ptr())
            : nullptr,
        stream);
    if (input.scalar_type() == torch::kBFloat16) {
      invokeQuantizeMatrix(
          quantized_input_ptr,
          reinterpret_cast<float*>(scales.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
          input.numel(),
          input.size(-1),
          stochastic_rounding,
          stream);
    } else {
      invokeQuantizeMatrix(
          quantized_input_ptr,
          reinterpret_cast<float*>(scales.data_ptr()),
          reinterpret_cast<const float*>(input.data_ptr()),
          input.numel(),
          input.size(-1),
          stochastic_rounding,
          stream);
    }
  } else {
    invokeComputeScale(
        reinterpret_cast<float*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        input.numel(),
        input.size(-1),
        -1,
        nullptr,
        scale_ub.has_value()
            ? reinterpret_cast<float*>(scale_ub.value().data_ptr())
            : nullptr,
        stream);
    if (input.scalar_type() == torch::kBFloat16) {
      invokeQuantizeMatrix(
          quantized_input_ptr,
          reinterpret_cast<float*>(scales.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
          input.numel(),
          input.size(-1),
          stochastic_rounding,
          stream);
    } else {
      invokeQuantizeMatrix(
          quantized_input_ptr,
          reinterpret_cast<float*>(scales.data_ptr()),
          reinterpret_cast<const float*>(input.data_ptr()),
          input.numel(),
          input.size(-1),
          stochastic_rounding,
          stream);
    }
  }
  return std::vector<at::Tensor>{quantized_input, scales};
}

template <typename T>
__inline__ __device__ T blockAllReduceMax(T val) {
  static __shared__ T shared[32];
  auto lane = threadIdx.x & 0x1f;
  auto wid = threadIdx.x >> 5;
  val = warpReduceMax(val);
  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (lane < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

template <typename SCALE, typename T_OUT, typename T_S, typename T_IN>
__global__ void dynamicQuantizeMatrixRowwise(
    T_OUT* output,
    T_S* quant_ptr,
    T_IN const* input,
    int64_t numel,
    int64_t lda,
    const float* scale_ub) {
  extern __shared__ __align__(sizeof(float)) char _shmem[];
  T_IN* shmem = reinterpret_cast<T_IN*>(_shmem);
  constexpr float min_scaling_factor = 1.0f / (SCALE::value * 512.f);
  auto const nrows = numel / lda;
  for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x) {
    float max = 0.f;
    for (int64_t i = threadIdx.x; i < lda; i += blockDim.x) {
      auto const in = input[row * lda + i];
      shmem[i] = in;
      auto val = fabs(static_cast<float>(in));
      max = max > val ? max : val;
    }
    max = blockAllReduceMax<float>(max);
    auto bounded_max = max;
    if (scale_ub != nullptr) {
      bounded_max = std::min(max, *scale_ub);
    }
    auto const s =
        (T_S)std::max(bounded_max / SCALE::value, min_scaling_factor);
    for (int64_t i = threadIdx.x; i < lda; i += blockDim.x) {
      output[row * lda + i] = (T_OUT)scale<true>(
          static_cast<float>(shmem[i]), static_cast<float>(s));
    }
    if (threadIdx.x == 0) {
      quant_ptr[row] = s;
    }
  }
}

template <typename SCALE, typename T_OUT, typename T_S, typename T_IN>
__global__ void dynamicQuantizeMatrixRowwiseStoc(
    T_OUT* output,
    T_S* quant_ptr,
    T_IN const* input,
    int64_t numel,
    int64_t lda,
    const float* scale_ub,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  auto stoc_rounding_state = StochasticRoundingRNGState(
      stochastic_rounding_philox_args, threadIdx.x + blockIdx.x * blockDim.x);
  const auto random_bits = stoc_rounding_state.rand4();

  extern __shared__ __align__(sizeof(float)) char _shmem[];
  T_IN* shmem = reinterpret_cast<T_IN*>(_shmem);
  constexpr float min_scaling_factor = 1.0f / (SCALE::value * 512.f);
  auto const nrows = numel / lda;
  for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x) {
    float max = 0.f;
    for (int64_t i = threadIdx.x; i < lda; i += blockDim.x) {
      auto const in = input[row * lda + i];
      shmem[i] = in;
      auto val = fabs(static_cast<float>(in));
      max = max > val ? max : val;
    }
    max = blockAllReduceMax<float>(max);
    auto bounded_max = max;
    if (scale_ub != nullptr) {
      bounded_max = std::min(max, *scale_ub);
    }
    auto const s =
        (T_S)std::max(bounded_max / SCALE::value, min_scaling_factor);
    for (int64_t i = threadIdx.x; i < lda; i += blockDim.x) {
      output[row * lda + i] = (T_OUT)(stochastic_rounding_scalar_fp8(
          scale<true>(static_cast<float>(shmem[i]), static_cast<float>(s)),
          random_bits.x));
    }
    if (threadIdx.x == 0) {
      quant_ptr[row] = s;
    }
  }
}

template <typename SCALE, typename T_S, typename T_W>
__global__ void computeFP8QuantizeScaleRowwise(
    T_S* quant_ptr,
    const T_W* weights,
    const int64_t size,
    const int64_t n,
    const float* scale_ub) {
  constexpr float min_scaling_factor = 1.0f / (SCALE::value * 512.f);
  auto const nrows = size / n;
  for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x) {
    float max = 0.f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
      auto val = fabs(static_cast<float>(weights[row * n + i]));
      max = max > val ? max : val;
    }
    max = blockReduceMax<float>(max);
    if (threadIdx.x == 0) {
      auto bounded_max = max;
      if (scale_ub != nullptr) {
        bounded_max = std::min(max, *scale_ub);
      }
      auto const scale =
          (T_S)std::max(bounded_max / SCALE::value, min_scaling_factor);
      quant_ptr[row] = scale;
    }
  }
}

template <typename SCALE, typename T_OUT, typename T_S, typename T_IN>
void invokeComputeScalesAndQuantizeMatrix(
    T_OUT* output,
    T_S* quant_ptr,
    const T_IN* input,
    const int64_t numel,
    const int64_t lda,
    const float* scale_ub,
    bool stochastic_rounding,
    cudaStream_t stream) {
  dim3 grid(numel / lda);
  bool use_shmem = true;
  auto const shmem_size = lda * sizeof(T_IN);
  if (shmem_size >= (48 << 10)) {
    cudaError_t ret;
#ifndef USE_ROCM
    if (stochastic_rounding) {
      ret = cudaFuncSetAttribute(
          dynamicQuantizeMatrixRowwiseStoc<SCALE, T_OUT, T_S, T_IN>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shmem_size);
    } else {
      ret = cudaFuncSetAttribute(
          dynamicQuantizeMatrixRowwise<SCALE, T_OUT, T_S, T_IN>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shmem_size);
    }
    use_shmem = ret == cudaSuccess;
#else
    use_shmem = false;
#endif
  }
  if (use_shmem) {
    dim3 block(std::min((lda + 31) / 32 * 32, static_cast<int64_t>(1024)));

    if (stochastic_rounding) {
      at::PhiloxCudaState rng_engine_inputs;
      auto gen = at::cuda::detail::getDefaultCUDAGenerator();
      std::lock_guard<std::mutex> lock(gen.mutex());
      rng_engine_inputs =
          at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(4);

      dynamicQuantizeMatrixRowwiseStoc<SCALE>
          <<<grid, block, shmem_size, stream>>>(
              output,
              quant_ptr,
              input,
              numel,
              lda,
              scale_ub,
              rng_engine_inputs);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      dynamicQuantizeMatrixRowwise<SCALE><<<grid, block, shmem_size, stream>>>(
          output, quant_ptr, input, numel, lda, scale_ub);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  } else {
    dim3 block(CTA_SIZE);
    computeFP8QuantizeScaleRowwise<SCALE>
        <<<grid, block, 0, stream>>>(quant_ptr, input, numel, lda, scale_ub);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    invokeQuantizeMatrixRowwise(
        output, quant_ptr, input, numel, lda, stochastic_rounding, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeComputeScalesAndQuantizeMatrixCol(
    T_OUT* output,
    T_S* quant_ptr,
    const T_IN* input,
    const int64_t numel,
    const int64_t lda,
    cudaStream_t stream) {
  dim3 block(CTA_SIZE);
  dim3 grid((lda + CTA_SIZE - 1) / CTA_SIZE);
  C10_CUDA_CHECK(cudaMemsetAsync(quant_ptr, 0, lda * sizeof(T_S), stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  computeFP8QuantizeScaleColwise<<<grid, block, 0, stream>>>(
      quant_ptr, input, numel, lda);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  invokeQuantizeMatrixColwise(output, quant_ptr, input, numel, lda, stream);
}

std::vector<at::Tensor> quantize_fp8_per_row(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub, // scale upperbound
    std::optional<c10::ScalarType> output_dtype, // Quantization type
    bool stochastic_rounding) {
  TORCH_CHECK(
      input.dim() >= 2,
      "Invalid dim. The dim of input should be greater than or equal to 2");
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16 ||
          input.scalar_type() == torch::kFloat ||
          input.scalar_type() == torch::kHalf,
      "Invalid datatype. input must be BF16, FP16 or FP32");
  TORCH_CHECK(
      !stochastic_rounding || input.size(-1) % 4 == 0,
      "input row dim must be 4's multiple when stochastic_rounding is True");
  // Default data type is f8_e4m3fn.
  c10::ScalarType quantization_type = torch_fp8_e4m3;
  if (output_dtype.has_value()) {
    TORCH_CHECK(
        (output_dtype.value() == torch_fp8_e4m3 ||
         output_dtype.value() == torch_fp8_e5m2),
        "Invalid output type, must be e4m3 or e5m2.");
    quantization_type = output_dtype.value();
  }
  std::vector<long int> quantized_input_shape;
  for (int i = 0; i < input.dim(); i++)
    quantized_input_shape.push_back(input.size(i));
  std::vector<int64_t> scale_shape;
  for (int i = 0; i < input.dim() - 1; i++)
    scale_shape.push_back(input.size(i));

  input = input.cuda();
  at::Tensor quantized_input = torch::empty(
      quantized_input_shape,
      torch::dtype(quantization_type)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));
  at::Tensor scales = torch::empty(
      scale_shape,
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));

  if (input.numel() == 0) {
    return std::vector<at::Tensor>{quantized_input, scales};
  }

  // Templatize implementation based on output type.
  if (quantization_type == torch_fp8_e4m3) {
    auto* const quantized_input_ptr =
        reinterpret_cast<__nv_fp8_e4m3*>(quantized_input.data_ptr());
    const auto stream = at::cuda::getCurrentCUDAStream();
    invokeComputeScalesAndQuantizeMatrix<FP8_E4M3_MAX>(
        quantized_input_ptr,
        reinterpret_cast<float*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        input.numel(),
        input.size(-1),
        scale_ub.has_value()
            ? reinterpret_cast<float*>(scale_ub.value().data_ptr())
            : nullptr,
        stochastic_rounding,
        stream);

    return std::vector<at::Tensor>{quantized_input, scales};
  } else {
    auto* const quantized_input_ptr =
        reinterpret_cast<__nv_fp8_e5m2*>(quantized_input.data_ptr());
    const auto stream = at::cuda::getCurrentCUDAStream();
    invokeComputeScalesAndQuantizeMatrix<FP8_E5M2_MAX>(
        quantized_input_ptr,
        reinterpret_cast<float*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        input.numel(),
        input.size(-1),
        scale_ub.has_value()
            ? reinterpret_cast<float*>(scale_ub.value().data_ptr())
            : nullptr,
        stochastic_rounding,
        stream);

    return std::vector<at::Tensor>{quantized_input, scales};
  }
}

std::vector<at::Tensor> quantize_fp8_per_col(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub) // scale upperbound)
{
  CUDA_DEVICE_GUARD(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Invalid dim. The dim of input should be greater than or equal to 2");
  auto _st = input.scalar_type();
  TORCH_CHECK(_st == torch::kBFloat16, "Invalid datatype. input must be BF16");
  std::vector<long int> quantized_input_shape;
  for (int i = 0; i < input.dim(); i++)
    quantized_input_shape.push_back(input.size(i));
  std::vector<int64_t> scale_shape;
  for (int i = 1; i < input.dim(); i++)
    scale_shape.push_back(input.size(i));

  input = input.cuda();
  at::Tensor quantized_input = torch::empty(
      quantized_input_shape,
      torch::dtype(torch_fp8_e4m3)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));
  at::Tensor scales = torch::empty(
      scale_shape,
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));
  // When input is empty, return empty tensors.
  if (input.numel() == 0) {
    return std::vector<at::Tensor>{quantized_input, scales};
  }
  auto* const quantized_input_ptr =
      reinterpret_cast<__nv_fp8_e4m3*>(quantized_input.data_ptr());
  const auto stream = at::cuda::getCurrentCUDAStream();
  invokeComputeScalesAndQuantizeMatrixCol(
      quantized_input_ptr,
      reinterpret_cast<float*>(scales.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      input.numel(),
      input.size(-1),
      stream);

  return std::vector<at::Tensor>{quantized_input, scales};
}

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]),
        "f"(array[1]),
        "f"(array[2]),
        "f"(array[3]),
        "f"(array[4]),
        "f"(array[5]),
        "f"(array[6]),
        "f"(array[7]));
  return val;
#else
  return 0;
#endif
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x),
        "f"(array[0].y),
        "f"(array[1].x),
        "f"(array[1].y),
        "f"(array[2].x),
        "f"(array[2].y),
        "f"(array[3].x),
        "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    int rowIdx,
    int colIdx,
    int numCols,
    SFType* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(
      CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int32_t mTileIdx = mIdx / (32 * 4);
    // SF vector size 16.
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile layout [32, 4] is column-major.
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    // Compute the global offset.
    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride +
        outerMIdx * outerMStride + innerMIdx * innerMStride +
        innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
#endif
  return nullptr;
}

// Define a 16 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t
cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // Get the absolute maximum among all 16 values (two threads).
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    // Extract the 8 exponent bits from float32.
    // float 32bits = 1 sign bit + 8 exponent bits + 23 mantissa bits.
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8SFVal = tmp & 0xff;
    // Convert back to fp32.
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    // Convert back to fp32.
    SFValue = float(tmp);
  }
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) *
  //                       reciprocal(SFScaleVal))
  float outputScale = SFValue != 0
      ? reciprocal_approximate_ftz(
            SFValue * reciprocal_approximate_ftz(SFScaleVal))
      : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,
    int32_t numCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(
      sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
      "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // Input tensor row/col loops.
  for (int64_t rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int64_t colIdx = threadIdx.x;
         colIdx < numCols / CVT_FP4_ELTS_PER_THREAD;
         colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      // Get the output tensor offset.
      // Same as inOffset because 8 elements are packed into one uint32_t.
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<
          uint32_t,
          CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

      out_pos =
          cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
#endif
}

template <typename T>
void invokeFP4Quantization(
    int m,
    int n,
    T const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream) {
  // Grid, Block size.
  // Each thread converts 8 values.
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  // Get number of blocks per SM (assume we can fully utilize the SM).
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  // Launch the cvt kernel.
  if (useUE8M0) {
    cvt_fp16_to_fp4<T, true><<<grid, block, 0, stream>>>(
        m,
        n,
        input,
        SFScale,
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOuput));
  } else {
    cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
        m,
        n,
        input,
        SFScale,
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOuput));
  }
}

// Instantiate the function.
template void invokeFP4Quantization(
    int m,
    int n,
    half const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream);

template void invokeFP4Quantization(
    int m,
    int n,
    __nv_bfloat16 const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream);

int64_t get_device_attribute(int64_t attribute, int64_t device_id) {
  // Return the cached value on subsequent calls
  static int value = [=]() {
    int device = static_cast<int>(device_id);
    if (device < 0) {
      CUDA_CHECK(cudaGetDevice(&device));
    }
    int value;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &value, static_cast<cudaDeviceAttr>(attribute), device));
    return static_cast<int>(value);
  }();

  return value;
}

// This is originally from
// https://github.com/vllm-project/vllm/blob/7fcc4223dca53ea1531093cbbfcf59ddfea1800d/csrc/quantization/fp4/nvfp4_quant_kernels.cu
void scaled_fp4_quant(
    at::Tensor const& output,
    at::Tensor const& input,
    at::Tensor const& output_sf,
    at::Tensor const& input_sf) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  auto stream = at::cuda::getStreamFromPool(false, input.get_device());
  if (stream == nullptr) {
    std::cerr << "Warning: Null CUDA stream" << std::endl;
  }

  // We don't support e8m0 scales at this moment.
  bool useUE8M0 = false;

  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      invokeFP4Quantization(
          m,
          n,
          input_ptr,
          input_sf_ptr,
          output_ptr,
          sf_out,
          useUE8M0,
          multiProcessorCount,
          stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      invokeFP4Quantization(
          m,
          n,
          input_ptr,
          input_sf_ptr,
          output_ptr,
          sf_out,
          useUE8M0,
          multiProcessorCount,
          stream);
      break;
    }
    default: {
      std::cerr << "Observing: " << input.scalar_type()
                << " for the input datatype which is invalid";
      throw std::runtime_error(
          "Unsupported input data type for quantize_to_fp4.");
    }
  }
}
#else
void scaled_fp4_quant(
    at::Tensor const& output,
    at::Tensor const& input,
    at::Tensor const& output_sf,
    at::Tensor const& input_sf) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}
#endif

#else
std::vector<at::Tensor> quantize_fp8_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub,
    bool stochastic_rounding) { // scale upperbound
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

std::vector<at::Tensor> quantize_fp8_per_row(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub, // scale upperbound
    std::optional<c10::ScalarType> output_dtype,
    bool stochastic_rounding) { // quantization type
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor quantize_fp8_per_tensor_fixed_scale(
    at::Tensor input,
    at::Tensor scale,
    std::optional<at::Tensor> bs,
    bool stochastic_rounding) { // batch size
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor get_fp8_per_tensor_scale(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub) { // scale upperbound
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

std::vector<at::Tensor> quantize_fp8_per_col(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub) { // scale upperbound
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

void scaled_fp4_quant(
    at::Tensor const& output,
    at::Tensor const& input,
    at::Tensor const& output_sf,
    at::Tensor const& input_sf) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}

#endif
} // namespace fbgemm_gpu
