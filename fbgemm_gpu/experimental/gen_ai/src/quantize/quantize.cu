/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
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

#include <fbgemm_gpu/sparse_ops_utils.h>
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"

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
#endif

#if (                         \
    defined(__CUDA_ARCH__) && \
    ((__CUDA_ARCH__ == 800) || (__CUDA_ARCH__ == 900)))
#define USE_WMMA_FRAG
#endif

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

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

static __host__ DEVICE_INLINE int32_t div_up(int32_t a, int32_t b) {
  return (a + b - 1) / b;
};

static __host__ DEVICE_INLINE int32_t round_up(int32_t a, int32_t b) {
  return ((a + b - 1) / b) * b;
}

#ifdef __HIP_PLATFORM_AMD__
// constexpr int32_t kThreadsPerWarp = 64;
// constexpr int32_t kWarpsPerBlock = 16;
#else
constexpr int32_t kThreadsPerWarp = 32;
// constexpr int32_t kWarpsPerBlock = 32;
#endif

// constexpr int32_t D_H = 128;
// MAX_T: max seq len. We need to make sure shared memory size
// (https://fburl.com/code/ruc41vc7) <= limit of V100/A100/H100 GPUs
// (https://fburl.com/code/gh9j9go4).
// constexpr int32_t MAX_T = 16384;
// constexpr int SMEM_ADJUST_THRESHOLD = 48 * 1024;

#ifdef __HIP_PLATFORM_AMD__
static __host__ __device__ float __bfloat162float(__nv_bfloat16 f) {
  // float output;
  // https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/hip__bfloat16_8h_source.html
  return float(f);
}

static __host__ __device__ __nv_bfloat162
__floats2bfloat162_rn(float x, float y) {
  __nv_bfloat162 output;
  output.x = __float2bfloat16_rn(x);
  output.y = __float2bfloat16_rn(y);
  return output;
}
#endif

struct __align__(16) bf16x8 {
  __nv_bfloat162 vals[4];
};

struct __align__(16) fx4 {
  float x;
  float y;
  float z;
  float w;
  __host__ __device__ fx4() {
    x = 0;
    y = 0;
    z = 0;
    w = 0;
  }
};

struct __align__(8) bfx4 {
  __nv_bfloat162 vals[2];
};

struct __align__(16) bfx8 {
  __nv_bfloat162 vals[4];
};

DEVICE_INLINE bfx4 dequantize_packed_int4(uint16_t vs, __half2 shift_scale_0);
DEVICE_INLINE bfx8 dequantize_packed_int4(
    uint32_t v,
    __half2 shift_scale_0,
    __half2 shift_scale_1);

DEVICE_INLINE float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#elif defined(USE_ROCM)
  float2 f_val;
  f_val.x = __bfloat162float(val.x);
  f_val.y = __bfloat162float(val.y);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}

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
  dim3 blocks = cuda_calc_block_count(div_up(X.numel(), 8), kThreadsPerBlock);
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
  dim3 blocks = cuda_calc_block_count(div_up(X.numel(), 8), kThreadsPerBlock);

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

#if CUDART_VERSION >= 12000
class FP8_E4M3_MAX {
 public:
  static constexpr float value = 448.0;
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
    const cudaStream_t stream) {
  constexpr dim3 grid(1024);
  const dim3 block(CTA_SIZE);
  scaleMatrix<true>
      <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeQuantizeMatrixRowwise(
    T_OUT* const output,
    T_S const* const input_scale,
    T_IN const* const input,
    const int64_t numel,
    const int64_t lda,
    const cudaStream_t stream) {
  constexpr dim3 grid(1024);
  const dim3 block(CTA_SIZE);
  scaleMatrixRowwise<true>
      <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

#define FINAL_MASK 0xffffffff

template <typename T>
DEVICE_INLINE T shfl_xor(
    unsigned shfl_sync_mask,
    const T val,
    int laneMask,
    int width = kThreadsPerWarp) {
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < 9000
  return __shfl_xor(val, laneMask, width);
#else
  return __shfl_xor_sync(shfl_sync_mask, val, laneMask, width);
#endif
}

template <typename T>
DEVICE_INLINE T warpReduceMax(T val, uint32_t warp_mask = FINAL_MASK) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, shfl_xor(warp_mask, val, mask, 32));
  return val;
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
  TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
  TORCH_CHECK(
      input.dim() >= 2,
      "Invalid dim. The dim of input should be greater than or equal to 2");
  auto _st = input.scalar_type();
  TORCH_CHECK(_st == torch::kBFloat16, "Invalid datatype. input must be BF16");

  at::Tensor scale = torch::empty(
      {1},
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));

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
    std::optional<at::Tensor> bs) // batch size
{
  CUDA_DEVICE_GUARD(input);
  TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
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
      torch::dtype(torch::kFloat8_e4m3fn)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));

  const auto stream = at::cuda::getCurrentCUDAStream();
  invokeQuantizeMatrix(
      reinterpret_cast<__nv_fp8_e4m3*>(quantized_input.data_ptr()),
      reinterpret_cast<float*>(scale.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      input.numel(),
      input.size(-1),
      stream);

  return quantized_input;
}

// TODO: Extend to support other data types for other
// usecases/models when needed
std::vector<at::Tensor> quantize_fp8_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub) // scale upperbound)
{
  CUDA_DEVICE_GUARD(input);
  TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
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
  std::vector<long int> scale_shape = {1};
  input = input.cuda();
  at::Tensor quantized_input = torch::empty(
      quantized_input_shape,
      torch::dtype(torch::kFloat8_e4m3fn)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));
  at::Tensor scales = torch::empty(
      scale_shape,
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));
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
    invokeQuantizeMatrix(
        quantized_input_ptr,
        reinterpret_cast<float*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        input.numel(),
        input.size(-1),
        stream);
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
    invokeQuantizeMatrix(
        quantized_input_ptr,
        reinterpret_cast<float*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        input.numel(),
        input.size(-1),
        stream);
  }
  return std::vector<at::Tensor>{quantized_input, scales};
}

template <typename T>
__inline__ __device__ T blockAllReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
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
    cudaStream_t stream) {
  dim3 grid(numel / lda);
  bool use_shmem = true;
  auto const shmem_size = lda * sizeof(T_IN);
  if (shmem_size >= (48 << 10)) {
    cudaError_t ret = cudaFuncSetAttribute(
        dynamicQuantizeMatrixRowwise<SCALE, T_OUT, T_S, T_IN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size);
    use_shmem = ret == cudaSuccess;
  }
  if (use_shmem) {
    dim3 block(std::min((lda + 31) / 32 * 32, static_cast<int64_t>(1024)));
    dynamicQuantizeMatrixRowwise<SCALE><<<grid, block, shmem_size, stream>>>(
        output, quant_ptr, input, numel, lda, scale_ub);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    dim3 block(CTA_SIZE);
    computeFP8QuantizeScaleRowwise<SCALE>
        <<<grid, block, 0, stream>>>(quant_ptr, input, numel, lda, scale_ub);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    invokeQuantizeMatrixRowwise(output, quant_ptr, input, numel, lda, stream);
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
    std::optional<c10::ScalarType> output_dtype) // Quantization type
{
  TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
  TORCH_CHECK(
      input.dim() >= 2,
      "Invalid dim. The dim of input should be greater than or equal to 2");
  auto _st = input.scalar_type();
  TORCH_CHECK(_st == torch::kBFloat16, "Invalid datatype. input must be BF16");
  // Default data type is f8_e4m3fn.
  c10::ScalarType quantization_type = torch::kFloat8_e4m3fn;
  if (output_dtype.has_value()) {
    TORCH_CHECK(
        (output_dtype.value() == torch::kFloat8_e4m3fn ||
         output_dtype.value() == torch::kFloat8_e5m2),
        "Invalid output type, must be e4m3fn or e5m2.");
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
  // Templatize implementation based on output type.
  if (quantization_type == torch::kFloat8_e4m3fn) {
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
  TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
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
      torch::dtype(torch::kFloat8_e4m3fn)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));
  at::Tensor scales = torch::empty(
      scale_shape,
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, at::cuda::current_device())
          .requires_grad(false));
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

#else
std::vector<at::Tensor> quantize_fp8_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub) { // scale upperbound
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

std::vector<at::Tensor> quantize_fp8_per_row(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub, // scale upperbound
    std::optional<c10::ScalarType> output_dtype) { // quantization type
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor quantize_fp8_per_tensor_fixed_scale(
    at::Tensor input,
    at::Tensor scale,
    std::optional<at::Tensor> bs) { // batch size
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
#endif
} // namespace fbgemm_gpu
