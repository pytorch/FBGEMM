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
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include "c10/core/ScalarType.h"
#include "c10/util/BFloat16.h"
#include "kv_cache.cuh"
#include "kv_cache.h"

#ifndef USE_ROCM
#include <mma.h>
#endif

#include "fbgemm_gpu/utils/cuda_block_count.h"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/vec_quant.cuh"

#include <torch/torch.h>

namespace fbgemm_gpu {

#if (defined(USE_ROCM))
/**
 * Converts the contents of a FP8 KV cache from e4m3fn (NV) to e4m3fnuz (AMD).
 * These formats differ in their support for negative zero, and in their
 * exponent bias. Negative zeros are replaced with positive zero, and the scale
 * qparam is multiplied by 2.0, because we know that the scale will be applied
 * to the k/v value and is equivalent to recomputing the exponent bias.
 *
 * This in an inplace operation.
 *
 * It is assumed that inputs will have been generated with scale_ub = max(fp16)
 * / 2 to avoid overflow. Some debug mode assertions are in place, but there are
 * no runtime guarantees.
 *
 * As written, this kernel is only valid on AMD, because it relies on threads
 * 32-63 to convert the V tensors. NV only has threads 0-31 per warp.
 */
__global__ void convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace_kernel(
    pta::PackedTensorAccessor64<uint8_t, 5, at::RestrictPtrTraits>
        cache_K, // [N_H_L][B][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<uint8_t, 5, at::RestrictPtrTraits>
        cache_V, // [N_H_L][B][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<int32_t, 5, at::RestrictPtrTraits> qparam_K,
    pta::PackedTensorAccessor64<int32_t, 5, at::RestrictPtrTraits> qparam_V) {
  auto N_KVH = cache_K.size(3);
  auto MAX_T = cache_K.size(2);
  auto D_H = cache_K.size(4);
  CUDA_KERNEL_ASSERT(D_H == 128);

  auto l = blockIdx.x;
  auto b = blockIdx.y;
  int h = 0, t = 0;
  uint8_t* head;
  __half2* shift_scale;

  for (auto t_h = threadIdx.y + blockIdx.z * blockDim.y; t_h < MAX_T * N_KVH;
       t_h += blockDim.y * gridDim.z) {
    h = t_h % N_KVH;
    t = t_h / N_KVH;

    auto tidx = threadIdx.x;
    if (threadIdx.x < 32) {
      head = &cache_K[l][b][t][h][0];
      shift_scale = reinterpret_cast<__half2*>(&qparam_K[l][b][t][h][0]);
    } else {
      head = &cache_V[l][b][t][h][0];
      shift_scale = reinterpret_cast<__half2*>(&qparam_V[l][b][t][h][0]);
      tidx -= 32;
    }
    auto D_H_idx = tidx * 4; // Reading 4 bytes at once.
    auto negative_zero = 0x80;

    // Our only goal here is to detect negative zeros that are valid
    // in e4m3fn, but not valid in e4m3fnuz, and overwrite them with positive
    // zeros.
    uint32_t packed_fp8x4_vals = *reinterpret_cast<uint32_t*>(&head[D_H_idx]);
    if (((packed_fp8x4_vals >> 24) & 0xff) == negative_zero) {
      packed_fp8x4_vals &= 0x00ffffff;
    }
    if (((packed_fp8x4_vals >> 16) & 0xff) == negative_zero) {
      packed_fp8x4_vals &= 0xff00ffff;
    }
    if (((packed_fp8x4_vals >> 8) & 0xff) == negative_zero) {
      packed_fp8x4_vals &= 0xffff00ff;
    }
    if ((packed_fp8x4_vals & 0xff) == negative_zero) {
      packed_fp8x4_vals &= 0xffffff00;
    }
    *reinterpret_cast<uint32_t*>(&head[D_H_idx]) = packed_fp8x4_vals;

    // Multiply qparam scale (member x) by 2 to compensate for the exponent
    // bias difference (1) between e4m3fn and e4m3fnuz. We only need to do
    // this once per row. In debug mode, assert that 2.0*scale as a float would
    // not exceed the max value of __half.
    if (tidx == 0) {
      __half shift = __high2half(*shift_scale);
      __half scale = __low2half(*shift_scale);

      CUDA_KERNEL_ASSERT(__half2float(scale) * 2.0f <= 65504.0f);

      __half new_scale = __hmul(scale, __float2half(2.0f));
      *shift_scale = __half2(new_scale, shift);
    }
  }
}

void convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor qparam_K,
    at::Tensor qparam_V) {
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());
  TORCH_CHECK(qparam_K.is_cuda());
  TORCH_CHECK(qparam_V.is_cuda());

  auto N_H_L = cache_K.size(0);
  auto B = cache_K.size(1);

  constexpr int32_t kMaxBlocks = 512;
  // Blocks: (N_H_L, B, residual from max blocks)
  dim3 blocks(N_H_L, B, std::max<int32_t>(1, kMaxBlocks / (B * N_H_L)));
  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);

  FBGEMM_LAUNCH_KERNEL(
      (convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace_kernel),
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(cache_K, uint8_t, 5, 64),
      PTA_B(cache_V, uint8_t, 5, 64),
      PTA_B(qparam_K, int32_t, 5, 64),
      PTA_B(qparam_V, int32_t, 5, 64));
}
#else
void convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor qparam_K,
    at::Tensor qparam_V) {
  throw std::runtime_error(
      "convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace is only supported on AMD");
}
#endif

} // namespace fbgemm_gpu
