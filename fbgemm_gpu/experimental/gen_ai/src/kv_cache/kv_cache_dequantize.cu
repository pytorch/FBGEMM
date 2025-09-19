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
#include <cub/cub.cuh>

#include "fbgemm_gpu/utils/cuda_block_count.h"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/vec_quant.cuh"

#include <torch/torch.h>

namespace fbgemm_gpu {

template <int KVQuantNumGroups = 1>
__global__ void dequantize_int4_cache_kernel(
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H // G]
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> kv_seqlen,
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K_dq, // [B][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V_dq // [B][MAX_T][N_KVH][D_H]
) {
  auto N_KVH = cache_K.size(2);
  auto D_H = cache_K_dq.size(3);

  auto b = blockIdx.x;
  // only need to dequantize this far.
  auto max_t = kv_seqlen[b];

  // one warp per T/H
  for (auto t_h = threadIdx.y + blockIdx.y * blockDim.y; t_h < max_t * N_KVH;
       t_h += blockDim.y * gridDim.y) {
    auto h = t_h % N_KVH;
    auto t = t_h / N_KVH;

    auto* row_k = &cache_K[b][t][h][0];
    auto* row_v = &cache_V[b][t][h][0];
    bfx8 kv_dq;
    if (KVQuantNumGroups == 1) {
      __half2 k_shift_scale;
      __half2 v_shift_scale;
      *reinterpret_cast<uint*>(&k_shift_scale) =
          *reinterpret_cast<uint*>(&row_k[0]);
      *reinterpret_cast<uint*>(&v_shift_scale) =
          *reinterpret_cast<uint*>(&row_v[0]);
      if (4 * threadIdx.x >= D_H) {
        continue;
      }
      uint32_t kq = *reinterpret_cast<uint16_t*>(&row_k[threadIdx.x * 2 + 4]);
      uint32_t vq = *reinterpret_cast<uint16_t*>(&row_v[threadIdx.x * 2 + 4]);

      uint32_t packed = kq | (vq << 16);
      kv_dq = dequantize_packed_int4(packed, k_shift_scale, v_shift_scale);

    } else {
      __half2 k_shift_scale;
      __half2 v_shift_scale;
      auto group_size = D_H / KVQuantNumGroups;
      auto group_idx = threadIdx.x * 4 / group_size;

      *reinterpret_cast<uint*>(&k_shift_scale) =
          *reinterpret_cast<uint*>(&row_k[4 * group_idx]);
      *reinterpret_cast<uint*>(&v_shift_scale) =
          *reinterpret_cast<uint*>(&row_v[4 * group_idx]);

      int32_t int4_qparam_offset = 4 * KVQuantNumGroups;

      if (4 * threadIdx.x >= D_H) {
        continue;
      }

      uint32_t kq = *reinterpret_cast<uint16_t*>(
          &row_k[threadIdx.x * 2 + int4_qparam_offset]);
      uint32_t vq = *reinterpret_cast<uint16_t*>(
          &row_v[threadIdx.x * 2 + int4_qparam_offset]);

      uint32_t packed = kq | (vq << 16);

      kv_dq = dequantize_packed_int4(packed, k_shift_scale, v_shift_scale);
    }
    // now, write our outputs
    auto* row_k_dq = &cache_K_dq[b][t][h][0];
    auto* row_v_dq = &cache_V_dq[b][t][h][0];

    *reinterpret_cast<uint2*>(&row_k_dq[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&kv_dq.vals[0]);
    *reinterpret_cast<uint2*>(&row_v_dq[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&kv_dq.vals[2]);
  }
}

#define CALL_DEQUANTIZE_INT4_CACHE_GROUPWISE_KERNEL(NUM_GROUPS, ...) \
  FBGEMM_LAUNCH_KERNEL(                                              \
      (dequantize_int4_cache_kernel<NUM_GROUPS>),                    \
      blocks,                                                        \
      threads,                                                       \
      0,                                                             \
      at::cuda::getCurrentCUDAStream(),                              \
      PTA_B(cache_K, uint8_t, 4, 64),                                \
      PTA_B(cache_V, uint8_t, 4, 64),                                \
      PTA_B(kv_seqlen, int32_t, 1, 32),                              \
      PTA_B(cache_K_dq, at::BFloat16, 4, 64),                        \
      PTA_B(cache_V_dq, at::BFloat16, 4, 64));

std::tuple<at::Tensor, at::Tensor> dequantize_int4_cache(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor kv_seqlen,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v) {
  // allocate DQ outputs
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());
  TORCH_CHECK(kv_seqlen.is_cuda());
  TORCH_CHECK(
      !qparam_k.has_value(),
      "CUDA doesn't support external qparams in dequantize_int4_cache");
  TORCH_CHECK(
      !qparam_v.has_value(),
      "CUDA doesn't support external qparams in dequantize_int4_cache");
  auto B = cache_K.size(0);
  auto MAX_T = cache_K.size(1);
  auto N_KVH = cache_K.size(2);
  auto D_HQ = cache_K.size(3);
  // D_HQ == D_H // 2 + 8 (int4 + 4xhalf qparams)
  auto num_groups_ = num_groups ? num_groups.value() : 1;
  auto int4_qparam_offset = 4 * num_groups_;
  auto D_H = (D_HQ - int4_qparam_offset) * 2;

  auto cache_K_dq =
      at::zeros({B, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));
  auto cache_V_dq =
      at::zeros({B, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));

  if (B == 0) {
    return {cache_K_dq, cache_V_dq};
  }

  constexpr int32_t kMaxBlocks = 256;
  dim3 blocks(B, std::max<int32_t>(1, kMaxBlocks / B));
  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
  CALL_INT4_KERNEL_WITH_KV_GROUPWISE_QUANT_CHECK(
      CALL_DEQUANTIZE_INT4_CACHE_GROUPWISE_KERNEL, num_groups_)
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {cache_K_dq, cache_V_dq};
}

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
template <bool ExternalQParam>
__global__ void dequantize_fp8_cache_kernel(
    // This code currently represents FP8 version not int4
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H // G]
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> kv_seqlen,
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K_dq, // [B][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V_dq, // [B][MAX_T][N_KVH][D_H]
    int32_t* qparam_k_ptr,
    int32_t* qparam_v_ptr) {
  auto N_KVH = cache_K.size(2);
  auto MAX_T = cache_K.size(1);
  auto D_H = cache_K_dq.size(3);
  auto D_H_q = cache_K.size(3);
  // TODO: support D_H < 128 for small model used in testing.
  CUDA_KERNEL_ASSERT(D_H == 128);
  const uint8_t offset_bytes = (ExternalQParam) ? 0 : 4;
  CUDA_KERNEL_ASSERT(D_H_q - D_H == offset_bytes);

  auto b = blockIdx.x;
  // only need to dequantize this far.
  auto max_t = kv_seqlen[b];

  // one warp per T/H
  int h = 0, t = 0;
  uint8_t* row;
  c10::BFloat16* row_dq{};
  bfx4 kv_dq;
  long t_h{};
  __half2* qparam_src;
  // On AMD, we have 64 threads per warp.
  // We use the first 32 threads to process K
  // and the second 32 threads to process V
  for (t_h = threadIdx.y + blockIdx.y * blockDim.y; t_h < max_t * N_KVH;
       t_h += blockDim.y * gridDim.y) {
    h = t_h % N_KVH;
    t = t_h / N_KVH;
    size_t idx = b * (MAX_T * N_KVH) + t * N_KVH + h;
    auto tidx = threadIdx.x;
    if (threadIdx.x < 32) {
      row = &cache_K[b][t][h][0];
      row_dq = &cache_K_dq[b][t][h][0];
      if constexpr (ExternalQParam) {
        qparam_src = reinterpret_cast<__half2*>(&qparam_k_ptr[idx]);
      } else {
        qparam_src = reinterpret_cast<__half2*>(&row[0]);
      }
    } else {
      row = &cache_V[b][t][h][0];
      row_dq = &cache_V_dq[b][t][h][0];
      if constexpr (ExternalQParam) {
        qparam_src = reinterpret_cast<__half2*>(&qparam_v_ptr[idx]);
      } else {
        qparam_src = reinterpret_cast<__half2*>(&row[0]);
      }
      tidx -= 32;
    }
    uint32_t q = *reinterpret_cast<uint32_t*>(&row[tidx * 4 + offset_bytes]);
    kv_dq = dequantize_packed_fp8(q, *qparam_src);
    // now, write our outputs
    // each thread writes 4 elements of type bf16
    *reinterpret_cast<uint2*>(&row_dq[4 * tidx]) =
        *reinterpret_cast<uint2*>(&kv_dq.vals[0]);
  }

  max_t = (max_t + 127) / 128 * 128;
  max_t = max_t > MAX_T ? MAX_T : max_t;
  for (; t_h < max_t * N_KVH; t_h += blockDim.y * gridDim.y) {
    h = t_h % N_KVH;
    t = t_h / N_KVH;
    auto tidx = threadIdx.x;
    if (threadIdx.x < 32) {
      row_dq = &cache_K_dq[b][t][h][0];
    } else {
      row_dq = &cache_V_dq[b][t][h][0];
      tidx -= 32;
    }
    memset(&row_dq[4 * tidx], 0, sizeof(uint2));
  }
}

__global__ void dequantize_fp8_cache_kernel_paged(
    // This code currently represents FP8 version not int4
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [1][MAX_PAGE * PAGE_SIZE][N_KVH][D_H]
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [1][MAX_PAGE * PAGE_SIZE][N_KVH][D_H // G]
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> kv_seqlen,
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K_dq, // [1][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V_dq, // [1][MAX_T][N_KVH][D_H]
    int32_t* qparam_k_ptr,
    int32_t* qparam_v_ptr,
    int32_t* block_tables,
    int32_t block_tables_b_stride,
    int32_t page_size) {
  CUDA_KERNEL_ASSERT(0 && "unimplemented");
}

#else
template <bool ExternalQParam>
__global__ void dequantize_fp8_cache_kernel(
    // This code currently represents FP8 version not int4
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H // G]
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> kv_seqlen,
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K_dq, // [B][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V_dq, // [B][MAX_T][N_KVH][D_H]
    int32_t* qparam_k_ptr,
    int32_t* qparam_v_ptr) {
  auto N_KVH = cache_K.size(2);
  auto MAX_T = cache_K.size(1);
  auto D_H = cache_K_dq.size(3);
  auto D_H_q = cache_K.size(3);
  // TODO: support D_H < 128 for small model used in testing.
  CUDA_KERNEL_ASSERT(D_H == 128);
  const uint8_t offset_bytes = (ExternalQParam) ? 0 : 4;
  CUDA_KERNEL_ASSERT(D_H_q - D_H == offset_bytes);

  auto b = blockIdx.x;
  // only need to dequantize this far.
  auto max_t = kv_seqlen[b];

  // one warp per T/H
  int h = 0, t = 0;
  uint8_t *row_k{}, *row_v{};
  c10::BFloat16 *row_k_dq{}, *row_v_dq{};
  uint64_t packed{};
  bfx8 kv_dq;
  long t_h{};
  for (t_h = threadIdx.y + blockIdx.y * blockDim.y; t_h < max_t * N_KVH;
       t_h += blockDim.y * gridDim.y) {
    h = t_h % N_KVH;
    t = t_h / N_KVH;

    row_k = &cache_K[b][t][h][0];
    row_v = &cache_V[b][t][h][0];
    row_k_dq = &cache_K_dq[b][t][h][0];
    row_v_dq = &cache_V_dq[b][t][h][0];
    // Calculate kv_dq for this row
    {
      __half2* qparam_k_src;
      __half2* qparam_v_src;
      if (ExternalQParam) {
        size_t idx = b * (MAX_T * N_KVH) + t * N_KVH + h;
        qparam_k_src = reinterpret_cast<__half2*>(&qparam_k_ptr[idx]);
        qparam_v_src = reinterpret_cast<__half2*>(&qparam_v_ptr[idx]);
      } else {
        qparam_k_src = reinterpret_cast<__half2*>(&row_k[0]);
        qparam_v_src = reinterpret_cast<__half2*>(&row_v[0]);
      }
      uint64_t kq =
          *reinterpret_cast<uint32_t*>(&row_k[threadIdx.x * 4 + offset_bytes]);
      uint64_t vq =
          *reinterpret_cast<uint32_t*>(&row_v[threadIdx.x * 4 + offset_bytes]);

      packed = kq | (vq << 32);

      kv_dq = dequantize_packed_fp8(packed, *qparam_k_src, *qparam_v_src);
    }
    // now, write our outputs
    // each thread writes 4 elements of type bf16
    *reinterpret_cast<uint2*>(&row_k_dq[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&kv_dq.vals[0]);
    *reinterpret_cast<uint2*>(&row_v_dq[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&kv_dq.vals[2]);
  }

  max_t = (max_t + 127) / 128 * 128;
  max_t = max_t > MAX_T ? MAX_T : max_t;
  for (; t_h < max_t * N_KVH; t_h += blockDim.y * gridDim.y) {
    h = t_h % N_KVH;
    t = t_h / N_KVH;
    row_k_dq = &cache_K_dq[b][t][h][0];
    row_v_dq = &cache_V_dq[b][t][h][0];

    memset(&row_k_dq[4 * threadIdx.x], 0, sizeof(uint2));
    memset(&row_v_dq[4 * threadIdx.x], 0, sizeof(uint2));
  }
}

// Cloned from dequantize_fp8_cache_kernel because
// branching inside the original kernel runs into
// "too many resources requested for launch" which
// necessitates decreasing the number of warps per block,
// which might have performance implications. Also we
// might have more diverging behaviors for paged kernel
// as noted in the comment below so we will keep a separate
// kernel for now.
__global__ void dequantize_fp8_cache_kernel_paged(
    // This code currently represents FP8 version not int4
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [1][MAX_PAGE * PAGE_SIZE][N_KVH][D_H]
    pta::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [1][MAX_PAGE * PAGE_SIZE][N_KVH][D_H // G]
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> kv_seqlen,
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K_dq, // [1][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V_dq, // [1][MAX_T][N_KVH][D_H]
    int32_t* qparam_k_ptr,
    int32_t* qparam_v_ptr,
    int32_t* block_tables,
    int32_t block_tables_b_stride,
    int32_t page_size) {
  auto N_KVH = cache_K.size(2);
  auto D_H = cache_K_dq.size(3);
  auto D_H_q = cache_K.size(3);
  CUDA_KERNEL_ASSERT(D_H == 128);

  auto b = blockIdx.x;
  // only need to dequantize this far.
  auto max_t = kv_seqlen[b];

  // one warp per T/H
  auto t_h = threadIdx.y + blockIdx.y * blockDim.y;
  for (; t_h < max_t * N_KVH; t_h += blockDim.y * gridDim.y) {
    auto h = t_h % N_KVH;
    auto t = t_h / N_KVH;

    int page_logical_idx = t / page_size;
    int page_offset = t % page_size;
    int page_physical_idx =
        block_tables[b * block_tables_b_stride + page_logical_idx];
    int physical_t = page_physical_idx * page_size + page_offset;

    uint8_t* row_k = &cache_K[0][physical_t][h][0];
    uint8_t* row_v = &cache_V[0][physical_t][h][0];

    bfx8 kv_dq;
    uint8_t qparam_offset_bytes;
    __half2* qparam_k_src;
    __half2* qparam_v_src;
    if (qparam_k_ptr) {
      // read from standalone qparam tensor
      qparam_offset_bytes = 0;
      auto idx = physical_t * N_KVH + h;
      qparam_k_src = reinterpret_cast<__half2*>(&qparam_k_ptr[idx]);
      qparam_v_src = reinterpret_cast<__half2*>(&qparam_v_ptr[idx]);
    } else {
      // read from first row
      qparam_offset_bytes = 4;
      qparam_k_src = reinterpret_cast<__half2*>(&row_k[0]);
      qparam_v_src = reinterpret_cast<__half2*>(&row_v[0]);
    }
    // Assert the quantized row dim is as expected
    CUDA_KERNEL_ASSERT(D_H_q - D_H == qparam_offset_bytes);
    if (4 * threadIdx.x >= D_H) {
      continue;
    }
    // each thread reads 4 x 8 bits

    uint64_t kq = *reinterpret_cast<uint32_t*>(
        &row_k[threadIdx.x * 4 + qparam_offset_bytes]);
    uint64_t vq = *reinterpret_cast<uint32_t*>(
        &row_v[threadIdx.x * 4 + qparam_offset_bytes]);

    uint64_t packed = kq | (vq << 32);

    kv_dq = dequantize_packed_fp8(packed, *qparam_k_src, *qparam_v_src);

    // now, write our outputs
    auto* row_k_dq = &cache_K_dq[0][physical_t][h][0];
    auto* row_v_dq = &cache_V_dq[0][physical_t][h][0];
    // each thread writes 4 elements of type bf16
    *reinterpret_cast<uint2*>(&row_k_dq[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&kv_dq.vals[0]);
    *reinterpret_cast<uint2*>(&row_v_dq[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&kv_dq.vals[2]);
  }

  // zero out the rest of the page, because FA3 can be affected by
  // NaN values beyond the sequence length.
  max_t = (max_t + page_size - 1) / page_size * page_size;
  for (; t_h < max_t * N_KVH; t_h += blockDim.y * gridDim.y) {
    if (4 * threadIdx.x >= D_H) {
      continue;
    }
    auto h = t_h % N_KVH;
    auto t = t_h / N_KVH;

    int page_logical_idx = t / page_size;
    int page_offset = t % page_size;
    int page_physical_idx =
        block_tables[b * block_tables_b_stride + page_logical_idx];
    int physical_t = page_physical_idx * page_size + page_offset;

    auto* row_k_dq = &cache_K_dq[0][physical_t][h][0];
    auto* row_v_dq = &cache_V_dq[0][physical_t][h][0];

    memset(&row_k_dq[4 * threadIdx.x], 0, sizeof(uint2));
    memset(&row_v_dq[4 * threadIdx.x], 0, sizeof(uint2));
  }
}
#endif

std::tuple<at::Tensor, at::Tensor> dequantize_fp8_cache(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor kv_seqlen,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    std::optional<at::Tensor> block_tables,
    int64_t page_size) {
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());
  TORCH_CHECK(kv_seqlen.is_cuda());
  auto B = kv_seqlen.size(0);
  // vanilla: B_KV = B, paged: B_KV = 1
  auto B_KV = cache_K.size(0);
  // vanilla: MAX_T = MAX_T, paged: MAX_T = MAX_PAGE * PAGE_SIZE
  auto MAX_T = cache_K.size(1);
  auto N_KVH = cache_K.size(2);
  auto D_HQ = cache_K.size(3);
  auto fp8_qparam_offset = 4;
  int32_t* qparam_k_ptr = nullptr;
  int32_t* qparam_v_ptr = nullptr;
  if (qparam_k.has_value()) {
    qparam_k_ptr = static_cast<int32_t*>(qparam_k.value().data_ptr());
    qparam_v_ptr = static_cast<int32_t*>(qparam_v.value().data_ptr());
    fp8_qparam_offset = 0;
  }
  auto D_H = (D_HQ - fp8_qparam_offset);

  // TODO:
  // The below allocates Tensors that have the same shape as cache_K and
  // cache_V to store their dequantize results. For paged KV cache, this can
  // be a bit inefficient because it has the shape of [1 x (MAX_PAGES *
  // PAGE_SIZE) x N_KVH x D_H] to accommodate pages globally across batch
  // instances, and if we have very large MAX_PAGES then we are essentially
  // allocating a very huge Tensor here. The benefit is that the following
  // users of this dequantized results can reuse the existing block_tables to
  // access their elements. If we want to be more efficient, there are two
  // possible approaches: (1) Allocate a shorter Tensor here and store the
  // dequantize results in a more compact manner, but that requires creating a
  // new block_tables here and making sure the following users all use the
  // correct block_tables. (2) From outside, keep a persistent buffer that has
  // a matching shape with the original paged KV and feed the same buffer into
  // this function at every layer to reuse it and prevent allocation.

  auto cache_K_dq = at::empty(
      {B_KV, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));
  auto cache_V_dq = at::empty(
      {B_KV, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));

  if (B == 0) {
    return {cache_K_dq, cache_V_dq};
  }

  int32_t* block_tables_ptr = nullptr;
  int32_t block_tables_b_stride = 0;
  if (block_tables.has_value()) {
    block_tables_ptr = static_cast<int32_t*>(block_tables.value().data_ptr());
    block_tables_b_stride = block_tables.value().stride(0);
  }

  constexpr int32_t kMaxBlocks = 512;
  dim3 blocks(B, std::max<int32_t>(1, kMaxBlocks / B));
  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
#define CALL_DEQUANTIZE_FP8_CACHE(EXTERNAL_Q_PARAM)    \
  FBGEMM_LAUNCH_KERNEL(                                \
      (dequantize_fp8_cache_kernel<EXTERNAL_Q_PARAM>), \
      blocks,                                          \
      threads,                                         \
      0,                                               \
      at::cuda::getCurrentCUDAStream(),                \
      PTA_B(cache_K, uint8_t, 4, 64),                  \
      PTA_B(cache_V, uint8_t, 4, 64),                  \
      PTA_B(kv_seqlen, int32_t, 1, 32),                \
      PTA_B(cache_K_dq, at::BFloat16, 4, 64),          \
      PTA_B(cache_V_dq, at::BFloat16, 4, 64),          \
      qparam_k_ptr,                                    \
      qparam_v_ptr);
  if (block_tables_ptr == nullptr) {
    if (qparam_k_ptr) {
      CALL_DEQUANTIZE_FP8_CACHE(true);
    } else {
      CALL_DEQUANTIZE_FP8_CACHE(false);
    }
#undef CALL_DEQUANTIZE_FP8_CACHE
  } else {
    FBGEMM_LAUNCH_KERNEL(
        (dequantize_fp8_cache_kernel_paged),
        blocks,
        threads,
        0,
        at::cuda::getCurrentCUDAStream(),
        PTA_B(cache_K, uint8_t, 4, 64),
        PTA_B(cache_V, uint8_t, 4, 64),
        PTA_B(kv_seqlen, int32_t, 1, 32),
        PTA_B(cache_K_dq, at::BFloat16, 4, 64),
        PTA_B(cache_V_dq, at::BFloat16, 4, 64),
        qparam_k_ptr,
        qparam_v_ptr,
        block_tables_ptr,
        block_tables_b_stride,
        page_size);
  }

  return {cache_K_dq, cache_V_dq};
}

// Function to convert and pack a single component
DEVICE_INLINE uint32_t
convertAndPack(float component, float inv_scale, float shift = 0.0) {
  // auto val = (component - shift) * inv_scale;
  auto val = fmaf(component, inv_scale, -shift * inv_scale);
  val = fmaxf(val, -FP8_E4M3_MAX::value);
  val = fminf(val, FP8_E4M3_MAX::value);
  auto x = __nv_fp8_e4m3(val);
  return *reinterpret_cast<uint32_t*>(&x);
}
// Function to pack four components into a single uint32_t
DEVICE_INLINE uint32_t packComponents(uint32_t x_bits[4]) {
  uint32_t packed = 0;
  packed |= (x_bits[0] << 0);
  packed |= (x_bits[1] << 8);
  packed |= (x_bits[2] << 16);
  packed |= (x_bits[3] << 24);
  return packed;
}

__global__ void quantizeQKVPerHead(
    const float* xqkv_amax_head, // [B, HH]
    at::BFloat16* xqkv, // [B_T, HH, D_H]
    const int32_t* varseq_seqpos, // [B_T]
    const int32_t* varseq_batch, // [B_T]
    const bool* is_precalculated_qparam, // [B_T]
    pta::PackedTensorAccessor64<at::Float8_e4m3fn, 3, at::RestrictPtrTraits>
        XQ_O, // [B_T][N_H][D]
    pta::PackedTensorAccessor64<at::Float8_e4m3fn, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H]
    pta::PackedTensorAccessor64<at::Float8_e4m3fn, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H]
    float* const scale_q,
    float* const scale_k,
    float* const scale_v,
    float kv_multiplier = 64.f) {
  // Launch one warp per token. Each thread handles 4 elements.
  // warps = B_T
  auto N_KVH = cache_K.size(2);
  auto N_H = XQ_O.size(1);
  auto B_T = XQ_O.size(0);
  // TODO: Support N_KVH > 1
  // CUDA_KERNEL_ASSERT(N_KVH == 1);

  auto HH = N_H + N_KVH * 2;
  auto maxHH = scale_k ? HH : N_H;

  uint2 buffer;

  // warps_per_block = blockDim.y
  // warp_id = threadIdx.y
  // block_id = blockIdx.x

  // Calculate scaling factor
  constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX::value * 512.f);
  int b = 0;
  int last_b = -1;
  int h = 0;
  float* qparam = nullptr;
  at::Float8_e4m3fn* dst_row_q = nullptr;
  float val = 0;
  float inv_scale = 0;

  uint d = 4 * threadIdx.x;

  auto b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  for (int b_t = b_t_start; b_t < B_T; b_t += blockDim.y * gridDim.x) {
    b = varseq_batch ? varseq_batch[b_t] : b_t;
    if (b_t > 0) {
      last_b = varseq_batch ? varseq_batch[b_t - 1] : b_t - 1;
    } else {
      last_b = -1;
    }
    {
      // Skip quantization of KV if scale is pre-calculated for K/V
      // as in decode and partial prefill cases
      bool is_precalculated_qparam_b_t =
          is_precalculated_qparam ? is_precalculated_qparam[b_t] : true;
      if (is_precalculated_qparam_b_t)
        maxHH = N_H;
    }
    val = 0;
    for (auto hh = 0; hh < N_H; hh++) {
      val = fmaxf(val, xqkv_amax_head[b * HH + hh]);
    }

    for (auto hh = 0; hh < maxHH; hh++) {
      float val_h = 0;
      {
        at::BFloat16* src_row = &xqkv[(b_t * HH + hh + 0) * D_H];
        buffer = *reinterpret_cast<uint2*>(&src_row[d]);
        val_h = (hh < N_H) ? val : xqkv_amax_head[b * HH + hh];
      }

      {
        int seqpos_t = varseq_seqpos[b_t];
        if (hh < N_H) {
          h = hh;
          qparam = scale_q + b * N_KVH + hh / (N_H / N_KVH);
          dst_row_q = &XQ_O[b_t][h][0];
          // val_h = val_h * 8;
        } else if (hh < N_H + N_KVH) {
          h = hh - N_H;

          qparam = scale_k + b * N_KVH + h;
          dst_row_q = &cache_K[b][seqpos_t][h][0];
          val_h = kv_multiplier * val_h;
        } else {
          h = hh - N_H - N_KVH;

          qparam = scale_v + b * N_KVH + h;
          dst_row_q = &cache_V[b][seqpos_t][h][0];
          val_h = kv_multiplier * val_h;
        }
      }
      {
        float scale = 0;
        val_h = fminf(val_h, 12000);
        scale = fmaxf(val_h / FP8_E4M3_MAX::value, min_scaling_factor);
        bool is_first_token = b != last_b;
        if (threadIdx.x == 0 && h == 0 && is_first_token) {
          *qparam = scale;
        }
        inv_scale = 1 / scale;
      }

      {
        bfx4 src;
        fx4 dst;
        uint32_t x_bits[4];
        // Convert and pack data
        // 8 bytes are 4 elements of type bf16
        *reinterpret_cast<uint2*>(&src) = buffer;
        dst = bfx4_to_fx4(src);
        x_bits[0] = convertAndPack(dst.x, inv_scale);
        x_bits[1] = convertAndPack(dst.y, inv_scale);
        x_bits[2] = convertAndPack(dst.z, inv_scale);
        x_bits[3] = convertAndPack(dst.w, inv_scale);
        uint32_t packed = packComponents(x_bits);
        // CUDA_KERNEL_ASSERT(uintptr_t(&dst_row_q[d]) % 4 == 0);
        *reinterpret_cast<uint32_t*>(&dst_row_q[d]) = packed;
      }
    }
  }
}

at::Tensor quantize_qkv_per_head(
    at::Tensor xqkv_amax_row, // [B, HH]
    at::Tensor xqkv, // [B_T, HH, D_H]
    at::Tensor varseq_seqpos, // [B_T]
    std::optional<at::Tensor> varseq_batch, // [B_T]
    std::optional<at::Tensor> is_precalculated_qparam, // [B_T]
    at::Tensor cache_K, // [B][MAX_T][N_KVH][D_H]
    at::Tensor cache_V, // [B][MAX_T][N_KVH][D_H]
    at::Tensor XQ_O, // [B_T][N_H][D]
    int64_t B, // Batch size
    std::optional<at::Tensor> qparam_k = std::nullopt,
    std::optional<at::Tensor> qparam_v = std::nullopt) {
  auto N_KVH_L = cache_K.size(2);

  float* qparam_k_ptr = nullptr;
  float* qparam_v_ptr = nullptr;
  if (qparam_k.has_value()) {
    // prefill case
    qparam_k_ptr = qparam_k.value().data_ptr<float>();
    qparam_v_ptr = qparam_v.value().data_ptr<float>();
  }

  constexpr int32_t kMaxBlocks = 512;
  dim3 block_size(kThreadsPerWarp, kWarpsPerBlock);
  dim3 grid_size(kMaxBlocks);
  auto scale_q = at::zeros({B, N_KVH_L}, XQ_O.options().dtype(at::kFloat));
  float* const scale_q_ptr = scale_q.data_ptr<float>();
  // Launch the kernel

  FBGEMM_LAUNCH_KERNEL(
      (quantizeQKVPerHead),
      grid_size,
      block_size,
      0,
      at::cuda::getCurrentCUDAStream(),
      xqkv_amax_row.data_ptr<float>(),
      xqkv.data_ptr<at::BFloat16>(),
      varseq_seqpos.data_ptr<int32_t>(),
      varseq_batch.has_value() ? varseq_batch.value().data_ptr<int32_t>()
                               : nullptr, // not needed for decode
      is_precalculated_qparam.has_value()
          ? is_precalculated_qparam.value().data_ptr<bool>()
          : nullptr,
      PTA_B(XQ_O, at::Float8_e4m3fn, 3, 64),
      PTA_B(cache_K, at::Float8_e4m3fn, 4, 64),
      PTA_B(cache_V, at::Float8_e4m3fn, 4, 64),
      scale_q_ptr,
      qparam_k_ptr,
      qparam_v_ptr,
      64.f);
  return scale_q;
}
#else

std::tuple<at::Tensor, at::Tensor> dequantize_fp8_cache(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor kv_seqlen,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    std::optional<at::Tensor> block_tables,
    int64_t page_size) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor quantize_qkv_per_head(
    at::Tensor xqkv_amax_row, // [B_T, HH]
    at::Tensor xqkv, // [B_T, HH, D_H]
    at::Tensor varseq_seqpos, // [B_T]
    std::optional<at::Tensor> varseq_batch, // [B_T]
    at::Tensor q_seqstarts, // [B+1]
    at::Tensor cache_K, // [B][MAX_T][N_KVH][D_H]
    at::Tensor cache_V, // [B][MAX_T][N_KVH][D_H]
    at::Tensor XQ_O, // [B_T][N_H][D]
    int64_t max_seq_length, // Length of the sequence
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
#endif

} // namespace fbgemm_gpu
