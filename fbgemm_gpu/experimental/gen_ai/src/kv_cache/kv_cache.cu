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
#include <algorithm>
#include "c10/core/ScalarType.h"
#include "c10/util/BFloat16.h"

#ifndef USE_ROCM
#include <mma.h>
#endif
#include <cub/cub.cuh>

#include "fbgemm_gpu/utils/cuda_block_count.h"
#include "fbgemm_gpu/utils/vec_quant.cuh"

#include <torch/torch.h>

template <typename func_t>
void set_gpu_max_dynamic_shared_memory(
    func_t kernel,
    const int smem_bytes,
    const int device) {
  // V100: 96 KB; A100: 160 KB; H100: 228 KB.
  int max_shared_bytes = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared_bytes,
#ifndef __HIP_PLATFORM_AMD__
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
#else
      hipDeviceAttributeMaxSharedMemoryPerBlock,
#endif
      device));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  TORCH_CHECK(
      smem_bytes <= max_shared_bytes,
      "Try to allocate ",
      smem_bytes / 1024,
      " KB of shared memory but only ",
      max_shared_bytes / 1024,
      " KB is available");

  C10_CUDA_CHECK(cudaFuncSetAttribute(
      (void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace fbgemm_gpu {

template <int KVQuantNumGroups = 1>
__global__ void dequantize_int4_cache_kernel(
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H // G]
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> kv_seqlen,
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K_dq, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V_dq // [B][MAX_T][N_KVH][D_H]
) {
  auto N_KVH = cache_K.size(2);
  auto MAX_T = cache_K.size(1);
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

#define CALL_DEQUANTIZE_INT4_CACHE_GROUPWISE_KERNEL(NUM_GROUPS, ...)          \
  dequantize_int4_cache_kernel<                                               \
      NUM_GROUPS><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(  \
      cache_K.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),         \
      cache_V.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),         \
      kv_seqlen.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),       \
      cache_K_dq.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(), \
      cache_V_dq.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>());

std::tuple<at::Tensor, at::Tensor> dequantize_int4_cache(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor kv_seqlen,
    std::optional<int64_t> num_groups) {
  // allocate DQ outputs
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());
  TORCH_CHECK(kv_seqlen.is_cuda());
  auto B = cache_K.size(0);
  auto MAX_T = cache_K.size(1);
  auto N_KVH = cache_K.size(2);
  auto D_HQ = cache_K.size(3);
  // D_HQ == D_H // 2 + 8 (int4 + 4xhalf qparams)
  auto num_groups_ = num_groups ? num_groups.value() : 1;
  auto int4_qparam_offset = 4 * num_groups_;
  auto D_H = (D_HQ - int4_qparam_offset) * 2;

  auto cache_K_dq =
      at::empty({B, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));
  auto cache_V_dq =
      at::empty({B, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));

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

template <typename T>
__device__ void get_dst_row(
    T** dst_row,
    at::PackedTensorAccessor64<T, 4, at::RestrictPtrTraits>&
        cache_KV, // [B][MAX_T][N_KVH][D_H +4 or D_H]
    int32_t b,
    int32_t h,
    int32_t cache_loc_t,
    int32_t page_size,
    int32_t* block_tables,
    int32_t block_tables_b_stride) {
  if (block_tables == nullptr) {
    *dst_row = &cache_KV[b][cache_loc_t][h][0];
  } else {
    int page_logical_idx = cache_loc_t / page_size;
    int page_offset = cache_loc_t % page_size;
    int page_physical_idx =
        block_tables[b * block_tables_b_stride + page_logical_idx];
    *dst_row = &cache_KV[0][page_physical_idx * page_size + page_offset][h][0];
  }
}

enum class PositionEmbeddingMode { ROPE = 0, XPOS = 1 };
enum class QKV { Q, K, V };
DEVICE_INLINE void quantize_fp8_kv(
    fx4 dst,
    uint8_t* dst_row_q,
    __half2* qparam = nullptr,
    bool do_norm = false);
__global__ void nope_qkv_varseq_prefill_kernel_quantized(
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XQ, // [B_T][N_H][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XK, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XV, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H] or
                 // [1][MAX_PAGES * PAGE_SIZE][N_KVH][D_H] for paged attention
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H] or
                 // [1][MAX_PAGES * PAGE_SIZE][N_KVH][D_H] for paged attention
    int32_t* qparam_k_ptr,
    int32_t* qparam_v_ptr,
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XQ_O, // [B_T][N_H][D]
    int32_t* varseq_batch, // in decoding case we have T == 1 and so just pass
                           // nullptr
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> varseq_seqpos,
    int32_t* block_tables, // [B][MAX_PAGES], maps logical pages to physical
                           // ones for paged attention
    int32_t page_size,
    int32_t block_tables_b_stride,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        varseq_cache_seqpos,
    int64_t* actual_batch_size =
        nullptr, // When running in CUDA graph mode, the actual batch size
                 // can be smaller than block_tables.size(0). In this case
                 // rows of block_tables beyond actual_batch_size are not
                 // initialized, and using them wil cause undefined
                 // behavior. To prevent this, when actual_batch_size is
                 // provided, the kernel exits if the current batch index is
                 // larger of equal to actual_batch_size
    bool k_norm = false) {
  auto b_t_hh = blockIdx.x * blockDim.y + threadIdx.y;
  auto B_T = XQ.size(0);
  auto N_KVH = XK.size(1);
  auto N_H = XQ.size(1);
  auto D_H = XQ.size(2);
  auto HH = 2 * N_KVH + N_H;

  auto hh = b_t_hh % HH;
  auto b_t = b_t_hh / HH;
  if (b_t >= B_T) {
    return;
  }
  auto seqpos_t = varseq_seqpos[b_t];
  if (seqpos_t == -1) {
    return;
  }
  auto cache_loc_t = varseq_cache_seqpos[b_t];
  auto b = varseq_batch ? varseq_batch[b_t] : b_t;

  if (actual_batch_size != nullptr && b_t >= *actual_batch_size) {
    return;
  }

  at::BFloat16* src_row = nullptr;
  at::BFloat16* dst_row = nullptr;
  uint8_t* dst_row_q = nullptr;
  auto h = 0;
  QKV qkv;
  if (hh < N_H) {
    h = hh;
    src_row = &XQ[b_t][h][0];
    dst_row = &XQ_O[b_t][h][0];
    qkv = QKV::Q;
  } else if (hh < N_H + N_KVH) {
    h = hh - N_H;
    src_row = &XK[b_t][h][0];

    get_dst_row(
        &dst_row_q,
        cache_K,
        b,
        h,
        cache_loc_t,
        page_size,
        block_tables,
        block_tables_b_stride);
    qkv = QKV::K;
  } else {
    h = hh - N_H - N_KVH;
    src_row = &XV[b_t][h][0];
    get_dst_row(
        &dst_row_q,
        cache_V,
        b,
        h,
        cache_loc_t,
        page_size,
        block_tables,
        block_tables_b_stride);
    qkv = QKV::V;
  }
  CUDA_KERNEL_ASSERT(D_H <= 4 * kThreadsPerWarp);
  if (4 * threadIdx.x >= D_H) {
    return;
  }

  bfx4 src;
  *reinterpret_cast<uint2*>(&src) =
      *reinterpret_cast<uint2*>(&src_row[4 * threadIdx.x]);
  if (qkv == QKV::Q) {
    *reinterpret_cast<uint2*>(&dst_row[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&src);
  } else {
    fx4 dst = bfx4_to_fx4(src);
    // Assert Fp8 kv
    auto D_H = XQ.size(2);
    auto D_H_q = cache_K.size(3);
    __half2* qparam_row = nullptr;
    if (qparam_k_ptr == nullptr) {
      CUDA_KERNEL_ASSERT(D_H_q - D_H == 4);
      // quantize_fp8_kv(dst, dst_row_q, nullptr, (qkv == QKV::K &&
      // k_norm));
    } else {
      auto T = cache_K.size(1);
      size_t idx{};
      if (block_tables == nullptr) {
        idx = b * (T * N_KVH) + (size_t)cache_loc_t * N_KVH + h;
      } else {
        // This is duplicate computation with get_dst_row above.
        // TODO: Maybe clean up and merge later.
        int page_logical_idx = cache_loc_t / page_size;
        int page_offset = cache_loc_t % page_size;
        int page_physical_idx =
            block_tables[b * block_tables_b_stride + page_logical_idx];
        int physical_t = page_physical_idx * page_size + page_offset;
        idx = physical_t * N_KVH + h;
      }
      if (qkv == QKV::K) {
        qparam_row = reinterpret_cast<__half2*>(&qparam_k_ptr[idx]);
      } else {
        qparam_row = reinterpret_cast<__half2*>(&qparam_v_ptr[idx]);
      }
      // quantize_fp8_kv(
      //     dst, dst_row_q, qparam_row, (qkv == QKV::K && k_norm));
    }
    quantize_fp8_kv(dst, dst_row_q, qparam_row, (qkv == QKV::K && k_norm));
  }
}
__global__ void nope_qkv_varseq_prefill_kernel(
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XQ, // [B_T][N_H][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XK, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XV, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H] or
                 // [1][MAX_PAGES * PAGE_SIZE][N_KVH][D_H] for paged attention
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H] or
                 // [1][MAX_PAGES * PAGE_SIZE][N_KVH][D_H] for paged attention
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XQ_O, // [B_T][N_H][D]
    int32_t* varseq_batch, // in decoding case we have T == 1 and so just pass
                           // nullptr
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> varseq_seqpos,
    int32_t* block_tables, // [B][MAX_PAGES], maps logical pages to physical
                           // ones for paged attention
    int32_t page_size,
    int32_t block_tables_b_stride,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        varseq_cache_seqpos,
    int64_t* actual_batch_size =
        nullptr // When running in CUDA graph mode, the actual batch size
                // can be smaller than block_tables.size(0). In this case
                // rows of block_tables beyond actual_batch_size are not
                // initialized, and using them wil cause undefined
                // behavior. To prevent this, when actual_batch_size is
                // provided, the kernel exits if the current batch index is
                // larger of equal to actual_batch_size,
) {
  // Launch b_t_(sum(h)) warps.
  auto b_t_hh = blockIdx.x * blockDim.y + threadIdx.y;
  auto B_T = XQ.size(0);
  auto N_KVH = XK.size(1);
  auto N_H = XQ.size(1);
  auto D_H = XQ.size(2);
  auto HH = 2 * N_KVH + N_H;

  auto hh = b_t_hh % HH;
  auto b_t = b_t_hh / HH;
  if (b_t >= B_T) {
    return;
  }
  auto seqpos_t = varseq_seqpos[b_t];
  if (seqpos_t == -1) {
    return;
  }
  auto cache_loc_t = varseq_cache_seqpos[b_t];
  auto b = varseq_batch ? varseq_batch[b_t] : b_t;

  if (actual_batch_size != nullptr && b_t >= *actual_batch_size) {
    return;
  }

  at::BFloat16* src_row;
  at::BFloat16* dst_row;
  auto h = 0;
  if (hh < N_H) {
    h = hh;
    src_row = &XQ[b_t][h][0];
    dst_row = &XQ_O[b_t][h][0];
  } else if (hh < N_H + N_KVH) {
    h = hh - N_H;
    src_row = &XK[b_t][h][0];

    get_dst_row(
        &dst_row,
        cache_K,
        b,
        h,
        cache_loc_t,
        page_size,
        block_tables,
        block_tables_b_stride);
  } else {
    h = hh - N_H - N_KVH;
    src_row = &XV[b_t][h][0];
    get_dst_row(
        &dst_row,
        cache_V,
        b,
        h,
        cache_loc_t,
        page_size,
        block_tables,
        block_tables_b_stride);
  }

  for (auto head_id = 4 * threadIdx.x; head_id < D_H;
       head_id += kThreadsPerWarp * 4) {
    // assert D_H % 4 == 0;
    // load 4 elements per thread in a warp.
    if (head_id >= D_H) {
      return;
    }
    bfx4 src;
    *reinterpret_cast<uint2*>(&src) =
        *reinterpret_cast<uint2*>(&src_row[head_id]);
    *reinterpret_cast<uint2*>(&dst_row[head_id]) =
        *reinterpret_cast<uint2*>(&src);
  }
}

template <PositionEmbeddingMode Mode>
__global__ void rope_xpos_qkv_varseq_prefill_kernel(
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XQ, // [B_T][N_H][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XK, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XV, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H] or
                 // [1][MAX_PAGES * PAGE_SIZE][N_KVH][D_H] for paged attention
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H] or
                 // [1][MAX_PAGES * PAGE_SIZE][N_KVH][D_H] for paged attention
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XQ_O, // [B_T][N_H][D]
    int32_t* varseq_batch, // in decoding case we have T == 1 and so just pass
                           // nullptr
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> varseq_seqpos,
    double theta,
    double gamma,
    double scale_base,
    double exponent_offset,
    int32_t* block_tables, // [B][MAX_PAGES], maps logical pages to physical
                           // ones for paged attention
    int32_t page_size,
    int32_t block_tables_b_stride,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        varseq_cache_seqpos,
    int64_t* actual_batch_size =
        nullptr, // When running in CUDA graph mode, the actual batch size
                 // can be smaller than block_tables.size(0). In this case
                 // rows of block_tables beyond actual_batch_size are not
                 // initialized, and using them wil cause undefined
                 // behavior. To prevent this, when actual_batch_size is
                 // provided, the kernel exits if the current batch index is
                 // larger of equal to actual_batch_size,
    bool rope_scaling = false,
    int64_t old_context_len = 8192,
    double scaling_factor = 16,
    double lo_freq_factor = 1,
    double hi_freq_factor = 32,
    bool write_k_back = false) {
  // Launch b_t_(sum(h)) warps.
  auto b_t_hh = blockIdx.x * blockDim.y + threadIdx.y;
  auto B_T = XQ.size(0);
  auto N_KVH = XK.size(1);
  auto N_H = XQ.size(1);
  auto D_H = XQ.size(2);
  auto HH = 2 * N_KVH + N_H;

  auto hh = b_t_hh % HH;
  auto b_t = b_t_hh / HH;
  if (b_t >= B_T) {
    return;
  }
  auto seqpos_t = varseq_seqpos[b_t];
  if (seqpos_t == -1) {
    return;
  }
  auto cache_loc_t = varseq_cache_seqpos[b_t];
  auto b = varseq_batch ? varseq_batch[b_t] : b_t;

  if (actual_batch_size != nullptr && b_t >= *actual_batch_size) {
    return;
  }

  at::BFloat16* src_row;
  at::BFloat16* dst_row;
  auto h = 0;
  QKV qkv;
  if (hh < N_H) {
    h = hh;
    src_row = &XQ[b_t][h][0];
    dst_row = &XQ_O[b_t][h][0];
    qkv = QKV::Q;
  } else if (hh < N_H + N_KVH) {
    h = hh - N_H;
    src_row = &XK[b_t][h][0];

    get_dst_row(
        &dst_row,
        cache_K,
        b,
        h,
        cache_loc_t,
        page_size,
        block_tables,
        block_tables_b_stride);
    qkv = QKV::K;
  } else {
    h = hh - N_H - N_KVH;
    src_row = &XV[b_t][h][0];
    get_dst_row(
        &dst_row,
        cache_V,
        b,
        h,
        cache_loc_t,
        page_size,
        block_tables,
        block_tables_b_stride);
    qkv = QKV::V;
  }

  for (auto head_id = 4 * threadIdx.x; head_id < D_H;
       head_id += kThreadsPerWarp * 4) {
    // assert D_H % 4 == 0;
    // load 4 elements per thread in a warp.
    if (head_id >= D_H) {
      return;
    }

    bfx4 src;
    *reinterpret_cast<uint2*>(&src) =
        *reinterpret_cast<uint2*>(&src_row[head_id]);
    if (qkv == QKV::V) {
      *reinterpret_cast<uint2*>(&dst_row[head_id]) =
          *reinterpret_cast<uint2*>(&src);
    } else {
      int32_t offset_0 = ((head_id) / 2 + 0);
      int32_t offset_1 = ((head_id) / 2 + 1);

      double powers_0 = offset_0 * 2;
      double powers_1 = offset_1 * 2;

      double freqs_0 = pow(theta, powers_0 / -static_cast<double>(D_H));
      double freqs_1 = pow(theta, powers_1 / -static_cast<double>(D_H));
      if (rope_scaling) {
        double lo_freq_wavelen = old_context_len / lo_freq_factor;
        double hi_freq_wavelen = old_context_len / hi_freq_factor;
        double wavelen_0 = 2 * M_PI / freqs_0;
        if (wavelen_0 >= hi_freq_wavelen && wavelen_0 > lo_freq_wavelen) {
          freqs_0 = freqs_0 / scaling_factor;
        } else if (wavelen_0 >= hi_freq_wavelen) {
          double smooth = (old_context_len / wavelen_0 - lo_freq_factor) /
              (hi_freq_factor - lo_freq_factor);
          freqs_0 = (1 - smooth) * freqs_0 / scaling_factor + smooth * freqs_0;
        }
        double wavelen_1 = 2 * M_PI / freqs_1;
        if (wavelen_1 >= hi_freq_wavelen && wavelen_1 > lo_freq_wavelen) {
          freqs_1 = freqs_1 / scaling_factor;
        } else if (wavelen_1 >= hi_freq_wavelen) {
          double smooth = (old_context_len / wavelen_1 - lo_freq_factor) /
              (hi_freq_factor - lo_freq_factor);
          freqs_1 = (1 - smooth) * freqs_1 / scaling_factor + smooth * freqs_1;
        }
      }
      freqs_0 = static_cast<double>(seqpos_t) * freqs_0;
      freqs_1 = static_cast<double>(seqpos_t) * freqs_1;

      double sin_0, sin_1, cos_0, cos_1;
      sincos(freqs_0, &sin_0, &cos_0);
      sincos(freqs_1, &sin_1, &cos_1);

      auto src_0 = bf1622float2(src.vals[0]);
      auto src_1 = bf1622float2(src.vals[1]);

      double dst_x, dst_y, dst_z, dst_w;

      dst_x = static_cast<double>(src_0.x) * cos_0 -
          static_cast<double>(src_0.y) * sin_0;
      dst_y = static_cast<double>(src_0.y) * cos_0 +
          static_cast<double>(src_0.x) * sin_0;

      dst_z = static_cast<double>(src_1.x) * cos_1 -
          static_cast<double>(src_1.y) * sin_1;
      dst_w = static_cast<double>(src_1.y) * cos_1 +
          static_cast<double>(src_1.x) * sin_1;

      if (Mode == PositionEmbeddingMode::XPOS) {
        double gamma_0 = (powers_0 + gamma * D_H) / (D_H + gamma * D_H);
        double gamma_1 = (powers_1 + gamma * D_H) / (D_H + gamma * D_H);
        double scale_base_ = (qkv == QKV::Q) ? scale_base : -scale_base;
        double factor_0 = pow(
            gamma_0,
            (static_cast<double>(seqpos_t) - exponent_offset) / scale_base_);
        double factor_1 = pow(
            gamma_1,
            (static_cast<double>(seqpos_t) - exponent_offset) / scale_base_);

        dst_x *= factor_0;
        dst_y *= factor_0;
        dst_z *= factor_1;
        dst_w *= factor_1;
      }

      fx4 dst;
      dst.x = __double2float_rn(dst_x);
      dst.y = __double2float_rn(dst_y);
      dst.z = __double2float_rn(dst_z);
      dst.w = __double2float_rn(dst_w);

      bfx4 dst_;
      dst_.vals[0] = __floats2bfloat162_rn(dst.x, dst.y);
      dst_.vals[1] = __floats2bfloat162_rn(dst.z, dst.w);
      *reinterpret_cast<uint2*>(&dst_row[head_id]) =
          *reinterpret_cast<uint2*>(&dst_);

      if (write_k_back && qkv == QKV::K) {
        // Also write back to the source row
        *reinterpret_cast<uint2*>(&src_row[head_id]) =
            *reinterpret_cast<uint2*>(&dst_);
      }
    }
  }
}

template <PositionEmbeddingMode EmbMode>
DEVICE_INLINE fx4 rope_xpos(
    bfx4 src,
    int32_t seqpos_t,
    QKV head,
    double theta,
    double gamma,
    double scale_base,
    double exponent_offset,
    bool rope_scaling = false,
    int64_t old_context_len = 8192,
    double scaling_factor = 16,
    double lo_freq_factor = 1,
    double hi_freq_factor = 32) {
  fx4 dst; // read 4 bf16 from src and store in 4 float registers
  if (head == QKV::V) {
    return bfx4_to_fx4(src);
  }
  int32_t offset_0 = ((4 * threadIdx.x) / 2 + 0);
  int32_t offset_1 = ((4 * threadIdx.x) / 2 + 1);

  double powers_0 = offset_0 * 2;
  double powers_1 = offset_1 * 2;

  double freqs_0 = pow(theta, powers_0 / -static_cast<double>(D_H));
  double freqs_1 = pow(theta, powers_1 / -static_cast<double>(D_H));

  if (rope_scaling) {
    // From https://github.com/fairinternal/llm_inference/pull/391
    // See https://arxiv.org/pdf/2309.16039 , https://fburl.com/eyhqrzhn
    double lo_freq_wavelen = old_context_len / lo_freq_factor;
    double hi_freq_wavelen = old_context_len / hi_freq_factor;
    double wavelen_0 = 2 * M_PI / freqs_0;
    if (wavelen_0 >= hi_freq_wavelen && wavelen_0 > lo_freq_wavelen) {
      freqs_0 = freqs_0 / scaling_factor;
    } else if (wavelen_0 >= hi_freq_wavelen) {
      double smooth = (old_context_len / wavelen_0 - lo_freq_factor) /
          (hi_freq_factor - lo_freq_factor);
      freqs_0 = (1 - smooth) * freqs_0 / scaling_factor + smooth * freqs_0;
    }
    double wavelen_1 = 2 * M_PI / freqs_1;
    if (wavelen_1 >= hi_freq_wavelen && wavelen_1 > lo_freq_wavelen) {
      freqs_1 = freqs_1 / scaling_factor;
    } else if (wavelen_1 >= hi_freq_wavelen) {
      double smooth = (old_context_len / wavelen_1 - lo_freq_factor) /
          (hi_freq_factor - lo_freq_factor);
      freqs_1 = (1 - smooth) * freqs_1 / scaling_factor + smooth * freqs_1;
    }
  }
  freqs_0 = static_cast<double>(seqpos_t) * freqs_0;
  freqs_1 = static_cast<double>(seqpos_t) * freqs_1;

  double sin_0, sin_1, cos_0, cos_1;
  sincos(freqs_0, &sin_0, &cos_0);
  sincos(freqs_1, &sin_1, &cos_1);

  auto src_0 = bf1622float2(src.vals[0]);
  auto src_1 = bf1622float2(src.vals[1]);

  double dst_x, dst_y, dst_z, dst_w;

  dst_x = static_cast<double>(src_0.x) * cos_0 -
      static_cast<double>(src_0.y) * sin_0;
  dst_y = static_cast<double>(src_0.y) * cos_0 +
      static_cast<double>(src_0.x) * sin_0;

  dst_z = static_cast<double>(src_1.x) * cos_1 -
      static_cast<double>(src_1.y) * sin_1;
  dst_w = static_cast<double>(src_1.y) * cos_1 +
      static_cast<double>(src_1.x) * sin_1;

  if (EmbMode == PositionEmbeddingMode::XPOS) {
    double gamma_0 = (powers_0 + gamma * D_H) / (D_H + gamma * D_H);
    double gamma_1 = (powers_1 + gamma * D_H) / (D_H + gamma * D_H);
    double scale_base_ = (head == QKV::Q) ? scale_base : -scale_base;
    double factor_0 =
        pow(gamma_0,
            (static_cast<double>(seqpos_t) - exponent_offset) / scale_base_);
    double factor_1 =
        pow(gamma_1,
            (static_cast<double>(seqpos_t) - exponent_offset) / scale_base_);
    dst_x *= factor_0;
    dst_y *= factor_0;
    dst_z *= factor_1;
    dst_w *= factor_1;
  }

  dst.x = __double2float_rn(dst_x);
  dst.y = __double2float_rn(dst_y);
  dst.z = __double2float_rn(dst_z);
  dst.w = __double2float_rn(dst_w);

  return dst;
}

template <int KVQuantNumGroups = 1>
DEVICE_INLINE void quantize_int4_kv(fx4 dst, uint8_t* dst_row_q) {
  auto thread_min = fminf(fminf(fminf(dst.x, dst.y), dst.z), dst.w);
  auto thread_max = fmaxf(fmaxf(fmaxf(dst.x, dst.y), dst.z), dst.w);

  float warp_min, warp_max;

  int32_t int4_qparam_offset = 4;
  if (KVQuantNumGroups == 1) {
    unsigned mask = ballot_sync(4 * threadIdx.x < D_H, 0xFFFFFFFF);
    warp_min = -warpReduceMax(-thread_min, mask);
    warp_max = warpReduceMax(thread_max, mask);
  } else {
    auto group_size = D_H / KVQuantNumGroups;
    auto group_idx = threadIdx.x * 4 / group_size;
    int4_qparam_offset = 4 * KVQuantNumGroups;
    unsigned masks[KVQuantNumGroups];
    for (int i = 0; i < KVQuantNumGroups; ++i) {
      masks[i] = ballot_sync(group_idx == i, 0xFFFFFFFF);
    }
    warp_min = -warpReduceMax(-thread_min, masks[group_idx]);
    warp_max = warpReduceMax(thread_max, masks[group_idx]);
  }

  auto scale = (warp_max - warp_min) / 15.0f;
  auto inv_scale = 15.0 / (scale * 15.0 + 1.0e-8);
  auto shift = warp_min;

  auto x_0 = __float2int_rn((dst.x - shift) * inv_scale) & 0xF;
  auto x_1 = __float2int_rn((dst.y - shift) * inv_scale) & 0xF;
  auto x_2 = __float2int_rn((dst.z - shift) * inv_scale) & 0xF;
  auto x_3 = __float2int_rn((dst.w - shift) * inv_scale) & 0xF;

  uint16_t packed = 0;

  packed |= (x_0 << 0);
  packed |= (x_1 << 4);
  packed |= (x_2 << 8);
  packed |= (x_3 << 12);

  // each threadIdx.x writes 2 bytes with 4+4 byte offset for scale/shift

  CUDA_KERNEL_ASSERT(
      uintptr_t(&dst_row_q[2 * threadIdx.x + int4_qparam_offset]) % 2 == 0);

  *reinterpret_cast<uint16_t*>(
      &dst_row_q[2 * threadIdx.x + int4_qparam_offset]) = packed;
  if (threadIdx.x == 0) {
    CUDA_KERNEL_ASSERT(uintptr_t(&dst_row_q[0]) % 4 == 0);
    __half2 qparams = __floats2half2_rn(scale, shift);
    *reinterpret_cast<__half2*>(&dst_row_q[0]) = qparams;
  }
  if (KVQuantNumGroups > 1) {
    int32_t group_size = D_H / KVQuantNumGroups;
    if (threadIdx.x > 0 && threadIdx.x * 4 % group_size == 0) {
      auto group_idx = threadIdx.x * 4 / group_size;
      int32_t qparam_offset = 4 * group_idx;
      CUDA_KERNEL_ASSERT(uintptr_t(&dst_row_q[qparam_offset]) % 4 == 0);
      __half2 qparams = __floats2half2_rn(scale, shift);
      *reinterpret_cast<__half2*>(&dst_row_q[qparam_offset]) = qparams;
    }
  }
}

#define CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL(                 \
    NUM_GROUPS,                                                             \
    DTYPE,                                                                  \
    EMB_MODE,                                                               \
    VARSEQ_BATCH,                                                           \
    VARSEQ_SEQPOS,                                                          \
    THETA,                                                                  \
    GAMMA,                                                                  \
    SCALE_BASE,                                                             \
    EXPO_OFFSET,                                                            \
    block_tables,                                                           \
    page_size,                                                              \
    block_tables_b_stride,                                                  \
    varseq_cache_seqpos,                                                    \
    actual_batch_size,                                                      \
    rope_scaling,                                                           \
    old_context_len,                                                        \
    scaling_factor,                                                         \
    lo_freq_factor,                                                         \
    hi_freq_factor,                                                         \
    write_k_back,                                                           \
    k_norm)                                                                 \
  rope_xpos_qkv_varseq_prefill_kernel_<EMB_MODE, DTYPE, NUM_GROUPS>         \
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(           \
          XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),   \
          XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),   \
          XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),   \
          cache_K.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),   \
          cache_V.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),   \
          qparam_k_ptr,                                                     \
          qparam_v_ptr,                                                     \
          XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(), \
          VARSEQ_BATCH,                                                     \
          VARSEQ_SEQPOS,                                                    \
          THETA,                                                            \
          GAMMA,                                                            \
          SCALE_BASE,                                                       \
          EXPO_OFFSET,                                                      \
          block_tables,                                                     \
          page_size,                                                        \
          block_tables_b_stride,                                            \
          varseq_cache_seqpos,                                              \
          actual_batch_size,                                                \
          rope_scaling,                                                     \
          old_context_len,                                                  \
          scaling_factor,                                                   \
          lo_freq_factor,                                                   \
          hi_freq_factor,                                                   \
          write_k_back,                                                     \
          k_norm);

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
#endif

template <
    PositionEmbeddingMode EmbMode,
    CacheLogicalDtype kCacheDtype,
    int KVQuantNumGroups = 1>
__global__ void rope_xpos_qkv_varseq_prefill_kernel_(
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XQ, // [B_T][N_H][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XK, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XV, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H +4]
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H + 4]
    int32_t* qparam_k_ptr,
    int32_t* qparam_v_ptr,
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XQ_O, // [B_T][N_H][D]
    int32_t* varseq_batch, // in decoding case we have T == 1 and so just
                           // pass nullptr
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> varseq_seqpos,
    double theta,
    double gamma,
    double scale_base,
    double exponent_offset,
    int32_t* block_tables, // [B][MAX_PAGES], maps logical pages to physical
                           // ones for paged attention
    int32_t page_size,
    int32_t block_tables_b_stride,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        varseq_cache_seqpos,
    int64_t* actual_batch_size =
        nullptr, // When running in CUDA graph mode, the actual batch size
                 // can be smaller than block_tables.size(0). In this case
                 // rows of block_tables beyond actual_batch_size are not
                 // initialized, and using them wil cause undefined
                 // behavior. To prevent this, when actual_batch_size is
                 // provided, the kernel exits if the current batch index is
                 // larger of equal to actual_batch_size,
    bool rope_scaling = false,
    int64_t old_context_len = 8192,
    double scaling_factor = 16,
    double lo_freq_factor = 1,
    double hi_freq_factor = 32,
    bool write_k_back = false,
    bool k_norm = false) {
  // Launch b_t_(sum(h)) warps.
  auto b_t_hh = blockIdx.x * blockDim.y +
      threadIdx.y; // Block = [kThreadsPerWarp, kWarpsPerBlock]
  // Each warp handles a single head XQ or XK or XV of a single token..
  // That would be 1 x 128 distributed among 32 threads in the warp.
  // Each thread should handle 4 elements.
  auto B_T = XQ.size(0);
  auto N_KVH = XK.size(1);
  auto N_H = XQ.size(1);
  auto D_H = XQ.size(2);

  auto HH = 2 * N_KVH + N_H;

  auto hh = b_t_hh % HH;
  auto b_t = b_t_hh / HH;
  if (b_t >= B_T) {
    return;
  }
  auto seqpos_t = varseq_seqpos[b_t];
  if (seqpos_t == -1) {
    return;
  }
  auto cache_loc_t = varseq_cache_seqpos[b_t];
  auto b = varseq_batch ? varseq_batch[b_t] : b_t;

  if (actual_batch_size != nullptr && b_t >= *actual_batch_size) {
    return;
  }

  at::BFloat16* src_row = nullptr;
  at::BFloat16* dst_row = nullptr;
  uint8_t* dst_row_q = nullptr;
  auto h = 0;
  QKV qkv;
  if (hh < N_H) {
    h = hh;
    src_row = &XQ[b_t][h][0];
    dst_row = &XQ_O[b_t][h][0];
    qkv = QKV::Q;
  } else if (hh < N_H + N_KVH) {
    h = hh - N_H;
    src_row = &XK[b_t][h][0];
    get_dst_row(
        &dst_row_q,
        cache_K,
        b,
        h,
        cache_loc_t,
        page_size,
        block_tables,
        block_tables_b_stride);
    qkv = QKV::K;
  } else {
    h = hh - N_H - N_KVH;
    src_row = &XV[b_t][h][0];
    get_dst_row(
        &dst_row_q,
        cache_V,
        b,
        h,
        cache_loc_t,
        page_size,
        block_tables,
        block_tables_b_stride);
    qkv = QKV::V;
  }

  // load 4 elements per thread in a warp.

  // Each thread should handle D_H//32 = 4 elements.
  CUDA_KERNEL_ASSERT(D_H <= 4 * kThreadsPerWarp);
  if (4 * threadIdx.x >= D_H) {
    return;
  }
  bfx4 src;
  *reinterpret_cast<uint2*>(&src) =
      *reinterpret_cast<uint2*>(&src_row[4 * threadIdx.x]);

  fx4 dst = rope_xpos<EmbMode>(
      src,
      seqpos_t,
      qkv,
      theta,
      gamma,
      scale_base,
      exponent_offset,
      rope_scaling,
      old_context_len,
      scaling_factor,
      lo_freq_factor,
      hi_freq_factor);
  // now we have our output.
  if (qkv == QKV::Q) { // is_q // store to Qo without quantization
    bfx4 dst_ = fx4_to_bfx4(dst);
    CUDA_KERNEL_ASSERT(uintptr_t(&dst_row[4 * threadIdx.x]) % 8 == 0);

    *reinterpret_cast<uint2*>(&dst_row[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&dst_);
  } else {
    if (write_k_back && qkv == QKV::K) {
      // Also write back to the source row
      bfx4 dst_ = fx4_to_bfx4(dst);
      *reinterpret_cast<uint2*>(&src_row[4 * threadIdx.x]) =
          *reinterpret_cast<uint2*>(&dst_);
    }
    // quantize and write to dst_row
    auto D_H = XQ.size(2);
    auto D_H_q = cache_K.size(3);
    if (kCacheDtype == CacheLogicalDtype::FP8) {
      if (qparam_k_ptr == nullptr) {
        CUDA_KERNEL_ASSERT(D_H_q - D_H == 4);
        quantize_fp8_kv(dst, dst_row_q, nullptr, (qkv == QKV::K && k_norm));
      } else {
        __half2* qparam_row = nullptr;
        auto T = cache_K.size(1);
        size_t idx = 0;
        if (block_tables == nullptr) {
          idx = b * (T * N_KVH) + (size_t)cache_loc_t * N_KVH + h;
        } else {
          // This is duplicate computation with get_dst_row above.
          // TODO: Maybe clean up and merge later.
          int page_logical_idx = cache_loc_t / page_size;
          int page_offset = cache_loc_t % page_size;
          int page_physical_idx =
              block_tables[b * block_tables_b_stride + page_logical_idx];
          int physical_t = page_physical_idx * page_size + page_offset;
          idx = physical_t * N_KVH + h;
        }
        if (qkv == QKV::K) {
          qparam_row = reinterpret_cast<__half2*>(&qparam_k_ptr[idx]);
        } else {
          qparam_row = reinterpret_cast<__half2*>(&qparam_v_ptr[idx]);
        }
        quantize_fp8_kv(dst, dst_row_q, qparam_row, (qkv == QKV::K && k_norm));
      }

    } else if (kCacheDtype == CacheLogicalDtype::INT4) {
      CUDA_KERNEL_ASSERT(D_H_q - D_H / 2 == 4 * KVQuantNumGroups);
      quantize_int4_kv<KVQuantNumGroups>(dst, dst_row_q);
    }
  }
}

at::Tensor nope_qkv_varseq_prefill(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor varseq_batch,
    at::Tensor varseq_seqpos,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> varseq_cache_seqpos,
    std::optional<at::Tensor> qparam_k = std::nullopt,
    std::optional<at::Tensor> qparam_v = std::nullopt,
    bool k_norm = false) {
  auto B_T = XQ.size(0);
  auto N_H = XQ.size(1);
  auto N_KVH = XK.size(1);

  TORCH_CHECK(XQ.size(2) % 4 == 0);
  TORCH_CHECK(XQ.size(2) <= 512);

  int32_t num_warps = B_T * (2 * N_KVH + N_H);
  TORCH_CHECK(num_warps > 0);

  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
  dim3 blocks(cuda_calc_xblock_count(num_warps, kWarpsPerBlock));

  TORCH_CHECK(varseq_batch.is_contiguous());
  TORCH_CHECK(varseq_batch.numel() == B_T);
  auto XQ_O = at::empty_like(XQ);

  auto varseq_cache_seqpos_ = varseq_cache_seqpos.value_or(varseq_seqpos);

  int32_t* block_tables_ptr = nullptr;
  int32_t block_tables_b_stride = 0;
  if (block_tables.has_value()) {
    block_tables_ptr = static_cast<int32_t*>(block_tables.value().data_ptr());
    block_tables_b_stride = block_tables.value().stride(0);
  }

  // Current NOPE kernel only supports BF16
  if (cache_K.dtype() == at::kBFloat16) {
    nope_qkv_varseq_prefill_kernel<<<
        blocks,
        threads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
        XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
        XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
        cache_K.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
        cache_V.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
        XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
        varseq_batch.data_ptr<int32_t>(),
        varseq_seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
        block_tables_ptr,
        page_size,
        block_tables_b_stride,
        varseq_cache_seqpos_
            .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
        nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return XQ_O;
  }
  int32_t* qparam_k_ptr = nullptr;
  int32_t* qparam_v_ptr = nullptr;
  if (qparam_k.has_value()) {
    qparam_k_ptr = static_cast<int32_t*>(qparam_k.value().data_ptr());
    qparam_v_ptr = static_cast<int32_t*>(qparam_v.value().data_ptr());
  }
  nope_qkv_varseq_prefill_kernel_quantized<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
      XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
      XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
      cache_K.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
      cache_V.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
      qparam_k_ptr,
      qparam_v_ptr,
      XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
      varseq_batch.data_ptr<int32_t>(),
      varseq_seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      block_tables_ptr,
      page_size,
      block_tables_b_stride,
      varseq_cache_seqpos_
          .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      nullptr,
      k_norm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return XQ_O;
}

at::Tensor nope_qkv_decoding(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seqpos,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> actual_batch_size,
    std::optional<at::Tensor> batch,
    std::optional<at::Tensor> cache_seqpos,
    std::optional<at::Tensor> qparam_k = std::nullopt,
    std::optional<at::Tensor> qparam_v = std::nullopt,
    bool k_norm = false) {
  auto B = XQ.size(0);
  auto N_H = XQ.size(1);
  auto N_KVH = XK.size(1);

  TORCH_CHECK(XQ.size(2) % 4 == 0);
  int32_t num_warps = B * (2 * N_KVH + N_H);
  TORCH_CHECK(num_warps > 0);

  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
  dim3 blocks(cuda_calc_xblock_count(num_warps, kWarpsPerBlock));
  auto XQ_O = at::empty_like(XQ);

  int32_t* block_tables_ptr = nullptr;
  int32_t block_tables_b_stride = 0;
  if (block_tables.has_value()) {
    block_tables_ptr = static_cast<int32_t*>(block_tables.value().data_ptr());
    block_tables_b_stride = block_tables.value().stride(0);
  }
  int64_t* actual_batch_size_ptr = nullptr;
  if (actual_batch_size.has_value()) {
    actual_batch_size_ptr =
        static_cast<int64_t*>(actual_batch_size.value().data_ptr());
  }
  auto cache_seqpos_ = cache_seqpos.value_or(seqpos);

  // Current NOPE kernel only supports BF16
  // num_groups  == 1
  if (cache_K.dtype() == at::kBFloat16) {
    nope_qkv_varseq_prefill_kernel<<<
        blocks,
        threads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
        XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
        XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
        cache_K.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
        cache_V.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
        XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
        batch.has_value() ? batch.value().data_ptr<int32_t>() : nullptr,
        seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
        block_tables_ptr,
        page_size,
        block_tables_b_stride,
        cache_seqpos_.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
        actual_batch_size_ptr);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return XQ_O;
  }
  int32_t* qparam_k_ptr = nullptr;
  int32_t* qparam_v_ptr = nullptr;
  if (qparam_k.has_value()) {
    qparam_k_ptr = static_cast<int32_t*>(qparam_k.value().data_ptr());
    qparam_v_ptr = static_cast<int32_t*>(qparam_v.value().data_ptr());
  }
  // cache_logical_dtype == CacheLogicalDtype::FP8
  nope_qkv_varseq_prefill_kernel_quantized<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
      XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
      XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
      cache_K.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
      cache_V.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
      qparam_k_ptr,
      qparam_v_ptr,
      XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
      batch.has_value() ? batch.value().data_ptr<int32_t>() : nullptr,
      seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      block_tables_ptr,
      page_size,
      block_tables_b_stride,
      cache_seqpos_.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      actual_batch_size_ptr,
      k_norm);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return XQ_O;
}

at::Tensor rope_qkv_varseq_prefill(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor varseq_batch,
    at::Tensor varseq_seqpos,
    double theta,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> varseq_cache_seqpos,
    int64_t cache_logical_dtype_int,
    bool rope_scaling = false,
    int64_t old_context_len = 8192,
    double scaling_factor = 16,
    double lo_freq_factor = 1,
    double hi_freq_factor = 32,
    std::optional<at::Tensor> qparam_k = std::nullopt,
    std::optional<at::Tensor> qparam_v = std::nullopt,
    bool write_k_back = false,
    bool k_norm = false) {
  auto B_T = XQ.size(0);
  auto N_H = XQ.size(1);
  auto N_KVH = XK.size(1);

  TORCH_CHECK(XQ.size(2) % 4 == 0);
  TORCH_CHECK(XQ.size(2) <= 512);

  int32_t num_warps = B_T * (2 * N_KVH + N_H);
  TORCH_CHECK(num_warps > 0);

  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
  dim3 blocks(cuda_calc_xblock_count(num_warps, kWarpsPerBlock));

  TORCH_CHECK(varseq_batch.is_contiguous());
  TORCH_CHECK(varseq_batch.numel() == B_T);
  auto XQ_O = at::empty_like(XQ);

  auto varseq_cache_seqpos_ = varseq_cache_seqpos.value_or(varseq_seqpos);

  CacheLogicalDtype cache_logical_dtype =
      static_cast<CacheLogicalDtype>(cache_logical_dtype_int);

  int32_t* block_tables_ptr = nullptr;
  int32_t block_tables_b_stride = 0;
  if (block_tables.has_value()) {
    block_tables_ptr = static_cast<int32_t*>(block_tables.value().data_ptr());
    block_tables_b_stride = block_tables.value().stride(0);
  }
  if (cache_K.dtype() == at::kBFloat16) {
    rope_xpos_qkv_varseq_prefill_kernel<PositionEmbeddingMode::ROPE>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
            XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            varseq_batch.data_ptr<int32_t>(),
            varseq_seqpos
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            theta,
            0,
            0,
            0,
            block_tables_ptr,
            page_size,
            block_tables_b_stride,
            varseq_cache_seqpos_
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            nullptr,
            rope_scaling,
            old_context_len,
            scaling_factor,
            lo_freq_factor,
            hi_freq_factor,
            write_k_back);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto num_groups_ = num_groups ? num_groups.value() : 1;
    auto varseq_batch_ = varseq_batch.data_ptr<int32_t>();
    auto varseq_seqpos_ =
        varseq_seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>();
    int32_t* qparam_k_ptr = nullptr;
    int32_t* qparam_v_ptr = nullptr;
    if (qparam_k.has_value()) {
      qparam_k_ptr = static_cast<int32_t*>(qparam_k.value().data_ptr());
      qparam_v_ptr = static_cast<int32_t*>(qparam_v.value().data_ptr());
    }
    if (cache_logical_dtype == CacheLogicalDtype::FP8) {
#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
      CUDA_KERNEL_ASSERT(num_groups_ == 1);
      CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL(
          1,
          CacheLogicalDtype::FP8,
          PositionEmbeddingMode::ROPE,
          varseq_batch_,
          varseq_seqpos_,
          theta,
          0,
          0,
          0,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (varseq_cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          nullptr,
          rope_scaling,
          old_context_len,
          scaling_factor,
          lo_freq_factor,
          hi_freq_factor,
          write_k_back,
          k_norm);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
      throw std::runtime_error("CUDA version is older than 12.0");
#endif
    } else {
      CALL_INT4_KERNEL_WITH_KV_GROUPWISE_QUANT_CHECK(
          CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL,
          num_groups_,
          CacheLogicalDtype::INT4,
          PositionEmbeddingMode::ROPE,
          varseq_batch_,
          varseq_seqpos_,
          theta,
          0,
          0,
          0,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (varseq_cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          nullptr,
          rope_scaling,
          old_context_len,
          scaling_factor,
          lo_freq_factor,
          hi_freq_factor,
          write_k_back,
          false);

      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  return XQ_O;
}

at::Tensor xpos_qkv_varseq_prefill(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor varseq_batch,
    at::Tensor varseq_seqpos,
    double theta,
    double gamma,
    double scale_base,
    double exponent_offset,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> varseq_cache_seqpos,
    int64_t cache_logical_dtype_int,
    bool rope_scaling = false,
    int64_t old_context_len = 8192,
    double scaling_factor = 16,
    double lo_freq_factor = 1,
    double hi_freq_factor = 32,
    std::optional<at::Tensor> qparam_k = {},
    std::optional<at::Tensor> qparam_v = {}) {
  auto B_T = XQ.size(0);
  auto N_H = XQ.size(1);
  auto N_KVH = XK.size(1);

  TORCH_CHECK(XQ.size(2) % 4 == 0);
  TORCH_CHECK(XQ.size(2) <= 512);

  int32_t num_warps = B_T * (2 * N_KVH + N_H);
  TORCH_CHECK(num_warps > 0);

  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
  dim3 blocks(cuda_calc_xblock_count(num_warps, kWarpsPerBlock));

  auto XQ_O = at::empty_like(XQ);
  TORCH_CHECK(varseq_batch.is_contiguous());
  TORCH_CHECK(varseq_batch.numel() == B_T);
  auto varseq_cache_seqpos_ = varseq_cache_seqpos.value_or(varseq_seqpos);
  CacheLogicalDtype cache_logical_dtype =
      static_cast<CacheLogicalDtype>(cache_logical_dtype_int);

  int32_t* block_tables_ptr = nullptr;
  int32_t block_tables_b_stride = 0;
  if (block_tables.has_value()) {
    block_tables_ptr = static_cast<int32_t*>(block_tables.value().data_ptr());
    block_tables_b_stride = block_tables.value().stride(0);
  }

  if (cache_K.dtype() == at::kBFloat16) {
    rope_xpos_qkv_varseq_prefill_kernel<PositionEmbeddingMode::XPOS>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
            XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            varseq_batch.data_ptr<int32_t>(),
            varseq_seqpos
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            theta,
            gamma,
            scale_base,
            exponent_offset,
            block_tables_ptr,
            page_size,
            block_tables_b_stride,
            varseq_cache_seqpos_
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            nullptr,
            rope_scaling,
            old_context_len,
            scaling_factor,
            lo_freq_factor,
            hi_freq_factor);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto num_groups_ = num_groups ? num_groups.value() : 1;
    auto varseq_batch_ = varseq_batch.data_ptr<int32_t>();
    auto varseq_seqpos_ =
        varseq_seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>();
    int32_t* qparam_k_ptr = nullptr;
    int32_t* qparam_v_ptr = nullptr;
    if (qparam_k.has_value()) {
      qparam_k_ptr = static_cast<int32_t*>(qparam_k.value().data_ptr());
      qparam_v_ptr = static_cast<int32_t*>(qparam_v.value().data_ptr());
    }
    if (cache_logical_dtype == CacheLogicalDtype::FP8) {
#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
      CUDA_KERNEL_ASSERT(num_groups_ == 1);
      CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL(
          1,
          CacheLogicalDtype::FP8,
          PositionEmbeddingMode::XPOS,
          varseq_batch_,
          varseq_seqpos_,
          theta,
          gamma,
          scale_base,
          exponent_offset,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (varseq_cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          nullptr,
          rope_scaling,
          old_context_len,
          scaling_factor,
          lo_freq_factor,
          hi_freq_factor,
          false,
          false);

      C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
      throw std::runtime_error("CUDA version is older than 12.0");
#endif
    } else {
      CALL_INT4_KERNEL_WITH_KV_GROUPWISE_QUANT_CHECK(
          CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL,
          num_groups_,
          CacheLogicalDtype::INT4,
          PositionEmbeddingMode::XPOS,
          varseq_batch_,
          varseq_seqpos_,
          theta,
          gamma,
          scale_base,
          exponent_offset,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (varseq_cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          nullptr,
          rope_scaling,
          old_context_len,
          scaling_factor,
          lo_freq_factor,
          hi_freq_factor,
          false,
          false);

      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  return XQ_O;
}

at::Tensor rope_qkv_decoding(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seqpos,
    double theta,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> actual_batch_size,
    std::optional<at::Tensor> batch,
    std::optional<at::Tensor> cache_seqpos,
    int64_t cache_logical_dtype_int,
    bool rope_scaling = false,
    int64_t old_context_len = 8192,
    double scaling_factor = 16,
    double lo_freq_factor = 1,
    double hi_freq_factor = 32,
    std::optional<at::Tensor> qparam_k = std::nullopt,
    std::optional<at::Tensor> qparam_v = std::nullopt,
    bool k_norm = false) {
  auto B = XQ.size(0);
  auto N_H = XQ.size(1);
  auto N_KVH = XK.size(1);

  TORCH_CHECK(XQ.size(2) % 4 == 0);
  int32_t num_warps = B * (2 * N_KVH + N_H);
  TORCH_CHECK(num_warps > 0);

  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
  dim3 blocks(cuda_calc_xblock_count(num_warps, kWarpsPerBlock));
  auto XQ_O = at::empty_like(XQ);

  CacheLogicalDtype cache_logical_dtype =
      static_cast<CacheLogicalDtype>(cache_logical_dtype_int);

  int32_t* block_tables_ptr = nullptr;
  int32_t block_tables_b_stride = 0;
  if (block_tables.has_value()) {
    block_tables_ptr = static_cast<int32_t*>(block_tables.value().data_ptr());
    block_tables_b_stride = block_tables.value().stride(0);
  }
  int64_t* actual_batch_size_ptr = nullptr;
  if (actual_batch_size.has_value()) {
    actual_batch_size_ptr =
        static_cast<int64_t*>(actual_batch_size.value().data_ptr());
  }
  auto cache_seqpos_ = cache_seqpos.value_or(seqpos);
  if (cache_K.dtype() == at::kBFloat16) {
    rope_xpos_qkv_varseq_prefill_kernel<PositionEmbeddingMode::ROPE>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
            XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            batch.has_value() ? batch.value().data_ptr<int32_t>() : nullptr,
            seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            theta,
            0,
            0,
            0,
            block_tables_ptr,
            page_size,
            block_tables_b_stride,
            cache_seqpos_
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            actual_batch_size_ptr,
            rope_scaling,
            old_context_len,
            scaling_factor,
            lo_freq_factor,
            hi_freq_factor);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto seqpos_ =
        seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>();
    auto num_groups_ = num_groups ? num_groups.value() : 1;
    int32_t* qparam_k_ptr = nullptr;
    int32_t* qparam_v_ptr = nullptr;
    if (qparam_k.has_value()) {
      qparam_k_ptr = static_cast<int32_t*>(qparam_k.value().data_ptr());
      qparam_v_ptr = static_cast<int32_t*>(qparam_v.value().data_ptr());
    }
    if (cache_logical_dtype == CacheLogicalDtype::FP8) {
#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
      CUDA_KERNEL_ASSERT(num_groups_ == 1);
      CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL(
          1,
          CacheLogicalDtype::FP8,
          PositionEmbeddingMode::ROPE,
          nullptr,
          seqpos_,
          theta,
          0,
          0,
          0,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          actual_batch_size_ptr,
          rope_scaling,
          old_context_len,
          scaling_factor,
          lo_freq_factor,
          hi_freq_factor,
          false,
          k_norm);

      C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
      throw std::runtime_error("CUDA version is older than 12.0");
#endif
    } else {
      CALL_INT4_KERNEL_WITH_KV_GROUPWISE_QUANT_CHECK(
          CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL,
          num_groups_,
          CacheLogicalDtype::INT4,
          PositionEmbeddingMode::ROPE,
          nullptr,
          seqpos_,
          theta,
          0,
          0,
          0,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          actual_batch_size_ptr,
          rope_scaling,
          old_context_len,
          scaling_factor,
          lo_freq_factor,
          hi_freq_factor,
          false,
          false);

      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  return XQ_O;
}

at::Tensor xpos_qkv_decoding(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seqpos,
    double theta,
    double gamma,
    double scale_base,
    double exponent_offset,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> actual_batch_size,
    std::optional<at::Tensor> batch,
    std::optional<at::Tensor> cache_seqpos,
    int64_t cache_logical_dtype_int,
    bool rope_scaling = false,
    int64_t old_context_len = 8192,
    double scaling_factor = 16,
    double lo_freq_factor = 1,
    double hi_freq_factor = 32,
    std::optional<at::Tensor> qparam_k = {},
    std::optional<at::Tensor> qparam_v = {}) {
  auto B = XQ.size(0);
  auto N_H = XQ.size(1);
  auto N_KVH = XK.size(1);

  TORCH_CHECK(XQ.size(2) % 4 == 0);
  int32_t num_warps = B * (2 * N_KVH + N_H);
  TORCH_CHECK(num_warps > 0);

  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
  dim3 blocks(cuda_calc_xblock_count(num_warps, kWarpsPerBlock));
  auto XQ_O = at::empty_like(XQ);
  CacheLogicalDtype cache_logical_dtype =
      static_cast<CacheLogicalDtype>(cache_logical_dtype_int);

  int32_t* block_tables_ptr = nullptr;
  int32_t block_tables_b_stride = 0;
  if (block_tables.has_value()) {
    block_tables_ptr = static_cast<int32_t*>(block_tables.value().data_ptr());
    block_tables_b_stride = block_tables.value().stride(0);
  }

  int64_t* actual_batch_size_ptr = nullptr;
  if (actual_batch_size.has_value()) {
    actual_batch_size_ptr =
        static_cast<int64_t*>(actual_batch_size.value().data_ptr());
  }
  auto cache_seqpos_ = cache_seqpos.value_or(seqpos);
  if (cache_K.dtype() == at::kBFloat16) {
    rope_xpos_qkv_varseq_prefill_kernel<PositionEmbeddingMode::XPOS>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
            XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            batch.has_value() ? batch.value().data_ptr<int32_t>() : nullptr,
            seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            theta,
            gamma,
            scale_base,
            exponent_offset,
            block_tables_ptr,
            page_size,
            block_tables_b_stride,
            cache_seqpos_
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            actual_batch_size_ptr,
            rope_scaling,
            old_context_len,
            scaling_factor,
            lo_freq_factor,
            hi_freq_factor);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto num_groups_ = num_groups ? num_groups.value() : 1;
    auto seqpos_ =
        seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>();
    int32_t* qparam_k_ptr = nullptr;
    int32_t* qparam_v_ptr = nullptr;
    if (qparam_k.has_value()) {
      qparam_k_ptr = static_cast<int32_t*>(qparam_k.value().data_ptr());
      qparam_v_ptr = static_cast<int32_t*>(qparam_v.value().data_ptr());
    }
    if (cache_logical_dtype == CacheLogicalDtype::FP8) {
#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
      CUDA_KERNEL_ASSERT(num_groups_ == 1);
      CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL(
          1,
          CacheLogicalDtype::FP8,
          PositionEmbeddingMode::XPOS,
          nullptr,
          seqpos_,
          theta,
          gamma,
          scale_base,
          exponent_offset,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          actual_batch_size_ptr,
          rope_scaling,
          old_context_len,
          scaling_factor,
          lo_freq_factor,
          hi_freq_factor,
          false,
          false);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
      throw std::runtime_error("CUDA version is older than 12.0");
#endif
    } else {
      CALL_INT4_KERNEL_WITH_KV_GROUPWISE_QUANT_CHECK(
          CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL,
          num_groups_,
          CacheLogicalDtype::INT4,
          PositionEmbeddingMode::XPOS,
          nullptr,
          seqpos_,
          theta,
          gamma,
          scale_base,
          exponent_offset,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          actual_batch_size_ptr,
          rope_scaling,
          old_context_len,
          scaling_factor,
          lo_freq_factor,
          hi_freq_factor,
          false,
          false);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  return XQ_O;
}

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
template <bool ExternalQParam>
__global__ void dequantize_fp8_cache_kernel(
    // This code currently represents FP8 version not int4
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H // G]
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> kv_seqlen,
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K_dq, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
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
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [1][MAX_PAGE * PAGE_SIZE][N_KVH][D_H]
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [1][MAX_PAGE * PAGE_SIZE][N_KVH][D_H // G]
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> kv_seqlen,
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_K_dq, // [1][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<at::BFloat16, 4, at::RestrictPtrTraits>
        cache_V_dq, // [1][MAX_T][N_KVH][D_H]
    int32_t* qparam_k_ptr,
    int32_t* qparam_v_ptr,
    int32_t* block_tables,
    int32_t block_tables_b_stride,
    int32_t page_size) {
  auto N_KVH = cache_K.size(2);
  auto MAX_T = cache_K.size(1);
  auto D_H = cache_K_dq.size(3);
  auto D_H_q = cache_K.size(3);
  CUDA_KERNEL_ASSERT(D_H == 128);

  auto b = blockIdx.x;
  // only need to dequantize this far.
  auto max_t = kv_seqlen[b];

  // one warp per T/H
  for (auto t_h = threadIdx.y + blockIdx.y * blockDim.y; t_h < max_t * N_KVH;
       t_h += blockDim.y * gridDim.y) {
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
}
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
  // The below allocates Tensors that have the same shape as cache_K and cache_V
  // to store their dequantize results. For paged KV cache, this can be a bit
  // inefficient because it has the shape of [1 x (MAX_PAGES * PAGE_SIZE) x
  // N_KVH x D_H] to accommodate pages globally across batch instances, and
  // if we have very large MAX_PAGES then we are essentially allocating a very
  // huge Tensor here. The benefit is that the following users of this
  // dequantized results can reuse the existing block_tables to access their
  // elements. If we want to be more efficient, there are two possible
  // approaches: (1) Allocate a shorter Tensor here and store the dequantize
  // results in a more compact manner, but that requires creating a new
  // block_tables here and making sure the following users all use the
  // correct block_tables. (2) From outside, keep a persistent buffer that has a
  // matching shape with the original paged KV and feed the same buffer
  // into this function at every layer to reuse it and prevent allocation.

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
#define CALL_DEQUANTIZE_FP8_CACHE(EXTERNAL_Q_PARAM)                           \
  const auto deq_fn = dequantize_fp8_cache_kernel<EXTERNAL_Q_PARAM>;          \
  deq_fn<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(           \
      cache_K.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),         \
      cache_V.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),         \
      kv_seqlen.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),       \
      cache_K_dq.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(), \
      cache_V_dq.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(), \
      qparam_k_ptr,                                                           \
      qparam_v_ptr);                                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK()
  if (block_tables_ptr == nullptr) {
    if (qparam_k_ptr) {
      CALL_DEQUANTIZE_FP8_CACHE(true);
    } else {
      CALL_DEQUANTIZE_FP8_CACHE(false);
    }
#undef CALL_DEQUANTIZE_FP8_CACHE
  } else {
    dequantize_fp8_cache_kernel_paged<<<
        blocks,
        threads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        cache_K.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
        cache_V.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
        kv_seqlen.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
        cache_K_dq.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
        cache_V_dq.packed_accessor64<at::BFloat16, 4, at::RestrictPtrTraits>(),
        qparam_k_ptr,
        qparam_v_ptr,
        block_tables_ptr,
        block_tables_b_stride,
        page_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {cache_K_dq, cache_V_dq};
}

DEVICE_INLINE void
quantize_fp8_kv(fx4 dst, uint8_t* dst_row_q, __half2* qparam, bool do_norm) {
  if (do_norm) {
    float sum = fx4_dot(dst, dst);
    // Warp reduce sum
    sum = warpReduceSum(sum);
    float rsqr = rsqrtf(sum / 128);
    dst = fx4_scale(dst, rsqr);
  }
  auto thread_min = fminf(fminf(fminf(dst.x, dst.y), dst.z), dst.w);
  auto thread_max = fmaxf(fmaxf(fmaxf(dst.x, dst.y), dst.z), dst.w);

  float warp_min, warp_max;

  int32_t fp8_qparam_offset = 0;
  if (qparam == nullptr) {
    fp8_qparam_offset = 4;
  }
  unsigned mask = ballot_sync(4 * threadIdx.x < D_H, 0xFFFFFFFF);
  warp_min = -warpReduceMax(-thread_min, mask);
  warp_max = warpReduceMax(thread_max, mask);

  auto bounded_max = (warp_max - warp_min) / 2;
  //  TODO: Pass scale_ub
  const float* scale_ub = nullptr;
  constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX::value * 512.f);
  if (scale_ub != nullptr) {
    bounded_max = std::min(bounded_max, *scale_ub);
  }
  float scale = static_cast<float>(
      std::max(bounded_max / FP8_E4M3_MAX::value, min_scaling_factor));
  float inv_scale = 1 / scale;
  float shift = warp_min + FP8_E4M3_MAX::value * scale;

  auto x_0 = __nv_fp8_e4m3((dst.x - shift) * inv_scale);
  auto x_1 = __nv_fp8_e4m3((dst.y - shift) * inv_scale);
  auto x_2 = __nv_fp8_e4m3((dst.z - shift) * inv_scale);
  auto x_3 = __nv_fp8_e4m3((dst.w - shift) * inv_scale);

  uint32_t x_bits[4];
  x_bits[0] = *reinterpret_cast<uint32_t*>(&x_0);
  x_bits[1] = *reinterpret_cast<uint32_t*>(&x_1);
  x_bits[2] = *reinterpret_cast<uint32_t*>(&x_2);
  x_bits[3] = *reinterpret_cast<uint32_t*>(&x_3);

  uint32_t packed = 0;

  packed |= (x_bits[0] << 0);
  packed |= (x_bits[1] << 8);
  packed |= (x_bits[2] << 16);
  packed |= (x_bits[3] << 24);

  CUDA_KERNEL_ASSERT(
      uintptr_t(&dst_row_q[4 * threadIdx.x + fp8_qparam_offset]) % 4 == 0);

  *reinterpret_cast<uint32_t*>(
      &dst_row_q[4 * threadIdx.x + fp8_qparam_offset]) = packed;
  // write qparams
  if (threadIdx.x == 0) {
    __half2* param_store = qparam;
    if (param_store == nullptr) {
      // If no external qparam, store the params at beginning of the quantized
      // cache.
      param_store = reinterpret_cast<__half2*>(&dst_row_q[0]);
    }
    CUDA_KERNEL_ASSERT(uintptr_t(param_store) % 4 == 0);
    *param_store = __floats2half2_rn(scale, shift);
  }
}
#else
DEVICE_INLINE void
quantize_fp8_kv(fx4 dst, uint8_t* dst_row_, __half2* qparam, bool do_norm) {}
std::vector<at::Tensor> quantize_fp8_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub) { // scale upperbound
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

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
#endif
} // namespace fbgemm_gpu
