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
#include "kv_cache.h"

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

enum class PositionEmbeddingMode { ROPE = 0, XPOS = 1, NOPE = 2 };
enum class KVQuantRecipe { perTokenScaling = 0, perHeadScaling = 1 };
enum class QKV { Q, K, V };

template <typename T, KVQuantRecipe recipe = KVQuantRecipe::perTokenScaling>
DEVICE_INLINE void quantize_fp8_kv(
    fx4 dst,
    T* dst_row_q,
    __half2* qparam = nullptr,
    bool do_norm = false);

DEVICE_INLINE void per_row_norm(fx4& dst) {
  float sum = fx4_dot(dst, dst);
  // Warp reduce sum
  sum = warpReduceSum(sum);
  float rsqr = rsqrtf(sum / D_H);
  dst = fx4_scale(dst, rsqr);
}

DEVICE_INLINE void per_head_amax(fx4& dst, float* amax) {
  dst = fx4_abs(dst);
  auto thread_max = fmaxf(fmaxf(fmaxf(dst.x, dst.y), dst.z), dst.w);
  CUDA_KERNEL_ASSERT(thread_max >= 0)
  unsigned mask = ballot_sync(4 * threadIdx.x < D_H, 0xFFFFFFFF);
  float warp_max = warpReduceMax(thread_max, mask);
  // Use atomic operation to update the global maximum
  if (threadIdx.x == 0) {
    // CUDA_KERNEL_ASSERT(warp_max >= 0);
    atomicMax(reinterpret_cast<int*>(amax), __float_as_int(warp_max));
  }
}
DEVICE_INLINE void per_row_amax(fx4& dst, float* amax) {
  dst = fx4_abs(dst);
  auto thread_max = fmaxf(fmaxf(fmaxf(dst.x, dst.y), dst.z), dst.w);
  unsigned mask = ballot_sync(4 * threadIdx.x < D_H, 0xFFFFFFFF);
  float warp_max = warpReduceMax(thread_max, mask);
  // write amax
  if (threadIdx.x == 0) {
    CUDA_KERNEL_ASSERT(uintptr_t(amax) % 4 == 0);
    *amax = warp_max;
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
        nullptr, // When running in CUDA graph mode, the actual batch size
                 // can be smaller than block_tables.size(0). In this case
                 // rows of block_tables beyond actual_batch_size are not
                 // initialized, and using them wil cause undefined
                 // behavior. To prevent this, when actual_batch_size is
                 // provided, the kernel exits if the current batch index is
                 // larger of equal to actual_batch_size,
    bool update_kv = true) {
  // Launch b_t_(sum(h)) warps.
  auto b_t_hh = blockIdx.x * blockDim.y + threadIdx.y;
  auto B_T = XQ.size(0);
  int N_KVH = 0;
  if (update_kv) {
    N_KVH = XK.size(1);
  }
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
    qkv = QKV::K;
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
    qkv = QKV::V;
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

    if (qkv == QKV::Q || update_kv) {
      *reinterpret_cast<uint2*>(&dst_row[head_id]) =
          *reinterpret_cast<uint2*>(&src);
    }
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
    bool write_k_back = false,
    bool update_kv = true) {
  // Launch b_t_(sum(h)) warps.
  auto b_t_hh = blockIdx.x * blockDim.y + threadIdx.y;
  auto B_T = XQ.size(0);
  int N_KVH = 0;
  if (update_kv) {
    N_KVH = XK.size(1);
  } else {
    assert(!write_k_back);
  }
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
      if (update_kv) {
        *reinterpret_cast<uint2*>(&dst_row[head_id]) =
            *reinterpret_cast<uint2*>(&src);
      }
    } else { // qk requires rope
      if (update_kv || qkv == QKV::Q) {
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
            freqs_0 =
                (1 - smooth) * freqs_0 / scaling_factor + smooth * freqs_0;
          }
          double wavelen_1 = 2 * M_PI / freqs_1;
          if (wavelen_1 >= hi_freq_wavelen && wavelen_1 > lo_freq_wavelen) {
            freqs_1 = freqs_1 / scaling_factor;
          } else if (wavelen_1 >= hi_freq_wavelen) {
            double smooth = (old_context_len / wavelen_1 - lo_freq_factor) /
                (hi_freq_factor - lo_freq_factor);
            freqs_1 =
                (1 - smooth) * freqs_1 / scaling_factor + smooth * freqs_1;
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
        if (update_kv || qkv == QKV::Q) {
          *reinterpret_cast<uint2*>(&dst_row[head_id]) =
              *reinterpret_cast<uint2*>(&dst_);
        }

        if (write_k_back && qkv == QKV::K) {
          // Also write back to the source row
          *reinterpret_cast<uint2*>(&src_row[head_id]) =
              *reinterpret_cast<uint2*>(&dst_);
        }
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
  if (head == QKV::V || EmbMode == PositionEmbeddingMode::NOPE) {
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
DEVICE_INLINE void
quantize_int4_kv(fx4 dst, uint8_t* dst_row_q, bool do_norm = false) {
  if (do_norm) {
    per_row_norm(dst);
  }
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

#define CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL(                  \
    NUM_GROUPS,                                                              \
    DTYPE,                                                                   \
    EMB_MODE,                                                                \
    VARSEQ_BATCH,                                                            \
    VARSEQ_SEQPOS,                                                           \
    THETA,                                                                   \
    GAMMA,                                                                   \
    SCALE_BASE,                                                              \
    EXPO_OFFSET,                                                             \
    block_tables,                                                            \
    page_size,                                                               \
    block_tables_b_stride,                                                   \
    varseq_cache_seqpos,                                                     \
    actual_batch_size,                                                       \
    rope_scaling,                                                            \
    old_context_len,                                                         \
    scaling_factor,                                                          \
    lo_freq_factor,                                                          \
    hi_freq_factor,                                                          \
    write_k_back,                                                            \
    k_norm)                                                                  \
  rope_xpos_qkv_varseq_prefill_kernel_quantized<EMB_MODE, DTYPE, NUM_GROUPS> \
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(            \
          XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),    \
          XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),    \
          XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),    \
          cache_K.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),    \
          cache_V.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),    \
          qparam_k_ptr,                                                      \
          qparam_v_ptr,                                                      \
          XQ_O.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),  \
          VARSEQ_BATCH,                                                      \
          VARSEQ_SEQPOS,                                                     \
          THETA,                                                             \
          GAMMA,                                                             \
          SCALE_BASE,                                                        \
          EXPO_OFFSET,                                                       \
          block_tables,                                                      \
          page_size,                                                         \
          block_tables_b_stride,                                             \
          varseq_cache_seqpos,                                               \
          actual_batch_size,                                                 \
          rope_scaling,                                                      \
          old_context_len,                                                   \
          scaling_factor,                                                    \
          lo_freq_factor,                                                    \
          hi_freq_factor,                                                    \
          write_k_back,                                                      \
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
__global__ void rope_xpos_qkv_varseq_prefill_kernel_quantized(
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
  if (k_norm && qkv != QKV::V) {
    // fusing K/Q norm into the kernel
    per_row_norm(dst);
  }

  if (qkv == QKV::K && write_k_back) {
    // Also write back to the source row
    bfx4 dst_bf16 = fx4_to_bfx4(dst);
    *reinterpret_cast<uint2*>(&src_row[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&dst_bf16);
  }
  if (qkv == QKV::Q) {
    // write to dst_row
    bfx4 dst_bf16 = fx4_to_bfx4(dst);
    CUDA_KERNEL_ASSERT(uintptr_t(&dst_row[4 * threadIdx.x]) % 8 == 0);
    *reinterpret_cast<uint2*>(&dst_row[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&dst_bf16);
  } else {
    // quantize and write to dst_row
    auto D_H = XQ.size(2);
    auto D_H_q = cache_K.size(3);
    __half2* qparam_row = nullptr;
    if (kCacheDtype == CacheLogicalDtype::FP8) {
      if (qparam_k_ptr == nullptr) {
        CUDA_KERNEL_ASSERT(D_H_q - D_H == 4);
      } else {
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
      }
      quantize_fp8_kv(dst, dst_row_q, qparam_row);
    } else if (kCacheDtype == CacheLogicalDtype::INT4) {
      CUDA_KERNEL_ASSERT(D_H_q - D_H / 2 == 4 * KVQuantNumGroups);
      quantize_int4_kv<KVQuantNumGroups>(dst, dst_row_q);
    }
  }
}

template <
    PositionEmbeddingMode EmbMode,
    CacheLogicalDtype kCacheDtype,
    int KVQuantNumGroups = 1>
__global__ void rope_xpos_qkv_varseq_prefill_kernel_fp8(
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XQ, // [B_T][N_H][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XK, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor32<at::BFloat16, 3, at::RestrictPtrTraits>
        XV, // [B_T][N_KVH][D_H]
    at::PackedTensorAccessor64<at::Float8_e4m3fn, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H +4]
    at::PackedTensorAccessor64<at::Float8_e4m3fn, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H + 4]
    int32_t* qparam_k_ptr,
    int32_t* qparam_v_ptr,
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
    bool k_norm = false,
    float* amax = nullptr) {
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
  // dst_row_q dtype is fp8 not uint8_t
  at::Float8_e4m3fn* dst_row_q = nullptr;
  auto h = 0;
  QKV qkv;
  if (hh < N_H) {
    h = hh;
    src_row = &XQ[b_t][h][0];
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
  if (k_norm && qkv != QKV::V) {
    // fusing K/Q norm into the kernel
    per_row_norm(dst);
  }
  // Write back to the source row for Q and K
  if (qkv != QKV::V && write_k_back) {
    // Also write back to the source row
    bfx4 dst_bf16 = fx4_to_bfx4(dst);
    *reinterpret_cast<uint2*>(&src_row[4 * threadIdx.x]) =
        *reinterpret_cast<uint2*>(&dst_bf16);
  }
  // This kernel does not write to xq_o
  if (qkv != QKV::Q) {
    // only write to cache if batch lane has a pre-calculated qparam
    // quantize and write to dst_row
    CUDA_KERNEL_ASSERT(qparam_k_ptr != nullptr)
    // TODO: Change qparam ptr to float*
    __half2* qparam_row = nullptr;

    size_t idx = b * N_KVH + h;
    if (qkv == QKV::K) {
      qparam_row = reinterpret_cast<__half2*>(&qparam_k_ptr[idx]);
    } else {
      qparam_row = reinterpret_cast<__half2*>(&qparam_v_ptr[idx]);
    }
    quantize_fp8_kv<at::Float8_e4m3fn, KVQuantRecipe::perHeadScaling>(
        dst, dst_row_q, qparam_row);
  }
  // amax calculation for QKV
  // TODO: avoid amax calculation for KV if batch lane has a pre-calculated
  // qparam if (qkv != QKV::Q && is_precalculated_qparam[b_t] ){
  //   return;
  // }
  if (amax != nullptr) {
    // per_row_amax(dst, &amax[b_t * HH + hh]);
    per_head_amax(dst, &amax[b * HH + hh]);
  }
}

at::Tensor nope_qkv_varseq_prefill(
    at::Tensor XQ,
    std::optional<at::Tensor> XK_,
    std::optional<at::Tensor> XV_,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor varseq_batch,
    at::Tensor varseq_seqpos,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> varseq_cache_seqpos,
    int64_t cache_logical_dtype_int,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> qparam_k = std::nullopt,
    std::optional<at::Tensor> qparam_v = std::nullopt,
    bool k_norm = false,
    bool update_kv = true,
    std::optional<at::Tensor> amax_qkv = std::nullopt) {
  auto B_T = XQ.size(0);
  auto N_H = XQ.size(1);

  auto N_KVH = 0;

  at::Tensor XK, XV;
  if (!update_kv) {
    assert(XK_.has_value() == false);
    XK = at::empty_like(XQ);
    // at::zeros({0, 0, 0}, at::BFloat16); // at::zeros(0);
    XV = at::empty_like(XQ);
    // at::zeros({0, 0, 0}, at::BFloat16);
  } else {
    XK = XK_.value();
    XV = XV_.value();
    N_KVH = XK.size(1);
  }

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
  CacheLogicalDtype cache_logical_dtype =
      static_cast<CacheLogicalDtype>(cache_logical_dtype_int);
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
        nullptr,
        update_kv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto num_groups_ = num_groups ? num_groups.value() : 1;
    int32_t* qparam_k_ptr = nullptr;
    int32_t* qparam_v_ptr = nullptr;
    if (qparam_k.has_value()) {
      qparam_k_ptr = static_cast<int32_t*>(qparam_k.value().data_ptr());
      qparam_v_ptr = static_cast<int32_t*>(qparam_v.value().data_ptr());
    }
    auto varseq_batch_ = varseq_batch.data_ptr<int32_t>();
    auto varseq_seqpos_ =
        varseq_seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>();
    if (cache_logical_dtype == CacheLogicalDtype::FP8) {
#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
      CUDA_KERNEL_ASSERT(num_groups_ == 1);
      if (cache_K.dtype() == at::kFloat8_e4m3fn) {
        float* amax_ptr = nullptr;
        if (amax_qkv.has_value()) {
          amax_ptr = static_cast<float*>(amax_qkv.value().data_ptr());
        }
        rope_xpos_qkv_varseq_prefill_kernel_fp8<
            PositionEmbeddingMode::NOPE,
            CacheLogicalDtype::FP8,
            1><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<
                at::Float8_e4m3fn,
                4,
                at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<
                at::Float8_e4m3fn,
                4,
                at::RestrictPtrTraits>(),
            qparam_k_ptr,
            qparam_v_ptr,
            varseq_batch_,
            varseq_seqpos_,
            0,
            0,
            0,
            0,
            block_tables_ptr,
            page_size,
            block_tables_b_stride,
            varseq_cache_seqpos_
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            nullptr,
            false,
            0,
            0,
            0,
            0,
            true, // write_k_back and q too if we are doing norm.
            k_norm,
            amax_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL(
            1,
            CacheLogicalDtype::FP8,
            PositionEmbeddingMode::NOPE,
            varseq_batch_,
            varseq_seqpos_,
            0,
            0,
            0,
            0,
            block_tables_ptr,
            page_size,
            block_tables_b_stride,
            (varseq_cache_seqpos_
                 .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
            nullptr,
            false,
            0,
            0,
            0,
            0,
            false,
            k_norm);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
#else
      throw std::runtime_error("CUDA version is older than 12.0");
#endif
    } else {
      CALL_INT4_KERNEL_WITH_KV_GROUPWISE_QUANT_CHECK(
          CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL,
          num_groups_,
          CacheLogicalDtype::INT4,
          PositionEmbeddingMode::NOPE,
          varseq_batch_,
          varseq_seqpos_,
          0,
          0,
          0,
          0,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (varseq_cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          nullptr,
          false,
          0,
          0,
          0,
          0,
          false,
          k_norm);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  return XQ_O;
}

at::Tensor nope_qkv_decoding(
    at::Tensor XQ,
    std::optional<at::Tensor> XK_,
    std::optional<at::Tensor> XV_,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seqpos,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> actual_batch_size,
    std::optional<at::Tensor> batch,
    std::optional<at::Tensor> cache_seqpos,
    int64_t cache_logical_dtype_int,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> qparam_k = std::nullopt,
    std::optional<at::Tensor> qparam_v = std::nullopt,
    bool k_norm = false,
    bool update_kv = true,
    std::optional<at::Tensor> amax_qkv = std::nullopt) {
  auto B = XQ.size(0);
  auto N_H = XQ.size(1);
  // auto N_KVH = XK.size(1);
  auto N_KVH = 0;
  at::Tensor XK, XV;
  if (!update_kv) {
    assert(XK_.has_value() == false);
    XK = at::empty_like(XQ);
    // at::zeros({0, 0, 0}, at::BFloat16); // at::zeros(0);
    XV = at::empty_like(XQ);
    // at::zeros({0, 0, 0}, at::BFloat16);
  } else {
    XK = XK_.value();
    XV = XV_.value();
    N_KVH = XK.size(1);
  }

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
  CacheLogicalDtype cache_logical_dtype =
      static_cast<CacheLogicalDtype>(cache_logical_dtype_int);
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
        actual_batch_size_ptr,
        update_kv);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto num_groups_ = num_groups ? num_groups.value() : 1;
    int32_t* qparam_k_ptr = nullptr;
    int32_t* qparam_v_ptr = nullptr;
    if (qparam_k.has_value()) {
      qparam_k_ptr = static_cast<int32_t*>(qparam_k.value().data_ptr());
      qparam_v_ptr = static_cast<int32_t*>(qparam_v.value().data_ptr());
    }
    auto batch_ =
        batch.has_value() ? batch.value().data_ptr<int32_t>() : nullptr;
    auto seqpos_ =
        seqpos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>();
    if (cache_logical_dtype == CacheLogicalDtype::FP8) {
#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
      CUDA_KERNEL_ASSERT(num_groups_ == 1);
      if (cache_K.dtype() == at::kFloat8_e4m3fn) {
        float* amax_ptr = nullptr;
        if (amax_qkv.has_value()) {
          amax_ptr = static_cast<float*>(amax_qkv.value().data_ptr());
        }
        rope_xpos_qkv_varseq_prefill_kernel_fp8<
            PositionEmbeddingMode::NOPE,
            CacheLogicalDtype::FP8,
            1><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<
                at::Float8_e4m3fn,
                4,
                at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<
                at::Float8_e4m3fn,
                4,
                at::RestrictPtrTraits>(),
            qparam_k_ptr,
            qparam_v_ptr,
            batch_,
            seqpos_,
            0,
            0,
            0,
            0,
            block_tables_ptr,
            page_size,
            block_tables_b_stride,
            cache_seqpos_
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            actual_batch_size_ptr,
            false,
            0,
            0,
            0,
            0,
            true, // write_k_back and q too if we are doing norm.
            k_norm,
            amax_ptr);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

      } else {
        CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL(
            1,
            CacheLogicalDtype::FP8,
            PositionEmbeddingMode::NOPE,
            batch_,
            seqpos_,
            0,
            0,
            0,
            0,
            block_tables_ptr,
            page_size,
            block_tables_b_stride,
            (cache_seqpos_
                 .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
            actual_batch_size_ptr,
            false,
            0,
            0,
            0,
            0,
            false,
            k_norm);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
#else
      throw std::runtime_error("CUDA version is older than 12.0");
#endif
    } else {
      CALL_INT4_KERNEL_WITH_KV_GROUPWISE_QUANT_CHECK(
          CALL_ROPE_XPOS_QKV_VARSEQ_PREFILL_GROUPWISE_KERNEL,
          num_groups_,
          CacheLogicalDtype::INT4,
          PositionEmbeddingMode::NOPE,
          batch_,
          seqpos_,
          0,
          0,
          0,
          0,
          block_tables_ptr,
          page_size,
          block_tables_b_stride,
          (cache_seqpos_
               .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()),
          actual_batch_size_ptr,
          false,
          0,
          0,
          0,
          0,
          false,
          k_norm);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  return XQ_O;
}

at::Tensor rope_qkv_varseq_prefill(
    at::Tensor XQ,
    std::optional<at::Tensor> XK_,
    std::optional<at::Tensor> XV_,
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
    bool k_norm = false,
    bool update_kv = true,
    std::optional<at::Tensor> amax_qkv = std::nullopt) {
  auto B_T = XQ.size(0);
  auto N_H = XQ.size(1);
  auto N_KVH = 0;

  at::Tensor XK, XV;
  if (!update_kv) {
    assert(XK_.has_value() == false);
    XK = at::empty_like(XQ);
    // at::zeros({0, 0, 0}, at::BFloat16); // at::zeros(0);
    XV = at::empty_like(XQ);
    // at::zeros({0, 0, 0}, at::BFloat16);
  } else {
    XK = XK_.value();
    XV = XV_.value();
    N_KVH = XK.size(1);
  }

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
            write_k_back,
            update_kv);
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
      if (cache_K.dtype() == at::kFloat8_e4m3fn) {
        float* amax_ptr = nullptr;
        if (amax_qkv.has_value()) {
          amax_ptr = static_cast<float*>(amax_qkv.value().data_ptr());
        }
        rope_xpos_qkv_varseq_prefill_kernel_fp8<
            PositionEmbeddingMode::ROPE,
            CacheLogicalDtype::FP8,
            1><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<
                at::Float8_e4m3fn,
                4,
                at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<
                at::Float8_e4m3fn,
                4,
                at::RestrictPtrTraits>(),
            qparam_k_ptr,
            qparam_v_ptr,
            varseq_batch_,
            varseq_seqpos_,
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
            true,
            k_norm,
            amax_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

      } else {
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
      }
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
          k_norm);

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
    std::optional<at::Tensor> qparam_k = std::nullopt,
    std::optional<at::Tensor> qparam_v = std::nullopt) {
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
    std::optional<at::Tensor> XK_,
    std::optional<at::Tensor> XV_,
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
    bool k_norm = false,
    bool update_kv = true,
    std::optional<at::Tensor> amax_qkv = std::nullopt) {
  auto B = XQ.size(0);
  auto N_H = XQ.size(1);
  auto N_KVH = 0;
  at::Tensor XK, XV;
  if (!update_kv) {
    assert(XK_.has_value() == false);
    XK = at::empty_like(XQ);
    // at::zeros({0, 0, 0}, at::BFloat16); // at::zeros(0);
    XV = at::empty_like(XQ);
    // at::zeros({0, 0, 0}, at::BFloat16);
  } else {
    XK = XK_.value();
    XV = XV_.value();
    N_KVH = XK.size(1);
  }

  TORCH_CHECK(XQ.size(2) % 4 == 0);
  int32_t num_warps = B * (2 * N_KVH + N_H);
  TORCH_CHECK(num_warps > 0);

  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);
  dim3 blocks(cuda_calc_xblock_count(num_warps, kWarpsPerBlock));
  // TODO: not needed for fp8 attn case
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
            hi_freq_factor,
            false,
            update_kv);
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
      if (cache_K.dtype() == at::kFloat8_e4m3fn) {
        float* amax_ptr = nullptr;
        if (amax_qkv.has_value()) {
          amax_ptr = static_cast<float*>(amax_qkv.value().data_ptr());
        }
        rope_xpos_qkv_varseq_prefill_kernel_fp8<
            PositionEmbeddingMode::ROPE,
            CacheLogicalDtype::FP8,
            1><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            XQ.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XK.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            XV.packed_accessor32<at::BFloat16, 3, at::RestrictPtrTraits>(),
            cache_K.packed_accessor64<
                at::Float8_e4m3fn,
                4,
                at::RestrictPtrTraits>(),
            cache_V.packed_accessor64<
                at::Float8_e4m3fn,
                4,
                at::RestrictPtrTraits>(),
            qparam_k_ptr,
            qparam_v_ptr,
            batch.has_value() ? batch.value().data_ptr<int32_t>() : nullptr,
            seqpos_,
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
            hi_freq_factor,
            true,
            k_norm,
            amax_ptr);

        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
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
      }
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
          k_norm);

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
    std::optional<at::Tensor> qparam_k = std::nullopt,
    std::optional<at::Tensor> qparam_v = std::nullopt) {
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
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<uint8_t, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<uint32_t, 4, at::RestrictPtrTraits> qparam_K,
    at::PackedTensorAccessor64<uint32_t, 4, at::RestrictPtrTraits> qparam_V) {
  auto N_KVH = cache_K.size(2);
  auto MAX_T = cache_K.size(1);
  auto D_H = cache_K.size(3);
  CUDA_KERNEL_ASSERT(D_H == 128);

  auto b = blockIdx.x;
  int h = 0, t = 0;
  uint8_t* head;
  __half* qparam_scale_ptr;

  for (auto t_h = threadIdx.y + blockIdx.y * blockDim.y; t_h < MAX_T * N_KVH;
       t_h += blockDim.y * gridDim.y) {
    h = t_h % N_KVH;
    t = t_h / N_KVH;

    auto tidx = threadIdx.x;
    if (threadIdx.x < 32) {
      head = &cache_K[b][t][h][0];
      qparam_scale_ptr = reinterpret_cast<__half*>(&qparam_K[b][t][h][0]);
    } else {
      head = &cache_V[b][t][h][0];
      qparam_scale_ptr = reinterpret_cast<__half*>(&qparam_V[b][t][h][0]);
      tidx -= 32;
    }
    auto D_H_idx = tidx * 4; // Reading 4 bytes at once.

    // Our only goal here is to detect negative zeros that are valid
    // in e4m3fn, but not valid in e4m3fnuz, and overwrite them with zeros.
    // E.g. 0x80708070u ^ 0x80808080u = 0xf000f0u & 0x80708070u = 0x00700070u
    uint32_t packed_fp8x4_vals = *reinterpret_cast<uint32_t*>(&head[D_H_idx]);
    *reinterpret_cast<uint32_t*>(&head[D_H_idx]) =
        (packed_fp8x4_vals ^ 0x80808080) & packed_fp8x4_vals;

    // Multiply qparam scale (member x) by 2 to compensate for the exponent
    // bias difference (1) between e4m3fn and e4m3fnuz. We only need to do this
    // once per row. In debug mode, assert that 2.0*scale as a float would not
    // exceed the max value of __half.
    if (tidx == 0) {
      CUDA_KERNEL_ASSERT(
          __half2float(*qparam_scale_ptr) * 2.0f <=
          __half2float(std::numeric_limits<__half>::max()));
      *qparam_scale_ptr = __float2half(__half2float(*qparam_scale_ptr) * 2.0f);
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

  auto B = cache_K.size(0);

  constexpr int32_t kMaxBlocks = 1;
  dim3 blocks(B, std::max<int32_t>(1, kMaxBlocks / B));
  dim3 threads(kThreadsPerWarp, kWarpsPerBlock);

  convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      cache_K.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
      cache_V.packed_accessor64<uint8_t, 4, at::RestrictPtrTraits>(),
      qparam_K.packed_accessor64<uint32_t, 4, at::RestrictPtrTraits>(),
      qparam_V.packed_accessor64<uint32_t, 4, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
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
  CUDA_KERNEL_ASSERT(0 && "unimplemented");
}

#else
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

// Function to convert and pack a single component
DEVICE_INLINE uint32_t
convertAndPack(float component, float inv_scale, float shift = 0.0) {
  auto val = (component - shift) * inv_scale;
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
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        q_seqstarts, // [B+1]
    at::PackedTensorAccessor64<at::Float8_e4m3fn, 3, at::RestrictPtrTraits>
        XQ_O, // [B_T][N_H][D]
    at::PackedTensorAccessor64<at::Float8_e4m3fn, 4, at::RestrictPtrTraits>
        cache_K, // [B][MAX_T][N_KVH][D_H]
    at::PackedTensorAccessor64<at::Float8_e4m3fn, 4, at::RestrictPtrTraits>
        cache_V, // [B][MAX_T][N_KVH][D_H]
    float* const scale_q,
    float* const scale_k,
    float* const scale_v,
    float kv_multiplier = 64.f) {
  extern __shared__ float
      shared_max[]; // Shared memory for block-level reduction
  // Launch one warp per token. Each thread handles 4 elements.
  // warps = B_T * HH
  auto b_t_hh = blockIdx.x * blockDim.y + threadIdx.y;
  auto N_KVH = cache_K.size(2);
  auto N_H = XQ_O.size(1);

  auto B_T = XQ_O.size(0);
  // TODO: Support N_KVH > 1
  CUDA_KERNEL_ASSERT(N_KVH == 1);

  // TODO: Support amax_head of size [B, N_H] for decode case
  // auto HH = (scale_k == nullptr) ? N_H : N_KVH * 2 + N_H;
  auto HH = N_H + N_KVH * 2;

  auto hh = b_t_hh % HH;
  auto b_t = b_t_hh / HH;

  // Boundary check
  if (b_t >= B_T || hh >= HH)
    return;

  auto seqpos_t = varseq_seqpos[b_t];
  auto b = varseq_batch ? varseq_batch[b_t] : b_t;

  int h = 0;
  float* qparam = nullptr;
  QKV qkv;
  at::Float8_e4m3fn* dst_row_q = nullptr;
  if (hh < N_H) {
    qkv = QKV::Q;
    h = hh;
    qparam = scale_q + b * N_KVH + hh / (N_H / N_KVH);
    dst_row_q = &XQ_O[b_t][h][0];
  } else if (hh < N_H + N_KVH) {
    qkv = QKV::K;
    h = hh - N_H;
    if (scale_k == nullptr)
      return;
    qparam = scale_k + b * N_KVH + h;
    dst_row_q = &cache_K[b][seqpos_t][h][0];
  } else {
    qkv = QKV::V;
    h = hh - N_H - N_KVH;
    if (scale_k == nullptr)
      return;
    qparam = scale_v + b * N_KVH + h;
    dst_row_q = &cache_V[b][seqpos_t][h][0];
  }
  // Skip quantization if scale is pre-calculated for K/V
  // as in decode and partial prefill cases
  int start = q_seqstarts[b];
  if (qkv != QKV::Q && varseq_seqpos[start] > 0)
    return;

  CUDA_KERNEL_ASSERT(uintptr_t(qparam) % 4 == 0);
  float val = 0;
  if (qkv == QKV::Q) {
    // assert N_KVH == 1
    for (int _h = 0; _h < N_H; _h++) {
      val = fmaxf(val, xqkv_amax_head[b * HH + _h]);
    }
    val *= 8; // [Experimental] improves accuracy
  } else {
    val = kv_multiplier * xqkv_amax_head[b * HH + hh];
  }
  // Calculate scaling factor
  constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX::value * 512.f);
  val = fminf(val, 12000);
  float scale = fmaxf(val / FP8_E4M3_MAX::value, min_scaling_factor);
  bool is_first_token =
      (b_t == 0 || !varseq_batch || varseq_batch[b_t - 1] != b);
  if (threadIdx.x == 0 && h == 0 && is_first_token) {
    *qparam = scale;
  }
  // Write scaling factor
  auto inv_scale = 1 / scale;
  CUDA_KERNEL_ASSERT(uintptr_t(&xqkv[b_t * HH * D_H + hh * D_H]) % 16 == 0);
  at::BFloat16* src_row = &xqkv[b_t * HH * D_H + hh * D_H];

  // Convert and pack data
  bfx4 src;
  fx4 dst;
  uint32_t x_bits[4];
  auto d = threadIdx.x * 4;
  CUDA_KERNEL_ASSERT(uintptr_t(&src_row[d]) % 8 == 0);
  // 8 bytes are 4 elements of type bf16
  *reinterpret_cast<uint2*>(&src) = *reinterpret_cast<uint2*>(&src_row[d]);
  dst = bfx4_to_fx4(src);
  x_bits[0] = convertAndPack(dst.x, inv_scale);
  x_bits[1] = convertAndPack(dst.y, inv_scale);
  x_bits[2] = convertAndPack(dst.z, inv_scale);
  x_bits[3] = convertAndPack(dst.w, inv_scale);
  uint32_t packed = packComponents(x_bits);
  CUDA_KERNEL_ASSERT(uintptr_t(&dst_row_q[d]) % 4 == 0);
  *reinterpret_cast<uint32_t*>(&dst_row_q[d]) = packed;
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
  auto B_T = XQ_O.size(0);
  auto N_KVH_L = cache_K.size(2);
  auto N_H_L = XQ_O.size(1);
  auto HH = N_H_L + N_KVH_L * 2;
  float* qparam_k_ptr = nullptr;
  float* qparam_v_ptr = nullptr;
  if (qparam_k.has_value()) {
    // prefill case
    // HH += N_KVH_L * 2;
    qparam_k_ptr = qparam_k.value().data_ptr<float>();
    qparam_v_ptr = qparam_v.value().data_ptr<float>();
  }
  auto num_warps = B_T * HH;
  dim3 block_size(kThreadsPerWarp, kWarpsPerBlock);
  dim3 grid_size(cuda_calc_xblock_count(num_warps, kWarpsPerBlock));

  auto scale_q = at::zeros(
      {q_seqstarts.size(0) - 1, N_KVH_L}, XQ_O.options().dtype(at::kFloat));
  float* const scale_q_ptr = scale_q.data_ptr<float>();
  // Launch the kernel
  // TODO: Launch the kernel with B_T * N_H_L blocks only in case of decode.
  // Currently, we are launching the kernel with B_T * HH blocks for decode
  // and KV blocks just return when qparam_k is passed as nullptr.
  quantizeQKVPerHead<<<
      grid_size,
      block_size,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      xqkv_amax_row.data_ptr<float>(),
      xqkv.data_ptr<at::BFloat16>(),
      varseq_seqpos.data_ptr<int32_t>(),
      varseq_batch.has_value() ? varseq_batch.value().data_ptr<int32_t>()
                               : nullptr, // not needed for decode
      q_seqstarts.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      XQ_O.packed_accessor64<at::Float8_e4m3fn, 3, at::RestrictPtrTraits>(),
      cache_K.packed_accessor64<at::Float8_e4m3fn, 4, at::RestrictPtrTraits>(),
      cache_V.packed_accessor64<at::Float8_e4m3fn, 4, at::RestrictPtrTraits>(),
      scale_q_ptr,
      qparam_k_ptr,
      qparam_v_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scale_q;
}

template <typename T, KVQuantRecipe recipe>
DEVICE_INLINE void
quantize_fp8_kv(fx4 dst, T* dst_row_q, __half2* qparam, bool do_norm) {
  if (do_norm) {
    per_row_norm(dst);
  }
  float shift, scale;
  int32_t fp8_qparam_offset = 0;
  if (qparam == nullptr) {
    fp8_qparam_offset = 4;
  }
  if (recipe == KVQuantRecipe::perTokenScaling) {
    auto thread_min = fminf(fminf(fminf(dst.x, dst.y), dst.z), dst.w);
    auto thread_max = fmaxf(fmaxf(fmaxf(dst.x, dst.y), dst.z), dst.w);

    float warp_min, warp_max;
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
    scale = static_cast<float>(
        std::max(bounded_max / FP8_E4M3_MAX::value, min_scaling_factor));
    shift = warp_min + FP8_E4M3_MAX::value * scale;
  } else {
    // Support of per-head scaling is limited to reading a
    // pre-calculated scale from qparam tensor and using it for scaling the
    // row.
    scale = *reinterpret_cast<float*>(qparam);
    shift = 0.0f;
  }
  float inv_scale = 1 / scale;
  uint32_t x_bits[4];
  x_bits[0] = convertAndPack(dst.x, inv_scale, shift);
  x_bits[1] = convertAndPack(dst.y, inv_scale, shift);
  x_bits[2] = convertAndPack(dst.z, inv_scale, shift);
  x_bits[3] = convertAndPack(dst.w, inv_scale, shift);
  uint32_t packed = packComponents(x_bits);
  CUDA_KERNEL_ASSERT(
      uintptr_t(&dst_row_q[4 * threadIdx.x + fp8_qparam_offset]) % 4 == 0);

  *reinterpret_cast<uint32_t*>(
      &dst_row_q[4 * threadIdx.x + fp8_qparam_offset]) = packed;
  // write qparams
  if (threadIdx.x == 0 && recipe == KVQuantRecipe::perTokenScaling) {
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
template <typename T, KVQuantRecipe recipe>
DEVICE_INLINE void
quantize_fp8_kv(fx4 dst, T* dst_row_, __half2* qparam, bool do_norm) {}
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
