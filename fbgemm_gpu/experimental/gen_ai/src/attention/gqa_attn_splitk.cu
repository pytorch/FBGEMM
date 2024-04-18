/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#if !(                                                  \
    defined(USE_ROCM) ||                                \
    ((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#elif (defined(USE_ROCM))
#include <hip/hip_bfloat16.h> // @manual
#endif

#ifndef USE_ROCM
#include <mma.h>
#endif

#if (                         \
    defined(__CUDA_ARCH__) && \
    ((__CUDA_ARCH__ == 800) || (__CUDA_ARCH__ == 900)))
#define USE_WMMA_FRAG
#endif

#ifdef USE_ROCM
constexpr int32_t kThreadsPerWarp = 64;
constexpr int32_t kWarpsPerBlock = 16;
#else
constexpr int32_t kThreadsPerWarp = 32;
constexpr int32_t kWarpsPerBlock = 32;
#endif

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))
#define FINAL_MASK 0xffffffff

namespace fbgemm_gpu::gen_ai::attention {

constexpr int32_t D_H = 128;
constexpr int32_t MAX_T = 16384;
constexpr int SMEM_ADJUST_THRESHOLD = 48 * 1024;

constexpr int kMaxHeads = 8;
// Fragments shapes used for wmma tensor core operations
constexpr int F_M = 8, F_N = 32, F_K = 16;
constexpr int SMEM_K_PAD = 2;
constexpr int SMEM_V_PAD = 2;
constexpr int SMEM_K_STRIDE = F_K + SMEM_K_PAD;
constexpr int SMEM_V_STRIDE = F_N + SMEM_V_PAD;

// Use fewer warps for gqa_attn_splitk_wmma_kernel
constexpr int32_t kSplitKWarpsPerBlock = 4;

namespace {

static __host__ DEVICE_INLINE int32_t div_up(int32_t a, int32_t b) {
  return (a + b - 1) / b;
};

static __host__ DEVICE_INLINE int32_t round_up(int32_t a, int32_t b) {
  return ((a + b - 1) / b) * b;
}

template <typename func_t>
void set_gpu_max_dynamic_shared_memory(
    func_t kernel,
    const int smem_bytes,
    const int device) {
  // V100: 96 KB; A100: 160 KB; H100: 228 KB.
  int max_shared_bytes = 0;
  cudaDeviceGetAttribute(
      &max_shared_bytes,
#ifndef __HIP_PLATFORM_AMD__
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
#else
      hipDeviceAttributeMaxSharedMemoryPerBlock,
#endif
      device);
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

// TODO: Include the following code from fbgemm_gpu header
struct __align__(16) bfx8 {
  __nv_bfloat162 vals[4];
};

struct __align__(16) halfx8 {
  __half2 vals[4];
};

// Reinterpret a  pair of uint16_t (packed into a uint32_t) as half2, and
// multiply by rhs.
DEVICE_INLINE __half2 hmul_short2(uint32_t lhs, __half rhs) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
#ifndef __HALF2_TO_UI
// cuda_fp16.hpp
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif
#ifndef __HALF2_TO_CUI
// cuda_fp16.hpp
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int*>(&(var)))
#endif
  __half2 ret;
  __half2 rhsp = make_half2(rhs, rhs);
  asm("mul.f16x2 %0, %1, %2;"
      : "=r"(__HALF2_TO_UI(ret))
      : "r"(__HALF2_TO_CUI(lhs)), "r"(__HALF2_TO_CUI(rhsp)));
  return ret;
#else
#ifndef __HALF2_TO_UI
// cuda_fp16.hpp
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif
  __half2 lhs_h2;
  __HALF2_TO_UI(lhs_h2) = lhs;
  float2 fx = __half22float2(lhs_h2);
  float2 fy = __half22float2(make_half2(rhs, rhs));
  float2 fr;
  fr.x = fx.x * fy.x;
  fr.y = fx.y * fy.y;
  return __float22half2_rn(fr);
#endif
}

__forceinline__ __device__ bfx8
dequantize_permuted_int4(uint32_t packedVals, __half2 shift_scale) {
  halfx8 res;
  uint32_t v = packedVals;
  // What's going on here, you might ask? We extra out 4-bit pairs of integers
  // as 2xuint16 packed into an int32 via the mask operation, and then we
  // convert them to half precision values. As these are all integers in [0,
  // 15], we can actually just interpret the 4-bit integer values as
  // half-precision values. We multiply by 4096 x 4096 to go from the 4-bit
  // representation to the equivalent fp16 value, or alternatively 32768 * 512
  // (or 32 when we have shifted the 4-bit value up). See e.g.
  // https://gist.github.com/ajtulloch/021254a291a95966bc509db4e34ffeff for a
  // NumPy implementation. We do this dance because: a) doing bitwise operations
  // on each 4-bit value is expensive on the ALU, and 4-bit to half is expensive
  // on the XU. b) doing a 256-entry shared memory LUT on 8-bit pairs is
  // expensive on SMEM throughput. Credit to @jhj.
  res.vals[0] = hmul_short2(v & 0x000F000F, __float2half(32768));
  res.vals[1] = hmul_short2(v & 0x00F000F0, __float2half(32768));
  v >>= 8;
  res.vals[2] = hmul_short2(v & 0x000F000F, __float2half(32768));
  res.vals[3] = hmul_short2(v & 0x00F000F0, __float2half(32768));

  // ~5% perf gain is observed with the explicit type conversions using
  // __float2half on Nvidia A100 GPUs (https://fburl.com/diff/ss8372zw) using
  // NVCC 11.0. Additionally, HIP compiler requires these explicit type
  // conversions.
  half shift_scale_x = __low2half(shift_scale);
  half shift_scale_y = __high2half(shift_scale);

  // now, dequantize
  auto shifts = __half2(shift_scale_y, shift_scale_y);
  auto scales_lower_temp = __hmul(shift_scale_x, __float2half(512));
  auto scales_lower = __half2(scales_lower_temp, scales_lower_temp);
  auto scales_upper_temp = __hmul(shift_scale_x, __float2half(32));
  auto scales_upper = __half2(scales_upper_temp, scales_upper_temp);

  auto r0 = __half22float2(__hfma2(res.vals[0], scales_lower, shifts));
  auto r1 = __half22float2(__hfma2(res.vals[1], scales_upper, shifts));
  auto r2 = __half22float2(__hfma2(res.vals[2], scales_lower, shifts));
  auto r3 = __half22float2(__hfma2(res.vals[3], scales_upper, shifts));

  bfx8 result;
  result.vals[0] = __floats2bfloat162_rn(r0.x, r1.x);
  result.vals[1] = __floats2bfloat162_rn(r2.x, r3.x);
  result.vals[2] = __floats2bfloat162_rn(r0.y, r1.y);
  result.vals[3] = __floats2bfloat162_rn(r2.y, r3.y);

  return result;
}

template <
    typename kv_t,
    int KVQuantNumGroups = 1,
    typename kv_load_t = uint32_t>
__global__ void __launch_bounds__(kThreadsPerWarp* kSplitKWarpsPerBlock, 1)
    gqa_attn_splitk_wmma_kernel(
        const at::PackedTensorAccessor32<at::BFloat16, 4, at::RestrictPtrTraits>
            XQ,
        const at::PackedTensorAccessor64<kv_t, 4, at::RestrictPtrTraits>
            cache_K,
        const at::PackedTensorAccessor64<kv_t, 4, at::RestrictPtrTraits>
            cache_V,
        at::PackedTensorAccessor32<float, 4, at::RestrictPtrTraits> out_splitK,
        const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
            seq_positions,
        at::PackedTensorAccessor32<float, 4, at::RestrictPtrTraits> metadata,
        float qk_scale) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  // Need kWarpsPerBlock == blockDim.y;
  // Need D_H == 128
  static_assert(kWarpsPerBlock <= kThreadsPerWarp, "");

  extern __shared__ __align__(16) float smem[];

  // Each block handles a single query, split-K partition, and max of 8 query
  // heads
  const int32_t b = blockIdx.x;
  // Head block
  const int32_t h_block = blockIdx.y;
  // Split-K block
  const int32_t s_block = blockIdx.z;

  const int32_t H_max = XQ.size(2);
  const int32_t num_split_ks = gridDim.z;
  const int32_t warp_idx = threadIdx.y;

  // Note: this is decoding case where we attent to current and all previous
  // tokens.
  const auto t_max = seq_positions[b] + 1;

  // Assume cache_K/cache_V is contiguous
  const auto* cache_K_base = &cache_K[b][0][0][0];
  const auto* cache_V_base = &cache_V[b][0][0][0];
  constexpr bool USE_INT4 = std::is_same<kv_t, uint8_t>::value;

  // Only used for int4
  constexpr int32_t INT4_PARAM_BYTES = 4 * KVQuantNumGroups;
  constexpr int32_t D_H_bytes = D_H / 2 + INT4_PARAM_BYTES;
  constexpr int32_t INT4_GROUP_SIZE = D_H / KVQuantNumGroups;

  // Compute S[MAX_T] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
  // Split T across warps in a block.
  const int32_t t_total = round_up(t_max, num_split_ks);
  const int32_t t_per_block = t_total / num_split_ks;
  const int32_t t_per_block_start = t_per_block * s_block;
  const int32_t t_per_block_end = min(t_per_block * (s_block + 1), t_max);
  const int32_t t_total_per_block = t_per_block_end - t_per_block_start;

  // Compute start and end heads
  const int32_t h_per_block_start = kMaxHeads * h_block;
  const int32_t h_per_block_end = min(kMaxHeads * (h_block + 1), H_max);
  const int32_t h_total_per_block = h_per_block_end - h_per_block_start;

  // Early return if there is no work to do
  if (t_total_per_block <= 0) {
    return;
  }

  using namespace nvcuda;
  // Number of vectors for K and V (vector type in this case is uint32_t)
  constexpr int KV_NUM_VECS = 2;
  // Number of elements to load when using the kv_load_t type (kv_load_t is 32
  // bits for KVQuantNumGroups = 1 and 64 bits for KVQuantNumGroups = 4)
  constexpr int KV_LD_NUM_ELS =
      (KV_NUM_VECS * sizeof(uint32_t)) / sizeof(kv_load_t);

  wmma::fragment<wmma::matrix_a, F_M, F_N, F_K, __nv_bfloat16, wmma::row_major>
      q_frag;
  wmma::fragment<wmma::matrix_b, F_M, F_N, F_K, __nv_bfloat16, wmma::col_major>
      k_frag;
  wmma::fragment<wmma::accumulator, F_M, F_N, F_K, float> c_frag;

  // Get shared memory pointers
  static_assert(
      F_N >= F_K, "F_N must be >= F_K because we allocate smem based on F_N");
  const int ldc = round_up(t_total_per_block, F_N);
  auto smem_max = smem + max(h_total_per_block, F_M) * ldc;
  __nv_bfloat16* smem_staging = reinterpret_cast<__nv_bfloat16*>(
      smem_max + max(h_total_per_block, F_M) * kSplitKWarpsPerBlock);
  float* smem_out = reinterpret_cast<float*>(
      smem_staging +
      kSplitKWarpsPerBlock * max(F_N * SMEM_K_STRIDE, F_K * SMEM_V_STRIDE));
  constexpr float NEG_FINF = -std::numeric_limits<float>::infinity();

#ifdef USE_WMMA_FRAG
  // The kernel can compute max_qk directly from the WMMA fragment on A100/H100
  // Each thread handles 2 heads according to the tensor core layout
  constexpr int HEADS_PER_THREAD_QK = 2;
  float max_qk[HEADS_PER_THREAD_QK];
  max_qk[0] = NEG_FINF;
  max_qk[1] = NEG_FINF;
#else
  // TODO: Support computing max_qk from the WMMA fragment on other GPUs
  float max_qk = NEG_FINF;
#endif

  // Compute Q @ K^T
  for (auto t_start = t_per_block_start + warp_idx * F_N;
       t_start < t_per_block_end;
       t_start += kSplitKWarpsPerBlock * F_N) {
    constexpr int32_t K_UNROLLS = 4;
    __half2 k_scales;
    kv_load_t k_vals[KV_LD_NUM_ELS * K_UNROLLS];

    // Init the accumulator with zeros
    wmma::fill_fragment(c_frag, 0.0f);

    // Intra-warp reduction within across D_H
    for (auto d_start = 0; d_start < D_H; d_start += F_K) {
      if (USE_INT4 && d_start % INT4_GROUP_SIZE == 0) {
        // Load K scales for INT4 K
        // Each thread operates on a single row (T dim). Columns are split into
        // KVQuantNumGroups groups and each group has the same K scales
        if (t_start + threadIdx.x < min(t_start + F_N, t_per_block_end)) {
          auto* k_ = cache_K_base + (t_start + threadIdx.x) * D_H_bytes;
          const int group_id = d_start / INT4_GROUP_SIZE;
          k_scales = reinterpret_cast<const __half2*>(k_)[group_id];
        }
      }

      // Load Q fragment
      wmma::load_matrix_sync(
          q_frag,
          reinterpret_cast<const __nv_bfloat16*>(
              &XQ[b][0][h_per_block_start][d_start]),
          D_H);

      // Load K fragment
      if (USE_INT4) {
        // Load and dequantize INT4 K
        // Each thread loads 16 columns (D dim) from one row (T dim).
        // Each row is handled by one thread.
        const auto t = t_start + threadIdx.x;
        const auto t_scope = min(t_start + F_N, t_per_block_end);

        // Prefetch 4 sets of Ks (load every 4 d_starts)
        if (d_start % (K_UNROLLS * F_K) == 0) {
          // Since F_N = 32, each thread handles only one row (T dim). Thus a
          // for-loop is not required
          if (t < t_scope) {
            // Ratio between the INT4 bytes and kv_load_t bytes
            constexpr int KV_LOAD_T_INT4_RATIO = 2 * sizeof(kv_load_t);
            const auto k_offset =
                t * D_H_bytes + INT4_PARAM_BYTES + d_start / 2;
            const auto* cache_k_ =
                reinterpret_cast<const kv_load_t*>(cache_K_base + k_offset);
#pragma unroll K_UNROLLS
            for (int k_unroll = 0; k_unroll < K_UNROLLS; k_unroll++) {
              auto* k_vals_ = k_vals + k_unroll * KV_LD_NUM_ELS;
              const auto* k_ =
                  cache_k_ + ((k_unroll * F_K) / KV_LOAD_T_INT4_RATIO);
#pragma unroll KV_LD_NUM_ELS
              for (auto k_i = 0; k_i < KV_LD_NUM_ELS; k_i++) {
                k_vals_[k_i] = k_[(k_i * 8) / KV_LOAD_T_INT4_RATIO];
              }
            }
          }
        }

        if (t < t_scope) {
          // Shift pointers
          const auto k_offset =
              ((d_start % (K_UNROLLS * F_K)) / F_K) * KV_NUM_VECS;
          const auto smem_offset =
              (warp_idx * F_N + t - t_start) * SMEM_K_STRIDE;
          const auto* k_vals_ = reinterpret_cast<uint32_t*>(k_vals) + k_offset;
          auto* smem_staging_ = smem_staging + smem_offset;
#pragma unroll KV_NUM_VECS
          for (int vec = 0; vec < KV_NUM_VECS; ++vec) {
            // Dequantize 8 INT4s to 8 BF16s and store the results in shared
            // memory
            const auto k_deq = dequantize_permuted_int4(k_vals_[vec], k_scales);
            auto* smem_s =
                reinterpret_cast<__nv_bfloat162*>(smem_staging_ + vec * 8);
#pragma unroll
            for (int i = 0; i < 4; i++) {
              smem_s[i] = k_deq.vals[i];
            }
          }
        }
        // Load BF16 values to K fragment
        wmma::load_matrix_sync(
            k_frag,
            smem_staging + warp_idx * F_N * SMEM_K_STRIDE,
            SMEM_K_STRIDE);
      } else if (t_start + F_N <= MAX_T) {
        // Load BF16 K to K fragment
        wmma::load_matrix_sync(
            k_frag,
            reinterpret_cast<const __nv_bfloat16*>(cache_K_base) +
                t_start * D_H + d_start,
            D_H);
      } else {
        // Handle the remainder of T to avoid load_matrix_sync to K will OOB
        // Load 8 bfloat16s at a time for 16B loads
        constexpr int kThreadsPerF_K = F_K / 8;
        for (int t = t_start + threadIdx.x / kThreadsPerF_K;
             t < min(t_start + F_N, t_per_block_end);
             t += kThreadsPerWarp / kThreadsPerF_K) {
          const int d = d_start + threadIdx.x % kThreadsPerF_K * 8;
          const auto smem_offset =
              (warp_idx * F_N + t - t_start) * F_K + d - d_start;
          *(reinterpret_cast<uint4*>(smem_staging + smem_offset)) =
              *(reinterpret_cast<const uint4*>(cache_K_base + t * D_H + d));
        }
        // Load BF16 values to K fragment
        wmma::load_matrix_sync(
            k_frag, smem_staging + warp_idx * F_N * F_K, F_K);
      }
      // Compute matrix multiplication
      wmma::mma_sync(c_frag, q_frag, k_frag, c_frag);
    }

#ifdef USE_WMMA_FRAG
    // The following fragment (tensor core) layout is specific to the A100/H100
    // GPU Compute max_qk directly from the fragment
    constexpr int C_FRAG_SIZE = F_M * F_N;
    // A quadrant has 64 elements
    constexpr int C_QUAD_SIZE = (F_M * F_N) / 4;
    // A quadrant of a quadrant has 16 elements
    constexpr int C_DOUBLE_QUAD_SIZE = C_QUAD_SIZE / 4;
    // Half of a quadrant of a quadrant has 8 elements
    constexpr int C_HALF_DOUBLE_QUAD_SIZE = C_DOUBLE_QUAD_SIZE / 2;
    if (t_start < t_per_block_end) {
      const auto max_col = min(t_start + F_N, t_per_block_end) - t_start;
      // The column stride that each thread processes is 8
      // The number of threads processing each column is 4
      const int col_group = max_col / 8;
      const int cols_in_group = max_col % 8;
      const int max_elements =
          threadIdx.x < cols_in_group * 4 ? (col_group + 1) * 2 : col_group * 2;
      const int h_start = (threadIdx.x % 4) * 2;

      const int frag_offset =
          static_cast<int>((t_start - t_per_block_start) / F_N) * C_FRAG_SIZE;
      const int doub_quad_offset = threadIdx.x % 4 * C_DOUBLE_QUAD_SIZE;
      const int pos = threadIdx.x >> 2;
      auto* smem_ = smem + frag_offset + doub_quad_offset + pos;

      for (auto i = 0; i < max_elements && i < c_frag.num_elements; i++) {
        const int h_i = i % 2;
        if (h_i < h_total_per_block - h_start) {
          const auto qk = c_frag.x[i];
          const auto qk_acc = qk * qk_scale;
          max_qk[h_i] = max(max_qk[h_i], qk_acc);

          const int quad_offset = (i >> 1) * C_QUAD_SIZE;
          const int half_doub_quad_offset = (i % 2) * C_HALF_DOUBLE_QUAD_SIZE;
          smem_[quad_offset + half_doub_quad_offset] = qk_acc;
        }
      }
    }
#else
    // Store matrix multiplication results to shared memory
    wmma::store_matrix_sync(
        smem + t_start - t_per_block_start, c_frag, ldc, wmma::mem_row_major);

    // Scale the results and compute max for each head from shared memory
    const int nThreadsPerH = kThreadsPerWarp / h_total_per_block;
    // Each thread computes only one head
    const int h = threadIdx.x / nThreadsPerH;
    if (h < h_total_per_block) {
      for (int t = t_start + (threadIdx.x % nThreadsPerH);
           t < min(t_start + F_N, t_per_block_end);
           t += nThreadsPerH) {
        const float qk_acc = smem[h * ldc + t - t_per_block_start] * qk_scale;
        max_qk = max(max_qk, qk_acc);
      }
    }

    // Compute max within a warp
    // threadIdx.x % nThreadsPerH == 0 are master threads
    for (int offset = nThreadsPerH >> 1; offset >= 1; offset >>= 1) {
      max_qk = max(max_qk, __shfl_down_sync(FINAL_MASK, max_qk, offset));
    }
#endif
  }

#ifdef USE_WMMA_FRAG
  // At this point, every thread has their local max_qk's
  // Compute max_qk within a warp
#pragma unroll HEADS_PER_THREAD_QK
  for (auto h_i = 0; h_i < HEADS_PER_THREAD_QK; h_i++) {
    for (auto offset = 4; offset < kThreadsPerWarp; offset <<= 1) {
      max_qk[h_i] =
          max(max_qk[h_i],
              __shfl_sync(FINAL_MASK, max_qk[h_i], threadIdx.x + offset));
    }
  }

  // Use shared reduction to compute max and compute softmax on shared memory.
  // write max acc
  if (threadIdx.x < 4) {
    const auto h = threadIdx.x * 2;
    if (t_per_block_start + warp_idx * F_N < t_per_block_end) {
      smem_max[warp_idx * h_total_per_block + h] = max_qk[0];
      smem_max[warp_idx * h_total_per_block + h + 1] = max_qk[1];
    } else {
      smem_max[warp_idx * h_total_per_block + h] = NEG_FINF;
      smem_max[warp_idx * h_total_per_block + h + 1] = NEG_FINF;
    }
  }
#else
  // Use shared reduction to compute max and compute softmax on shared memory.
  // write max acc
  const int max_qk_threads_per_h = kThreadsPerWarp / h_total_per_block;
  if (threadIdx.x % max_qk_threads_per_h == 0) {
    const auto h = threadIdx.x / max_qk_threads_per_h;
    smem_max[warp_idx * h_total_per_block + h] =
        (t_per_block_start + warp_idx * F_N < t_per_block_end) ? max_qk
                                                               : NEG_FINF;
  }
#endif

  __syncthreads();

  const auto h = threadIdx.x;
  for (int w = kSplitKWarpsPerBlock >> 1; w >= 1; w >>= 1) {
    if (warp_idx < w && h < h_total_per_block) {
      smem_max[warp_idx * h_total_per_block + h] =
          max(smem_max[warp_idx * h_total_per_block + h],
              smem_max[(warp_idx + w) * h_total_per_block + h]);
    }
    __syncthreads();
  }

  const int hPerWarp = div_up(h_total_per_block, kSplitKWarpsPerBlock);
  const int h_begin = warp_idx * hPerWarp;
  const int h_end = min(h_begin + hPerWarp, h_total_per_block);

  // Complete max computation for each head at this point
  const int threads_per_h = kThreadsPerWarp / (h_end - h_begin);
  float head_sum = 0;
#ifdef USE_WMMA_FRAG
  // A100/H100 GPU will only store max here and compute head sum later
  // Only master thread sets the max metadata
  if (threadIdx.x % threads_per_h == 0 && h_end > h_begin) {
    const int h = h_begin + (threadIdx.x / threads_per_h);
    const auto max_qk_ = smem_max[h];
    metadata[b][0][s_block][h_per_block_start + h] = max_qk_;
    smem_max[h] = max_qk_;
  }
  __syncthreads();
  const auto max_qk_ = smem_max[threadIdx.x / 4];
#else
  // Non-A100/H100 GPUs will store both max and head sum here
  if (h_begin + threadIdx.x / threads_per_h < h_end) {
    const int h = h_begin + threadIdx.x / threads_per_h;
    const auto max_qk_ = smem_max[h];
    auto* smem_ = smem + h * ldc;
    for (int t = threadIdx.x % threads_per_h; t < t_total_per_block;
         t += threads_per_h) {
      const float p = __expf(smem_[t] * qk_scale - max_qk_);
      // Compute the sum value for each head
      head_sum += p;
      smem_[t] = p;
    }
  }
  // Compute sum within a warp
  for (int offset = threads_per_h >> 1; offset >= 1; offset >>= 1) {
    head_sum += __shfl_down_sync(FINAL_MASK, head_sum, offset);
  }

  // Store max and sum to global memory
  if (threadIdx.x % threads_per_h == 0 && h_end > h_begin) {
    const int h = h_begin + (threadIdx.x / threads_per_h);
    metadata[b][0][s_block][h_per_block_start + h] = smem_max[h];
    metadata[b][1][s_block][h_per_block_start + h] = head_sum;
  }
#endif

  // Each thread loads two uint32_t's in each iteration
  kv_load_t v_vals[KV_LD_NUM_ELS];
  __half2 v_scales;

  // Prefetch V
  if (USE_INT4) {
    const auto d_start = warp_idx * F_N;
    const int t_chunk_id = threadIdx.x % 2;
    const int group_id = d_start / INT4_GROUP_SIZE;
    int t = t_per_block_start + threadIdx.x / 2;
    if (t < min(t_per_block_start + F_K, t_per_block_end)) {
      const auto* v_ = cache_V_base + t * D_H_bytes;
      v_scales = reinterpret_cast<const __half2*>(v_)[group_id];
#pragma unroll KV_LD_NUM_ELS
      for (int vec = 0; vec < KV_LD_NUM_ELS; vec++) {
        int d = d_start + (vec + t_chunk_id * KV_NUM_VECS) * 8;
        v_vals[vec] =
            *reinterpret_cast<const kv_load_t*>(&v_[d / 2 + INT4_PARAM_BYTES]);
      }
    }
  }

#ifndef USE_WMMA_FRAG
  // Non-A100/H100 GPUs convert P from FP32 to BF16 inplace (i.e., using the
  // same shared memory space) here
  constexpr int32_t CONV_UNROLLS = 4;
  __nv_bfloat16* smem_bf16 = reinterpret_cast<__nv_bfloat16*>(smem);
  float2 p[CONV_UNROLLS];
  const int t_stride = blockDim.x * blockDim.y * 2;
  const int t_rounds = div_up(t_total_per_block, t_stride);
  const int global_tid = warp_idx * blockDim.x + threadIdx.x;

  // Ensure that all threads finish writing to smem before modifying it in the
  // loop below
  __syncthreads();

  // All threads work on the same head in every iteration
  for (int t_i = 0; t_i < t_rounds; t_i++) {
    const int t_start = t_i * t_stride + global_tid * 2;
    const int global_t_start = t_per_block_start + t_start;
    auto* smem_fp32_ = smem + t_start;
    auto* smem_bf16_ = smem_bf16 + t_start;

    for (int h_i = 0; h_i < div_up(h_total_per_block, CONV_UNROLLS); h_i++) {
      // Read FP32
#pragma unroll
      for (int h_j = 0; h_j < CONV_UNROLLS; h_j++) {
        const int h = h_i * CONV_UNROLLS + h_j;
        const int smem_idx = h * ldc;

        p[h_j].x = global_t_start < t_per_block_end ? smem_fp32_[smem_idx] : 0;
        p[h_j].y =
            global_t_start + 1 < t_per_block_end ? smem_fp32_[smem_idx + 1] : 0;
      }

      // Sync threads to make sure that all threads finish reading data before
      // overwriting the memory with new values
      __syncthreads();

      // Convert and write BF16
#pragma unroll
      for (int h_j = 0; h_j < CONV_UNROLLS; h_j++) {
        // It is safe to use nv_bfloat162 because smem was float
        if (global_t_start < t_per_block_end) {
          const int h = h_i * CONV_UNROLLS + h_j;
          const int smem_idx = h * ldc;

          *reinterpret_cast<nv_bfloat162*>(&smem_bf16_[smem_idx]) =
              __float22bfloat162_rn(p[h_j]);
        }
      }
    }
  }
  __syncthreads();

  // Fill smem with zeros for t_total <= t < F_K to avoid nan
  // Round up to the nv_bfloat162 granularity because if t_total_per_block is
  // an odd number, the FP32->BF16 conversion should already take care of
  // writing zero to .y
  const int t_zero_start = round_up(t_total_per_block, 2);
  const nv_bfloat162 zero_bf162 = {0, 0};
  const int t_mul_F_K = round_up(t_total_per_block, F_K);

  for (auto h = warp_idx; h < h_total_per_block; h += blockDim.y) {
    // Each thread operates on two BF16 values to avoid smem bank conflict
    for (auto t = t_zero_start + threadIdx.x * 2; t < t_mul_F_K;
         t += kThreadsPerWarp * 2) {
      *reinterpret_cast<nv_bfloat162*>(&smem_bf16[h * ldc + t]) = zero_bf162;
    }
  }

  __syncthreads();
#endif

  // Split D_H across warps in a block
  // each warp compute sum(t_subset) P[H, t] * V[t_subset, d_subset]
  // outputs are of size float[H, D]

  wmma::fragment<wmma::matrix_b, F_M, F_N, F_K, __nv_bfloat16, wmma::row_major>
      v_frag;

  // Compute P @ V
  // Parallelize D_H among warps. Note only 4 warps will do the work here.
  for (auto d_start = warp_idx * F_N; d_start < D_H;
       d_start += kSplitKWarpsPerBlock * F_N) {
    // Init the accumulator with zeros
    wmma::fill_fragment(c_frag, 0.0f);

    // Intra-warp reduction across T
    for (auto t_start = t_per_block_start; t_start < t_per_block_end;
         t_start += F_K) {
#ifdef USE_WMMA_FRAG
      // A100/H100 GPU reads FP32 from shared memory, convert it into BF16, and
      // writes data directly to the WMMA fragment.
      const int head = threadIdx.x / 4;
      constexpr int NUM_COL_VECS = 2;
      constexpr int P_FRAG_SIZE = F_M * F_K;
      constexpr int P_HALF_SIZE = P_FRAG_SIZE / 2;
      const int frag_offset =
          static_cast<int>((t_start - t_per_block_start) / F_K) * P_FRAG_SIZE;
      const int pos = threadIdx.x * 2;
      const auto* smem_ = smem + frag_offset + pos;
      const auto t_start_ = t_start + (threadIdx.x % 4) * 2;
      const auto t_scope = min(t_start + F_K, t_per_block_end);

      for (auto vec = 0; vec < NUM_COL_VECS; vec++) {
        float2 p;
        const int t = t_start_ + 8 * vec;
        if (head < h_total_per_block && t < t_scope) {
          p = *(reinterpret_cast<const float2*>(&smem_[vec * P_HALF_SIZE]));
          p.x = __expf(p.x - max_qk_);
          p.y = t + 1 < t_scope ? __expf(p.y - max_qk_) : 0;

          // Compute head sum here
          if (d_start == 0) {
            head_sum += p.x + p.y;
          }
        } else {
          p.x = 0;
          p.y = 0;
        }

        // TODO: store BF16 results in smem for D_H > 128 or F_N < 32
        // TODO: use vector store?
        // FP32->BF16 conversion is implicit
        q_frag.x[vec * 2] = p.x;
        q_frag.x[vec * 2 + 1] = p.y;
      }
      __syncwarp();
#else
      // Non-A100/H100 GPUs already did the FP32->BF16 conversion before
      // entering this loop. Thus, data can be loaded from shared memory
      wmma::load_matrix_sync(
          q_frag, smem_bf16 + t_start - t_per_block_start, ldc);
#endif

      // Load V fragment
      if (USE_INT4) {
        // Load and dequantize INT4 V
        // Each thread loads 16 columns (D dim) from one row (T dim).
        // Each row is handled by two threads
        const auto t_scope = min(t_start + F_K, t_per_block_end);
        const int t = t_start + threadIdx.x / 2;
        const int t_chunk_id = threadIdx.x % 2;
        if (t < t_scope) {
          const auto smem_offset =
              (warp_idx * F_K + t - t_start) * SMEM_V_STRIDE;
          auto* smem_staging_ = smem_staging + smem_offset;
#pragma unroll KV_NUM_VECS
          for (int vec = 0; vec < KV_NUM_VECS; ++vec) {
            const int smem_d = vec + t_chunk_id * KV_NUM_VECS;
            // Dequantize 8 INT4s to 8 BF16s and store the results in shared
            // memory
            const auto v_vals_ = reinterpret_cast<uint32_t*>(v_vals)[vec];
            const auto v_deq = dequantize_permuted_int4(v_vals_, v_scales);
            auto* smem_s =
                reinterpret_cast<__nv_bfloat162*>(smem_staging_ + smem_d * 8);
#pragma unroll
            for (int i = 0; i < 4; i++) {
              smem_s[i] = v_deq.vals[i];
            }
          }
        } else {
          // Need to fill zeros to avoid nan
          if (t < t_start + F_K) {
            const auto smem_offset =
                (warp_idx * F_K + t - t_start) * SMEM_V_STRIDE;
            auto* smem_staging_ = smem_staging + smem_offset;
#pragma unroll KV_NUM_VECS
            for (int vec = 0; vec < KV_NUM_VECS; ++vec) {
              const int smem_d = vec + t_chunk_id * KV_NUM_VECS;
              auto* smem_s =
                  reinterpret_cast<uint32_t*>(smem_staging_ + smem_d * 8);
#pragma unroll
              for (int i = 0; i < 4; i++) {
                smem_s[i] = 0;
              }
            }
          }
        }

        int t_start_next = t_start + F_K;
        t_start_next =
            t_start_next < t_per_block_end ? t_start_next : t_per_block_start;
        const int d_start_next = t_start_next < t_per_block_end
            ? d_start
            : d_start + kSplitKWarpsPerBlock * F_N;
        const int t_next = t_start_next + threadIdx.x / 2;

        if (t_next < min(t_start_next + F_K, t_per_block_end) &&
            d_start_next < D_H) {
          auto* v_ = cache_V_base + t_next * D_H_bytes;
          const auto group_id = d_start_next / INT4_GROUP_SIZE;
          v_scales = reinterpret_cast<const __half2*>(v_)[group_id];
#pragma unroll KV_LD_NUM_ELS
          for (int vec = 0; vec < KV_LD_NUM_ELS; vec++) {
            const int d = d_start_next + (vec + t_chunk_id * KV_NUM_VECS) * 8;
            v_vals[vec] = *reinterpret_cast<const kv_load_t*>(
                &v_[d / 2 + INT4_PARAM_BYTES]);
          }
        }
        // Load BF16 values to V fragment
        wmma::load_matrix_sync(
            v_frag,
            smem_staging + warp_idx * SMEM_V_STRIDE * F_K,
            SMEM_V_STRIDE);
      } else if (t_start + F_K <= t_per_block_end) {
        // Load BF16 V to V fragment
        wmma::load_matrix_sync(
            v_frag,
            reinterpret_cast<const __nv_bfloat16*>(cache_V_base) +
                t_start * D_H + d_start,
            D_H);
      } else {
        // Handle the remainder of T to avoid load_matrix_sync to V will OOB
        int t = t_start;
        const auto smem_offset = (warp_idx * F_K - t_start) * F_N - d_start;
        auto* smem_staging_ = smem_staging + smem_offset;
        for (; t < min(t_start + F_K, t_per_block_end); ++t) {
          auto* smem_staging_t_ = smem_staging_ + t * F_N;
          auto* v_ =
              reinterpret_cast<const __nv_bfloat16*>(cache_V_base) + t * D_H;
          for (int d = d_start + threadIdx.x; d < d_start + F_N;
               d += kThreadsPerWarp) {
            smem_staging_t_[d] = v_[d];
          }
        }
        // Need to fill zeros to avoid nan
        for (; t < t_start + F_K; ++t) {
          auto* smem_staging_t_ = smem_staging_ + t * F_N;
          for (int d = d_start + threadIdx.x; d < d_start + F_N;
               d += kThreadsPerWarp) {
            smem_staging_t_[d] = 0;
          }
        }
        // Load BF16 values to V fragment
        wmma::load_matrix_sync(
            v_frag, smem_staging + warp_idx * F_N * F_K, F_N);
      }
      // Compute matrix multiplication
      wmma::mma_sync(c_frag, q_frag, v_frag, c_frag);
    }

    // Store final results in global memory
    if (h_total_per_block == F_M) {
      // For this common case, no need to worry about OOB.
      auto* o_ = &out_splitK[b][s_block][h_per_block_start][d_start];
      wmma::store_matrix_sync(o_, c_frag, D_H, wmma::mem_row_major);
    } else {
      wmma::store_matrix_sync(
          smem_out + F_M * d_start, c_frag, F_N, wmma::mem_row_major);

      for (int h = 0; h < h_total_per_block; ++h) {
        // [B, H, num_split_ks, 1, D_H]
        auto* o_ = &out_splitK[b][s_block][h_per_block_start + h][d_start];
        for (int d = threadIdx.x; d < F_N; d += kThreadsPerWarp) {
          o_[d] = smem_out[F_M * d_start + h * F_N + d];
        }
      }
    }
  } // d_start

#ifdef USE_WMMA_FRAG
  // A100/H100 GPU has to store head sum in global memory here because it
  // computes this value during the P @ V computation
  if (warp_idx == 0) {
    for (int offset = 2; offset >= 1; offset >>= 1) {
      head_sum += __shfl_sync(FINAL_MASK, head_sum, threadIdx.x + offset);
    }

    const int head = threadIdx.x / 4;
    if (threadIdx.x % 4 == 0 && head < h_total_per_block) {
      metadata[b][1][s_block][h_per_block_start + head] = head_sum;
    }
  }
#endif
#endif
}

__global__ void gqa_attn_splitk_reduce_wmma_kernel(
    // {B, H, num_split_ks, D_H}
    const at::PackedTensorAccessor32<float, 4, at::RestrictPtrTraits>
        out_splitK,
    // {B, H, 2, num_split_ks, 1},
    const at::PackedTensorAccessor32<float, 4, at::RestrictPtrTraits> metadata,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        seq_positions,
    // [B, 1, H, D]
    at::PackedTensorAccessor32<at::BFloat16, 4, at::RestrictPtrTraits> O) {
  const int32_t b = blockIdx.x;
  const int32_t h = blockIdx.y;
  const auto num_split_ks = out_splitK.size(1);
  const auto d = threadIdx.y * kThreadsPerWarp + threadIdx.x;

  float m = metadata[b][0][0][h];
  float l_sum = metadata[b][1][0][h];
  float acc = out_splitK[b][0][h][d];

  const int32_t t_max = seq_positions[b] + 1;
  const int32_t t_total = round_up(t_max, num_split_ks);
  const int32_t t_per_block = t_total / num_split_ks;
  const int32_t num_split_ks_max = div_up(t_max, t_per_block);

  for (int k = 1; k < num_split_ks_max; ++k) {
    float m_k = metadata[b][0][k][h];
    float l_k = metadata[b][1][k][h];
    float acc_k = out_splitK[b][k][h][d];

    float m_new = max(m, m_k);
    float alpha;
    if (m_k < m) {
      alpha = __expf(m_k - m_new);
      acc_k *= alpha;
      l_k *= alpha;
    } else {
      alpha = __expf(m - m_new);
      acc *= alpha;
      l_sum *= alpha;
    }

    m = m_new;
    l_sum += l_k;
    acc += acc_k;
  }

  O[b][0][h][d] = acc / l_sum;
}
} // namespace

/// @ingroup experimental-gen-ai
///
/// @brief Decoding Grouped Query Attention Split-K w/ BF16/INT4 KV
///
/// The CUDA implementation of decoding Grouped Query Attention (GQA)
/// that supports BF16 and INT4 KV cache and BF16 input query.  It
/// currently only supports the max context length of 16384, the fixed
/// head dimension of 128, and only one KV cache head.  It supports an
/// arbitrary number of query heads.
///
/// @param XQ Input query; shape = (B, 1, H_Q, D), where B = batch
///           size, H_Q = num query heads, D = head dimension (fixed
///           to 128)
/// @param cache_K K cache; shape = (B, MAX_T, H_KV, D), where MAX_T =
///                max context length (fixed to 16384), and H_KV = num
///                KV cache heads (fixed to 1)
/// @param cache_V V cache; shape = (B, MAX_T, H_KV, D)
/// @param seq_positions Sequence position (contains the actual
///                      length of each token); shape = (B)
/// @param qk_scale The scale that is applied after QK^T
/// @param num_split_ks The number of split Ks (controlling the
///                     amount of parallelism in the context length
///                     dimension (MAX_T))
/// @param num_int4_kv_groups The number of groups for group-wise INT4
///                           quantization for each KV token (each
///                           group uses the same scale and bias for
///                           quantization)
///
/// @return    A tuple of the combined split-K output, the
///            non-combined split-K output, and the split-K metadata
///            (containing max QK^T, and softmax(QK^T) head sum)
std::tuple<at::Tensor, at::Tensor, at::Tensor> gqa_attn_splitk_cuda(
    const at::Tensor& XQ,
    const at::Tensor& cache_K,
    const at::Tensor& cache_V,
    const at::Tensor& seq_positions,
    const double qk_scale,
    const int64_t num_split_ks,
    const int64_t num_int4_kv_groups) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(
      dprops->major >= 8,
      "Too old compute capability major version to run gqa_attn_splitk_wmma ",
      dprops->major);

  at::OptionalDeviceGuard guard(XQ.device());
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(XQ.is_contiguous());
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());
  TORCH_CHECK(cache_K.is_contiguous());
  TORCH_CHECK(cache_V.is_contiguous());
  TORCH_CHECK(seq_positions.is_cuda());

  // Check input shapes
  TORCH_CHECK(cache_K.size(1) <= MAX_T);
  TORCH_CHECK(
      cache_K.size(2) == 1,
      "Currently gqa_attn_splitk only support num KV heads = 1");
  TORCH_CHECK(
      cache_V.size(2) == 1,
      "Currently gqa_attn_splitk only support num KV heads = 1");

  if (cache_K.dtype() == at::kBFloat16) {
    TORCH_CHECK(cache_K.size(3) == D_H);
  } else {
    TORCH_CHECK(
        num_int4_kv_groups == 1 || num_int4_kv_groups == 4,
        "Invalid num_int4_kv_groups ",
        num_int4_kv_groups);
    auto qparam_offset = 4 * num_int4_kv_groups;
    TORCH_CHECK(cache_K.size(3) == D_H / 2 + qparam_offset);
  }

  const auto B = XQ.size(0);
  const auto H = XQ.size(2);

  auto out_splitK =
      at::empty({B, num_split_ks, H, D_H}, XQ.options().dtype(at::kFloat));
  auto O = at::empty_like(XQ);
  auto metadata = at::empty({B, 2, num_split_ks, H}, out_splitK.options());

  // TODO: Check if the grid size is valid
  const int32_t H_blocks = div_up(H, kMaxHeads);
  dim3 blocks(B, H_blocks, num_split_ks);
  dim3 threads(kThreadsPerWarp, kSplitKWarpsPerBlock);

  if (B == 0) {
    return {O, out_splitK, metadata};
  }

  const int32_t t_per_block = div_up(cache_K.size(1), num_split_ks);
  // This is called ldc inside gqa_attn_splitk_wmma_kernel kernel
  const int32_t t_per_block_round_up = round_up(t_per_block, F_N);

  // QK^T and P smem: max(kMaxHeads, F_M) * t_per_block_round_up floats
  // Max QK^T smem: max(kMaxHeads, F_M) * kSplitKWarpsPerBlock floats
  // Stagging smem: smem_staging_size bfloat16s
  // Output smem: max(kMaxHeads, F_M) * D_H floats
  const int32_t smem_staging_size = kSplitKWarpsPerBlock *
      max(F_N * SMEM_K_STRIDE, F_K * SMEM_V_STRIDE) * sizeof(at::BFloat16);
  int32_t smem = max(kMaxHeads, F_M) *
          (t_per_block_round_up + kSplitKWarpsPerBlock + D_H) * sizeof(float) +
      smem_staging_size;

#define CALL_GQA_ATTN_SPLITK_WMMA(CACHE_TYPE, NUM_GROUPS, KV_LOAD_T)        \
  const auto gqa_fn =                                                       \
      gqa_attn_splitk_wmma_kernel<CACHE_TYPE, NUM_GROUPS, KV_LOAD_T>;       \
  if (smem > SMEM_ADJUST_THRESHOLD) {                                       \
    set_gpu_max_dynamic_shared_memory(gqa_fn, smem, XQ.get_device());       \
  }                                                                         \
  gqa_fn<<<blocks, threads, smem, at::cuda::getCurrentCUDAStream()>>>(      \
      XQ.packed_accessor32<at::BFloat16, 4, at::RestrictPtrTraits>(),       \
      cache_K.packed_accessor64<CACHE_TYPE, 4, at::RestrictPtrTraits>(),    \
      cache_V.packed_accessor64<CACHE_TYPE, 4, at::RestrictPtrTraits>(),    \
      out_splitK.packed_accessor32<float, 4, at::RestrictPtrTraits>(),      \
      seq_positions.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
      metadata.packed_accessor32<float, 4, at::RestrictPtrTraits>(),        \
      qk_scale);                                                            \
  C10_CUDA_KERNEL_LAUNCH_CHECK()

  if (cache_K.dtype() == at::kBFloat16) {
    CALL_GQA_ATTN_SPLITK_WMMA(at::BFloat16, 1, uint32_t);
  } else {
    if (num_int4_kv_groups == 1) {
      CALL_GQA_ATTN_SPLITK_WMMA(uint8_t, 1, uint32_t);
    } else {
      CALL_GQA_ATTN_SPLITK_WMMA(uint8_t, 4, uint2);
    }
  }

#undef CALL_GQA_ATTN_SPLITK_WMMA

  gqa_attn_splitk_reduce_wmma_kernel<<<
      dim3(B, H),
      dim3(kThreadsPerWarp, D_H / kThreadsPerWarp),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      out_splitK.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
      metadata.packed_accessor32<float, 4, at::RestrictPtrTraits>(),
      seq_positions.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      O.packed_accessor32<at::BFloat16, 4, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {O, out_splitK, metadata};
}

} // namespace fbgemm_gpu::gen_ai::attention
