/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "fbgemm/UtilsAvx2.h"
#include "fbgemm/Utils.h"
#include "src/FbgemmI8DepthwiseAvx2-inl.h"
#include "src/MaskAvx2.h"

namespace fbgemm {

template <int S = 3, bool SUM_A = false, bool REMAINDER = false>
static ALWAYS_INLINE void inner_prod_2d_packed_(
    const __m256i* a_v,
    const __m256i* Bp,
    std::int32_t* C,
    int remainder,
    __m256i* a_sum = nullptr) {
  return inner_prod_packed_<S * S, SUM_A, REMAINDER>(
      a_v, Bp, C, remainder, a_sum);
}

template <
    bool SUM_A,
    bool REMAINDER = false,
    bool PER_CHANNEL_QUANTIZATION = false>
static ALWAYS_INLINE void inner_prod_3x3_packed_(
    int H,
    int W,
    int K,
    int h_in,
    int w_in,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    const std::int8_t* Bp,
    const std::int32_t* B_zero_point,
    std::int32_t* C,
    int remainder,
    std::int32_t* row_offsets) {
  __m256i A_zero_point_v =
      _mm256_set1_epi8(static_cast<std::uint8_t>(A_zero_point));
  __m256i mask_v = _mm256_setzero_si256();
  if (REMAINDER) {
    mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_masks[remainder / 4]));
  }

  // The code below can be written as a simple R*S loop but the compiler
  // doesn't unroll so we're manually unrolling it.
  // constexpr int R = 3, S = 3;
  // array<__m256i, R * S> a_v;
  // for (int r = 0; r < R; ++r) {
  //   for (int s = 0; s < S; ++s) {
  //     if (h_in + r >= 0 && h_in + r < H && w_in + s >= 0 && w_in + s < W) {
  //       if (REMAINDER) {
  //         a_v[r * S + s] =
  //             _mm256_maskload_epi32((const int *)(A + (r * W + s) * K),
  //             mask_v);
  //       } else {
  //         a_v[r * S + s] =
  //             _mm256_lddqu_si256((const __m256i *)(A + (r * W + s) * K));
  //       }
  //     } else {
  //       a_v[r * S + s] = A_zero_point_v;
  //     }
  //   }
  // }
  __m256i a_v[9] = {
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
      A_zero_point_v,
  };

  if (h_in >= 0 && h_in < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[0] = load_a<REMAINDER>(A + (0 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[1] = load_a<REMAINDER>(A + (0 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[2] = load_a<REMAINDER>(A + (0 * W + 2) * K, mask_v);
    }
  }

  if (h_in + 1 >= 0 && h_in + 1 < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[3] = load_a<REMAINDER>(A + (1 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[4] = load_a<REMAINDER>(A + (1 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[5] = load_a<REMAINDER>(A + (1 * W + 2) * K, mask_v);
    }
  }

  if (h_in + 2 >= 0 && h_in + 2 < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[6] = load_a<REMAINDER>(A + (2 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[7] = load_a<REMAINDER>(A + (2 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[8] = load_a<REMAINDER>(A + (2 * W + 2) * K, mask_v);
    }
  }

  __m256i a_sum[4];
  inner_prod_2d_packed_<3, SUM_A, REMAINDER>(
      a_v, reinterpret_cast<const __m256i*>(Bp), C, remainder, a_sum);
  if (SUM_A) {
    __m256i B_zero_point_v;
    for (int i = 0; i < (REMAINDER ? (remainder / 8) : 4); ++i) {
      if (PER_CHANNEL_QUANTIZATION) {
        B_zero_point_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B_zero_point + i * 8));
      } else {
        B_zero_point_v = _mm256_set1_epi32(B_zero_point[0]);
      }
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&row_offsets[i * 8]),
          _mm256_mullo_epi32(a_sum[i], B_zero_point_v));
    }
  }
}

template <
    bool SUM_A,
    bool REMAINDER = false,
    bool PER_CHANNEL_QUANTIZATION = false>
static ALWAYS_INLINE void inner_prod_5x5_packed_(
    int H,
    int W,
    int K,
    int h_in,
    int w_in,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    const std::int8_t* Bp,
    const std::int32_t* B_zero_point,
    std::int32_t* C,
    int remainder,
    std::int32_t* row_offsets) {
  __m256i A_zero_point_v =
      _mm256_set1_epi8(static_cast<std::uint8_t>(A_zero_point));
  __m256i mask_v = _mm256_setzero_si256();
  if (REMAINDER) {
    mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(
        internal::avx2_ps_or_epi32_masks[remainder / 4]));
  }

  // The code below can be written as a simple R*S loop but the compiler
  // doesn't unroll so we're manually unrolling it.
  // constexpr int R = 5, S = 5;
  // array<__m256i, R * S> a_v;
  // for (int r = 0; r < R; ++r) {
  //   for (int s = 0; s < S; ++s) {
  //     if (h_in + r >= 0 && h_in + r < H && w_in + s >= 0 && w_in + s < W) {
  //       if (REMAINDER) {
  //         a_v[r * S + s] =
  //             _mm256_maskload_epi32((const int *)(A + (r * W + s) * K),
  //             mask_v);
  //       } else {
  //         a_v[r * S + s] =
  //             _mm256_lddqu_si256((const __m256i *)(A + (r * W + s) * K));
  //       }
  //     } else {
  //       a_v[r * S + s] = A_zero_point_v;
  //     }
  //   }
  // }
  __m256i a_v[25] = {
      A_zero_point_v, A_zero_point_v, A_zero_point_v, A_zero_point_v,
      A_zero_point_v, A_zero_point_v, A_zero_point_v, A_zero_point_v,
      A_zero_point_v, A_zero_point_v, A_zero_point_v, A_zero_point_v,
      A_zero_point_v, A_zero_point_v, A_zero_point_v, A_zero_point_v,
      A_zero_point_v, A_zero_point_v, A_zero_point_v, A_zero_point_v,
      A_zero_point_v, A_zero_point_v, A_zero_point_v, A_zero_point_v,
      A_zero_point_v,
  };

  if (h_in >= 0 && h_in < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[0] = load_a<REMAINDER>(A + (0 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[1] = load_a<REMAINDER>(A + (0 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[2] = load_a<REMAINDER>(A + (0 * W + 2) * K, mask_v);
    }
    if (w_in + 3 >= 0 && w_in + 3 < W) {
      a_v[3] = load_a<REMAINDER>(A + (0 * W + 3) * K, mask_v);
    }
    if (w_in + 4 >= 0 && w_in + 4 < W) {
      a_v[4] = load_a<REMAINDER>(A + (0 * W + 4) * K, mask_v);
    }
  }

  if (h_in + 1 >= 0 && h_in + 1 < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[5] = load_a<REMAINDER>(A + (1 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[6] = load_a<REMAINDER>(A + (1 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[7] = load_a<REMAINDER>(A + (1 * W + 2) * K, mask_v);
    }
    if (w_in + 3 >= 0 && w_in + 3 < W) {
      a_v[8] = load_a<REMAINDER>(A + (1 * W + 3) * K, mask_v);
    }
    if (w_in + 4 >= 0 && w_in + 4 < W) {
      a_v[9] = load_a<REMAINDER>(A + (1 * W + 4) * K, mask_v);
    }
  }

  if (h_in + 2 >= 0 && h_in + 2 < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[10] = load_a<REMAINDER>(A + (2 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[11] = load_a<REMAINDER>(A + (2 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[12] = load_a<REMAINDER>(A + (2 * W + 2) * K, mask_v);
    }
    if (w_in + 3 >= 0 && w_in + 3 < W) {
      a_v[13] = load_a<REMAINDER>(A + (2 * W + 3) * K, mask_v);
    }
    if (w_in + 4 >= 0 && w_in + 4 < W) {
      a_v[14] = load_a<REMAINDER>(A + (2 * W + 4) * K, mask_v);
    }
  }

  if (h_in + 3 >= 0 && h_in + 3 < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[15] = load_a<REMAINDER>(A + (3 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[16] = load_a<REMAINDER>(A + (3 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[17] = load_a<REMAINDER>(A + (3 * W + 2) * K, mask_v);
    }
    if (w_in + 3 >= 0 && w_in + 3 < W) {
      a_v[18] = load_a<REMAINDER>(A + (3 * W + 3) * K, mask_v);
    }
    if (w_in + 4 >= 0 && w_in + 4 < W) {
      a_v[19] = load_a<REMAINDER>(A + (3 * W + 4) * K, mask_v);
    }
  }

  if (h_in + 4 >= 0 && h_in + 4 < H) {
    if (w_in >= 0 && w_in < W) {
      a_v[20] = load_a<REMAINDER>(A + (4 * W + 0) * K, mask_v);
    }
    if (w_in + 1 >= 0 && w_in + 1 < W) {
      a_v[21] = load_a<REMAINDER>(A + (4 * W + 1) * K, mask_v);
    }
    if (w_in + 2 >= 0 && w_in + 2 < W) {
      a_v[22] = load_a<REMAINDER>(A + (4 * W + 2) * K, mask_v);
    }
    if (w_in + 3 >= 0 && w_in + 3 < W) {
      a_v[23] = load_a<REMAINDER>(A + (4 * W + 3) * K, mask_v);
    }
    if (w_in + 4 >= 0 && w_in + 4 < W) {
      a_v[24] = load_a<REMAINDER>(A + (4 * W + 4) * K, mask_v);
    }
  }

  __m256i a_sum[4];
  inner_prod_2d_packed_<5, SUM_A, REMAINDER>(
      a_v, reinterpret_cast<const __m256i*>(Bp), C, remainder, a_sum);
  if (SUM_A) {
    __m256i B_zero_point_v;
    for (int i = 0; i < (REMAINDER ? (remainder / 8) : 4); ++i) {
      if (PER_CHANNEL_QUANTIZATION) {
        B_zero_point_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B_zero_point + i * 8));
      } else {
        B_zero_point_v = _mm256_set1_epi32(B_zero_point[0]);
      }
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(&row_offsets[i * 8]),
          _mm256_mullo_epi32(a_sum[i], B_zero_point_v));
    }
  }
}

template <
    int S,
    bool SUM_A,
    bool REMAINDER = false,
    bool PER_CHANNEL_QUANTIZATION = false>
static ALWAYS_INLINE void inner_prod_2d_packed_(
    int H,
    int W,
    int K,
    int h_in,
    int w_in,
    const std::uint8_t* A,
    std::int32_t A_zero_point,
    const std::int8_t* Bp,
    const std::int32_t* B_zero_point,
    std::int32_t* C,
    int remainder,
    std::int32_t* row_offsets) {
  if (S == 3) {
    inner_prod_3x3_packed_<SUM_A, REMAINDER, PER_CHANNEL_QUANTIZATION>(
        H,
        W,
        K,
        h_in,
        w_in,
        A,
        A_zero_point,
        Bp,
        B_zero_point,
        C,
        remainder,
        row_offsets);
  } else {
    assert(S == 5);
    inner_prod_5x5_packed_<SUM_A, REMAINDER, PER_CHANNEL_QUANTIZATION>(
        H,
        W,
        K,
        h_in,
        w_in,
        A,
        A_zero_point,
        Bp,
        B_zero_point,
        C,
        remainder,
        row_offsets);
  }
}

template <
    int S,
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_2d_kernel_(
    int H,
    int W,
    int K,
    int h,
    int w,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    std::int32_t B_zero_point,
    const std::int8_t* Bp,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    float act_times_w_scale) {
  constexpr int PAD_T = (S - 1) / 2, PAD_L = (S - 1) / 2, PAD_R = (S - 1) / 2;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  constexpr int KERNEL_PROD_ALIGNED = (S * S + 1) / 2 * 2;

  int k;
  for (k = 0; k < K / 32 * 32; k += 32) {
    inner_prod_2d_packed_<S, !B_SYMMETRIC /*SUM_A*/>(
        H,
        W,
        K,
        h_in,
        w_in,
        A + (h_in * W + w_in) * K + k,
        A_zero_point,
        Bp + k * KERNEL_PROD_ALIGNED,
        &B_zero_point,
        C_int32 + k,
        0,
        B_SYMMETRIC ? nullptr : &row_offsets[k]);
  }
  int remainder = K - k;
  if (remainder) {
    inner_prod_2d_packed_<S, !B_SYMMETRIC, true>(
        H,
        W,
        K,
        h_in,
        w_in,
        A + (h_in * W + w_in) * K + k,
        A_zero_point,
        Bp + k * KERNEL_PROD_ALIGNED,
        &B_zero_point,
        C_int32 + k,
        remainder,
        B_SYMMETRIC ? nullptr : &row_offsets[k]);
  }

  requantize_<
      FUSE_RELU,
      HAS_BIAS,
      false, /*PER_CHAN_QUANT*/
      A_SYMMETRIC,
      B_SYMMETRIC,
      BIAS_TYPE>(
      A_zero_point,
      &C_multiplier,
      C_zero_point,
      C_int32,
      C_uint8 + (h * W_OUT + w) * K,
      K,
      row_offsets,
      col_offsets,
      bias,
      &act_times_w_scale);
}

template <
    int S,
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_2d_per_channel_quantization_kernel_(
    int H,
    int W,
    int K,
    int h,
    int w,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const std::int8_t* Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale) {
  constexpr int PAD_T = (S - 1) / 2, PAD_L = (S - 1) / 2, PAD_R = (S - 1) / 2;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  constexpr int KERNEL_PROD_ALIGNED = (S * S + 1) / 2 * 2;

  int k;
  for (k = 0; k < K / 32 * 32; k += 32) {
    inner_prod_2d_packed_<
        S,
        true, /*SUM_A*/
        false, /*remainder*/
        true /*per-channel*/>(
        H,
        W,
        K,
        h_in,
        w_in,
        A + (h_in * W + w_in) * K + k,
        A_zero_point,
        Bp + k * KERNEL_PROD_ALIGNED,
        B_zero_point + k,
        C_int32 + k,
        0,
        &row_offsets[k]);
  }
  int remainder = K - k;
  if (remainder) {
    inner_prod_2d_packed_<
        S,
        true, /*SUM_A*/
        true, /*remainder*/
        true /*per-channel*/>(
        H,
        W,
        K,
        h_in,
        w_in,
        A + (h_in * W + w_in) * K + k,
        A_zero_point,
        Bp + k * KERNEL_PROD_ALIGNED,
        B_zero_point + k,
        C_int32 + k,
        remainder,
        &row_offsets[k]);
  }

  requantize_<
      FUSE_RELU,
      HAS_BIAS,
      true, /*PER_CHAN_QUANT*/
      A_SYMMETRIC,
      false, /*B_SYMM*/
      BIAS_TYPE>(
      A_zero_point,
      C_multiplier,
      C_zero_point,
      C_int32,
      C_uint8 + (h * W_OUT + w) * K,
      K,
      row_offsets,
      col_offsets,
      bias,
      act_times_w_scale);
}

// TODO: short-circuit when B_zero_point is 0 or A_zero_point is 0
// This implemntation should be general enough to handle not just 3x3 but other
// filter shapes by parameterizing with R and S but restricting it to just 3x3
// for now.
template <
    int S,
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_2d_(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    std::int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  assert(K % 8 == 0);
  constexpr int R = S;
  constexpr int PAD_T = (R - 1) / 2, PAD_B = PAD_T, PAD_L = (S - 1) / 2,
                PAD_R = PAD_L;
  int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  const std::int8_t* Bp = B.PackedMat();

  int32_t* row_offsets = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));

  int n_begin, n_end, h_begin, h_end, w_begin, w_end;
  // Reuse the 3-dim partition scheme for parallelization in matrix
  // multiplication.
  thread_type_t th_info =
      fbgemmGetThreadPartition(N, H_OUT, W_OUT, thread_id, num_threads);
  // Calculate the begin and end index along the batch (N) dimension
  fbgemmPartition1D(
      th_info.g_thread_id, th_info.g_num_threads, N, n_begin, n_end);
  // Calculate the begin and end index along the H dimension
  fbgemmPartition1D(
      th_info.m_thread_id, th_info.m_num_threads, H_OUT, h_begin, h_end);
  // Calculate the begin and end index along the W dimension
  fbgemmPartition1D(
      th_info.n_thread_id, th_info.n_num_threads, W_OUT, w_begin, w_end);

  for (int n = n_begin; n < n_end; ++n) {
    const std::uint8_t* A_base = A + n * H * W * K;
    std::uint8_t* C_uint8_base = C_uint8 + n * H_OUT * W_OUT * K;

    int h = 0;
    int w = 0;

    for (h = h_begin; h < std::max(PAD_T, h_begin); ++h) {
      for (w = w_begin; w < std::max(PAD_L, w_begin); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < std::min(W_OUT - PAD_R, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < w_end; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }
    }

    for (; h < std::min(H - PAD_B, h_end); ++h) {
      for (w = w_begin; w < std::max(PAD_L, w_begin); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < std::min(W_OUT - PAD_R, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < w_end; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }
    }

    for (; h < h_end; ++h) {
      for (w = w_begin; w < std::max(PAD_L, w_begin); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < std::min(W_OUT - PAD_R, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < w_end; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }
    }
  } // for each n

  fbgemmAlignedFree(row_offsets);
};

template <
    int S,
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_2d_per_channel_quantization_(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  assert(K % 8 == 0);
  constexpr int R = S;
  constexpr int PAD_T = (R - 1) / 2, PAD_B = PAD_T, PAD_L = (S - 1) / 2,
                PAD_R = PAD_L;
  int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  const std::int8_t* Bp = B.PackedMat();

  int32_t* row_offsets = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));

  int n_begin, n_end, h_begin, h_end, w_begin, w_end;
  // Reuse the 3-dim partition scheme for parallelization in matrix
  // multiplication.
  thread_type_t th_info =
      fbgemmGetThreadPartition(N, H_OUT, W_OUT, thread_id, num_threads);
  // Calculate the begin and end index along the batch (N) dimension
  fbgemmPartition1D(
      th_info.g_thread_id, th_info.g_num_threads, N, n_begin, n_end);
  // Calculate the begin and end index along the H dimension
  fbgemmPartition1D(
      th_info.m_thread_id, th_info.m_num_threads, H_OUT, h_begin, h_end);
  // Calculate the begin and end index along the W dimension
  fbgemmPartition1D(
      th_info.n_thread_id, th_info.n_num_threads, W_OUT, w_begin, w_end);

  for (int n = n_begin; n < n_end; ++n) {
    const std::uint8_t* A_base = A + n * H * W * K;
    std::uint8_t* C_uint8_base = C_uint8 + n * H_OUT * W_OUT * K;

    int h = 0;
    int w = 0;

    for (h = h_begin; h < std::max(PAD_T, h_begin); ++h) {
      for (w = w_begin; w < std::max(PAD_L, w_begin); ++w) {
        depthwise_2d_per_channel_quantization_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < std::min(W_OUT - PAD_R, w_end); ++w) {
        depthwise_2d_per_channel_quantization_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < w_end; ++w) {
        depthwise_2d_per_channel_quantization_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }
    }

    for (; h < std::min(H - PAD_B, h_end); ++h) {
      for (w = w_begin; w < std::max(PAD_L, w_begin); ++w) {
        depthwise_2d_per_channel_quantization_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < std::min(W_OUT - PAD_R, w_end); ++w) {
        depthwise_2d_per_channel_quantization_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < w_end; ++w) {
        depthwise_2d_per_channel_quantization_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }
    }

    for (; h < h_end; ++h) {
      for (w = w_begin; w < std::max(PAD_L, w_begin); ++w) {
        depthwise_2d_per_channel_quantization_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < std::min(W_OUT - PAD_R, w_end); ++w) {
        depthwise_2d_per_channel_quantization_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }

      for (; w < w_end; ++w) {
        depthwise_2d_per_channel_quantization_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            BIAS_TYPE>(
            H,
            W,
            K,
            h,
            w,
            stride_h,
            stride_w,
            A_zero_point,
            A_base,
            B_zero_point,
            Bp,
            C_multiplier,
            C_zero_point,
            C_int32,
            C_uint8_base,
            row_offsets,
            col_offsets,
            bias,
            act_times_w_scale);
      }
    }
  } // for each n

  fbgemmAlignedFree(row_offsets);
};

// Dispatch A_SYMMETRIC and B_SYMMETRIC
template <int S, bool FUSE_RELU, bool HAS_BIAS, typename BIAS_TYPE>
static void depthwise_2d_(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    std::int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  int32_t* C_int32_temp = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));
  if (A_zero_point == 0 || col_offsets == nullptr) {
    if (B_zero_point == 0) {
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          true /*B_symmetric*/,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    }
  } else {
    if (B_zero_point == 0) {
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          false /*A_symmetric*/,
          true /*B_symmetric*/,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          false /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    }
  }
  fbgemmAlignedFree(C_int32_temp);
}

// Dispatch HAS_BIAS
template <int S, bool FUSE_RELU, typename BIAS_TYPE>
static void depthwise_2d_(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    std::int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (bias) {
    depthwise_2d_<S, FUSE_RELU, true /*HAS_BIAS*/, BIAS_TYPE>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  } else {
    depthwise_2d_<S, FUSE_RELU, false /*HAS_BIAS*/, BIAS_TYPE>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  }
}

// Dispatch A_SYMMETRIC
template <int S, bool FUSE_RELU, bool HAS_BIAS, typename BIAS_TYPE>
static void depthwise_2d_per_channel_quantization_(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  int32_t* C_int32_temp = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));
  if (A_zero_point == 0 || col_offsets == nullptr) {
    depthwise_2d_per_channel_quantization_<
        S,
        FUSE_RELU,
        HAS_BIAS,
        true /*A_SYMM*/,
        BIAS_TYPE>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        Bp,
        C_multiplier,
        C_zero_point,
        C_int32_temp,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  } else {
    depthwise_2d_per_channel_quantization_<
        S,
        FUSE_RELU,
        HAS_BIAS,
        false /*A_SYMM*/,
        BIAS_TYPE>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        Bp,
        C_multiplier,
        C_zero_point,
        C_int32_temp,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  }
  fbgemmAlignedFree(C_int32_temp);
}

// Dispatch HAS_BIAS
template <int S, bool FUSE_RELU, typename BIAS_TYPE>
static void depthwise_2d_per_channel_quantization_(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (bias) {
    depthwise_2d_per_channel_quantization_<
        S,
        FUSE_RELU,
        true /* HAS_BIAS */,
        BIAS_TYPE>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        Bp,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  } else {
    depthwise_2d_per_channel_quantization_<
        S,
        FUSE_RELU,
        false /* HAS_BIAS */,
        BIAS_TYPE>(
        N,
        H,
        W,
        K,
        stride_h,
        stride_w,
        A_zero_point,
        A,
        B_zero_point,
        Bp,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  }
}

template <typename BIAS_TYPE = std::int32_t>
FBGEMM_API void depthwise_3x3_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    std::int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    float C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    float act_times_w_scale = 1.0f,
    int thread_id = 0,
    int num_threads = 1);

template <typename BIAS_TYPE = std::int32_t>
FBGEMM_API void depthwise_3x3_per_channel_quantization_pad_1(
    int N,
    int H,
    int W,
    int K,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu = false,
    const float* act_times_w_scale = nullptr,
    int thread_id = 0,
    int num_threads = 1);

} // namespace fbgemm
