/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"

#include <stdexcept> // for logic_error
#include <string>

#include "./FbgemmI8DepthwiseAvx2-inl.h"
#include "./MaskAvx2.h"
#include "fbgemm/Utils.h"
#include "fbgemm/UtilsAvx2.h"

using namespace std;

namespace fbgemm {

template <
    bool SUM_A,
    bool REMAINDER = false,
    bool PER_CHANNEL_QUANTIZATION = false>
static ALWAYS_INLINE void inner_prod_3x3x3_packed_(
    int T,
    int H,
    int W,
    int K,
    int t_in,
    int h_in,
    int w_in,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* Bp,
    const int32_t* B_zero_point,
    int32_t* C,
    int remainder,
    int32_t* row_offsets) {
  __m256i A_zero_point_v = _mm256_set1_epi8(static_cast<uint8_t>(A_zero_point));
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
  __m256i a_v[8];
  a_v[0] = A_zero_point_v;
  a_v[1] = A_zero_point_v;
  a_v[2] = A_zero_point_v;
  a_v[3] = A_zero_point_v;
  a_v[4] = A_zero_point_v;
  a_v[5] = A_zero_point_v;
  a_v[6] = A_zero_point_v;
  a_v[7] = A_zero_point_v;

  if (t_in >= 0 && t_in < T) {
    if (h_in >= 0 && h_in < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[0] = load_a<REMAINDER>(A + ((0 * H + 0) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[1] = load_a<REMAINDER>(A + ((0 * H + 0) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[2] = load_a<REMAINDER>(A + ((0 * H + 0) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 1 >= 0 && h_in + 1 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[3] = load_a<REMAINDER>(A + ((0 * H + 1) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[4] = load_a<REMAINDER>(A + ((0 * H + 1) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[5] = load_a<REMAINDER>(A + ((0 * H + 1) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[6] = load_a<REMAINDER>(A + ((0 * H + 2) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[7] = load_a<REMAINDER>(A + ((0 * H + 2) * W + 1) * K, mask_v);
      }
    }
  }

  __m256i a_sum[4];
  inner_prod_packed_<8, SUM_A, REMAINDER>(
      a_v, reinterpret_cast<const __m256i*>(Bp), C, remainder, a_sum);

  a_v[0] = A_zero_point_v;
  a_v[1] = A_zero_point_v;
  a_v[2] = A_zero_point_v;
  a_v[3] = A_zero_point_v;
  a_v[4] = A_zero_point_v;
  a_v[5] = A_zero_point_v;
  a_v[6] = A_zero_point_v;
  a_v[7] = A_zero_point_v;

  if (t_in >= 0 && t_in < T) {
    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[0] = load_a<REMAINDER>(A + ((0 * H + 2) * W + 2) * K, mask_v);
      }
    }
  }

  if (t_in + 1 >= 0 && t_in + 1 < T) {
    if (h_in >= 0 && h_in < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[1] = load_a<REMAINDER>(A + ((1 * H + 0) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[2] = load_a<REMAINDER>(A + ((1 * H + 0) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[3] = load_a<REMAINDER>(A + ((1 * H + 0) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 1 >= 0 && h_in + 1 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[4] = load_a<REMAINDER>(A + ((1 * H + 1) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[5] = load_a<REMAINDER>(A + ((1 * H + 1) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[6] = load_a<REMAINDER>(A + ((1 * H + 1) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[7] = load_a<REMAINDER>(A + ((1 * H + 2) * W + 0) * K, mask_v);
      }
    }
  }

  __m256i a_sum_temp[4];
  inner_prod_packed_<8, SUM_A, REMAINDER, true /* acc */>(
      a_v, reinterpret_cast<const __m256i*>(Bp) + 8, C, remainder, a_sum_temp);
  if (SUM_A) {
    a_sum[0] = _mm256_add_epi32(a_sum[0], a_sum_temp[0]);
    a_sum[1] = _mm256_add_epi32(a_sum[1], a_sum_temp[1]);
    a_sum[2] = _mm256_add_epi32(a_sum[2], a_sum_temp[2]);
    a_sum[3] = _mm256_add_epi32(a_sum[3], a_sum_temp[3]);
  }

  a_v[0] = A_zero_point_v;
  a_v[1] = A_zero_point_v;
  a_v[2] = A_zero_point_v;
  a_v[3] = A_zero_point_v;
  a_v[4] = A_zero_point_v;
  a_v[5] = A_zero_point_v;
  a_v[6] = A_zero_point_v;
  a_v[7] = A_zero_point_v;

  if (t_in + 1 >= 0 && t_in + 1 < T) {
    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[0] = load_a<REMAINDER>(A + ((1 * H + 2) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[1] = load_a<REMAINDER>(A + ((1 * H + 2) * W + 2) * K, mask_v);
      }
    }
  }

  if (t_in + 2 >= 0 && t_in + 2 < T) {
    if (h_in >= 0 && h_in < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[2] = load_a<REMAINDER>(A + ((2 * H + 0) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[3] = load_a<REMAINDER>(A + ((2 * H + 0) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[4] = load_a<REMAINDER>(A + ((2 * H + 0) * W + 2) * K, mask_v);
      }
    }

    if (h_in + 1 >= 0 && h_in + 1 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[5] = load_a<REMAINDER>(A + ((2 * H + 1) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[6] = load_a<REMAINDER>(A + ((2 * H + 1) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[7] = load_a<REMAINDER>(A + ((2 * H + 1) * W + 2) * K, mask_v);
      }
    }
  }

  inner_prod_packed_<8, SUM_A, REMAINDER, true /* acc */>(
      a_v, reinterpret_cast<const __m256i*>(Bp) + 16, C, remainder, a_sum_temp);
  if (SUM_A) {
    a_sum[0] = _mm256_add_epi32(a_sum[0], a_sum_temp[0]);
    a_sum[1] = _mm256_add_epi32(a_sum[1], a_sum_temp[1]);
    a_sum[2] = _mm256_add_epi32(a_sum[2], a_sum_temp[2]);
    a_sum[3] = _mm256_add_epi32(a_sum[3], a_sum_temp[3]);
  }

  a_v[0] = A_zero_point_v;
  a_v[1] = A_zero_point_v;
  a_v[2] = A_zero_point_v;

  if (t_in + 2 >= 0 && t_in + 2 < T) {
    if (h_in + 2 >= 0 && h_in + 2 < H) {
      if (w_in >= 0 && w_in < W) {
        a_v[0] = load_a<REMAINDER>(A + ((2 * H + 2) * W + 0) * K, mask_v);
      }
      if (w_in + 1 >= 0 && w_in + 1 < W) {
        a_v[1] = load_a<REMAINDER>(A + ((2 * H + 2) * W + 1) * K, mask_v);
      }
      if (w_in + 2 >= 0 && w_in + 2 < W) {
        a_v[2] = load_a<REMAINDER>(A + ((2 * H + 2) * W + 2) * K, mask_v);
      }
    }
  }

  inner_prod_packed_<3, SUM_A, REMAINDER, true /* acc */>(
      a_v, reinterpret_cast<const __m256i*>(Bp) + 24, C, remainder, a_sum_temp);

  if (SUM_A) {
    a_sum[0] = _mm256_add_epi32(a_sum[0], a_sum_temp[0]);
    a_sum[1] = _mm256_add_epi32(a_sum[1], a_sum_temp[1]);
    a_sum[2] = _mm256_add_epi32(a_sum[2], a_sum_temp[2]);
    a_sum[3] = _mm256_add_epi32(a_sum[3], a_sum_temp[3]);

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
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_3x3x3_kernel_(
    int T,
    int H,
    int W,
    int K,
    int t,
    int h,
    int w,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const int8_t* Bp,
    float C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    int32_t* row_offsets,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    float act_times_w_scale) {
  constexpr int R = 3, S = 3;
  constexpr int PAD_P = 1, PAD_T = 1, PAD_B = 1, PAD_L = 1, PAD_R = 1;
  int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int t_in = -PAD_P + t * stride_t;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  int k;
  for (k = 0; k < K / 32 * 32; k += 32) {
    inner_prod_3x3x3_packed_<!B_SYMMETRIC /*SUM_A*/>(
        T,
        H,
        W,
        K,
        t_in,
        h_in,
        w_in,
        A + ((t_in * H + h_in) * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 28,
        &B_zero_point,
        C_int32 + k,
        0,
        B_SYMMETRIC ? nullptr : &row_offsets[k]);
  }
  int remainder = K - k;
  if (remainder) {
    inner_prod_3x3x3_packed_<!B_SYMMETRIC /*SUM_A*/, true>(
        T,
        H,
        W,
        K,
        t_in,
        h_in,
        w_in,
        A + ((t_in * H + h_in) * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 28,
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
      B_SYMMETRIC>(
      A_zero_point,
      &C_multiplier,
      C_zero_point,
      C_int32,
      C_uint8 + ((t * H_OUT + h) * W_OUT + w) * K,
      K,
      row_offsets,
      col_offsets,
      bias,
      &act_times_w_scale);
}

template <bool FUSE_RELU, bool HAS_BIAS, bool A_SYMMETRIC, typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_3x3x3_per_channel_quantization_kernel_(
    int T,
    int H,
    int W,
    int K,
    int t,
    int h,
    int w,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const int8_t* Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    int32_t* row_offsets,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale) {
  constexpr int R = 3, S = 3;
  constexpr int PAD_P = 1, PAD_T = 1, PAD_B = 1, PAD_L = 1, PAD_R = 1;
  int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int t_in = -PAD_P + t * stride_t;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  int k;
  for (k = 0; k < K / 32 * 32; k += 32) {
    inner_prod_3x3x3_packed_<
        true, /*SUM_A*/
        false, /*remainder*/
        true /*per-channel*/>(
        T,
        H,
        W,
        K,
        t_in,
        h_in,
        w_in,
        A + ((t_in * H + h_in) * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 28,
        B_zero_point + k,
        C_int32 + k,
        0,
        &row_offsets[k]);
  }
  int remainder = K - k;
  if (remainder) {
    inner_prod_3x3x3_packed_<
        true, /*SUM_A*/
        true, /*remainder*/
        true /*per-channel*/>(
        T,
        H,
        W,
        K,
        t_in,
        h_in,
        w_in,
        A + ((t_in * H + h_in) * W + w_in) * K + k,
        A_zero_point,
        Bp + k * 28,
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
      false /*B_SYMM*/>(
      A_zero_point,
      C_multiplier,
      C_zero_point,
      C_int32,
      C_uint8 + ((t * H_OUT + h) * W_OUT + w) * K,
      K,
      row_offsets,
      col_offsets,
      bias,
      act_times_w_scale);
}

template <
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_3x3x3_pad_1_(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  assert(K % 8 == 0);
  constexpr int K_T = 3, K_H = 3, K_W = 3;
  constexpr int PAD_P = 1, PAD_N = 1, PAD_T = 1, PAD_B = 1, PAD_L = 1,
                PAD_R = 1;
  int T_OUT = (T + PAD_P + PAD_N - K_T) / stride_t + 1;
  int H_OUT = (H + PAD_T + PAD_B - K_H) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - K_W) / stride_w + 1;
  const int8_t* Bp = B.PackedMat();

  int32_t* row_offsets = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));

  int n_begin, n_end, t_begin, t_end, h_begin, h_end;
  // Reuse the 3-dim partition scheme for parallelization in matrix
  // multiplication.
  thread_type_t th_info =
      fbgemmGetThreadPartition(N, T_OUT, H_OUT, thread_id, num_threads);
  // Calculate the begin and end index along the batch (N) dimension
  fbgemmPartition1D(
      th_info.g_thread_id, th_info.g_num_threads, N, n_begin, n_end);
  // Calculate the begin and end index along the T dimension
  fbgemmPartition1D(
      th_info.m_thread_id, th_info.m_num_threads, T_OUT, t_begin, t_end);
  // Calculate the begin and end index along the H dimension
  fbgemmPartition1D(
      th_info.n_thread_id, th_info.n_num_threads, H_OUT, h_begin, h_end);

  for (int n = n_begin; n < n_end; ++n) {
    const uint8_t* A_base = A + n * T * H * W * K;
    uint8_t* C_uint8_base = C_uint8 + n * T_OUT * H_OUT * W_OUT * K;

    for (int t = t_begin; t < t_end; ++t) {
      for (int h = h_begin; h < h_end; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          depthwise_3x3x3_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC>(
              T,
              H,
              W,
              K,
              t,
              h,
              w,
              stride_t,
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
        } // w
      } // h
    } // t
  } // for each n
  fbgemmAlignedFree(row_offsets);
};

template <bool FUSE_RELU, bool HAS_BIAS, bool A_SYMMETRIC, typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_3x3x3_per_channel_quantization_pad_1_(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  assert(K % 8 == 0);
  constexpr int K_T = 3, K_H = 3, K_W = 3;
  constexpr int PAD_P = 1, PAD_N = 1, PAD_T = 1, PAD_B = 1, PAD_L = 1,
                PAD_R = 1;
  int T_OUT = (T + PAD_P + PAD_N - K_T) / stride_t + 1;
  int H_OUT = (H + PAD_T + PAD_B - K_H) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - K_W) / stride_w + 1;
  const int8_t* Bp = B.PackedMat();

  int32_t* row_offsets = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));

  int n_begin, n_end, t_begin, t_end, h_begin, h_end;
  // Reuse the 3-dim partition scheme for parallelization in matrix
  // multiplication.
  thread_type_t th_info =
      fbgemmGetThreadPartition(N, T_OUT, H_OUT, thread_id, num_threads);
  // Calculate the begin and end index along the batch (N) dimension
  fbgemmPartition1D(
      th_info.g_thread_id, th_info.g_num_threads, N, n_begin, n_end);
  // Calculate the begin and end index along the T dimension
  fbgemmPartition1D(
      th_info.m_thread_id, th_info.m_num_threads, T_OUT, t_begin, t_end);
  // Calculate the begin and end index along the H dimension
  fbgemmPartition1D(
      th_info.n_thread_id, th_info.n_num_threads, H_OUT, h_begin, h_end);

  for (int n = n_begin; n < n_end; ++n) {
    const uint8_t* A_base = A + n * T * H * W * K;
    uint8_t* C_uint8_base = C_uint8 + n * T_OUT * H_OUT * W_OUT * K;

    for (int t = t_begin; t < t_end; ++t) {
      for (int h = h_begin; h < h_end; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          depthwise_3x3x3_per_channel_quantization_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              BIAS_TYPE>(
              T,
              H,
              W,
              K,
              t,
              h,
              w,
              stride_t,
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
        } // w
      } // h
    } // t
  } // for each n

  fbgemmAlignedFree(row_offsets);
};

// Dispatch A_SYMMETRIC and B_SYMMETRIC
template <bool FUSE_RELU, bool HAS_BIAS, typename BIAS_TYPE>
static void depthwise_3x3x3_pad_1_(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  int32_t* C_int32_temp = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));
  if (A_zero_point == 0 || col_offsets == nullptr) {
    if (B_zero_point == 0) {
      depthwise_3x3x3_pad_1_<
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          true /*B_symmetric*/,
          BIAS_TYPE>(
          N,
          T,
          H,
          W,
          K,
          stride_t,
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
      depthwise_3x3x3_pad_1_<
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE>(
          N,
          T,
          H,
          W,
          K,
          stride_t,
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
      depthwise_3x3x3_pad_1_<
          FUSE_RELU,
          HAS_BIAS,
          false /*A_symmetric*/,
          true /*B_symmetric*/,
          BIAS_TYPE>(
          N,
          T,
          H,
          W,
          K,
          stride_t,
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
      depthwise_3x3x3_pad_1_<
          FUSE_RELU,
          HAS_BIAS,
          false /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE>(
          N,
          T,
          H,
          W,
          K,
          stride_t,
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
template <bool FUSE_RELU, typename BIAS_TYPE>
static void depthwise_3x3x3_pad_1_(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (bias) {
    depthwise_3x3x3_pad_1_<FUSE_RELU, true /*HAS_BIAS*/, BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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
    depthwise_3x3x3_pad_1_<FUSE_RELU, false /*HAS_BIAS*/, BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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

// Dispatch FUSE_RELU
template <typename BIAS_TYPE>
void depthwise_3x3x3_pad_1(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (B.GetKernelProduct() != 3 * 3 * 3) {
    string msg =
        "[FBGEMM_CONV_ERROR] Packed weight is expected to have kernel_prod " +
        to_string(3 * 3 * 3) + " but has " + to_string(B.GetKernelProduct());
    throw logic_error(msg);
  }
  if (stride_t == 0 || stride_h == 0 || stride_w == 0 || num_threads == 0) {
    assert(
        0 &&
        "stride_t == 0 || stride_h == 0 || stride_w == 0 || num_threads == 0");
    return;
  }
  if (N == 0) {
    // In C2, batch size 0 is allowed, so we should just early return.
    return;
  }
  if (fuse_relu) {
    depthwise_3x3x3_pad_1_<true /*FUSE_RELU*/, BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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
    depthwise_3x3x3_pad_1_<false /*FUSE_RELU*/, BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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
template <bool FUSE_RELU, bool HAS_BIAS, typename BIAS_TYPE>
static void depthwise_3x3x3_per_channel_quantization_pad_1_(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
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
    depthwise_3x3x3_per_channel_quantization_pad_1_<
        FUSE_RELU,
        HAS_BIAS,
        true /*A_SYMM*/,
        BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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
    depthwise_3x3x3_per_channel_quantization_pad_1_<
        FUSE_RELU,
        HAS_BIAS,
        false /*A_SYMM*/,
        BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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
  fbgemmAlignedFree(C_int32_temp);
}

// Dispatch HAS_BIAS
template <bool FUSE_RELU, typename BIAS_TYPE>
static void depthwise_3x3x3_per_channel_quantization_pad_1_(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (bias) {
    depthwise_3x3x3_per_channel_quantization_pad_1_<
        FUSE_RELU,
        true /* HAS_BIAS */,
        BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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
    depthwise_3x3x3_per_channel_quantization_pad_1_<
        FUSE_RELU,
        false /* HAS_BIAS */,
        BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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

// Dispatch FUSE_RELU
template <typename BIAS_TYPE>
void depthwise_3x3x3_per_channel_quantization_pad_1(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (B.GetKernelProduct() != 3 * 3 * 3) {
    string msg =
        "[FBGEMM_CONV_ERROR] Packed weight is expected to have kernel_prod " +
        to_string(3 * 3 * 3) + " but has " + to_string(B.GetKernelProduct());
    throw logic_error(msg);
  }
  if (stride_t == 0 || stride_h == 0 || stride_w == 0 || num_threads == 0) {
    assert(
        0 &&
        "stride_t == 0 || stride_h == 0 || stride_w == 0 || num_threads == 0");
    return;
  }
  if (N == 0) {
    // In C2, batch size 0 is allowed, so we should just early return.
    return;
  }
  if (fuse_relu) {
    depthwise_3x3x3_per_channel_quantization_pad_1_<
        true /* FUSE_RELU */,
        BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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
    depthwise_3x3x3_per_channel_quantization_pad_1_<
        false /* FUSE_RELU */,
        BIAS_TYPE>(
        N,
        T,
        H,
        W,
        K,
        stride_t,
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

template FBGEMM_API void depthwise_3x3x3_pad_1(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3x3x3_pad_1(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3x3x3_per_channel_quantization_pad_1(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3x3x3_per_channel_quantization_pad_1(
    int N,
    int T,
    int H,
    int W,
    int K,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

} // namespace fbgemm
