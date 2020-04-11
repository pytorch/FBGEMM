/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "fbgemm/Utils.h"
#include "fbgemm/UtilsAvx2.h"
#include "src/FbgemmI8DepthwiseAvx2-inl.h"
#include "src/GenerateI8Depthwise.h"
#include "src/MaskAvx2.h"

namespace fbgemm {

template <
    int S,
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    bool PER_CHANNEL_QUANTIZAITON,
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
    const std::int32_t* B_zero_point,
    const std::int8_t* Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    GenI8Depthwise::jit_kernel_signature* pregenerated_kernel = nullptr) {
  constexpr int PAD_T = (S - 1) / 2, PAD_L = (S - 1) / 2, PAD_R = (S - 1) / 2;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  constexpr int KERNEL_PROD_ALIGNED = (S * S + 1) / 2 * 2;

  int remainder = K % 32;
  if (remainder == 0) {
    remainder = 32;
  }

  GenI8Depthwise::jit_kernel_signature kernel = pregenerated_kernel
      ? *pregenerated_kernel
      : GenI8Depthwise().getOrCreate(
            /*D=*/2,
            S,
            /*compute_a_sum=*/!B_SYMMETRIC,
            PER_CHANNEL_QUANTIZAITON,
            remainder,
            0,
            0,
            /*top_skip=*/std::max(-h_in, 0),
            /*bottom_skip=*/std::max(h_in + S - H, 0),
            /*left_skip=*/std::max(-w_in, 0),
            /*right_skip=*/std::max(w_in + S - W, 0));

  kernel(
      A + (h_in * W + w_in) * K,
      Bp,
      C_int32,
      B_SYMMETRIC ? nullptr : row_offsets,
      H,
      W,
      K,
      internal::avx2_ps_or_epi32_combined_mask,
      A_zero_point,
      B_zero_point);

  requantize_<
      FUSE_RELU,
      HAS_BIAS,
      PER_CHANNEL_QUANTIZAITON,
      A_SYMMETRIC,
      B_SYMMETRIC,
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
    typename BIAS_TYPE,
    bool PER_CHANNEL_QUANTIZATION>
static ALWAYS_INLINE void depthwise_2d_(
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

#ifdef _MSC_VER
  int32_t* row_offsets = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));
#else
  alignas(64) int32_t row_offsets[(K + 31) / 32 * 32];
#endif

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

  GenI8Depthwise::jit_kernel_signature middle_kernel;

  for (int n = n_begin; n < n_end; ++n) {
    const std::uint8_t* A_base = A + n * H * W * K;
    std::uint8_t* C_uint8_base = C_uint8 + n * H_OUT * W_OUT * K;

    int h = 0;
    int w = 0;

    for (h = h_begin; h < PAD_T; ++h) {
      for (w = w_begin; w < PAD_L; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            PER_CHANNEL_QUANTIZATION,
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

      for (; w < std::min(W_OUT - PAD_R - stride_w + 1, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            PER_CHANNEL_QUANTIZATION,
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
            PER_CHANNEL_QUANTIZATION,
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

    // h <= H_OUT - PAD_B - stride_h
    // h <= (H + PAD_T + PAD_B - S) / stride_h + 1 - PAD_B - stride_h
    // h_in <= -PAD_T +
    // ((H + PAD_T + PAD_B - S) / stride_h + 1 - PAD_B - stride_h) * stride_h
    // Case 1) For stride_h == 1,
    // h_in <= -PAD_T + H + PAD_T + PAD_B - S + 1 - PAD_B - 1
    // h_in + S - H <= 0
    // Case 2) For stride_h == 2,
    // h_in <= -PAD_L +
    // H + PAD_T + PAD_B - S + 1 + (1 - PAD_B - stride_h) * stride_h
    // h_in + S - H <= PAD_B * (1 - stride_h) + 1 + (1 - stride_h) * stride_h
    //              <= -PAD_B + 1 - stride_h <= 0
    for (; h < std::min(H_OUT - PAD_B - stride_h + 1, h_end); ++h) {
      for (w = w_begin; w < PAD_L; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            PER_CHANNEL_QUANTIZATION,
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

      for (; w < std::min(W_OUT - PAD_R - stride_w + 1, w_end); ++w) {
        if (n == n_begin && w == std::max(PAD_L, w_begin)) {
          int remainder = K % 32;
          if (remainder == 0) {
            remainder = 32;
          }
          middle_kernel = GenI8Depthwise().getOrCreate(
              /*D=*/2,
              S,
              /*compute_a_sum=*/!B_SYMMETRIC,
              PER_CHANNEL_QUANTIZATION,
              remainder,
              0,
              0,
              0,
              0,
              0,
              0);
        }
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            PER_CHANNEL_QUANTIZATION,
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
            act_times_w_scale,
            &middle_kernel);
      }

      for (; w < w_end; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            PER_CHANNEL_QUANTIZATION,
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
      for (w = w_begin; w < PAD_L; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            PER_CHANNEL_QUANTIZATION,
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

      for (; w < std::min(W_OUT - PAD_R - stride_w + 1, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            PER_CHANNEL_QUANTIZATION,
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
            PER_CHANNEL_QUANTIZATION,
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

#ifdef _MSC_VER
  fbgemmAlignedFree(row_offsets);
#endif
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
#ifdef _MSC_VER
  int32_t* C_int32_temp = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));
#else
  alignas(64) int32_t C_int32_temp[(K + 31) / 32 * 32];
#endif
  if (A_zero_point == 0 || col_offsets == nullptr) {
    if (B_zero_point == 0) {
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          true /*B_symmetric*/,
          BIAS_TYPE,
          false /*PER_CHANNEL_QUANTIZAITON*/>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          &B_zero_point,
          B,
          &C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          &act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE,
          false /*PER_CHANNEL_QUANTIZAITON*/>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          &B_zero_point,
          B,
          &C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          &act_times_w_scale,
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
          BIAS_TYPE,
          false /*PER_CHANNEL_QUANTIZAITON*/>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          &B_zero_point,
          B,
          &C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          &act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          false /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE,
          false /*PER_CHANNEL_QUANTIZAITON*/>(
          N,
          H,
          W,
          K,
          stride_h,
          stride_w,
          A_zero_point,
          A,
          &B_zero_point,
          B,
          &C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          &act_times_w_scale,
          thread_id,
          num_threads);
    }
  }
#ifdef _MSC_VER
  fbgemmAlignedFree(C_int32_temp);
#endif
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
#ifdef _MSC_VER
  int32_t* C_int32_temp = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (K + 31) / 32 * 32 * sizeof(int32_t)));
#else
  alignas(64) int32_t C_int32_temp[(K + 31) / 32 * 32];
#endif
  if (A_zero_point == 0 || col_offsets == nullptr) {
    depthwise_2d_<
        S,
        FUSE_RELU,
        HAS_BIAS,
        true /*A_SYMM*/,
        false /*B_SYMM*/,
        BIAS_TYPE,
        true /*PER_CHANNEL_QUANTIZAITON*/>(
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
    depthwise_2d_<
        S,
        FUSE_RELU,
        HAS_BIAS,
        false /*A_SYMM*/,
        false /*B_SYMM*/,
        BIAS_TYPE,
        true /*PER_CHANNEL_QUANTIZAITON*/>(
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
#ifdef _MSC_VER
  fbgemmAlignedFree(C_int32_temp);
#endif
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
