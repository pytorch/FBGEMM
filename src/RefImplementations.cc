/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "RefImplementations.h"

#include "fbgemm/Types.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>

using namespace std;

namespace fbgemm {

void requantize_u8acc32_ref(
    int M,
    int N,
    int ld,
    const int32_t* inp,
    uint8_t* out,
    int32_t C_multiplier,
    int32_t C_right_shift,
    int32_t C_zero_point,
    int32_t A_zero_point,
    int32_t B_zero_point,
    const int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu) {
  int64_t nudge = 1ll << std::max(0, C_right_shift - 1);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t raw = inp[i * ld + j];
      if (A_zero_point) {
        raw -= A_zero_point * col_offsets[j];
      }
      if (B_zero_point) {
        raw -= B_zero_point * row_offsets[i];
      }
      if (bias) {
        raw += bias[j];
      }

      int64_t ab_64 =
          static_cast<int64_t>(raw) * static_cast<int64_t>(C_multiplier);
      int64_t rounded = ((ab_64 + nudge) >> C_right_shift) + C_zero_point;

      out[i * ld + j] = std::max(
          fuse_relu ? static_cast<int64_t>(C_zero_point) : 0l,
          std::min(static_cast<int64_t>(255l), rounded));
    }
  }
}

void requantize_u8acc32_ref(
    int M,
    int N,
    int ld,
    const int32_t* inp,
    uint8_t* out,
    const float* C_multiplier,
    int32_t C_zero_point,
    int32_t A_zero_point,
    const int32_t* B_zero_point,
    const int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias,
    int ncols_per_quant_group,
    bool fuse_relu) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t raw = inp[i * ld + j];
      if (A_zero_point) {
        raw -= A_zero_point * col_offsets[j];
      }
      raw -= B_zero_point[j / ncols_per_quant_group] * row_offsets[i];
      if (bias) {
        raw += bias[j];
      }

      float result = raw * C_multiplier[j / ncols_per_quant_group];
      long rounded = lrintf(result) + C_zero_point;
      out[i * ld + j] = std::max(
          fuse_relu ? static_cast<long>(C_zero_point) : 0l,
          std::min(255l, rounded));
    }
  }
}

void matmul_u8i8acc32_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const uint8_t* Aint8,
    const int8_t* Bint8,
    int32_t* Cint32) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<int32_t>(Aint8[i * lda + k]) *
            static_cast<int32_t>(Bint8[k * ldb + j]);
      }
      Cint32[i * ldc + j] = sum;
    }
  }
}

void matmul_u8i8acc16_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int brow,
    const uint8_t* Aint8,
    const int8_t* Bint8,
    int32_t* Cint32) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t sum = 0, sum_32bit = 0;
      for (int k = 0; k < K; k += 2) {
        int a0 = Aint8[i * lda + k];
        int b0 = Bint8[k * ldb + j];
        int a1 = 0, b1 = 0;
        if (k + 1 < K) {
          a1 = Aint8[i * lda + k + 1];
          b1 = Bint8[(k + 1) * ldb + j];
        }
        sum = clip_16bit(sum + clip_16bit(a0 * b0 + a1 * b1));
        if ((k % brow) == (brow - 2)) {
          sum_32bit += sum;
          sum = 0;
        }
      }
      Cint32[i * ldc + j] = sum_32bit + sum;
    }
  }
}

void matmul_fp_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const float* Afp32,
    const float* Bfp32,
    float* Cfp32) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += Afp32[i * lda + k] * Bfp32[k * ldb + j];
      }
      Cfp32[i * ldc + j] = sum;
    }
  }
}

void cblas_sgemm_ref(
    const matrix_op_t transa,
    const matrix_op_t transb,
    const int m,
    const int n,
    const int k,
    float alpha,
    const float* Afp32,
    int lda,
    const float* Bfp32,
    int ldb,
    float beta,
    float* Cfp32,
    int ldc) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0;
      for (int p = 0; p < k; ++p) {
        float a =
            (transa == matrix_op_t::NoTranspose ? Afp32[i * lda + p]
                                                : Afp32[p * lda + i]);
        float b =
            (transb == matrix_op_t::NoTranspose ? Bfp32[p * ldb + j]
                                                : Bfp32[j * ldb + p]);
        sum += a * b;
      }
      if (beta == 0) {
        Cfp32[i * ldc + j] = alpha * sum;
      } else {
        Cfp32[i * ldc + j] = alpha * sum + beta * Cfp32[i * ldc + j];
      }
    }
  }
}

void row_offsets_u8acc32_ref(
    int M,
    int K,
    int ld,
    const uint8_t* Aint8,
    int32_t* row_offsets) {
  // row offset
  for (int i = 0; i < M; ++i) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += static_cast<int32_t>(Aint8[i * ld + k]);
    }
    row_offsets[i] = sum;
  }
}

void col_offsets_with_zero_pt_s8acc32_ref(
    int K,
    int N,
    int ld,
    const int8_t* Bint8,
    const int32_t* B_zero_point,
    int32_t* col_offsets,
    int ncols_per_quant_group) {
  for (int j = 0; j < N; ++j) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += Bint8[k * ld + j];
    }
    col_offsets[j] = sum - B_zero_point[j / ncols_per_quant_group] * K;
  }
}

void spmdm_ref(
    int M,
    const uint8_t* A,
    int lda,
    fbgemm::CompressedSparseColumn& B,
    bool accumulation,
    int32_t* C,
    int ldc,
    int groups /*=1*/) {
  int N = B.NumOfCols();
  assert(N % groups == 0);
  if (!accumulation) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] = 0;
      }
    }
  }
  for (int g = 0; g < groups; ++g) {
    for (int j = g * (N / groups); j < (g + 1) * (N / groups); ++j) {
      for (int k = B.ColPtr()[j]; k < B.ColPtr()[j + 1]; ++k) {
        int row = g * B.NumOfRows() + B.RowIdx()[k];
        int w = B.Values()[k];
        for (int i = 0; i < M; ++i) {
          C[i * ldc + j] += A[i * lda + row] * w;
        }
      }
    } // for each column of B
  } // for each group
}

int32_t clip_16bit(int32_t x) {
  if (x > numeric_limits<int16_t>::max()) {
    return std::min<int>(numeric_limits<int16_t>::max(), x);
  } else if (x < numeric_limits<int16_t>::min()) {
    return std::max<int>(numeric_limits<int16_t>::min(), x);
  } else {
    return x;
  }
}

/* Imitate the Im2Col<float, CPUContext, StorageOrder::NHWC> function
 * from caffe2/utils/math_cpu.cc
 * NHWC StorageOrder/Layout
 * A:  NHWC: NH_0W_0 x C_0
 * Ao: NHWC: NH_1W_1 x G RS C_0/G
 */
template <>
void im2col_ref(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    uint8_t* Ao) {
  int IC = conv_p.IC;
  int G = conv_p.G;
  assert(IC % G == 0);
  array<int, 2> IN_DIM = conv_p.IN_DIM;
  array<int, 2> OUT_DIM = conv_p.OUT_DIM;
  array<int, 2> K = conv_p.K;

  for (int n = 0; n < conv_p.MB; ++n) {
    for (int h = 0; h < OUT_DIM[0]; ++h) {
      for (int w = 0; w < OUT_DIM[1]; ++w) {
        for (int r = 0; r < K[0]; ++r) {
          int h_in =
              -conv_p.pad[0] + h * conv_p.stride[0] + r * conv_p.dilation[0];
          for (int s = 0; s < K[1]; ++s) {
            int w_in =
                -conv_p.pad[1] + w * conv_p.stride[1] + s * conv_p.dilation[1];
            if (h_in < 0 || h_in >= IN_DIM[0] || w_in < 0 ||
                w_in >= IN_DIM[1]) {
              for (int g = 0; g < G; ++g) {
                memset(
                    Ao +
                        (((((n * OUT_DIM[0] + h) * OUT_DIM[1] + w) * G + g) *
                              K[0] +
                          r) *
                             K[1] +
                         s) *
                            (IC / G),
                    A_zero_point,
                    sizeof(uint8_t) * (IC / G));
              }
            } else {
              for (int g = 0; g < G; ++g) {
                memcpy(
                    Ao +
                        (((((n * OUT_DIM[0] + h) * OUT_DIM[1] + w) * G + g) *
                              K[0] +
                          r) *
                             K[1] +
                         s) *
                            (IC / G),
                    A + ((n * IN_DIM[0] + h_in) * IN_DIM[1] + w_in) * IC +
                        g * (IC / G),
                    sizeof(uint8_t) * (IC / G));
              }
            }
          } // for each s
        } // for each r
      } // for each w
    } // for each h
  } // for each n
}

/* Imitate the Im2Col<float, CPUContext, StorageOrder::NHWC> function
 * from caffe2/utils/math_cpu.cc
 * NHWC StorageOrder/Layout
 * A:  NHWC: NT_0H_0W_0 x C_0
 * Ao: NHWC: NT_1H_1W_1 x G QRS C_0/G
 */
template <>
void im2col_ref(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    uint8_t* Ao) {
  int IC = conv_p.IC;
  int G = conv_p.G;
  assert(IC % G == 0);
  array<int, 3> IN_DIM = conv_p.IN_DIM;
  array<int, 3> OUT_DIM = conv_p.OUT_DIM;
  array<int, 3> K = conv_p.K;

  for (int n = 0; n < conv_p.MB; ++n) {
    for (int t = 0; t < OUT_DIM[0]; ++t) {
      for (int h = 0; h < OUT_DIM[1]; ++h) {
        for (int w = 0; w < OUT_DIM[2]; ++w) {
          for (int q = 0; q < K[0]; ++q) {
            int t_in =
                -conv_p.pad[0] + t * conv_p.stride[0] + q * conv_p.dilation[0];
            for (int r = 0; r < K[1]; ++r) {
              int h_in = -conv_p.pad[1] + h * conv_p.stride[1] +
                  r * conv_p.dilation[1];
              for (int s = 0; s < K[2]; ++s) {
                int w_in = -conv_p.pad[2] + w * conv_p.stride[2] +
                    s * conv_p.dilation[2];
                if (t_in < 0 || t_in >= IN_DIM[0] || h_in < 0 ||
                    h_in >= IN_DIM[1] || w_in < 0 || w_in >= IN_DIM[2]) {
                  for (int g = 0; g < G; ++g) {
                    memset(
                        Ao +
                            (((((((n * OUT_DIM[0] + t) * OUT_DIM[1] + h) *
                                     OUT_DIM[2] +
                                 w) *
                                    G +
                                g) *
                                   K[0] +
                               q) *
                                  K[1] +
                              r) *
                                 K[2] +
                             s) *
                                (IC / G),
                        A_zero_point,
                        sizeof(uint8_t) * (IC / G));
                  }
                } else {
                  for (int g = 0; g < G; ++g) {
                    memcpy(
                        Ao +
                            (((((((n * OUT_DIM[0] + t) * OUT_DIM[1] + h) *
                                     OUT_DIM[2] +
                                 w) *
                                    G +
                                g) *
                                   K[0] +
                               q) *
                                  K[1] +
                              r) *
                                 K[2] +
                             s) *
                                (IC / G),
                        A +
                            (((n * IN_DIM[0] + t_in) * IN_DIM[1] + h_in) *
                                 IN_DIM[2] +
                             w_in) *
                                IC +
                            g * (IC / G),
                        sizeof(uint8_t) * (IC / G));
                  }
                }
              } // for each s
            } // for each r
          } // for each q
        } // for each w
      } // for each h
    } // for each t
  } // for each n
}

// 2D Conv
template <>
void conv_ref(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* B,
    int32_t* C) {
  // filters are assumed to be in G RS C/G x K format
  int IC = conv_p.IC;
  int OC = conv_p.OC;
  int G = conv_p.G;
  assert(IC % G == 0);
  assert(OC % G == 0);
  array<int, 2> IN_DIM = conv_p.IN_DIM;
  array<int, 2> OUT_DIM = conv_p.OUT_DIM;
  array<int, 2> K = conv_p.K;

  for (int n = 0; n < conv_p.MB; ++n) {
    for (int h = 0; h < OUT_DIM[0]; ++h) {
      for (int w = 0; w < OUT_DIM[1]; ++w) {
        for (int g = 0; g < G; ++g) {
          for (int m = 0; m < OC / G; ++m) {
            int sum = 0;
            for (int r = 0; r < K[0]; ++r) {
              int h_in = -conv_p.pad[0] + h * conv_p.stride[0] +
                  r * conv_p.dilation[0];
              for (int s = 0; s < K[1]; ++s) {
                int w_in = -conv_p.pad[1] + w * conv_p.stride[1] +
                    s * conv_p.dilation[1];
                for (int c = 0; c < IC / G; ++c) {
                  int a = h_in < 0 || h_in >= IN_DIM[0] || w_in < 0 ||
                          w_in >= IN_DIM[1]
                      ? A_zero_point
                      : A[((n * IN_DIM[0] + h_in) * IN_DIM[1] + w_in) * IC +
                          g * (IC / G) + c];
                  int b =
                      B[(((g * K[0] + r) * K[1] + s) * (IC / G) + c) *
                            (OC / G) +
                        m];
                  sum += a * b;
                } // for each c
              } // for each s
            } // for each r
            C[((n * OUT_DIM[0] + h) * OUT_DIM[1] + w) * OC + g * (OC / G) + m] =
                sum;
          } // for each m
        } // for each group
      } // for each w
    } // for each h
  } // for each n
}

// 3D Conv
template <>
void conv_ref(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* B,
    int32_t* C) {
  // filters are assumed to be in G QRS C/G x K format
  int IC = conv_p.IC;
  int OC = conv_p.OC;
  int G = conv_p.G;
  assert(IC % G == 0);
  assert(OC % G == 0);
  array<int, 3> IN_DIM = conv_p.IN_DIM;
  array<int, 3> OUT_DIM = conv_p.OUT_DIM;
  array<int, 3> K = conv_p.K;

  for (int n = 0; n < conv_p.MB; ++n) {
    for (int t = 0; t < OUT_DIM[0]; ++t) {
      for (int h = 0; h < OUT_DIM[1]; ++h) {
        for (int w = 0; w < OUT_DIM[2]; ++w) {
          for (int g = 0; g < G; ++g) {
            for (int m = 0; m < OC / G; ++m) {
              int sum = 0;
              for (int q = 0; q < K[0]; ++q) {
                int t_in = -conv_p.pad[0] + t * conv_p.stride[0] +
                    q * conv_p.dilation[0];
                for (int r = 0; r < K[1]; ++r) {
                  int h_in = -conv_p.pad[1] + h * conv_p.stride[1] +
                      r * conv_p.dilation[1];
                  for (int s = 0; s < K[2]; ++s) {
                    int w_in = -conv_p.pad[2] + w * conv_p.stride[2] +
                        s * conv_p.dilation[2];
                    for (int c = 0; c < IC / G; ++c) {
                      int a = t_in < 0 || t_in >= IN_DIM[0] || h_in < 0 ||
                              h_in >= IN_DIM[1] || w_in < 0 || w_in >= IN_DIM[2]
                          ? A_zero_point
                          : A[(((n * IN_DIM[0] + t_in) * IN_DIM[1] + h_in) *
                                   IN_DIM[2] +
                               w_in) *
                                  IC +
                              g * (IC / G) + c];
                      int b =
                          B[((((g * K[0] + q) * K[1] + r) * K[2] + s) *
                                 (IC / G) +
                             c) *
                                (OC / G) +
                            m];
                      sum += a * b;
                    } // for each c
                  } // for each s
                } // for each r
              } // for each q
              C[(((n * OUT_DIM[0] + t) * OUT_DIM[1] + h) * OUT_DIM[2] + w) *
                    OC +
                g * (OC / G) + m] = sum;
            } // for each m
          } // for each group
        } // for each w
      } // for each h
    } // for each t
  } // for each n
}

template <int SPATIAL_DIM>
void transposeConvWeights(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest) {
  int G = conv_p.G;
  int IC_per_G = conv_p.IC / conv_p.G;
  int OC_per_G = conv_p.OC / conv_p.G;

  assert(
      (SPATIAL_DIM == 3 || SPATIAL_DIM == 2) &&
      "Only 2D and 3D convolutions are supported");
  if (SPATIAL_DIM == 2) {
    int R = conv_p.K[0];
    int S = conv_p.K[1];
    // Transforms weights from  G K/G (R S C/G) to G (R S C/G) K/G format.
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        for (int k = 0; k < OC_per_G; ++k) {
          for (int g = 0; g < G; ++g) {
            for (int c = 0; c < IC_per_G; ++c) {
              dest[(((g * R + r) * S + s) * IC_per_G + c) * OC_per_G + k] =
                  src[(((g * OC_per_G + k) * R + r) * S + s) * IC_per_G + c];
            }
          }
        }
      }
    }
  } else {
    // Transforms weights from  G K/G (T R S C/G) to G (T R S C/G) K/G format.
    int T = conv_p.K[0];
    int R = conv_p.K[1];
    int S = conv_p.K[2];
    for (int t = 0; t < T; ++t) {
      for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
          for (int k = 0; k < OC_per_G; ++k) {
            for (int g = 0; g < G; ++g) {
              for (int c = 0; c < IC_per_G; ++c) {
                dest
                    [((((g * T + t) * R + r) * S + s) * IC_per_G + c) *
                         OC_per_G +
                     k] =
                        src[((((g * OC_per_G + k) * T + t) * R + r) * S + s) *
                                IC_per_G +
                            c];
              }
            }
          }
        }
      }
    }
  }
}

template void transposeConvWeights(
    const conv_param_t<2>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest);

template void transposeConvWeights(
    const conv_param_t<3>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest);

} // namespace fbgemm
