/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmI8DirectconvAvx2.h"

#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>
#endif
#include <cassert>

#include "./DirectConv.h"
#include "./ExecuteKernel.h"
#include "./MaskAvx2.h"
#include "fbgemm/ConvUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/UtilsAvx2.h"

#include "./CodeGenHelpers.h"
#include "./OptimizedKernelsAvx2.h"
#include "./RefImplementations.h"
#include "./TransposeUtils.h"
#include "fbgemm/QuantUtilsAvx512.h"
namespace fbgemm {

PackedDirectConvMatrix::PackedDirectConvMatrix(
    int IC_per_G,
    int OC_per_G,
    int filter_prod,
    const int8_t* smat) {
  // Allocate packed arrays
  int kernel_prod_aligned = (filter_prod + 1) / 2 * 2;
  pmat_ = static_cast<int8_t*>(fbgemmAlignedAlloc(
      64,
      ((OC_per_G + 31) / 32 * 32) * kernel_prod_aligned * IC_per_G *
          sizeof(int8_t)));

  // the transposed weight layout: W[oc/8][r][s][ic/4][8][4]
  for (int g = 0; g < /* G */ 1; ++g) {
    for (int k = 0; k < OC_per_G; ++k) {
      for (int f = 0; f < filter_prod; ++f) {
        for (int c = 0; c < IC_per_G; ++c) {
          int ocB = k / 8;
          int ocb = k % 8;
          int icB = c / 4;
          int icb = c % 4;
          pmat_
              [((((g * (OC_per_G / 8) + ocB) * filter_prod + f) *
                     (IC_per_G / 4) +
                 icB) *
                    8 +
                ocb) *
                   4 +
               icb] =
                  smat[((g * OC_per_G + k) * filter_prod + f) * IC_per_G + c];
        }
      }
    }
  }
}

PackedDirectConvMatrix::~PackedDirectConvMatrix() {
  fbgemmAlignedFree(pmat_);
}

template <int kSpatialDim>
void PackedDirectConvMatrix::col_offsets_with_zero_pt_s8acc32_DirectConvT(
    const fbgemm::conv_param_t<kSpatialDim>& conv_p,
    std::int32_t* B_zero_point,
    std::vector<int32_t>& col_offsets,
    int ncols_per_quant_group) {
  // if use direct convolution implementation, compute the col_offsets
  // of the weight matrix at the first time of inference.
  // We need to know the shape of output matrix
  // to compute col_offsets for direct convolution.
  // Hence it cannot be called from inside weight packing function
  // at initialization stage like other quantized conv implementation.
  // Thus the col_offsets computation will be invoked at forward pass,
  // and only the first pass will prepare the col_offsets.
  if (first_call == false) {
    return;
  }
  int IC = conv_p.IC;
  int OC = conv_p.OC;

  int IN_DIM0 = conv_p.IN_DIM[0];
  int IN_DIM1 = conv_p.IN_DIM[1];
  int OUT_DIM0 = conv_p.OUT_DIM[0];
  int OUT_DIM1 = conv_p.OUT_DIM[1];
  int K0 = conv_p.K[0];
  int K1 = conv_p.K[1];
  int stride0 = conv_p.stride[0];
  int stride1 = conv_p.stride[1];

  int MDim = conv_p.MB * OUT_DIM0 * OUT_DIM1;
  int NDim = conv_p.OC / conv_p.G;
  // int KDim = K[0] * K[1] * conv_p.IC;

  col_offsets.resize(MDim * NDim, 0);
  std::fill(col_offsets.begin(), col_offsets.end(), 0);
  std::vector<int> count(MDim * NDim, 0);

  for (int oc = 0; oc < OC; oc++) {
    for (int ih = 0; ih < IN_DIM0; ih++) {
      for (int iw = 0; iw < IN_DIM1; iw++) {
        for (int kh = 0; kh < K0; kh++) {
          for (int kw = 0; kw < K1; kw++) {
            for (int ic = 0; ic < IC; ic++) {
              int oh = ih * stride0 + kh;
              int ow = iw * stride1 + kw;
              col_offsets[(oh * OUT_DIM1 + ow) * OC + oc] += pmat_
                  [(((((oc / 8) * K0 + kh) * K1 + kw) * (IC / 4) + ic / 4) * 8 +
                    (oc % 8)) *
                       4 +
                   (ic % 4)];
              count[(oh * OUT_DIM1 + ow) * OC + oc]++;
            }
          }
        }
      }
    }
  }

  for (int oc = 0; oc < OC; oc++) {
    for (int oh = 0; oh < OUT_DIM0; oh++) {
      for (int ow = 0; ow < OUT_DIM1; ow++) {
        col_offsets[(oh * OUT_DIM1 + ow) * OC + oc] -=
            B_zero_point[oc / ncols_per_quant_group] *
            count[(oh * OUT_DIM1 + ow) * OC + oc];
      }
    }
  }

  first_call = false;
}

template FBGEMM_API void
PackedDirectConvMatrix::col_offsets_with_zero_pt_s8acc32_DirectConvT<1>(
    const fbgemm::conv_param_t<1>& conv_p,
    std::int32_t* B_zero_point,
    std::vector<int32_t>& col_offsets,
    int ncols_per_quant_group);

template FBGEMM_API void
PackedDirectConvMatrix::col_offsets_with_zero_pt_s8acc32_DirectConvT<2>(
    const fbgemm::conv_param_t<2>& conv_p,
    std::int32_t* B_zero_point,
    std::vector<int32_t>& col_offsets,
    int ncols_per_quant_group);

template FBGEMM_API void
PackedDirectConvMatrix::col_offsets_with_zero_pt_s8acc32_DirectConvT<3>(
    const fbgemm::conv_param_t<3>& conv_p,
    std::int32_t* B_zero_point,
    std::vector<int32_t>& col_offsets,
    int ncols_per_quant_group);

template <int SPATIAL_DIM>
void directConvRowSum(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const uint8_t* A,
    int32_t* inSum,
    int32_t* rowSum) {
  int IN0 = conv_p.IN_DIM[0];
  int IN1 = conv_p.IN_DIM[1];
  int IC = conv_p.IC;
  int K0 = conv_p.K[0];
  int K1 = conv_p.K[1];
  int OUT0 = conv_p.OUT_DIM[0];
  int OUT1 = conv_p.OUT_DIM[1];
  int stride = conv_p.stride[1];

  memset(rowSum, 0, sizeof(int32_t) * OUT0 * OUT1);

  for (int ih = 0; ih < IN0; ++ih) {
    for (int iw = 0; iw < IN1; ++iw) {
      inSum[ih * IN1 + iw] = reduceAvx2(A + ih * IN1 * IC + iw * IC, IC);
    }
  }

  for (int ih = 0; ih < IN0; ++ih) {
    for (int iw = 0; iw < IN1; iw++) {
      for (int r = 0; r < K0; ++r) {
        for (int s = 0; s < K1; ++s) {
          rowSum[(ih + r) * OUT1 + iw * stride + s] += inSum[ih * IN1 + iw];
        }
      }
    }
  }
  /*
    compare_buffers(
        rowSum,
        rowoffsets,
        OUT0,
        OUT1,
        OUT1,
        5);
  */
}

template void directConvRowSum<1>(
    const conv_param_t<1>& conv_p,
    const uint8_t* A,
    int32_t* inSum,
    int32_t* rowSum);

template void directConvRowSum<2>(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t* inSum,
    int32_t* rowSum);

template void directConvRowSum<3>(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t* inSum,
    int32_t* rowSum);

template <
    int SPATIAL_DIM,
    QuantizationGranularity Q_GRAN,
    bool FUSE_RELU,
    typename BIAS_TYPE>
void fbgemmDirectConv(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const uint8_t* Aint8,
    PackedDirectConvMatrix& Bint8_tr,
    uint8_t* C,
    int32_t* C_buffer,
    const ReQuantizeOutput<FUSE_RELU, Q_GRAN, BIAS_TYPE>& outProcess,
    const BIAS_TYPE* bias,
    // const int32_t* bias,
    int thread_id,
    int num_threads) {
  // support for single thread now,
  // will enable multithread later
  if (thread_id > 0 || thread_id >= num_threads) {
    return;
  }

  if (SPATIAL_DIM != 2) {
    assert(false && "1d/3d direct conv not supported");
  } else {
    if (conv_p.transposed) {
      DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
          jit_micro_kernel_fp_convT fn;
      DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
      /*
         fn = codeObj.getOrCreateDirectConvTrans<inst_set_t::avx2>(
         true, conv_p.stride[1]);
         */
      fn = codeObj.getOrCreateDirectConvTrans<inst_set_t::avx2>(
          true, conv_p.stride[1], conv_p.K[1]);

      int32_t* inSum = static_cast<int32_t*>(fbgemmAlignedAlloc(
          64, conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * sizeof(int32_t)));
      int32_t* rowSum = static_cast<int32_t*>(fbgemmAlignedAlloc(
          64, conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * sizeof(int32_t)));

      directConvRowSum(conv_p, Aint8, inSum, rowSum);
      int kernel_dim = conv_p.K[0] * conv_p.K[1];

      std::memset(
          C_buffer,
          0,
          sizeof(int32_t) * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC);
      std::memset(
          C,
          0,
          sizeof(int8_t) * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC);
      // no-op output process objects
      for (int i = 0; i < conv_p.OC; i += 8) {
        for (int j = 0; j < conv_p.IN_DIM[0]; j++) {
          fn(Aint8 + j * conv_p.IC * conv_p.IN_DIM[1],
             Bint8_tr.PackedMat() + i * kernel_dim * conv_p.IC,
             C_buffer + j * conv_p.OUT_DIM[1] * conv_p.OC + i,
             conv_p.IC,
             conv_p.OC,
             (conv_p.OC * conv_p.OUT_DIM[1] - conv_p.OC * conv_p.K[1]) * 4,
             conv_p.IN_DIM[1]);
        }
      }

      int32_t A_zero_point = outProcess.getAZeroPoint();
      const int32_t* B_zero_point = outProcess.getBZeroPoint();
      // const float* C_multiplier = outProcess.getCMultiplier();
      const int32_t* col_offsets = outProcess.getColOffsets();

      /*
      int groups = 1;
      if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        groups = conv_p.OC;
      }
      */
      requantizationParams_t<BIAS_TYPE> reqObj = {
          outProcess.getAZeroPoint(),
          outProcess.getBZeroPoint(),
          outProcess.getCZeroPoint(),
          outProcess.getCMultiplier(),
          rowSum, // rowOffsetBuf,
          outProcess.getColOffsets(),
          (outProcess.getBias()),
          static_cast<std::uint32_t>(conv_p.OC), // outProcess.getNCols(),
          1, // groups
          outProcess.getActWScale()};

      // Dispatch HAS_BIAS
      if (bias == nullptr) {
        // Dispatch A_SYMMETRIC and B_SYMMETRIC
        if (A_zero_point == 0 || col_offsets == nullptr) {
          if (Q_GRAN == QuantizationGranularity::TENSOR &&
              B_zero_point[0] == 0) {
            requantizeOutputProcessingAvx2<
                true,
                true,
                QuantizationGranularity::TENSOR,
                false, // HAS_BIAS,
                FUSE_RELU,
                BIAS_TYPE,
                true>(
                C,
                C_buffer,
                {0, conv_p.OUT_DIM[1] * conv_p.OUT_DIM[0], 0, conv_p.OC},
                conv_p.OC,
                conv_p.OC,
                reqObj);
          } else {
            requantizeOutputProcessingAvx2<
                true,
                false,
                Q_GRAN,
                false, // HAS_BIAS,
                FUSE_RELU,
                BIAS_TYPE,
                true>(
                C,
                C_buffer,
                {0, conv_p.OUT_DIM[1] * conv_p.OUT_DIM[0], 0, conv_p.OC},
                conv_p.OC,
                conv_p.OC,
                reqObj);
          }
        } else {
          if (Q_GRAN == QuantizationGranularity::TENSOR &&
              B_zero_point[0] == 0) {
            requantizeOutputProcessingAvx2<
                false,
                true,
                QuantizationGranularity::TENSOR,
                false, // HAS_BIAS,
                FUSE_RELU,
                BIAS_TYPE,
                true>(
                C,
                C_buffer,
                {0, conv_p.OUT_DIM[1] * conv_p.OUT_DIM[0], 0, conv_p.OC},
                conv_p.OC,
                conv_p.OC,
                reqObj);
          } else {
            requantizeOutputProcessingAvx2<
                false,
                false,
                Q_GRAN,
                false, // HAS_BIAS,
                FUSE_RELU,
                BIAS_TYPE,
                true>(
                C,
                C_buffer,
                {0, conv_p.OUT_DIM[1] * conv_p.OUT_DIM[0], 0, conv_p.OC},
                conv_p.OC,
                conv_p.OC,
                reqObj);
          }
        }
      } else { // has_bias == true

        // dispatch A_SYMMETRIC and B_SYMMETRIC
        if (A_zero_point == 0 || col_offsets == nullptr) {
          if (Q_GRAN == QuantizationGranularity::TENSOR &&
              B_zero_point[0] == 0) {
            requantizeOutputProcessingAvx2<
                true,
                true,
                QuantizationGranularity::TENSOR,
                true, // HAS_BIAS,
                FUSE_RELU,
                BIAS_TYPE,
                true>(
                C,
                C_buffer,
                {0, conv_p.OUT_DIM[1] * conv_p.OUT_DIM[0], 0, conv_p.OC},
                conv_p.OC,
                conv_p.OC,
                reqObj);
          } else {
            requantizeOutputProcessingAvx2<
                true,
                false,
                Q_GRAN,
                true, // HAS_BIAS,
                FUSE_RELU,
                BIAS_TYPE,
                true>(
                C,
                C_buffer,
                {0, conv_p.OUT_DIM[1] * conv_p.OUT_DIM[0], 0, conv_p.OC},
                conv_p.OC,
                conv_p.OC,
                reqObj);
          }
        } else {
          if (Q_GRAN == QuantizationGranularity::TENSOR &&
              B_zero_point[0] == 0) {
            requantizeOutputProcessingAvx2<
                false,
                true,
                QuantizationGranularity::TENSOR,
                true, // HAS_BIAS,
                FUSE_RELU,
                BIAS_TYPE,
                true>(
                C,
                C_buffer,
                {0, conv_p.OUT_DIM[1] * conv_p.OUT_DIM[0], 0, conv_p.OC},
                conv_p.OC,
                conv_p.OC,
                reqObj);
          } else {
            requantizeOutputProcessingAvx2<
                false,
                false,
                Q_GRAN,
                true, // HAS_BIAS,
                FUSE_RELU,
                BIAS_TYPE,
                true>(
                C,
                C_buffer,
                {0, conv_p.OUT_DIM[1] * conv_p.OUT_DIM[0], 0, conv_p.OC},
                conv_p.OC,
                conv_p.OC,
                reqObj);
          }
        }
      }
      fbgemmAlignedFree(inSum);
      fbgemmAlignedFree(rowSum);
    } // transposed conv
    else { // non-transposed conv
      assert(false && "non-transposed direct conv not integrated yet.");
    }
  } // else SPATIAL_DIM
}

#define INSTANTIATE_REQUANTIZE_SPATIAL_DIM(                        \
    SPATIAL_DIM, Q_GRAN, RELU, BIAS_TYPE)                          \
  template void FBGEMM_API                                         \
  fbgemmDirectConv<SPATIAL_DIM, Q_GRAN, RELU, BIAS_TYPE>(          \
      const conv_param_t<SPATIAL_DIM>& conv_p,                     \
      const uint8_t* Aint8,                                        \
      PackedDirectConvMatrix& Bint8_tr,                            \
      uint8_t* C,                                                  \
      int32_t* C_buffer,                                           \
      const ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>& outProcess, \
      const BIAS_TYPE* bias,                                       \
      int thread_id,                                               \
      int num_threads);

#define INSTANTIATE_REQUANTIZE_BIAS_TYPE(Q_GRAN, RELU, BIAS_TYPE) \
  INSTANTIATE_REQUANTIZE_SPATIAL_DIM(1, Q_GRAN, RELU, BIAS_TYPE)  \
  INSTANTIATE_REQUANTIZE_SPATIAL_DIM(2, Q_GRAN, RELU, BIAS_TYPE)  \
  INSTANTIATE_REQUANTIZE_SPATIAL_DIM(3, Q_GRAN, RELU, BIAS_TYPE)

#define INSTANTIATE_REQUANTIZE(Q_GRAN, RELU)            \
  INSTANTIATE_REQUANTIZE_BIAS_TYPE(Q_GRAN, RELU, float) \
  INSTANTIATE_REQUANTIZE_BIAS_TYPE(Q_GRAN, RELU, int32_t)

#define INSTANTIATE_Q_GRANS(RELU)                               \
  INSTANTIATE_REQUANTIZE(QuantizationGranularity::TENSOR, RELU) \
  INSTANTIATE_REQUANTIZE(QuantizationGranularity::GROUP, RELU)  \
  INSTANTIATE_REQUANTIZE(QuantizationGranularity::OUT_CHANNEL, RELU)

INSTANTIATE_Q_GRANS(true)
INSTANTIATE_Q_GRANS(false)

#undef INSTANTIATE_REQUANTIZE_SPATIAL_DIM
#undef INSTANTIATE_REQUANTIZE_BIAS_TYPE
#undef INSTANTIATE_REQUANTIZE
#undef INSTANTIATE_Q_GRANS
} // namespace fbgemm
