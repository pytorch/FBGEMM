/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "GroupwiseConv.h"

#include "RefImplementations.h"
#include "TransposeUtils.h"
#include "fbgemm/Fbgemm.h"

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
#include <chrono>
#endif
#include <cpuinfo.h>

namespace fbgemm {

using namespace std;

namespace {

template <int SPATIAL_DIM>
void calculateRowOffsets_(
    const conv_param_t<SPATIAL_DIM>& conv_param,
    const uint8_t* activations,
    int32_t* rowOffsetBuf,
    int32_t a_zero_point,
    int groupNum) {
  assert(SPATIAL_DIM == 2 && "3D conv not supported yet");

  int H_IN = conv_param.IN_DIM[0];
  int W_IN = conv_param.IN_DIM[1];
  int H_OUT = conv_param.OUT_DIM[0];
  int W_OUT = conv_param.OUT_DIM[1];

  int G = conv_param.G;
  int C_per_G = conv_param.IC / conv_param.G;
  int H_PAD = conv_param.pad[0];
  int W_PAD = conv_param.pad[1];
  // calculate row offset
  for (int h = 0; h < H_OUT; ++h) {
    for (int w = 0; w < W_OUT; ++w) {
      int32_t sum = 0;
      for (int r = 0; r < conv_param.K[0]; ++r) {
        int h_in = -H_PAD + h * conv_param.stride[0] + r;
        for (int s = 0; s < conv_param.K[1]; ++s) {
          int w_in = -W_PAD + w * conv_param.stride[1] + s;
          for (int c = 0; c < C_per_G; ++c) {
            if (h_in < 0 || h_in >= H_IN || w_in < 0 || w_in >= W_IN) {
              sum += a_zero_point;
            } else {
              sum += activations
                  [((h_in * W_IN + w_in) * G + groupNum) * C_per_G + c];
            }
          }
        }
      }
      rowOffsetBuf[h * W_OUT + w] = sum;
    }
  }
}

template <
    typename packed_W,
    typename outType,
    typename processOutputType,
    int SPATIAL_DIM>
void fbgemmGroupwiseConvBase_(
    const conv_param_t<SPATIAL_DIM>& conv_param,
    const std::uint8_t* activations,
    std::int32_t a_zero_point,
    std::int32_t* rowOffsetBuf,
    packed_W& packed_weights,
    outType* out,
    int32_t* outBuffer,
    const processOutputType& outProcess,
    int /* thread_id unused */,
    int /* num_threads unused */) {
  int MB = conv_param.MB;
  int H_in = conv_param.IN_DIM[0];
  int W_in = conv_param.IN_DIM[1];
  int G = conv_param.G;
  int K_per_G = conv_param.OC / G;
  int C_per_G = conv_param.IC / G;
  int R = conv_param.K[0];
  int S = conv_param.K[1];
  int oh_ow = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];
  int ih_iw = H_in * W_in;

  bool useAvx512 = fbgemmHasAvx512Support() && G >= 16;
  int simd_width = useAvx512 ? 512 : 256;
  int simd_width_i32 = simd_width / 32;
  int g_vec_block = std::max(simd_width_i32 / K_per_G, 1);
  int g_block = std::min<int>(simd_width_i32, G);
  assert(G % g_block == 0);

  assert(SPATIAL_DIM == 2 && "3D conv not supported yet");

  if (fbgemmOptimizedGConv<SPATIAL_DIM>(conv_param)) {
    int32_t* rowOffsetTrDest =
        rowOffsetBuf ? rowOffsetBuf + g_block * oh_ow : nullptr;
    assert(G % 8 == 0);
    for (int i = 0; i < MB; ++i) {
      const uint8_t* actStartBatch = activations + i * ih_iw * conv_param.IC;
      for (int gOuter = 0; gOuter < G; gOuter += g_block) {
        int gLimit = gOuter + simd_width_i32;
        // Work on 8 output channels at a time (8 * sizeof(int32_t) == 32B VLEN
        // of AVX2), and we need multiple groups if a group has not enough
        // number of channels.
        for (int g = gOuter; g < gLimit; g += g_vec_block) {
          int32_t* currOutBuf =
              outBuffer + i * oh_ow * conv_param.OC + g * K_per_G;
          const uint8_t* actStartGroup = actStartBatch + g * C_per_G;
          const int8_t* B =
              packed_weights.getBuf() + g * R * S * K_per_G * C_per_G;
          int32_t* row_buf =
              (g == gOuter && rowOffsetBuf) ? rowOffsetBuf : nullptr;

          constexpr int H_PAD = 1;
          int h = 0;
          if (useAvx512) {
            groupConvAvx512<true, false, SPATIAL_DIM>(
                conv_param,
                actStartGroup,
                a_zero_point,
                h,
                B,
                currOutBuf,
                row_buf);
          } else {
            groupConvAvx2<true, false, SPATIAL_DIM>(
                conv_param,
                actStartGroup,
                a_zero_point,
                h,
                B,
                currOutBuf,
                row_buf);
          }

          for (h = H_PAD; h <= (H_in + H_PAD - R) / conv_param.stride[0]; ++h) {
            if (useAvx512) {
              groupConvAvx512<false, false, SPATIAL_DIM>(
                  conv_param,
                  actStartGroup,
                  a_zero_point,
                  h,
                  B,
                  currOutBuf,
                  row_buf);
            } else {
              groupConvAvx2<false, false, SPATIAL_DIM>(
                  conv_param,
                  actStartGroup,
                  a_zero_point,
                  h,
                  B,
                  currOutBuf,
                  row_buf);
            }
          }

          if (h < conv_param.OUT_DIM[0]) {
            if (useAvx512) {
              groupConvAvx512<false, true, SPATIAL_DIM>(
                  conv_param,
                  actStartGroup,
                  a_zero_point,
                  h,
                  B,
                  currOutBuf,
                  row_buf);
            } else {
              groupConvAvx2<false, true, SPATIAL_DIM>(
                  conv_param,
                  actStartGroup,
                  a_zero_point,
                  h,
                  B,
                  currOutBuf,
                  row_buf);
            }
          }

          if (row_buf) {
            // Transpose to get row offsets in the format G x OH*OW
            if (useAvx512) {
              internal::transpose_16x16(
                  oh_ow,
                  16,
                  reinterpret_cast<const float*>(rowOffsetBuf),
                  16,
                  reinterpret_cast<float*>(rowOffsetTrDest),
                  oh_ow);
            } else {
              internal::transpose_8x8(
                  oh_ow,
                  8,
                  reinterpret_cast<const float*>(rowOffsetBuf),
                  8,
                  reinterpret_cast<float*>(rowOffsetTrDest),
                  oh_ow);
            }
          }

          // Output processing should be called for each group
          for (int j = 0; j < g_vec_block; ++j) {
            // calculateRowOffsets(
            // conv_param, actStartGroup, rowOffsetBuf, a_zero_point, j);
            int32_t* rowOffsetForCurG = rowOffsetTrDest
                ? rowOffsetTrDest + ((g - gOuter) + j) * oh_ow
                : nullptr;
            // compare_buffers(rowOffsetBuf, rowOffsetForCurG,
            // conv_param.IN_DIM[0]*conv_param.IN_DIM[1], 1, 1, 100);

            // outProcess expects rowOffsetBuf to contain row offsets for the
            // current group
            if (rowOffsetBuf) {
              memcpy(rowOffsetBuf, rowOffsetForCurG, oh_ow * sizeof(int32_t));
            }

            if (fbgemmHasAvx512Support() || fbgemmHasAvx2Support()) {
              // Currently use avx2 code also for avx512
              outProcess.template f<inst_set_t::avx2>(
                  out,
                  currOutBuf + j * K_per_G,
                  {i * oh_ow, oh_ow, (g + j) * K_per_G, K_per_G},
                  K_per_G * G,
                  K_per_G * G);
            } else {
              // TODO: Have default slower path
              assert(0 && "unsupported architecure");
            }
          } // j loop
        } // g loop
      } // gOuter loop
    } // i loop
  } else {
    // for the not supported cases, just execute the naive C implementation
    conv_ref(
        conv_param,
        activations,
        a_zero_point,
        packed_weights.getBuf(),
        outBuffer);
    for (int i = 0; i < conv_param.MB; ++i) {
      for (int g = 0; g < conv_param.G; ++g) {
        if (rowOffsetBuf) {
          calculateRowOffsets_<SPATIAL_DIM>(
              conv_param,
              activations +
                  i * conv_param.IN_DIM[0] * conv_param.IN_DIM[1] *
                      conv_param.IC,
              rowOffsetBuf,
              a_zero_point,
              g);
        }
        outProcess.template f<inst_set_t::anyarch>(
            out,
            outBuffer + i * oh_ow * conv_param.OC + g * K_per_G,
            {i * oh_ow, oh_ow, g * K_per_G, K_per_G},
            K_per_G * G,
            K_per_G * G);
      }
    }
  }
}

// dispatch inst_set
template <
    inst_set_t inst_set,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    int C_PER_G>
void requantizeOutputProcessingGConv(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t& r) {
  if (inst_set == inst_set_t::avx512) {
    requantizeOutputProcessingGConvAvx512<
        A_SYMMETRIC,
        B_SYMMETRIC,
        Q_GRAN,
        HAS_BIAS,
        FUSE_RELU,
        C_PER_G>(out, inp, block, ld_out, ld_in, r);
  } else if (inst_set == inst_set_t::avx2) {
    requantizeOutputProcessingGConvAvx2<
        A_SYMMETRIC,
        B_SYMMETRIC,
        Q_GRAN,
        HAS_BIAS,
        FUSE_RELU,
        C_PER_G>(out, inp, block, ld_out, ld_in, r);
  } else {
    assert(false);
  }
}

// dispatch FUSE_BIAS
template <
    inst_set_t inst_set,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool FUSE_RELU,
    int C_PER_G>
void requantizeOutputProcessingGConv(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t& r) {
  if (r.bias == nullptr) {
    requantizeOutputProcessingGConv<
        inst_set,
        A_SYMMETRIC,
        B_SYMMETRIC,
        Q_GRAN,
        false,
        FUSE_RELU,
        C_PER_G>(out, inp, block, ld_out, ld_in, r);
  } else {
    requantizeOutputProcessingGConv<
        inst_set,
        A_SYMMETRIC,
        B_SYMMETRIC,
        Q_GRAN,
        true,
        FUSE_RELU,
        C_PER_G>(out, inp, block, ld_out, ld_in, r);
  }
}

// dispatch B_SYMMETRIC
template <
    inst_set_t inst_set,
    bool A_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool FUSE_RELU,
    int C_PER_G>
void requantizeOutputProcessingGConv(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t& r) {
  bool b_symmetric = (Q_GRAN == QuantizationGranularity::TENSOR &&
                      r.B_zero_point[0] == 0) ||
      r.row_offsets == nullptr;
  if (b_symmetric) {
    requantizeOutputProcessingGConv<
        inst_set,
        A_SYMMETRIC,
        true,
        Q_GRAN,
        FUSE_RELU,
        C_PER_G>(out, inp, block, ld_out, ld_in, r);
  } else {
    requantizeOutputProcessingGConv<
        inst_set,
        A_SYMMETRIC,
        false,
        Q_GRAN,
        FUSE_RELU,
        C_PER_G>(out, inp, block, ld_out, ld_in, r);
  }
}

// dispatch A_SYMMETRIC
template <
    inst_set_t inst_set,
    QuantizationGranularity Q_GRAN,
    bool FUSE_RELU,
    int C_PER_G>
void requantizeOutputProcessingGConv(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t& r) {
  if (r.A_zero_point == 0) {
    requantizeOutputProcessingGConv<inst_set, true, Q_GRAN, FUSE_RELU, C_PER_G>(
        out, inp, block, ld_out, ld_in, r);
  } else {
    requantizeOutputProcessingGConv<
        inst_set,
        false,
        Q_GRAN,
        FUSE_RELU,
        C_PER_G>(out, inp, block, ld_out, ld_in, r);
  }
}

// dispatch C_PER_G
template <
    inst_set_t inst_set,
    QuantizationGranularity Q_GRAN,
    bool FUSE_RELU>
void requantizeOutputProcessingGConv(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t& r,
    int C_per_G) {
  if (C_per_G == 4) {
    requantizeOutputProcessingGConv<inst_set, Q_GRAN, FUSE_RELU, 4>(
        out, inp, block, ld_out, ld_in, r);
  } else if (C_per_G == 8) {
    requantizeOutputProcessingGConv<inst_set, Q_GRAN, FUSE_RELU, 8>(
        out, inp, block, ld_out, ld_in, r);
  } else if (C_per_G == 16) {
    requantizeOutputProcessingGConv<inst_set, Q_GRAN, FUSE_RELU, 16>(
        out, inp, block, ld_out, ld_in, r);
  }
}

template <
    inst_set_t inst_set,
    typename outType,
    bool FUSE_RELU,
    QuantizationGranularity Q_GRAN,
    int SPATIAL_DIM>
void requantize_(
    const conv_param_t<SPATIAL_DIM>& conv_param,
    std::int32_t a_zero_point,
    std::int32_t* rowOffsetBuf,
    outType* out,
    int32_t* outBuffer,
    const ReQuantizeOutput<FUSE_RELU, Q_GRAN>& outProcess,
    int i,
    int h,
    int gOuter) {
  int G = conv_param.G;
  int K_per_G = conv_param.OC / G;
  int C_per_G = conv_param.IC / G;
  int oh = conv_param.OUT_DIM[0];
  int ow = conv_param.OUT_DIM[1];

  requantizationParams_t r = {
      a_zero_point,
      outProcess.getBZeroPoint(),
      outProcess.getCZeroPoint(),
      outProcess.getCMultiplier(),
      rowOffsetBuf,
      outProcess.getColOffsets(),
      outProcess.getBias(),
      outProcess.getNCols(),
      G};

  constexpr bool USE_AVX512 = inst_set == inst_set_t::avx512;
  int SIMD_WIDTH = USE_AVX512 ? 512 : 256;
  int SIMD_WIDTH_I32 = SIMD_WIDTH / 32;

  const std::int32_t* inp = outBuffer;
  block_type_t block{
      (i * oh + h) * ow, ow, gOuter * K_per_G, SIMD_WIDTH_I32 * K_per_G};
  int ld_out = K_per_G * G;
  int ld_in = K_per_G * G;

  requantizeOutputProcessingGConv<inst_set, Q_GRAN, FUSE_RELU>(
      out, inp, block, ld_out, ld_in, r, C_per_G);
}

} // namespace

template <
    typename packed_W,
    typename outType,
    typename processOutputType,
    int SPATIAL_DIM>
void fbgemmGroupwiseConv(
    const conv_param_t<SPATIAL_DIM>& conv_param,
    const std::uint8_t* activations,
    std::int32_t a_zero_point,
    std::int32_t* rowOffsetBuf,
    packed_W& packed_weights,
    outType* out,
    int32_t* outBuffer,
    const processOutputType& outProcess,
    int thread_id,
    int num_threads) {
  return fbgemmGroupwiseConvBase_<
      packed_W,
      outType,
      processOutputType,
      SPATIAL_DIM>(
      conv_param,
      activations,
      a_zero_point,
      rowOffsetBuf,
      packed_weights,
      out,
      outBuffer,
      outProcess,
      thread_id,
      num_threads);
}

template <
    typename packed_W,
    typename outType,
    bool FUSE_RELU,
    QuantizationGranularity Q_GRAN,
    int SPATIAL_DIM>
void fbgemmGroupwiseConv(
    const conv_param_t<SPATIAL_DIM>& conv_param,
    const std::uint8_t* activations,
    std::int32_t a_zero_point,
    std::int32_t* rowOffsetBuf,
    packed_W& packed_weights,
    outType* out,
    int32_t* outBuffer,
    const ReQuantizeOutput<FUSE_RELU, Q_GRAN>& outProcess,
    int thread_id,
    int num_threads) {
  typedef ReQuantizeOutput<FUSE_RELU, Q_GRAN> processOutputType;

  if (!fbgemmOptimizedGConv<SPATIAL_DIM>(conv_param) ||
      (!fbgemmHasAvx512Support() && !fbgemmHasAvx2Support())) {
    return fbgemmGroupwiseConvBase_<
        packed_W,
        outType,
        processOutputType,
        SPATIAL_DIM>(
        conv_param,
        activations,
        a_zero_point,
        rowOffsetBuf,
        packed_weights,
        out,
        outBuffer,
        outProcess,
        thread_id,
        num_threads);
  }

  int MB = conv_param.MB;
  int H_in = conv_param.IN_DIM[0];
  int W_in = conv_param.IN_DIM[1];
  int W_out = conv_param.OUT_DIM[1];
  int G = conv_param.G;
  int K_per_G = conv_param.OC / G;
  int C_per_G = conv_param.IC / G;
  int R = conv_param.K[0];
  int S = conv_param.K[1];
  int ih_iw = H_in * W_in;

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  chrono::time_point<chrono::high_resolution_clock> t_begin, t_end;
#endif

  assert(SPATIAL_DIM == 2 && "3D conv not supported yet");
  bool useAvx512 = fbgemmHasAvx512Support() && G >= 16;
  int simd_width = useAvx512 ? 512 : 256;
  int simd_width_i32 = simd_width / 32;
  int g_vec_block = std::max(simd_width_i32 / K_per_G, 1);

  assert(G % 8 == 0);
  // Overall loop order and how it correlates to the standard i, j, and k
  // loop variables for GEMM.
  // for each MB // corresponds to i loop in standard GEMM, first level
  //  for each G // first-level loop for groups
  //   for each H // corresponds to i loop in standard GEMM, second level
  //    for each g1 // second-level loop for groups. Iteration count is same as
  //                // SIMD_WIDTH_I32 to vectorize row_offset computation over
  //                // groups. For each iteration of this loop, we do output
  //                // processing.
  //     // --- Below is executed by groupConvAvx512/Avx2
  //     for each K // corresponds to j loop in standard GEMM, first level
  //      Load weights into RS SIMD registers.
  //      These registers are reused W times. Each SIMD register contains
  //      k x g x C weights where g is G_VEC_BLOCK
  //      (see PackWeightMatrixForGConv.cc for more details on G_VEC_BLOCK).
  //      for each W // corresponds to i loop in standard GEMM, third level
  //       // for each iteration at this level, we have a complete row offset
  //       for each RS // corresponds to k loop in standard GEMM, first level
  //        // Activation is broadcasted into registers and reused by k times
  //        // Output is accumulated in-register by RSC times
  //        --- Below is vectorization
  //        for each g2 // second-level loop for groups iterate G_VEC_BLOCK
  //                    // times
  //         for each k // corresponds to j loop in standard GEMM, second level
  //          for each C // corresponds to k loop in standard GEMM, second level
  for (int i = 0; i < MB; ++i) {
    const uint8_t* actStartBatch = activations + i * ih_iw * conv_param.IC;
    for (int gOuter = 0; gOuter < G; gOuter += simd_width_i32) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      t_begin = chrono::high_resolution_clock::now();
#endif

      int gLimit = gOuter + simd_width_i32;
      // Work on 8 output channels at a time (8 * sizeof(int32_t) == 32B VLEN
      // of AVX2), and we need multiple groups if a group has not enough
      // number of channels.
      constexpr int H_PAD = 1;
      int h = 0;
      for (int g = gOuter; g < gLimit; g += g_vec_block) {
        // Reusing the same region of outBuffer multiple times for locality
        int32_t* currOutBuf =
            outBuffer + (-h * W_out * G + (g - gOuter)) * K_per_G;
        const uint8_t* actStartGroup = actStartBatch + g * C_per_G;
        const int8_t* B =
            packed_weights.getBuf() + g * R * S * K_per_G * C_per_G;
        int32_t* row_buf =
            (g == gOuter && rowOffsetBuf && outProcess.getBZeroPoint())
            ? rowOffsetBuf - h * W_out * simd_width_i32
            : nullptr;
        if (useAvx512) {
          groupConvAvx512<true, false, SPATIAL_DIM>(
              conv_param,
              actStartGroup,
              a_zero_point,
              h,
              B,
              currOutBuf,
              row_buf);
        } else {
          groupConvAvx2<true, false, SPATIAL_DIM>(
              conv_param,
              actStartGroup,
              a_zero_point,
              h,
              B,
              currOutBuf,
              row_buf);
        }
      }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      t_end = chrono::high_resolution_clock::now();
      double dt =
          chrono::duration_cast<chrono::nanoseconds>(t_end - t_begin).count();
      kernel_time += dt;
      t_begin = chrono::high_resolution_clock::now();
#endif

      if (useAvx512) {
        requantize_<
            inst_set_t::avx512,
            outType,
            FUSE_RELU,
            Q_GRAN,
            SPATIAL_DIM>(
            conv_param,
            a_zero_point,
            rowOffsetBuf,
            out,
            outBuffer,
            outProcess,
            i,
            h,
            gOuter);
      } else {
        requantize_<
            inst_set_t::avx2,
            outType,
            FUSE_RELU,
            Q_GRAN,
            SPATIAL_DIM>(
            conv_param,
            a_zero_point,
            rowOffsetBuf,
            out,
            outBuffer,
            outProcess,
            i,
            h,
            gOuter);
      }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      t_end = chrono::high_resolution_clock::now();
      dt = chrono::duration_cast<chrono::nanoseconds>(t_end - t_begin).count();
      postprocessing_time += dt;
      t_begin = chrono::high_resolution_clock::now();
#endif

      for (h = H_PAD; h <= (H_in + H_PAD - R) / conv_param.stride[0]; ++h) {
        for (int g = gOuter; g < gLimit; g += g_vec_block) {
          int32_t* currOutBuf =
              outBuffer + (-h * W_out * G + (g - gOuter)) * K_per_G;
          const uint8_t* actStartGroup = actStartBatch + g * C_per_G;
          const int8_t* B =
              packed_weights.getBuf() + g * R * S * K_per_G * C_per_G;
          int32_t* row_buf =
              (g == gOuter && rowOffsetBuf && outProcess.getBZeroPoint())
              ? rowOffsetBuf - h * W_out * simd_width_i32
              : nullptr;
          if (useAvx512) {
            groupConvAvx512<false, false, SPATIAL_DIM>(
                conv_param,
                actStartGroup,
                a_zero_point,
                h,
                B,
                currOutBuf,
                row_buf);
          } else {
            groupConvAvx2<false, false, SPATIAL_DIM>(
                conv_param,
                actStartGroup,
                a_zero_point,
                h,
                B,
                currOutBuf,
                row_buf);
          }
        }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        t_end = chrono::high_resolution_clock::now();
        dt =
            chrono::duration_cast<chrono::nanoseconds>(t_end - t_begin).count();
        kernel_time += dt;
        t_begin = chrono::high_resolution_clock::now();
#endif

        if (useAvx512) {
          requantize_<
              inst_set_t::avx512,
              outType,
              FUSE_RELU,
              Q_GRAN,
              SPATIAL_DIM>(
              conv_param,
              a_zero_point,
              rowOffsetBuf,
              out,
              outBuffer,
              outProcess,
              i,
              h,
              gOuter);
        } else {
          requantize_<
              inst_set_t::avx2,
              outType,
              FUSE_RELU,
              Q_GRAN,
              SPATIAL_DIM>(
              conv_param,
              a_zero_point,
              rowOffsetBuf,
              out,
              outBuffer,
              outProcess,
              i,
              h,
              gOuter);
        }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        t_end = chrono::high_resolution_clock::now();
        dt =
            chrono::duration_cast<chrono::nanoseconds>(t_end - t_begin).count();
        postprocessing_time += dt;
        t_begin = chrono::high_resolution_clock::now();
#endif
      }

      if (h < conv_param.OUT_DIM[0]) {
        for (int g = gOuter; g < gLimit; g += g_vec_block) {
          int32_t* currOutBuf =
              outBuffer + (-h * W_out * G + (g - gOuter)) * K_per_G;
          const uint8_t* actStartGroup = actStartBatch + g * C_per_G;
          const int8_t* B =
              packed_weights.getBuf() + g * R * S * K_per_G * C_per_G;
          int32_t* row_buf =
              (g == gOuter && rowOffsetBuf && outProcess.getBZeroPoint())
              ? rowOffsetBuf - h * W_out * simd_width_i32
              : nullptr;
          if (useAvx512) {
            groupConvAvx512<false, true, SPATIAL_DIM>(
                conv_param,
                actStartGroup,
                a_zero_point,
                h,
                B,
                currOutBuf,
                row_buf);
          } else {
            groupConvAvx2<false, true, SPATIAL_DIM>(
                conv_param,
                actStartGroup,
                a_zero_point,
                h,
                B,
                currOutBuf,
                row_buf);
          }
        }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        t_end = chrono::high_resolution_clock::now();
        dt =
            chrono::duration_cast<chrono::nanoseconds>(t_end - t_begin).count();
        kernel_time += dt;
        t_begin = chrono::high_resolution_clock::now();
#endif

        if (useAvx512) {
          requantize_<
              inst_set_t::avx512,
              outType,
              FUSE_RELU,
              Q_GRAN,
              SPATIAL_DIM>(
              conv_param,
              a_zero_point,
              rowOffsetBuf,
              out,
              outBuffer,
              outProcess,
              i,
              h,
              gOuter);
        } else {
          requantize_<
              inst_set_t::avx2,
              outType,
              FUSE_RELU,
              Q_GRAN,
              SPATIAL_DIM>(
              conv_param,
              a_zero_point,
              rowOffsetBuf,
              out,
              outBuffer,
              outProcess,
              i,
              h,
              gOuter);
        }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        t_end = chrono::high_resolution_clock::now();
        dt =
            chrono::duration_cast<chrono::nanoseconds>(t_end - t_begin).count();
        postprocessing_time += dt;
#endif
      }
    } // gOuter loop
  } // i loop
}

template <int SPATIAL_DIM>
int rowOffsetBufferSizeGConv(const conv_param_t<SPATIAL_DIM>& conv_param) {
  // row offset buffer should be a able to hold row offsets for however
  // number of groups we process at a time.
  assert(SPATIAL_DIM == 2 && "Only 2D is supported currently");
  if (cpuinfo_initialize()) {
    int G = conv_param.G;
    bool useAvx512 = fbgemmHasAvx512Support() && conv_param.IC >= 64 && G >= 16;
    int simd_width = useAvx512 ? 512 : 256;
    int simd_width_i32 = simd_width / 32;
    if (useAvx512) {
      int bufferSize = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];
      int C_per_G = conv_param.IC / G;
      int K_per_G = conv_param.OC / G;
      if (C_per_G == K_per_G &&
          (C_per_G == 4 || C_per_G == 8 || C_per_G == 16)) {
        return 2 * simd_width_i32 * bufferSize;
      } else {
        return conv_param.G * bufferSize;
      }
    } else if (fbgemmHasAvx2Support()) {
      int bufferSize = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];
      int C_per_G = conv_param.IC / G;
      int K_per_G = conv_param.OC / G;
      if (C_per_G == K_per_G &&
          (C_per_G == 4 || C_per_G == 8 || C_per_G == 16)) {
        // row offset is calculated for 8 groups at a time
        // 2x is needed for transposing
        return 2 * simd_width_i32 * bufferSize;
      } else {
        return conv_param.G * bufferSize;
      }
    } else {
      // TODO: Have default slower path
      assert(0 && "unsupported architecture");
      return -1;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

template int rowOffsetBufferSizeGConv<2>(const conv_param_t<2>& conv_param);
template int rowOffsetBufferSizeGConv<3>(const conv_param_t<3>& conv_param);

#define INSTANTIATE_BASE(RELU, Q_GRAN, SPATIAL_DIM)                           \
  template void fbgemmGroupwiseConv(                                          \
      const conv_param_t<SPATIAL_DIM>& conv_param,                            \
      const uint8_t* activations,                                             \
      int32_t a_zero_point,                                                   \
      std::int32_t* rowOffsetBuf,                                             \
      PackWeightMatrixForGConv<int8_t, int32_t, SPATIAL_DIM>& packed_weights, \
      uint8_t* out,                                                           \
      int32_t* outBuffer,                                                     \
      const ReQuantizeOutput<RELU, Q_GRAN>& outProcess,                       \
      int thread_id,                                                          \
      int num_threads);

#define INSTANTIATE_SPATIAL_DIM(RELU, Q_GRAN) \
  INSTANTIATE_BASE(RELU, Q_GRAN, 2);          \
  INSTANTIATE_BASE(RELU, Q_GRAN, 3);

#define INSTANTIATE_Q_GRANS(RELU)                          \
  INSTANTIATE_SPATIAL_DIM(RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_SPATIAL_DIM(RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_SPATIAL_DIM(RELU, QuantizationGranularity::OUT_CHANNEL);

INSTANTIATE_Q_GRANS(false);
INSTANTIATE_Q_GRANS(true);

#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_BASE

template void fbgemmGroupwiseConv(
    const conv_param_t<2>& conv_param,
    const uint8_t* activations,
    int32_t a_zero_point,
    std::int32_t* rowOffsetBuf,
    PackWeightMatrixForGConv<int8_t, int32_t, 2>& packed_weights,
    int32_t* out,
    int32_t* outBuffer,
    const DoNothing<int32_t, int32_t>& outProcess,
    int thread_id,
    int num_threads);

} // namespace fbgemm
