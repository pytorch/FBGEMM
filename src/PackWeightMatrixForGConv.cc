/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cpuinfo.h>
#include <cassert>
#include <iomanip>
#include "RefImplementations.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T, typename accT, int SPATIAL_DIM>
PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::PackWeightMatrixForGConv(
    matrix_op_t trans,
    const conv_param_t<SPATIAL_DIM>& conv_param,
    const T* sdata,
    T* pdata)
    : trans_(trans), conv_param_(conv_param), sdata_(sdata) {
  assert(SPATIAL_DIM == 2 && "3D conv not supported yet");

  if (!pdata) {
    bufAllocatedHere_ = true;
    pdata_ = static_cast<T*>(fbgemmAlignedAlloc(
        64,
        conv_param_.G * conv_param_.K[0] * conv_param_.K[1] *
            (conv_param_.OC / conv_param_.G) *
            (conv_param_.IC / conv_param_.G) * sizeof(T)));
  } else {
    bufAllocatedHere_ = false;
    pdata_ = pdata;
  }
  pack();
}

/**
 * @brief Pack weight tensor in a suitable format required for the optimized
 * kernel.
 *
 * Let IC_per_G be number of input channels per group and OC_per_G be number of
 * output channels per group.
 * Let SIMD_WIDTH_I32 is the number of i32 per SIMD register (8 in AVX2 and 16
 * in AVX512).
 * When OC_per_G is small, we need to work on multiple groups at a time to fill
 * the whole output SIMD register. We call the number of groups we put at each
 * SIMD register G_VEC_BLOCK.
 * For example, in AVX2 and OC_per_G = 4, G_VEC_BLOCK = 8 / 4 = 2
 * In AVX512 and OC_per_G = 8, G_VEC_BLOCK = 16 / 8 = 2.
 *
 * The layout we're using is G/G_VEC_BLOCK x R x S x K x G_VEC_BLOCK x C .
 * That is input channel (C) is the fast moving dimension, G_VEC_BLOCK, output
 * channel (K), filter dimensions (S and R), and finally G/G_VEC_BLOCK .
 *
 * This layout roughly aligns with the access order in our group convolution
 * kernels. See GroupwiseConv.cc for more details on the access order.
 */
template <typename T, typename accT, int SPATIAL_DIM>
void PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::pack() {
  // filters are assumed to be in G RS C/G K/G format
  int R = conv_param_.K[0];
  int S = conv_param_.K[1];
  int G = conv_param_.G;
  int IC_per_G = conv_param_.IC / G;
  int OC_per_G = conv_param_.OC / G;

  // If transpose option is set, the weight matrix is in layout G K/G (R S C/G)
  // instead of G (R S C/G) K/G
  bool tr = (trans_ == matrix_op_t::Transpose);
  if (fbgemmOptimizedGConv(conv_param_)) {
    // currently only this case is supported
    bool useAvx512 =
        cpuinfo_initialize() && fbgemmHasAvx512Support() && G >= 16;
    int simd_width = useAvx512 ? 512 : 256;
    // The number of groups we have per simd vector
    // e.g., for IC_per_G == 4 and AVX2, we need to work on 2 groups at a time
    int g_vec_block = std::max(simd_width / 32 / OC_per_G, 1);
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        for (int k = 0; k < OC_per_G; ++k) {
          for (int g = 0; g < G; ++g) {
            for (int c = 0; c < IC_per_G; ++c) {
              inpType b = tr
                  ? sdata_
                        [(((g * OC_per_G + k) * R + r) * S + s) * IC_per_G + c]
                  : sdata_
                        [(((g * R + r) * S + s) * IC_per_G + c) * OC_per_G + k];
              pdata_
                  [(((((g / g_vec_block) * R + r) * S + s) * OC_per_G + k) *
                        g_vec_block +
                    (g % g_vec_block)) *
                       IC_per_G +
                   c] = b;
            }
          }
        }
      }
    }
  } else {
    if (tr) {
      // conv_ref expects weights to be in G (R S C/G) K/G format
      transposeConvWeights(conv_param_, sdata_, pdata_);
    } else {
      // just copy the data for not supported cases
      memcpy(pdata_, sdata_, G * R * S * OC_per_G * IC_per_G * sizeof(inpType));
    }
  }
}

template class PackWeightMatrixForGConv<int8_t, int32_t, 2>;
template class PackWeightMatrixForGConv<int8_t, int16_t, 2>;
template class PackWeightMatrixForGConv<int8_t, int32_t, 3>;
template class PackWeightMatrixForGConv<int8_t, int16_t, 3>;
} // namespace fbgemm
