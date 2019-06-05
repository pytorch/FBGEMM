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
 *
 * For IC_per_G == 4 && OC_per_G == 4 optimized
 * kernel works on 2 groups at a time hence input channels for g and g+1 group
 * are laid out sequentially for each output channel, i.e., the layout is (G/2)
 * R S K (2C) and K (2C) is in each 32B vector.
 * We work on two groups at a time to fully utilize the avx2 SIMD width of
 * 256-bits.
 *
 * For IC_per_G == 8, 16, 32 && OC_per_G == 8, 16, 32 there is no need to work
 * on 2 groups at a time and full SIMD width can be efficiently utilized even
 * while working on 1 group at a time.
 * In this case, the layout is G (C/4) R S K 4
 */
template <typename T, typename accT, int SPATIAL_DIM>
void PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::pack() {
  // filters are assumed to be in G RS C/G K/G format
  int R = conv_param_.K[0];
  int S = conv_param_.K[1];
  int G = conv_param_.G;
  int IC_per_G = conv_param_.IC / conv_param_.G;
  int OC_per_G = conv_param_.OC / conv_param_.G;

  // If transpose option is set, the weight matrix is in layout G K/G (R S C/G)
  // instead of G (R S C/G) K/G
  bool tr = (trans_ == matrix_op_t::Transpose);
  if (fbgemmOptimizedGConv(conv_param_)) {
    // currently only this case is supported
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
              if (IC_per_G == 4) {
                // For IC_per_G == 4, we need to work on 2 groups at a time
                pdata_
                    [(((((g / 2) * R + r) * S + s) * OC_per_G + k) * 2 +
                      (g % 2)) *
                         IC_per_G +
                     c] = b;
              } else {
                pdata_
                    [((((g * (IC_per_G / 4) + (c / 4)) * R + r) * S + s) *
                          OC_per_G +
                      k) *
                         4 +
                     (c % 4)] = b;
              }
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
