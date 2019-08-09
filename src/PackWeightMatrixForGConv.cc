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
 * @brief Get the index of the unpacked data for a given <r, s, k, g, c, tr>
 *
 * Non-transposed: G (R S C/G) K/G
 * Transposed: G K/G (R S C/G)
 * Using inline as this will be called frequently
 */
template <typename T, typename accT, int SPATIAL_DIM>
inline int PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::unpacked_index_(
    int r, int s, int k, int g, int c, bool tr) {
  // Get the full dimensions
  int R = conv_param_.K[0];
  int S = conv_param_.K[1];
  int G = conv_param_.G;
  int IC_per_G = conv_param_.IC / G;
  int OC_per_G = conv_param_.OC / G;

  int idx;
  if (tr) {
    idx = (((g * OC_per_G + k) * R + r) * S + s) * IC_per_G + c;
  } else {
    idx = (((g * R + r) * S + s) * IC_per_G + c) * OC_per_G + k;
  }
  return idx;
}

/**
 * @brief Get the index of the packed data for a given <r, s, k, g, c>
 *
 * The index may differ depending on IC_per_G.
 * Using inline as this will be called frequently
 */
template <typename T, typename accT, int SPATIAL_DIM>
inline int PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::packed_index_(
    int r, int s, int k, int g, int c) {
  // Get the full dimensions
  int R = conv_param_.K[0];
  int S = conv_param_.K[1];
  int G = conv_param_.G;
  int IC_per_G = conv_param_.IC / G;
  int OC_per_G = conv_param_.OC / G;

  int idx;
  // For IC_per_G == 4, we need to work on 2 groups at a time
  if (IC_per_G == 4) {
    idx = (((((g / 2) * R + r) * S + s) * OC_per_G + k) * 2 + (g % 2))
      * IC_per_G + c;
  } else {
    idx = ((((g * (IC_per_G / 4) + (c / 4)) * R + r) * S + s) * OC_per_G + k)
      * 4 + (c % 4);
  }
  return idx;
}

/**
 * @ brief Pack or unpack matrix
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
void PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::pack_unpack_(
    const T* src, T* dst, bool ispack) {
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
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        for (int k = 0; k < OC_per_G; ++k) {
          for (int g = 0; g < G; ++g) {
            for (int c = 0; c < IC_per_G; ++c) {
              int p_idx = packed_index_(r, s, k, g, c);
              int up_idx = unpacked_index_(r, s, k, g, c, tr);
              // Pack: src (unpacked) -> dst (packed)
              if (ispack) {
                dst[p_idx] = src[up_idx];
              } else {
                dst[up_idx] = src[p_idx];
              }
            }
          }
        }
      }
    }
  } else {
    // For pack & transposed, call transposeConvWeights()
    // G K/G (R S C/G) => G (R S C/G) K/G
    if (tr) {
      if (ispack) {
        transposeConvWeights(conv_param_, src, dst);
      } else {
        // TODO: Wrap this as a inverseTransposeConvWeights()?
        // For unpack & transposed, call transposeConvWeights()
        // G (R S C/G) K/G => G K/G (R S C/G)
        for (int r = 0; r < R; ++r) {
          for (int s = 0; s < S; ++s) {
            for (int k = 0; k < OC_per_G; ++k) {
              for (int g = 0; g < G; ++g) {
                for (int c = 0; c < IC_per_G; ++c) {
                  dst[(((g * OC_per_G + k) * R + r) * S + s)
                    * IC_per_G + c] =
                    src[(((g * R + r) * S + s) * IC_per_G + c)
                    * OC_per_G + k];
                }
              }
            }
          }
        }
      }  // end if(ispack)
    } else {
      // just copy the data for not supported cases
      memcpy(dst, src,
          G * R * S * OC_per_G * IC_per_G * sizeof(inpType));
    } //end if(tr)
  } // end if(fbgemmOptimizedGConv(conv_param_)
}

/**
 * @brief Pack weight tensor in a suitable format required for the optimized
 * kernel.
 */
template <typename T, typename accT, int SPATIAL_DIM>
void PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::pack() {
  pack_unpack_(sdata_, pdata_, true);
}

/**
 * @brief Unpack the packed weight tensor (for the optimized kernel)
 * to the original form.
 */
template <typename T, typename accT, int SPATIAL_DIM>
void PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>::unpack(T* origin_buf) {
  pack_unpack_(const_cast<const T*>(pdata_), origin_buf, false);
}

template class PackWeightMatrixForGConv<int8_t, int32_t, 2>;
template class PackWeightMatrixForGConv<int8_t, int16_t, 2>;
template class PackWeightMatrixForGConv<int8_t, int32_t, 3>;
template class PackWeightMatrixForGConv<int8_t, int16_t, 3>;
} // namespace fbgemm
