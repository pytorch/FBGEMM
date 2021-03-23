/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <cpuinfo.h>
#include <cassert>
#include <iomanip>
// #include <numeric>
//#include <string>
#include "./RefImplementations.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T, typename accT>
PackWeightMatrixFor1DConv<T, accT>::PackWeightMatrixFor1DConv(
    matrix_op_t trans,
    const conv_param_t<1>& conv_param,
    const T* sdata,
    T* pdata)
    : trans_(trans), conv_param_(conv_param), sdata_(sdata) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  isa_ = fbgemmInstructionSet();
  if (isa_ == inst_set_t::avx2) {
    vsize_ = simd_info<inst_set_t::avx2>::WIDTH_BYTES;
  } else {
    vsize_ = simd_info<inst_set_t::avx512>::WIDTH_BYTES;
  }
  vsize4_ = vsize_ / 4;

  nreg_ = getRegNumForConv1D();
  bsize_ = nreg_ * vsize4_;
  G_ = conv_param_.G;
  IC_per_G_ = conv_param_.IC / G_;
  OC_per_G_ = conv_param_.OC / G_;
  paddedICPerG_ = (IC_per_G_ + 3) / 4 * 4;
  paddedOCPerG_ = (OC_per_G_ + vsize4_ - 1) / vsize4_ * vsize4_;
  nblock_ = paddedOCPerG_ / bsize_;

  if (!pdata) {
    bufAllocatedHere_ = true;
    int kernel_prod = conv_param.K[0];
    int size = G_ * kernel_prod * paddedOCPerG_ * paddedICPerG_ * sizeof(T);
    pdata_ = static_cast<T*>(fbgemmAlignedAlloc(64, size));
    // bzero(pdata_, size);
  } else {
    bufAllocatedHere_ = false;
    pdata_ = pdata;
  }

  pack();
}

template <typename T, typename accT>
int PackWeightMatrixFor1DConv<T, accT>::getRegNumForConv1D() {
  const inst_set_t isa = fbgemmInstructionSet();
  if (isa == inst_set_t::avx2) {
    return 4;
  } else {
    return 6;
  }
}

/**
 * @brief Get the index of the unpacked data
 *        for a given <g, s, c, k, tr>
 *
 * Non-transposed: G (S C/G) K/G
 * Transposed: G K/G (S C/G)
 * Using inline as this will be called frequently
 */

template <typename T, typename accT>
inline int PackWeightMatrixFor1DConv<T, accT>::unpacked_index_(
    int g,
    int s,
    int c,
    int k,
    bool tr) {
  // Get the full dimensions
  int S = conv_param_.K[0];

  int idx;
  if (tr) {
    idx = ((g * OC_per_G_ + k) * S + s) * IC_per_G_ + c;
  } else {
    idx = ((g * S + s) * IC_per_G_ + c) * OC_per_G_ + k;
  }
  return idx;
}

/**
 * @brief Get the index of the packed data for a given <g, s, c, k>
 *
 * Using inline as this will be called frequently
 */
template <typename T, typename accT>
inline int PackWeightMatrixFor1DConv<T, accT>::packed_index_(
    int g,
    int s,
    int c,
    int k,
    int b,
    int bs) {
  // Get the full dimensions
  int S = conv_param_.K[0];

  int idx = (g * S + s) * (paddedICPerG_ * paddedOCPerG_) +
      paddedICPerG_ * bsize_ * b + (c / 4) * bs * 4 + (k % bsize_) * 4 +
      (c % 4);
  return idx;
}

/**
 * @brief Pack or unpack matrix
 *
 * Let IC_per_G be number of input channels per group and OC_per_G be number of
 * output channels per group.
 *
 * Currently, each K ispacked separately, may consider pack all K together later
 *
 */

template <typename T, typename accT>
void PackWeightMatrixFor1DConv<T, accT>::pack_unpack_(
    const T* src,
    T* dst,
    bool ispack) {
  // Can't use T as varname because T is a template parameter.
  int S = conv_param_.K[0];

  bool tr = (trans_ == matrix_op_t::Transpose);
  if (take1DFastPath<1>(conv_param_)) {
    for (int g = 0; g < G_; g++) {
      for (int s = 0; s < S; ++s) {
        for (int c = 0; c < IC_per_G_; c++) {
          for (int k = 0; k < OC_per_G_; k++) {
            int up_idx = unpacked_index_(g, s, c, k, tr);
            int block = k / bsize_;
            int bs = bsize_;
            if (block == nblock_) {
              bs = paddedOCPerG_ % bsize_;
            }
            int p_idx = packed_index_(g, s, c, k, block, bs);
            if (ispack) {
              dst[p_idx] = src[up_idx];
            } else {
              dst[up_idx] = src[p_idx];
            }
          }
        }

        if (ispack) {
          for (int c = IC_per_G_; c < paddedICPerG_; c++) {
            for (int k = 0; k < OC_per_G_; k++) {
              int block = k / bsize_;
              int bs = bsize_;
              if (block == nblock_) {
                bs = paddedOCPerG_ % bsize_;
              }
              int p_idx = packed_index_(g, s, c, k, block, bs);
              dst[p_idx] = 0;
            }
          }
        }
      }
    }
  } else {
    throw std::runtime_error(
        "Does not fit for Groupwise 1D fast implementation!");
  }
}

/**
 * @brief Pack weight tensor in a suitable format required for the optimized
 * kernel.
 */
template <typename T, typename accT>
void PackWeightMatrixFor1DConv<T, accT>::pack() {
  pack_unpack_(sdata_, pdata_, true);
}

/**
 * @brief Unpack the packed weight tensor (for the optimized kernel)
 * to the original form.
 */
template <typename T, typename accT>
void PackWeightMatrixFor1DConv<T, accT>::unpack(T* origin_buf) {
  pack_unpack_(const_cast<const T*>(pdata_), origin_buf, false);
}

template class FBGEMM_API PackWeightMatrixFor1DConv<int8_t, int32_t>;
template class FBGEMM_API PackWeightMatrixFor1DConv<int8_t, int16_t>;
} // namespace fbgemm
