/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <memory>
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <int SPATIAL_DIM, typename T, typename accT>
PackWeightsForConv<SPATIAL_DIM, T, accT>::PackWeightsForConv(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const T* sdata,
    const BlockingFactors* blocking_params) {
  static_assert(
      SPATIAL_DIM == 2 || SPATIAL_DIM == 3,
      "Only 2D and 3D convolutions are supported");
  // Note: The following logic should *exactly* match with what we have in
  // FbgemmConv.cc
  switch (ConvFastPath<SPATIAL_DIM, accT>(conv_p)) {
    case optimized_conv_t::depthwise: {
      if (SPATIAL_DIM == 3) {
        W_im2col_packed_ = nullptr;
        W_dw_2D_packed_ = nullptr;
        W_dw_3D_packed_ =
            std::make_shared<Packed3x3x3ConvMatrix>(conv_p.G, sdata);
        W_gconv_packed_ = nullptr;
      } else {
        W_im2col_packed_ = nullptr;
        W_dw_2D_packed_ =
            std::make_shared<Packed3x3ConvMatrix>(conv_p.G, sdata);
        W_dw_3D_packed_ = nullptr;
        W_gconv_packed_ = nullptr;
      }
      break;
    }
    case optimized_conv_t::groupwise: {
      W_im2col_packed_ = nullptr;
      W_dw_2D_packed_ = nullptr;
      W_dw_3D_packed_ = nullptr;
      W_gconv_packed_ =
          std::make_shared<PackWeightMatrixForGConv<T, accT, SPATIAL_DIM>>(
              matrix_op_t::Transpose, conv_p, sdata, nullptr);
      break;
    }
    case optimized_conv_t::im2col: {
      int NDim = conv_p.OC / conv_p.G;
      int KDim = conv_p.K[0] * conv_p.K[1] * conv_p.IC;
      W_im2col_packed_ = std::make_shared<PackBMatrix<T, accT>>(
          matrix_op_t::Transpose,
          KDim,
          NDim,
          sdata,
          KDim / conv_p.G,
          nullptr,
          conv_p.G,
          blocking_params);
      W_dw_2D_packed_ = nullptr;
      W_dw_3D_packed_ = nullptr;
      W_gconv_packed_ = nullptr;
      break;
    }
  } // switch
}

template class PackWeightsForConv<2, int8_t, int32_t>;
template class PackWeightsForConv<3, int8_t, int32_t>;

} // namespace fbgemm
