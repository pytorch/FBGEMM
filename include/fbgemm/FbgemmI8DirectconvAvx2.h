/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "fbgemm/ConvUtils.h"
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/UtilsAvx2.h"

namespace fbgemm {

class FBGEMM_API PackedDirectConvMatrix {
 public:
  /**
   * @param IC the number of input channels
   * @param OC the number of output channels
   * @param kernel_prod the product of all kernels. For example, kernel_prod =
   *                    9 for 3x3 conv, and 27 for 3x3x3 conv.
   * @param smat the source unpacked weight in GRS layout
   */
  PackedDirectConvMatrix(
      int IC_per_G,
      int OC_per_G,
      int filter_prod,
      const std::int8_t* smat);
  virtual ~PackedDirectConvMatrix();

  const std::int8_t* PackedMat() const {
    return pmat_;
  }

  const bool& is_first_call() const {
    return first_call;
  }

  /**
   compute the column offsets of the weight matrix.
   output of this function is the col_offsets vector
   col_offses dimension is the same as conv_p.OUT_DIM
  */
  template <int kSpatialDim>
  FBGEMM_API void col_offsets_with_zero_pt_s8acc32_DirectConvT(
      const fbgemm::conv_param_t<kSpatialDim>& conv_p,
      std::int32_t* B_zero_point,
      std::vector<int32_t>& col_offsets,
      int ncols_per_quant_group);

 private:
  std::int8_t* pmat_; /** packed weight */
  bool first_call{true};
};

} // namespace fbgemm
