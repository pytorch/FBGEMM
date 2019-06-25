/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "GroupwiseConvAcc32Intrinsic.h"

namespace fbgemm {

using namespace std;

template <bool TOP, bool BOTTOM, int SPATIAL_DIM>
void groupConvAvx2(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf) {
  gconv_kernel_<TOP, BOTTOM>(conv_p, A, A_zero_point, h, B, C, rowOffsetBuf);
}

template void groupConvAvx2<true, false, 2>(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx2<false, false, 2>(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx2<false, true, 2>(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx2<true, false, 3>(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx2<false, false, 3>(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

template void groupConvAvx2<false, true, 3>(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    int h,
    const int8_t* B,
    int32_t* C,
    int32_t* rowOffsetBuf);

} // namespace fbgemm
