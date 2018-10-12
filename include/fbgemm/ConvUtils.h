/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <string>

namespace fbgemm2 {

/**
 * @brief A struct to conveniently store all convolution parameters.
 */
struct conv_param_t {
  int MB; ///< Mini Batch size
  int IC; ///< Number of Input Channels
  int OC; ///< Number of Output Channels
  int IH; ///< Input Image Height
  int IW; ///< Input Image Width
  int G; ///< Number of Groups
  int KH; ///< Filter (Kernel) Height
  int KW; ///< Filter (Kernel) Width
  int stride_h; ///< Stride in Height Dimension
  int stride_w; ///< Stride in Width  Dimension
  int pad_h; ///< Padding in Height Dimension (top and bottom)
  int pad_w; ///< Padding in Width Dimension (left and right)
  int dilation_h; ///< Kernel dilation in Height Dimension
  int dilation_w; ///< Kernel dilation in Width Dimension

  // The following are derived parameters
  int OH; ///< Output Image Height
  int OW; ///< Output Image Width
  int IHP; ///< Input Height Padded
  int IWP; ///< Input Width Padded

  /**
   * @brief Constructor for initializing the convolution parameters.
   * TODO: Dilation is not handled correctly.
   */
  conv_param_t(
      int mb,
      int ic,
      int oc,
      int ih,
      int iw,
      int g = 1,
      int kh = 3,
      int kw = 3,
      int strd_h = 1,
      int strd_w = 1,
      int pd_h = 1,
      int pd_w = 1)
      : MB(mb),
        IC(ic),
        OC(oc),
        IH(ih),
        IW(iw),
        G(g),
        KH(kh),
        KW(kw),
        stride_h(strd_h),
        stride_w(strd_w),
        pad_h(pd_h),
        pad_w(pd_w),
        dilation_h(1),
        dilation_w(1) {
    IHP = IH + 2 * pad_h;
    IWP = IW + 2 * pad_w;
    OH = (IHP - KH) / stride_h + 1;
    OW = (IWP - KW) / stride_w + 1;
  }

  /**
   * @brief Helper function to get convolution parameters as string.
   */
  std::string toString() const {
    std::string out = "";
    out += "MB:" + std::to_string(MB) + ", ";
    out += "IC:" + std::to_string(IC) + ", ";
    out += "OC:" + std::to_string(OC) + ", ";
    out += "IH:" + std::to_string(IH) + ", ";
    out += "IW:" + std::to_string(IW) + ", ";
    out += "G:" + std::to_string(G) + ", ";
    out += "KH:" + std::to_string(KH) + ", ";
    out += "KW:" + std::to_string(KW) + ", ";
    out += "stride_h:" + std::to_string(stride_h) + ", ";
    out += "stride_w:" + std::to_string(stride_w) + ", ";
    out += "pad_h:" + std::to_string(pad_h) + ", ";
    out += "pad_w:" + std::to_string(pad_w);
    return out;
  }
};

} // namespace fbgemm2
