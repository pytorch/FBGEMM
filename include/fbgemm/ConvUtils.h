/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fbgemm {

template <int N, int... Vals>
constexpr std::enable_if_t<N == sizeof...(Vals), std::array<int, N>>
array_of_ones() {
  return std::array<int, N>{{Vals...}};
}

template <int N, int... Vals>
constexpr std::enable_if_t<N != sizeof...(Vals), std::array<int, N>>
array_of_ones() {
  return array_of_ones<N, Vals..., 1>();
}

template <int N, int... Vals>
constexpr std::enable_if_t<N == sizeof...(Vals), std::array<int, N>>
array_of_zeroes() {
  return std::array<int, N>{{Vals...}};
}

template <int N, int... Vals>
constexpr std::enable_if_t<N != sizeof...(Vals), std::array<int, N>>
array_of_zeroes() {
  return array_of_zeroes<N, Vals..., 0>();
}

/**
 * @brief A struct to conveniently store all convolution parameters.
 */
template <int SPATIAL_DIM = 2>
struct conv_param_t {
  int MB; ///< Mini Batch size
  int IC; ///< Number of Input Channels
  int OC; ///< Number of Output Channels
  std::array<int, SPATIAL_DIM> IN_DIM; ///< Input Image Dimension
  int G; ///< Number of Groups
  std::array<int, SPATIAL_DIM> K; ///< Filter (Kernel) dimensions
  std::array<int, SPATIAL_DIM> stride; //< Strides
  std::array<int, SPATIAL_DIM * 2>
      pad; //< Padding (first SPATIAL_DIM is for prev/top/left padding, second
           // SPATIAL_DIM is for next/bottom/right padding)
  std::array<int, SPATIAL_DIM> dilation; //< Kernel dilation

  // The following are derived parameters
  std::array<int, SPATIAL_DIM> OUT_DIM; //< Output Image Dimension
  std::array<int, SPATIAL_DIM> IN_DIMP; //< Input Image Dimension Padded

  // The following is for tranposed convolution
  std::array<int, SPATIAL_DIM>
      output_pad; //< Padding (next/bottom/right padding in output buffer)
  bool transposed;

  /**
   * @brief Constructor for initializing the convolution parameters.
   */
  conv_param_t(
      int mb,
      int ic,
      int oc,
      std::array<int, SPATIAL_DIM> in_dim,
      int g,
      std::array<int, SPATIAL_DIM> k,
      std::array<int, SPATIAL_DIM> strd,
      std::array<int, SPATIAL_DIM * 2> pd,
      std::array<int, SPATIAL_DIM> dilations = array_of_ones<SPATIAL_DIM>(),
      std::array<int, SPATIAL_DIM> otpt_pd = array_of_zeroes<SPATIAL_DIM>(),
      bool transposed = false)
      : MB(mb),
        IC(ic),
        OC(oc),
        IN_DIM(in_dim),
        G(g),
        K(k),
        stride(strd),
        pad(pd),
        dilation(dilations),
        output_pad(otpt_pd),
        transposed(transposed) {
    if (ic % g != 0) {
      throw std::runtime_error(
          "groups = " + std::to_string(g) +
          " does not divide number of input channels = " + std::to_string(ic));
    }
    if (oc % g != 0) {
      throw std::runtime_error(
          "groups = " + std::to_string(g) +
          " does not divide number of output channels = " + std::to_string(oc));
    }

    for (int d = 0; d < SPATIAL_DIM; ++d) {
      if (transposed) {
        this->IN_DIMP[d] = this->IN_DIM[d] +
            (this->dilation[d] * (this->K[d] - 1) - this->pad[d]) +
            (this->dilation[d] * (this->K[d] - 1) - this->pad[SPATIAL_DIM + d]);
        this->OUT_DIM[d] = (this->IN_DIM[d] - 1) * this->stride[d] -
            this->pad[d] - this->pad[SPATIAL_DIM + d] +
            this->dilation[d] * (this->K[d] - 1) + output_pad[d] + 1;
      } else {
        IN_DIMP[d] = IN_DIM[d] + pad[d] + pad[SPATIAL_DIM + d];
        OUT_DIM[d] =
            (IN_DIMP[d] - dilation[d] * (K[d] - 1) - 1) / stride[d] + 1;
      }
    }
  }

  /**
   * @brief Helper function to get convolution parameters as string.
   */
  std::string toString() const {
    std::string dim_string[3] = {"T", "H", "W"};

    std::string out;
    out += "MB:" + std::to_string(MB) + ", ";
    out += "IC:" + std::to_string(IC) + ", ";
    out += "OC:" + std::to_string(OC) + ", ";
    if constexpr (SPATIAL_DIM <= 3) {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "I" + dim_string[3 - SPATIAL_DIM + d] + ":" +
            std::to_string(IN_DIM[d]) + ", ";
      }
    } else {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "I" + std::to_string(d) + ":" + std::to_string(IN_DIM[d]) + ", ";
      }
    }
    out += "G:" + std::to_string(G) + ", ";
    if constexpr (SPATIAL_DIM <= 3) {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "K" + dim_string[3 - SPATIAL_DIM + d] + ":" +
            std::to_string(K[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "stride_" + dim_string[3 - SPATIAL_DIM + d] + ":" +
            std::to_string(stride[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM * 2; ++d) {
        out += "pad_" + dim_string[3 - SPATIAL_DIM + (d % SPATIAL_DIM)] + ":" +
            std::to_string(pad[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "dilation_" + dim_string[3 - SPATIAL_DIM + d] + ":" +
            std::to_string(dilation[d]);
        if (d < SPATIAL_DIM - 1) {
          out += ", ";
        }
      }
    } else {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "K" + std::to_string(d) + ":" + std::to_string(K[d]) + ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "stride_" + std::to_string(d) + ":" + std::to_string(stride[d]) +
            ", ";
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "pad_" + std::to_string(d) + ":" + std::to_string(pad[d]);
        if (d < SPATIAL_DIM * 2 - 1) {
          out += ", ";
        }
      }
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "dilation_" + std::to_string(d) + ":" +
            std::to_string(dilation[d]) + ", ";
      }
    }
    if (transposed) {
      for (int d = 0; d < SPATIAL_DIM; ++d) {
        out += "output_padding_" + std::to_string(d) + ":" +
            std::to_string(output_pad[d]) + ", ";
      }
    }
    return out;
  }
};
} // namespace fbgemm
