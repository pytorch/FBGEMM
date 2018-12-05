/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
// This file defines common utilities used in code compiled with avx2/avx512
// flags.

#include <string>

namespace fbgemm {

enum class QuantizationGranularity {
  TENSOR,
  GROUP,
  OUT_CHANNEL,
};

/**
 * @brief A struct to represent a block of a matrix.
 */
struct block_type_t {
  int row_start;
  int row_size;
  int col_start;
  int col_size;

  std::string toString() const {
    std::string out = "";
    out += "row start:" + std::to_string(row_start) + ", ";
    out += "row size:" + std::to_string(row_size) + ", ";
    out += "col start:" + std::to_string(col_start) + ", ";
    out += "col size:" + std::to_string(col_size);
    return out;
  }
};

/**
 * @brief A struct to represent all the requantization parameters.
 *
 * Please note that this is different from RequantizationParams in
 * QuantUtilsAvx2.h as it combines all the parameters needed for various
 * quantization granularities
 */
struct requantizationParams_t {
  std::int32_t A_zero_point;
  const std::int32_t* B_zero_point;
  std::int32_t C_zero_point;
  const float* C_multiplier;
  const std::int32_t* row_offsets;
  const std::int32_t* col_offsets;
  const std::int32_t* bias;
  std::uint32_t ncols;
  int groups;
};

} // namespace fbgemm
