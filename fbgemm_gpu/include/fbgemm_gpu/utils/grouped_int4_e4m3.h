/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Float8_e4m3fn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

/// Group-wise symmetric INT4 with one float8_e4m3fn scale per group of
/// `group_size` consecutive values. Per group, scale = e4m3fn(max|x| / 7),
/// clamped to [2^-9, 448] before rounding, and code = rint(x / scale) clipped
/// to [-7, 7]; the value is code * scale. A row of `dim` values packs into
/// dim / 2 code bytes (two's-complement nibbles, even column in the low
/// nibble) followed by dim / group_size scale bytes.
namespace fbgemm_gpu::grouped_int4_e4m3 {

constexpr float kMinScale = 1.0f / 512.0f;
constexpr float kMaxScale = 448.0f;
constexpr float kQMax = 7.0f;

constexpr int64_t packed_row_bytes(int64_t dim, int64_t group_size) {
  return dim / 2 + dim / group_size;
}

inline void check_dims(int64_t dim, int64_t group_size) {
  TORCH_CHECK(
      group_size > 0 && group_size % 2 == 0,
      "group_size must be a positive even number, got ",
      group_size);
  TORCH_CHECK(
      dim > 0 && dim % group_size == 0,
      "dim must be a positive multiple of group_size ",
      group_size,
      ", got ",
      dim);
}

inline uint8_t encode(float value, float scale) {
  const float code =
      std::nearbyint(std::fmin(std::fmax(value / scale, -kQMax), kQMax));
  return static_cast<uint8_t>(static_cast<int8_t>(code)) & 0x0F;
}

inline float decode(uint8_t nibble, float scale) {
  const int code = static_cast<int>(nibble ^ 0x08) - 0x08;
  return static_cast<float>(code) * scale;
}

template <typename input_t>
void quantize_row(
    const input_t* input,
    int64_t dim,
    int64_t group_size,
    uint8_t* output) {
  uint8_t* codes = output;
  uint8_t* scales = output + dim / 2;
  for (int64_t start = 0; start < dim; start += group_size) {
    float abs_max = 0.0f;
    for (int64_t j = start; j < start + group_size; ++j) {
      abs_max = std::fmax(abs_max, std::fabs(static_cast<float>(input[j])));
    }
    const c10::Float8_e4m3fn scale_bits(
        std::clamp(abs_max / kQMax, kMinScale, kMaxScale));
    scales[start / group_size] = scale_bits.x;
    const float scale = static_cast<float>(scale_bits);
    for (int64_t j = start; j < start + group_size; j += 2) {
      codes[j / 2] = encode(static_cast<float>(input[j]), scale) |
          static_cast<uint8_t>(
                         encode(static_cast<float>(input[j + 1]), scale) << 4);
    }
  }
}

template <typename output_t>
void dequantize_row(
    const uint8_t* input,
    int64_t dim,
    int64_t group_size,
    output_t* output) {
  const uint8_t* codes = input;
  const uint8_t* scales = input + dim / 2;
  for (int64_t start = 0; start < dim; start += group_size) {
    const float scale = static_cast<float>(c10::Float8_e4m3fn(
        scales[start / group_size], c10::Float8_e4m3fn::from_bits()));
    for (int64_t j = start; j < start + group_size; j += 2) {
      const uint8_t byte = codes[j / 2];
      output[j] = static_cast<output_t>(decode(byte & 0x0F, scale));
      output[j + 1] = static_cast<output_t>(decode(byte >> 4, scale));
    }
  }
}

} // namespace fbgemm_gpu::grouped_int4_e4m3
