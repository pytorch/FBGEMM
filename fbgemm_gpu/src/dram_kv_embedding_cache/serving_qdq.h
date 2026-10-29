/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <fbgemm/QuantUtils.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <vector>

/// Rewrites embedding rows as the values serving decodes after a rowwise
/// N-bit publish (fbgemm FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf, fp16
/// scale and bias), so training reads exactly what the published table serves.
namespace kv_mem::serving_qdq {

enum class RowResult {
  kAlreadyFixed,
  kChanged,
  kChangedMultiPass,
  kNonFinite,
  kNotConverged,
};

struct Stats {
  int64_t rows{0};
  int64_t changed{0};
  int64_t multi_pass{0};
  int64_t non_finite{0};
  int64_t not_converged{0};

  void record(RowResult result) {
    ++rows;
    switch (result) {
      case RowResult::kAlreadyFixed:
        break;
      case RowResult::kChanged:
        ++changed;
        break;
      case RowResult::kChangedMultiPass:
        ++changed;
        ++multi_pass;
        break;
      case RowResult::kNonFinite:
        ++non_finite;
        break;
      case RowResult::kNotConverged:
        ++changed;
        ++not_converged;
        break;
    }
  }

  Stats& operator+=(const Stats& other) {
    rows += other.rows;
    changed += other.changed;
    multi_pass += other.multi_pass;
    non_finite += other.non_finite;
    not_converged += other.not_converged;
    return *this;
  }
};

inline std::ostream& operator<<(std::ostream& os, const Stats& stats) {
  return os << "rows=" << stats.rows << " changed=" << stats.changed
            << " multi_pass=" << stats.multi_pass
            << " non_finite=" << stats.non_finite
            << " not_converged=" << stats.not_converged;
}

/// Reusable buffers for one thread's round trips.
struct Scratch {
  std::vector<std::uint8_t> packed;
  std::vector<fbgemm::float16> dequantized;
  std::vector<fbgemm::float16> original;
};

constexpr int64_t packed_row_bytes(int64_t dim, int bit_rate) {
  const int64_t elems_per_byte = 8 / bit_rate;
  return (dim + elems_per_byte - 1) / elems_per_byte +
      2 * static_cast<int64_t>(sizeof(fbgemm::float16));
}

/// The scratch buffers hold exactly `dim` values, so `dim` must fill whole
/// packed bytes.
inline void check_round_trip_args(int64_t dim, int bit_rate) {
  TORCH_CHECK(
      bit_rate == 2 || bit_rate == 4,
      "serving_qdq bit_rate must be 2 or 4, got ",
      bit_rate);
  TORCH_CHECK(
      dim > 0 && dim % (8 / bit_rate) == 0,
      "serving_qdq dim must be a positive multiple of ",
      8 / bit_rate,
      ", got ",
      dim);
}

inline bool is_finite_row(const fbgemm::float16* row, int64_t dim) {
  const auto* halves = reinterpret_cast<const c10::Half*>(row);
  for (int64_t j = 0; j < dim; ++j) {
    if (!std::isfinite(static_cast<float>(halves[j]))) {
      return false;
    }
  }
  return true;
}

/// Replaces row[0, dim) with a fixed point y of dequantize(quantize(.)):
/// publishing y re-derives the same codes, scale and bias, so serving decodes
/// y bit-for-bit. One pass is usually a fixed point; a row whose fp16 scale
/// re-rounds needs a second. Rows with inf/nan, and rows whose decode would
/// overflow fp16 (magnitudes near 65504), are left as they are.
inline RowResult round_trip_row(
    c10::Half* row,
    int64_t dim,
    int bit_rate,
    Scratch& scratch,
    int max_passes = 4) {
  auto* values = reinterpret_cast<fbgemm::float16*>(row);
  if (!is_finite_row(values, dim)) {
    return RowResult::kNonFinite;
  }
  const int64_t packed_bytes = packed_row_bytes(dim, bit_rate);
  scratch.packed.resize(packed_bytes);
  scratch.dequantized.resize(dim);
  scratch.original.assign(values, values + dim);
  const size_t row_bytes = dim * sizeof(fbgemm::float16);
  for (int pass = 0; pass < max_passes; ++pass) {
    fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<fbgemm::float16>(
        bit_rate, values, 1, static_cast<int>(dim), scratch.packed.data());
    fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<fbgemm::float16>(
        bit_rate,
        scratch.packed.data(),
        1,
        static_cast<int>(packed_bytes),
        scratch.dequantized.data());
    if (std::memcmp(scratch.dequantized.data(), values, row_bytes) == 0) {
      return pass == 0 ? RowResult::kAlreadyFixed
          : pass == 1  ? RowResult::kChanged
                       : RowResult::kChangedMultiPass;
    }
    if (!is_finite_row(scratch.dequantized.data(), dim)) {
      std::memcpy(values, scratch.original.data(), row_bytes);
      return RowResult::kNonFinite;
    }
    std::memcpy(values, scratch.dequantized.data(), row_bytes);
  }
  return RowResult::kNotConverged;
}

/// Round-trips the first `dim` columns of `num_rows` rows laid out `stride`
/// elements apart. Columns past `dim` (e.g. optimizer state) are untouched.
inline Stats round_trip_rows(
    c10::Half* rows,
    int64_t num_rows,
    int64_t stride,
    int64_t dim,
    int bit_rate) {
  check_round_trip_args(dim, bit_rate);
  TORCH_CHECK(dim <= stride, "serving_qdq dim ", dim, " > row stride ", stride);
  Scratch scratch;
  Stats stats;
  for (int64_t r = 0; r < num_rows; ++r) {
    stats.record(round_trip_row(rows + r * stride, dim, bit_rate, scratch));
  }
  return stats;
}

} // namespace kv_mem::serving_qdq
