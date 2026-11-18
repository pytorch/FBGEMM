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
#include <c10/util/Half.h>
#include <fbgemm/QuantUtils.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <vector>
#include "fbgemm_gpu/utils/grouped_int4_e4m3.h"

#ifdef __aarch64__
#include <arm_neon.h>
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#endif

/// Rewrites embedding rows as the values serving decodes after the table's
/// publish encoding (rowwise N-bit with an fp16 scale and bias, or group-wise
/// INT4 with float8_e4m3fn scales), so training reads exactly what the
/// published table serves.
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

/// The publish encoding: rowwise N-bit with an fp16 scale and bias when
/// group_size is 0, else group-wise INT4 with float8_e4m3fn scales
/// (fbgemm_gpu/utils/grouped_int4_e4m3.h).
struct Format {
  int bit_rate{0};
  int64_t group_size{0};
};

inline int64_t packed_row_bytes(int64_t dim, const Format& format) {
  if (format.group_size > 0) {
    return fbgemm_gpu::grouped_int4_e4m3::packed_row_bytes(
        dim, format.group_size);
  }
  const int64_t elems_per_byte = 8 / format.bit_rate;
  return (dim + elems_per_byte - 1) / elems_per_byte +
      2 * static_cast<int64_t>(sizeof(fbgemm::float16));
}

/// The scratch buffers hold exactly `dim` values, so `dim` must fill whole
/// packed bytes (and whole groups).
inline void check_round_trip_args(int64_t dim, const Format& format) {
  if (format.group_size > 0) {
    TORCH_CHECK(
        format.bit_rate == 4,
        "serving_qdq group_size needs bit_rate 4, got ",
        format.bit_rate);
    fbgemm_gpu::grouped_int4_e4m3::check_dims(dim, format.group_size);
    return;
  }
  TORCH_CHECK(
      format.bit_rate == 2 || format.bit_rate == 4,
      "serving_qdq bit_rate must be 2 or 4, got ",
      format.bit_rate);
  TORCH_CHECK(
      dim > 0 && dim % (8 / format.bit_rate) == 0,
      "serving_qdq dim must be a positive multiple of ",
      8 / format.bit_rate,
      ", got ",
      dim);
}

/// Encodes `values` into scratch.packed and decodes it into
/// scratch.dequantized.
inline void quantize_dequantize(
    const fbgemm::float16* values,
    int64_t dim,
    const Format& format,
    Scratch& scratch) {
  if (format.group_size > 0) {
    fbgemm_gpu::grouped_int4_e4m3::quantize_row(
        reinterpret_cast<const c10::Half*>(values),
        dim,
        format.group_size,
        scratch.packed.data());
    fbgemm_gpu::grouped_int4_e4m3::dequantize_row(
        scratch.packed.data(),
        dim,
        format.group_size,
        reinterpret_cast<c10::Half*>(scratch.dequantized.data()));
    return;
  }
  fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<fbgemm::float16>(
      format.bit_rate, values, 1, static_cast<int>(dim), scratch.packed.data());
  fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<fbgemm::float16>(
      format.bit_rate,
      scratch.packed.data(),
      1,
      static_cast<int>(scratch.packed.size()),
      scratch.dequantized.data());
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
/// publishing y re-derives the same codes and scales, so serving decodes y
/// bit-for-bit. One pass is usually a fixed point; a row whose scale re-rounds
/// needs a second. Rows with inf/nan, and rows whose decode would overflow fp16
/// (magnitudes near 65504), are left as they are.
inline RowResult round_trip_row(
    c10::Half* row,
    int64_t dim,
    const Format& format,
    Scratch& scratch,
    int max_passes = 4) {
  auto* values = reinterpret_cast<fbgemm::float16*>(row);
  if (!is_finite_row(values, dim)) {
    return RowResult::kNonFinite;
  }
  scratch.packed.resize(packed_row_bytes(dim, format));
  scratch.dequantized.resize(dim);
  scratch.original.assign(values, values + dim);
  const size_t row_bytes = dim * sizeof(fbgemm::float16);
  for (int pass = 0; pass < max_passes; ++pass) {
    quantize_dequantize(values, dim, format, scratch);
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
    const Format& format) {
  check_round_trip_args(dim, format);
  TORCH_CHECK(dim <= stride, "serving_qdq dim ", dim, " > row stride ", stride);
  Scratch scratch;
  Stats stats;
  for (int64_t r = 0; r < num_rows; ++r) {
    stats.record(round_trip_row(rows + r * stride, dim, format, scratch));
  }
  return stats;
}

/// Byte layout of a row stored in its group-wise INT4 publish encoding: the
/// first `dim` columns as dim / 2 code bytes then dim / group_size
/// float8_e4m3fn scale bytes, followed by the remaining `tail` columns (e.g.
/// optimizer state) as raw fp16.
struct PackedLayout {
  int64_t dim{0};
  int64_t group_size{0};
  int64_t tail{0};

  int64_t code_bytes() const {
    return dim / 2;
  }
  int64_t packed_bytes() const {
    return fbgemm_gpu::grouped_int4_e4m3::packed_row_bytes(dim, group_size);
  }
  int64_t payload_bytes() const {
    return packed_bytes() + tail * static_cast<int64_t>(sizeof(c10::Half));
  }
};

/// First scale byte of a row whose values are kept outside the packed slot.
/// It is an e4m3fn NaN, which the encoder never writes (scales are clamped to
/// [2^-9, 448], and an all-zero group gets scale 0).
constexpr uint8_t kFallbackScale = 0x7F;

/// fp16 bits of grouped_int4_e4m3::decode(nibble, scale) for every scale byte
/// and nibble, split into low and high bytes so a byte shuffle decodes a group.
struct DecodeTable {
  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays): 16-byte rows for SIMD loads
  alignas(16) uint8_t lo[256][16];
  alignas(16) uint8_t hi[256][16];
  // NOLINTEND(cppcoreguidelines-avoid-c-arrays)
};

inline const DecodeTable& decode_table() {
  static const DecodeTable table = [] {
    DecodeTable t{};
    for (int s = 0; s < 256; ++s) {
      const float scale = static_cast<float>(c10::Float8_e4m3fn(
          static_cast<uint8_t>(s), c10::Float8_e4m3fn::from_bits()));
      for (int n = 0; n < 16; ++n) {
        const uint16_t bits =
            c10::Half(fbgemm_gpu::grouped_int4_e4m3::decode(n, scale)).x;
        t.lo[s][n] = static_cast<uint8_t>(bits & 0xFF);
        t.hi[s][n] = static_cast<uint8_t>(bits >> 8);
      }
    }
    return t;
  }();
  return table;
}

inline void unpack_group_scalar(
    const DecodeTable& t,
    const uint8_t* codes,
    uint8_t scale,
    int64_t group_size,
    uint16_t* out) {
  for (int64_t j = 0; j < group_size / 2; ++j) {
    const uint8_t lo = codes[j] & 0x0F;
    const uint8_t hi = codes[j] >> 4;
    out[2 * j] = static_cast<uint16_t>(t.lo[scale][lo] | t.hi[scale][lo] << 8);
    out[2 * j + 1] =
        static_cast<uint16_t>(t.lo[scale][hi] | t.hi[scale][hi] << 8);
  }
}

/// Decodes `num_groups` groups of 8: 4 code bytes and 1 scale byte each make
/// 8 fp16 values (16 output bytes).
inline void unpack_g8(
    const uint8_t* codes,
    const uint8_t* scales,
    int64_t num_groups,
    uint8_t* out) {
  const auto& t = decode_table();
  int64_t g = 0;
#ifdef __aarch64__
  const uint8x8_t low_nibble = vdup_n_u8(0x0F);
  for (; g + 2 <= num_groups; g += 2) {
    const uint8x8_t bytes = vld1_u8(codes + g * 4);
    // val[0] holds group g's 8 nibbles in column order, val[1] group g + 1's.
    const uint8x8x2_t nibbles =
        vzip_u8(vand_u8(bytes, low_nibble), vshr_n_u8(bytes, 4));
    for (int k = 0; k < 2; ++k) {
      const uint8_t s = scales[g + k];
      const uint8x8_t lo = vqtbl1_u8(vld1q_u8(t.lo[s]), nibbles.val[k]);
      const uint8x8_t hi = vqtbl1_u8(vld1q_u8(t.hi[s]), nibbles.val[k]);
      const uint8x8x2_t halves = vzip_u8(lo, hi);
      vst1q_u8(out + (g + k) * 16, vcombine_u8(halves.val[0], halves.val[1]));
    }
  }
#elif defined(__SSSE3__)
  const __m128i low_nibble = _mm_set1_epi8(0x0F);
  for (; g < num_groups; ++g) {
    uint32_t word = 0;
    std::memcpy(&word, codes + g * 4, sizeof(word));
    const __m128i bytes = _mm_cvtsi32_si128(static_cast<int>(word));
    const __m128i nibbles = _mm_unpacklo_epi8(
        _mm_and_si128(bytes, low_nibble),
        _mm_and_si128(_mm_srli_epi16(bytes, 4), low_nibble));
    const uint8_t s = scales[g];
    const __m128i lo = _mm_shuffle_epi8(
        _mm_load_si128(reinterpret_cast<const __m128i*>(t.lo[s])), nibbles);
    const __m128i hi = _mm_shuffle_epi8(
        _mm_load_si128(reinterpret_cast<const __m128i*>(t.hi[s])), nibbles);
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(out + g * 16), _mm_unpacklo_epi8(lo, hi));
  }
#endif
  for (; g < num_groups; ++g) {
    std::array<uint16_t, 8> group;
    unpack_group_scalar(t, codes + g * 4, scales[g], 8, group.data());
    std::memcpy(out + g * 16, group.data(), sizeof(group));
  }
}

/// Decodes the packed first `layout.dim` columns; bit-identical to
/// grouped_int4_e4m3::dequantize_row.
inline void
unpack_row(const uint8_t* payload, const PackedLayout& layout, c10::Half* out) {
  const uint8_t* scales = payload + layout.code_bytes();
  if (layout.group_size == 8) {
    unpack_g8(payload, scales, layout.dim / 8, reinterpret_cast<uint8_t*>(out));
    return;
  }
  const auto& t = decode_table();
  auto* dst = reinterpret_cast<uint16_t*>(out);
  for (int64_t g = 0; g < layout.dim / layout.group_size; ++g) {
    unpack_group_scalar(
        t,
        payload + g * layout.group_size / 2,
        scales[g],
        layout.group_size,
        dst + g * layout.group_size);
  }
}

/// Encodes row[0, dim) into `payload` and returns whether it decodes back
/// bit-for-bit (true exactly for serving fixed points). An all-zero group gets
/// scale 0, so a zero row packs to zero bytes. On false the payload holds the
/// encoding of the row, which is not the row.
inline bool
pack_row(const c10::Half* row, const PackedLayout& layout, uint8_t* payload) {
  fbgemm_gpu::grouped_int4_e4m3::quantize_row(
      row, layout.dim, layout.group_size, payload);
  uint8_t* scales = payload + layout.code_bytes();
  const int64_t group_code_bytes = layout.group_size / 2;
  for (int64_t g = 0; g < layout.dim / layout.group_size; ++g) {
    const uint8_t* codes = payload + g * group_code_bytes;
    if (std::all_of(codes, codes + group_code_bytes, [](uint8_t b) {
          return b == 0;
        })) {
      scales[g] = 0;
    }
  }
  thread_local std::vector<c10::Half> decoded;
  decoded.resize(layout.dim);
  unpack_row(payload, layout, decoded.data());
  return std::memcmp(decoded.data(), row, layout.dim * sizeof(c10::Half)) == 0;
}

/// Byte layout of a row stored as row-wise INT8: the first `dim` columns as
/// dim code bytes then an fp32 scale and an fp32 bias (the INT8 publish row of
/// torch.ops.quantized.embedding_bag_byte_prepack, and the owner-local HBM
/// row), followed by the remaining `tail` columns (e.g. optimizer state) as
/// raw fp16.
struct Int8Layout {
  static constexpr int64_t kQparamsBytes = 2 * sizeof(float);

  int64_t dim{0};
  int64_t tail{0};

  int64_t row_bytes() const {
    return dim + kQparamsBytes;
  }
  int64_t payload_bytes() const {
    return row_bytes() + tail * static_cast<int64_t>(sizeof(c10::Half));
  }
};

enum class Int8Result {
  kFixed,
  kMultiPass,
  kNonFinite,
  kNotConverged,
};

/// Decodes an INT8 row as the owner-local HBM gather does: fp16 of
/// code * scale + bias.
inline void decode_int8_row(const uint8_t* src, int64_t dim, c10::Half* dst) {
  float qparams[2];
  std::memcpy(qparams, src + dim, sizeof(qparams));
  for (int64_t d = 0; d < dim; ++d) {
    dst[d] = c10::Half(static_cast<float>(src[d]) * qparams[0] + qparams[1]);
  }
}

/// One pass of the INT8 publish quantizer over row[0, dim).
inline void
encode_int8_row_once(const c10::Half* row, int64_t dim, uint8_t* dst) {
  fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<fbgemm::float16>(
      reinterpret_cast<const fbgemm::float16*>(row),
      1,
      static_cast<int>(dim),
      dst);
}

/// Replaces nan with 0 and +-inf with +-65504; returns whether all were
/// finite.
inline bool clamp_non_finite(c10::Half* row, int64_t dim) {
  bool finite = true;
  for (int64_t d = 0; d < dim; ++d) {
    const float value = static_cast<float>(row[d]);
    if (std::isfinite(value)) {
      continue;
    }
    finite = false;
    row[d] = std::isnan(value) ? c10::Half(0.0f)
                               : c10::Half(value > 0 ? 65504.0f : -65504.0f);
  }
  return finite;
}

/// Encodes row[0, dim) into `dst` (row_bytes() bytes) as codes that a publish
/// of the decoded row re-derives bit for bit: the row is quantized, then its
/// fp16 decode is re-quantized until the encoding stops changing (one extra
/// pass for almost every row). Non-finite values are clamped first.
inline Int8Result encode_int8_row(
    const c10::Half* row,
    int64_t dim,
    uint8_t* dst,
    int max_passes = 4) {
  thread_local std::vector<c10::Half> values;
  thread_local std::vector<uint8_t> next;
  values.assign(row, row + dim);
  const bool finite = clamp_non_finite(values.data(), dim);
  const auto row_bytes = static_cast<size_t>(dim + Int8Layout::kQparamsBytes);
  next.resize(row_bytes);
  encode_int8_row_once(values.data(), dim, dst);
  for (int pass = 1; pass < max_passes; ++pass) {
    decode_int8_row(dst, dim, values.data());
    encode_int8_row_once(values.data(), dim, next.data());
    if (std::memcmp(next.data(), dst, row_bytes) == 0) {
      if (!finite) {
        return Int8Result::kNonFinite;
      }
      return pass == 1 ? Int8Result::kFixed : Int8Result::kMultiPass;
    }
    std::memcpy(dst, next.data(), row_bytes);
  }
  return Int8Result::kNotConverged;
}

} // namespace kv_mem::serving_qdq
