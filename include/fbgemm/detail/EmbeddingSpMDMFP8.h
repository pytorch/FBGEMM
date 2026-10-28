/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FloatConversion.h"
#include "fbgemm/Utils.h"

#if defined(__clang__) && HAVE_SVE
#include <arm_neon.h>
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>
#endif

namespace fbgemm::detail {

enum class FP8OutputKind { Float, Half, BFloat };

FBGEMM_API bool use_fp8_autovec();
FBGEMM_API void record_fp8_embedding_pattern(
    int64_t rows,
    int64_t dims,
    int64_t batch,
    int64_t length,
    FP8OutputKind output_kind);

inline void prefetch_fp8(const void* ptr) {
#ifndef _WIN32
  __builtin_prefetch(ptr, 0, 0);
#else
  (void)ptr;
#endif
}

// Provide an alternative to `memset` that can be inlined. Contrary to a
// `memset` call with the default aarch64 calling convention this will only
// clobber a single vector register.
inline void zero_fp8_accumulator(float* ptr, int64_t count) {
#if defined(__clang__) && HAVE_SVE
  if (!__builtin_constant_p(count)) {
    float32x4_t zeroVec;
    // Inline asm prevents compiler from replacing with a call to memset
    asm volatile("movi  %[zeroVec].2d, #0000000000000000"
                 : [zeroVec] "=w"(zeroVec)
                 :
                 :);
    while (count >= 16) {
      vst1q_f32(ptr, zeroVec);
      vst1q_f32(ptr + 4, zeroVec);
      vst1q_f32(ptr + 8, zeroVec);
      vst1q_f32(ptr + 12, zeroVec);
      ptr += 16;
      count -= 16;
    }
    if (count >= 8) {
      vst1q_f32(ptr, zeroVec);
      vst1q_f32(ptr + 4, zeroVec);
      ptr += 8;
      count -= 8;
    }
    if (count > 0) {
      svbool_t predA = svwhilelt_b32_u64(0, count);
      svbool_t predB = svwhilelt_b32_u64(4, count);
      svst1_f32(predA, ptr, svset_neonq(svundef_f32(), zeroVec));
      svst1_f32(predB, ptr + 4, svset_neonq(svundef_f32(), zeroVec));
    }
  } else {
#endif
    std::memset(ptr, 0, sizeof(float) * count);
#if defined(__clang__) && HAVE_SVE
  }
#endif
}

template <
    typename InType,
    typename IndexType,
    typename OffsetType,
    typename OutType,
    typename Decode,
    typename RecordPattern>
inline bool embedding_spmdm_fp8(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights,
    OutType* out,
    const bool is_weight_positional,
    const bool normalize_by_lengths,
    const bool use_offsets,
    const int64_t output_stride,
    const int64_t input_stride,
    Decode decode,
    RecordPattern record_pattern,
    bool use_autovec) {
  static_assert(sizeof(InType) == 1);
  if (data_size < 0) {
    return false;
  }

  std::array<float, 512> local_storage;
  std::unique_ptr<float[]> heap_storage;
  float* buf = nullptr;
  if (std::cmp_less_equal(block_size, 512)) {
    buf = local_storage.data();
  } else {
    heap_storage.reset(new float[block_size]);
    buf = heap_storage.get();
  }

  // Prefetch up to 16 rows within a 4 KiB budget, tuned for Neoverse-V2.
  constexpr int64_t max_prefetch_bytes = 4096;
  constexpr int64_t max_initial_prefetch_rows = 16;
  const int64_t rows_to_prefetch = use_autovec && input_stride > 0
      ? std::min(max_initial_prefetch_rows, max_prefetch_bytes / input_stride)
      : 0;
  const int64_t prefetch_stride = std::min(rows_to_prefetch, index_size);
  for (int pf_idx = 0; pf_idx < prefetch_stride; ++pf_idx) {
    [[maybe_unused]] const InType* prefetch_addr =
        input + input_stride * indices[pf_idx];
    for (int64_t offset = 0; offset < input_stride; offset += 64) {
      prefetch_fp8(prefetch_addr + offset);
    }
  }

  // Reference implementation of FP8 SLS. The algorithm is similar to FP32 SLS
  // except for the FP8->FP32 conversion after reading the embedding weight.
  int64_t current = 0;

  for (int m = 0; m < output_size; ++m) {
    zero_fp8_accumulator(buf, block_size);
    const int len = use_offsets
        ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
        : offsets_or_lengths[m];
    int64_t end = current + len;
    if (end > index_size) {
      return false;
    }

    record_pattern(data_size, block_size, output_size, len);

    const float* weights_addr = weights != nullptr
        ? (is_weight_positional ? weights : weights + current)
        : nullptr;
    for (; current < end; ++current) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      if (use_autovec) {
        [[maybe_unused]] int64_t prefetch_idx =
            indices[std::min(current + prefetch_stride, index_size - 1)];

        prefetch_fp8(input + input_stride * prefetch_idx);
        if (input_stride > 64) {
          for (int64_t offset = 64; offset < input_stride; offset += 64) {
            prefetch_fp8(input + input_stride * prefetch_idx + offset);
          }
        }
      }

      float w = 1.f;
      if (weights != nullptr) {
        w = *weights_addr++;
      }
      const InType* input_row = input + input_stride * idx;
      for (int64_t j = 0; j < block_size; ++j) {
        buf[j] = std::fma(w, decode(input_row[j]), buf[j]);
      }
    }
    if (normalize_by_lengths) {
      if (len) {
        const float scale = 1.f / len;
        for (int j = 0; j < block_size; ++j) {
          buf[j] *= scale;
        }
      }
    }

    for (int64_t j = 0; j < block_size; ++j) {
      out[j] = convert_ref<OutType>(buf[j]);
    }
    out += output_stride;
  }
  return current == index_size;
}

} // namespace fbgemm::detail
