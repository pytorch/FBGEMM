/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/FbgemmEmbedding.h"

#include <immintrin.h>
#include <cassert>
#include <cmath>

#include "fbgemm/Types.h"

namespace fbgemm {
namespace internal {

template <typename inType, typename IndexType>
bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const inType* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional) {
  int64_t current = 0;
  for (int m = 0; m < output_size; ++m) {
    out[m] = 0;
    if (current + lengths[m] > index_size) {
      return false;
    }
    int i = 0;

    // The following code doesn't speedup
#if 0
    constexpr int VLEN = std::is_same<IndexType, std::int64_t>::value ? 4 : 8;
    for (; i < lengths[m] / VLEN * VLEN; i += VLEN) {
      if (std::is_same<IndexType, std::int64_t>::value) {
        __m256i idx_v = _mm256_lddqu_si256(
            reinterpret_cast<const __m256i*>(indices + current));
        // Should be none true
        int mask1 = _mm256_movemask_pd(_mm256_castsi256_pd(
            _mm256_cmpgt_epi64(_mm256_setzero_si256(), idx_v)));
        // Should be all true
        int mask2 = _mm256_movemask_pd(_mm256_castsi256_pd(
            _mm256_cmpgt_epi64(_mm256_set1_epi64x(data_size), idx_v)));
        if (mask1 || mask2 != 0x0f) {
          return false;
        }

        __m128 in_v = _mm256_i64gather_ps(input, idx_v, 4);
        alignas(64) float in_buf[VLEN];
        _mm_store_ps(in_buf, in_v);
        for (int j = 0; j < VLEN; ++j) {
          if (weights) {
            out[m] = std::fma(
                weights[is_weight_positional ? i + j : current + j],
                in_buf[j],
                out[m]);
          } else {
            out[m] += in_buf[j];
          }
        }
      } else {
        __m256i idx_v = _mm256_lddqu_si256(
            reinterpret_cast<const __m256i*>(indices + current));
        // Should be none true
        int mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(
            _mm256_cmpgt_epi32(_mm256_setzero_si256(), idx_v)));
        // Should be all true
        int mask2 = _mm256_movemask_ps(_mm256_castsi256_ps(
            _mm256_cmpgt_epi32(_mm256_set1_epi32(data_size), idx_v)));
        if (mask1 || mask2 != 0x00ff) {
          return false;
        }

        __m256 in_v = _mm256_i32gather_ps(input, idx_v, 4);
        alignas(64) float in_buf[VLEN];
        _mm256_store_ps(in_buf, in_v);
        for (int j = 0; j < VLEN; ++j) {
          if (weights) {
            out[m] = std::fma(
                weights[is_weight_positional ? i + j : current + j],
                in_buf[j],
                out[m]);
          } else {
            out[m] += in_buf[j];
          }
        }
      }

      current += VLEN;
    }
#endif

    for (; i < lengths[m]; ++i) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      float w = 1.f;
      if (weights) {
        w = weights[is_weight_positional ? i : current];
      }

      const inType* inptr = input + indices[current];
      out[m] = std::fma(
          w,
          std::is_same<inType, float16>::value ? cpu_half2float(*inptr)
                                               : *inptr,
          out[m]);

      ++current;
    }
    if (normalize_by_lengths && lengths[m]) {
      float scale = 1.f / lengths[m];
      out[m] *= scale;
    }
  }
  return current == index_size;
}

template bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const float* input,
    const std::int64_t* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional);

template bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const float* input,
    const std::int32_t* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional);

template bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const float16* input,
    const std::int64_t* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional);

template bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const float16* input,
    const std::int32_t* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional);

template bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const std::uint8_t* input,
    const std::int64_t* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional);

template bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const std::uint8_t* input,
    const std::int32_t* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional);

} // namespace internal
} // namespace fbgemm
