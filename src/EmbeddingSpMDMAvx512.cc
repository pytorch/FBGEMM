/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmEmbedding.h"

#include <immintrin.h>
#include <type_traits>

namespace fbgemm {
namespace internal {

template <typename T>
struct w_reg;

template <>
struct w_reg<int32_t> {
  using reg_t = __m512;
};

template <>
struct w_reg<int64_t> {
  using reg_t = __m256;
};

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static constexpr int get_vlen() {
  return 16;
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static constexpr int get_vlen() {
  return 8;
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i load(void const* addr) {
  return _mm512_loadu_si512(addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i load(void const* addr) {
  return _mm512_loadu_si512(addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512 load_weights(void const* addr) {
  return _mm512_loadu_ps(addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m256 load_weights(float const* addr) {
  return _mm256_loadu_ps(addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __mmask16 mask_from_rem(int rem) {
  __mmask16 mask_rem_v = (((long long)1) << rem) - 1;
  return mask_rem_v;
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __mmask8 mask_from_rem(int rem) {
  __mmask8 mask_rem_v = (((long long)1) << rem) - 1;
  return mask_rem_v;
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i
mask_load(__m512i zero_v, __mmask16 mask_rem_v, void const* addr) {
  return _mm512_mask_loadu_epi32(zero_v, mask_rem_v, addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i
mask_load(__m512i zero_v, __mmask8 mask_rem_v, void const* addr) {
  return _mm512_mask_loadu_epi64(zero_v, mask_rem_v, addr);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i mask_mov(__m512i src, __mmask16 mask_rem_v, __m512i a) {
  return _mm512_mask_mov_epi32(src, mask_rem_v, a);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i mask_mov(__m512i src, __mmask8 mask_rem_v, __m512i a) {
  return _mm512_mask_mov_epi64(src, mask_rem_v, a);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i gather(__m512i indices, void const* addr) {
  return _mm512_i32gather_epi32(indices, addr, 4);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i gather(__m512i indices, void const* addr) {
  // ToDo: Change this _mm512_i64gather_epi64 once mapping table is 64-bit
  __m256i res_32 = _mm512_i64gather_epi32(indices, addr, 4);
  return _mm512_cvtepi32_epi64(res_32);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512i mask_gather(
    __m512i src,
    __mmask16 mask_rem_v,
    __m512i indices,
    void const* addr) {
  return _mm512_mask_i32gather_epi32(src, mask_rem_v, indices, addr, 4);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512i mask_gather(
    __m512i src,
    __mmask8 mask_rem_v,
    __m512i indices,
    void const* addr) {
  // ToDo: Change this _mm512_mask_i64gather_epi64 once mapping table is 64-bit
  __m256i res_32 = _mm512_mask_i64gather_epi32(
      _mm512_castsi512_si256(src), mask_rem_v, indices, addr, 4);
  return _mm512_cvtepi32_epi64(res_32);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __mmask16 gen_mask(__m512i indices, __m512i zero_v) {
  return _mm512_cmpge_epi32_mask(indices, zero_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __mmask8 gen_mask(__m512i indices, __m512i zero_v) {
  return _mm512_cmpge_epi64_mask(indices, zero_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline void compress_store(void* addr, __mmask16 mask, __m512i src_v) {
  _mm512_mask_compressstoreu_ps(addr, mask, _mm512_castsi512_ps(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline void compress_store(void* addr, __mmask8 mask, __m512i src_v) {
  _mm512_mask_compressstoreu_pd(addr, mask, _mm512_castsi512_pd(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline void
compress_store_weights(void* addr, __mmask16 mask, __m512 src_v) {
  _mm512_mask_compressstoreu_ps(addr, mask, src_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline void
compress_store_weights(void* addr, __mmask8 mask, __m256 src_v) {
  _mm256_mask_compressstoreu_ps(addr, mask, src_v);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline __m512 compress(__m512i zero_v, __mmask16 mask, __m512i src_v) {
  return _mm512_mask_compress_ps(
      _mm512_castsi512_ps(zero_v), mask, _mm512_castsi512_ps(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline __m512d compress(__m512i zero_v, __mmask8 mask, __m512i src_v) {
  return _mm512_mask_compress_pd(
      _mm512_castsi512_pd(zero_v), mask, _mm512_castsi512_pd(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int32_t>::value, int>::type = 0>
static inline void mask_store(void* addr, __mmask16 mask, __m512 src_v) {
  _mm512_mask_storeu_epi32(addr, mask, _mm512_castps_si512(src_v));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, int64_t>::value, int>::type = 0>
static inline void mask_store(void* addr, __mmask8 mask, __m512d src_v) {
  _mm512_mask_storeu_epi64(addr, mask, _mm512_castpd_si512(src_v));
}

template <typename IndexType, bool HAS_WEIGHTS>
void compressed_indices_remap_avx512(
    std::int32_t offsets_len,
    const IndexType* indices,
    const int32_t* compressed_indices_mapping,
    const IndexType* offsets,
    const float* weights, // optional, can be null,
    IndexType* out_indices,
    IndexType* out_offsets,
    float* out_weights) {
  __m512i zero_v = _mm512_set1_epi32(0);
  __m512i minus1_v = _mm512_set1_epi32(-1);
  out_offsets[0] = offsets[0];
  for (int k = 1; k < offsets_len; ++k) {
    int32_t start_offset = offsets[k - 1];
    int32_t end_offset = offsets[k];
    int len = end_offset - start_offset;

    // address of inputs for this iteration
    const IndexType* cur_indices = indices + start_offset;
    const float* cur_weights = nullptr;

    // address of outputs for this iteration
    int32_t out_start_offset = out_offsets[k - 1];
    IndexType* cur_out_indices = out_indices + out_start_offset;
    float* cur_out_weights = nullptr;
    if (HAS_WEIGHTS) {
      cur_weights = weights + start_offset;
      cur_out_weights = out_weights + out_start_offset;
    }

    IndexType count_indices = 0;
    constexpr int VLEN = get_vlen<IndexType>();
    int i = 0;
    for (; i < len / VLEN * VLEN; i += VLEN) {
      __m512i indices_v =
          load<IndexType>(reinterpret_cast<void const*>(cur_indices + i));

      // gather remapped indices from the mapping table
      __m512i remapped_indices_v = gather<IndexType>(
          indices_v, reinterpret_cast<void const*>(compressed_indices_mapping));

      typename w_reg<IndexType>::reg_t weights_v;
      if (HAS_WEIGHTS) {
        weights_v = load_weights<IndexType>(cur_weights + i);
      }

      // Now remove -1 from the remapped indices
      auto mask_indices_v = gen_mask<IndexType>(remapped_indices_v, zero_v);

      compress_store<IndexType>(
          reinterpret_cast<void*>(cur_out_indices + count_indices),
          mask_indices_v,
          remapped_indices_v);

      if (HAS_WEIGHTS) {
        compress_store_weights<IndexType>(
            reinterpret_cast<void*>(cur_out_weights + count_indices),
            mask_indices_v,
            weights_v);
      }

      count_indices += _mm_popcnt_u32(mask_indices_v);
    }

    // remainder
    int rem = len - i;
    if (rem > 0) {
      auto mask_rem_v = mask_from_rem<IndexType>(rem);
      __m512i indices_v = mask_load<IndexType>(
          zero_v, mask_rem_v, reinterpret_cast<void const*>(cur_indices + i));

      // gather remapped indices from the mapping table
      __m512i remapped_indices_v = mask_gather<IndexType>(
          zero_v,
          mask_rem_v,
          indices_v,
          reinterpret_cast<void const*>(compressed_indices_mapping));
      // mov -1 to not used places in the vector
      remapped_indices_v =
          mask_mov<IndexType>(minus1_v, mask_rem_v, remapped_indices_v);

      __m512 weights_v;
      if (HAS_WEIGHTS) {
        weights_v = _mm512_mask_loadu_ps(
            _mm512_castsi512_ps(zero_v),
            mask_rem_v,
            reinterpret_cast<void const*>(cur_weights + i));
      }

      // Now remove -1 from the remapped indices
      auto mask_indices_v = gen_mask<IndexType>(remapped_indices_v, zero_v);

      auto out_indices_v =
          compress<IndexType>(zero_v, mask_indices_v, remapped_indices_v);

      mask_store<IndexType>(
          reinterpret_cast<void*>(cur_out_indices + count_indices),
          mask_rem_v,
          out_indices_v);

      if (HAS_WEIGHTS) {
        __m512 out_weights_v = _mm512_mask_compress_ps(
            _mm512_castsi512_ps(zero_v), mask_indices_v, weights_v);
        _mm512_mask_storeu_ps(
            reinterpret_cast<void*>(cur_out_weights + count_indices),
            mask_rem_v,
            out_weights_v);
      }

      count_indices += _mm_popcnt_u32(mask_indices_v);
    }

    out_offsets[k] = out_offsets[k - 1] + count_indices;
  }
}

#define INSTANTIATE_REMAP_BASE(INDEX_TYPE, HAS_WEIGHTS)                   \
  template void compressed_indices_remap_avx512<INDEX_TYPE, HAS_WEIGHTS>( \
      std::int32_t offsets_numel,                                         \
      const INDEX_TYPE* indices,                                          \
      const int32_t* compressed_indices_mapping,                          \
      const INDEX_TYPE* offsets,                                          \
      const float* weights,                                               \
      INDEX_TYPE* out_indices,                                            \
      INDEX_TYPE* out_offsets,                                            \
      float* out_weights);

INSTANTIATE_REMAP_BASE(int32_t, true);
INSTANTIATE_REMAP_BASE(int32_t, false);
INSTANTIATE_REMAP_BASE(int64_t, true);
INSTANTIATE_REMAP_BASE(int64_t, false);

#undef INSTANTIATE_REMAP_BASE

} // namespace internal
} // namespace fbgemm
