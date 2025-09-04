/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm/Utils.h"

#if HAVE_SVE

#define FBGEMM_EXPORTS
#include "./EmbeddingSpMDMAutovec.h"
#include "./RefImplementations.h"
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FloatConversion.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <new>
#include <numeric>
#include <thread>

#ifdef _WIN32
#define do_prefetch(...)
#else
#define do_prefetch(...) __builtin_prefetch(__VA_ARGS__)
#endif

#define FBGEMM_VECTOR_WIDTH 16

namespace fbgemm {
namespace internal {

static constexpr size_t LOCAL_STORAGE_SIZE = 512;

template <typename OutType, bool WithScale>
static inline void fill_output_sve(
    OutType* out,
    const float32x4x2_t* src,
    const uint64_t iters,
    const bool is_bf16_out,
    svbool_t lastPredA,
    svbool_t lastPredB,
    svbool_t lastPredC,
    float32x4_t scale) {
  const float32x4x2_t* srcPtr = reinterpret_cast<const float32x4x2_t*>(src);
  const float32x4x2_t* endPtr = srcPtr + iters;
  if constexpr (std::is_same<OutType, float>::value) {
    if constexpr (WithScale) {
      float32x4x2_t* outPtr = reinterpret_cast<float32x4x2_t*>(out);
      for (; srcPtr < endPtr;) {
        float32x4_t row_0 = srcPtr->val[0];
        float32x4_t row_1 = srcPtr->val[1];
        srcPtr += 1;

        row_0 = vmulq_f32(row_0, scale);
        row_1 = vmulq_f32(row_1, scale);

        outPtr->val[0] = row_0;
        outPtr->val[1] = row_1;
        outPtr += 1;
      }

      {
        const auto ptrIn = reinterpret_cast<const float*>(srcPtr);
        auto ptrOut = reinterpret_cast<float*>(outPtr);

        svfloat32_t trailing_row_0 = svld1_f32(lastPredA, ptrIn);
        svfloat32_t trailing_row_1 = svld1_f32(lastPredB, ptrIn + 4);

        trailing_row_0 = svmul_f32_x(
            lastPredA, trailing_row_0, svset_neonq_f32(svundef_f32(), scale));
        trailing_row_1 = svmul_f32_x(
            lastPredB, trailing_row_1, svset_neonq_f32(svundef_f32(), scale));

        svst1_f32(lastPredA, ptrOut, trailing_row_0);
        svst1_f32(lastPredB, ptrOut + 4, trailing_row_1);
      }
    }
  } else {
    if (std::is_same<OutType, uint16_t>::value && is_bf16_out) {
      auto ptrOut = reinterpret_cast<uint16_t*>(out);

      for (; srcPtr < endPtr;) {
        float32x4_t row_0 = srcPtr->val[0];
        float32x4_t row_1 = srcPtr->val[1];
        srcPtr += 1;

        if constexpr (WithScale) {
          row_0 = vmulq_f32(row_0, scale);
          row_1 = vmulq_f32(row_1, scale);
        }

        auto svrow_0 = svreinterpret_u32_u16(svrshrnb_n_u32(
            svreinterpret_u32_f32(svset_neonq_f32(svundef_f32(), row_0)), 16));
        auto svrow_1 = svreinterpret_u32_u16(svrshrnb_n_u32(
            svreinterpret_u32_f32(svset_neonq_f32(svundef_f32(), row_1)), 16));

        svst1h_u32(svptrue_b8(), ptrOut, svrow_0);
        svst1h_u32(svptrue_b8(), ptrOut + 4, svrow_1);

        ptrOut += 8;
      }

      {
        const auto ptrIn = reinterpret_cast<const float*>(srcPtr);

        svfloat32_t trailing_row_0 = svld1_f32(lastPredA, ptrIn);
        svfloat32_t trailing_row_1 = svld1_f32(lastPredB, ptrIn + 4);

        if constexpr (WithScale) {
          trailing_row_0 = svmul_f32_x(
              lastPredA, trailing_row_0, svset_neonq_f32(svundef_f32(), scale));
          trailing_row_1 = svmul_f32_x(
              lastPredB, trailing_row_1, svset_neonq_f32(svundef_f32(), scale));
        }

        auto trailing_row_0_u32 = svreinterpret_u32_u16(
            svrshrnb_n_u32(svreinterpret_u32_f32(trailing_row_0), 16));
        auto trailing_row_1_u32 = svreinterpret_u32_u16(
            svrshrnb_n_u32(svreinterpret_u32_f32(trailing_row_1), 16));

        svst1h_u32(lastPredA, ptrOut, trailing_row_0_u32);
        svst1h_u32(lastPredB, ptrOut + 4, trailing_row_1_u32);
      }
    } else {
      float16x8_t* outPtr = reinterpret_cast<float16x8_t*>(out);

      for (; srcPtr < endPtr;) {
        float32x4_t row_0 = srcPtr->val[0];
        float32x4_t row_1 = srcPtr->val[1];
        srcPtr += 1;

        if constexpr (WithScale) {
          row_0 = vmulq_f32(row_0, scale);
          row_1 = vmulq_f32(row_1, scale);
        }

        float16x4_t converted_row_0 = vcvt_f16_f32(row_0);
        float16x8_t combined_rows = vcvt_high_f16_f32(converted_row_0, row_1);

        *outPtr = combined_rows;
        outPtr += 1;
      }

      {
        const auto ptrIn = reinterpret_cast<const float*>(srcPtr);
        auto ptrOut = reinterpret_cast<float16_t*>(outPtr);

        svfloat32_t trailing_row_0 = svld1_f32(lastPredA, ptrIn);
        svfloat32_t trailing_row_1 = svld1_f32(lastPredB, ptrIn + 4);

        if constexpr (WithScale) {
          trailing_row_0 = svmul_f32_x(
              lastPredA, trailing_row_0, svset_neonq_f32(svundef_f32(), scale));
          trailing_row_1 = svmul_f32_x(
              lastPredB, trailing_row_1, svset_neonq_f32(svundef_f32(), scale));
        }

        float16x4_t converted_trailing_row_0 =
            vcvt_f16_f32(svget_neonq(trailing_row_0));
        float16x8_t combined_trailing_rows = vcvt_high_f16_f32(
            converted_trailing_row_0, svget_neonq(trailing_row_1));

        svst1_f16(
            lastPredC,
            ptrOut,
            svset_neonq_f16(svundef_f16(), combined_trailing_rows));
      }
    }
  }
}

template <bool FuseWithOutput>
static inline void sve_fma_round(
    const uint8_t* input_row,
    float32x4x2_t* buf,
    uint64_t iters,
    svfloat32_t scale,
    svfloat32_t bias,
    svbool_t fullRowPred,
    svbool_t lastPredA,
    svbool_t lastPredB) {
  const uint64_t* input_row_v_0 = reinterpret_cast<const uint64_t*>(input_row);
  const uint64_t* input_row_v_1 =
      reinterpret_cast<const uint64_t*>(input_row + 4);
  const uint64_t* endPtr = input_row_v_0 + iters;
  while (input_row_v_0 < endPtr) {
    svuint32_t in_v_0 = svld1ub_u32(
        fullRowPred, reinterpret_cast<const uint8_t*>(input_row_v_0));
    svuint32_t in_v_1 = svld1ub_u32(
        fullRowPred, reinterpret_cast<const uint8_t*>(input_row_v_1));

    input_row_v_0 += 1;
    input_row_v_1 += 1;

    svfloat32_t in_v_0_f = svcvt_f32_u32_x(fullRowPred, in_v_0);
    svfloat32_t in_v_1_f = svcvt_f32_u32_x(fullRowPred, in_v_1);

    if constexpr (FuseWithOutput) {
      float32x4_t buf_0 = buf->val[0];
      float32x4_t buf_1 = buf->val[1];

      buf_0 = vaddq_f32(buf_0, svget_neonq(bias));
      buf_1 = vaddq_f32(buf_1, svget_neonq(bias));

      in_v_0_f = svmad_f32_m(
          fullRowPred, in_v_0_f, scale, svset_neonq_f32(svundef_f32(), buf_0));
      in_v_1_f = svmad_f32_m(
          fullRowPred, in_v_1_f, scale, svset_neonq_f32(svundef_f32(), buf_1));
    } else {
      in_v_0_f = svmad_f32_m(fullRowPred, in_v_0_f, scale, bias);
      in_v_1_f = svmad_f32_m(fullRowPred, in_v_1_f, scale, bias);
    }

    buf->val[0] = svget_neonq(in_v_0_f);
    buf->val[1] = svget_neonq(in_v_1_f);

    buf += 1;
  }

  {
    auto bufPtr = reinterpret_cast<float*>(buf);

    svuint32_t in_v_0 =
        svld1ub_u32(lastPredA, reinterpret_cast<const uint8_t*>(input_row_v_0));
    svuint32_t in_v_1 =
        svld1ub_u32(lastPredB, reinterpret_cast<const uint8_t*>(input_row_v_1));

    svfloat32_t in_v_0_f = svcvt_f32_u32_x(lastPredA, in_v_0);
    svfloat32_t in_v_1_f = svcvt_f32_u32_x(lastPredB, in_v_1);

    if constexpr (FuseWithOutput) {
      svfloat32_t buf_0 = svld1_f32(lastPredA, bufPtr);
      svfloat32_t buf_1 = svld1_f32(lastPredB, bufPtr + 4);

      buf_0 = svadd_f32_x(lastPredA, buf_0, bias);
      buf_1 = svadd_f32_x(lastPredB, buf_1, bias);

      in_v_0_f = svmad_f32_m(lastPredA, in_v_0_f, scale, buf_0);
      in_v_1_f = svmad_f32_m(lastPredB, in_v_1_f, scale, buf_1);
    } else {
      in_v_0_f = svmad_f32_m(lastPredA, in_v_0_f, scale, bias);
      in_v_1_f = svmad_f32_m(lastPredB, in_v_1_f, scale, bias);
    }

    svst1_f32(lastPredA, bufPtr, in_v_0_f);
    svst1_f32(lastPredB, bufPtr + 4, in_v_1_f);
  }
}

template <
    typename IndexType,
    typename OffsetType,
    typename OutType,
    bool NoBag,
    bool EnablePrefetching>
bool EmbeddingSpMDM8Bit_Sve(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const uint8_t* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    const bool normalize_by_lengths,
    OutType* out,
    const bool is_weight_positional,
    const bool use_offsets,
    const int64_t output_stride,
    const int64_t input_stride,
    const bool scale_bias_last,
    const bool is_bf16_out) {
  constexpr bool isOutput8bit = std::is_same<OutType, uint8_t>::value;
  if (data_size < 0) {
    return false;
  }
  if constexpr (isOutput8bit) {
    assert(input_stride == output_stride);
  }

  constexpr int64_t CACHE_LINE_SIZE = 64;
  constexpr int64_t MAX_INITIAL_PREFETCH_ROWS = 16;
  const int64_t prefetch_stride =
      std::min(MAX_INITIAL_PREFETCH_ROWS, index_size);

  if constexpr (EnablePrefetching) {
    for (int64_t pf_idx = 0; pf_idx < prefetch_stride; ++pf_idx) {
      const uint8_t* prefetch_addr = input + input_stride * indices[pf_idx];
      for (int64_t offset = 0; offset < input_stride;
           offset += CACHE_LINE_SIZE) {
        do_prefetch(prefetch_addr + offset, 0, 0);
      }
    }
  }

  const int64_t scale_bias_size = 2 * sizeof(float16);
  const int64_t scale_bias_offset = scale_bias_last ? block_size : 0;
  const int64_t input_offset = scale_bias_last ? 0 : scale_bias_size;

  uint64_t iters = ((uint64_t)std::max<int64_t>(block_size, 0)) / 8ull;

  uint64_t block_size_mod = block_size % 8;

  svbool_t fullRowPred =
      svwhilelt_b32_u64(0, FBGEMM_VECTOR_WIDTH / sizeof(float));
  svbool_t lastPredA = svwhilelt_b32_u64(0, block_size_mod);
  svbool_t lastPredB = svwhilelt_b32_u64(4, block_size_mod);
  svbool_t lastPredC = svwhilelt_b16_u64(0, block_size_mod);

  if constexpr (NoBag) {
    for (int64_t m = 0; m < output_size; ++m) {
      const IndexType idx = indices[m];

      if (idx < 0 || idx >= data_size) {
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;
      if constexpr (isOutput8bit) {
        memcpy(out, input_row_base, sizeof(uint8_t) * input_stride);
      } else {
        svfloat32_t scale;
        svfloat32_t bias;
        const float* scale_bias_addr =
            reinterpret_cast<const float*>(input_row_base + scale_bias_offset);
        scale = svdup_n_f32(scale_bias_addr[0]);
        if (scale_bias_last) {
          bias = svdup_n_f32(scale_bias_addr[1]);
        } else {
          bias = svcvtlt_f32_f16_x(fullRowPred, svreinterpret_f16_f32(scale));
          scale = svcvt_f32_f16_x(fullRowPred, svreinterpret_f16_f32(scale));
        }
        if (weights) {
          svfloat32_t weight = svdup_n_f32(weights[m]);
          scale = svmul_f32_x(fullRowPred, scale, weight);
          bias = svmul_f32_x(fullRowPred, bias, weight);
        }

        sve_fma_round<false>(
            input_row_base + input_offset,
            reinterpret_cast<float32x4x2_t*>(out),
            iters,
            scale,
            bias,
            fullRowPred,
            lastPredA,
            lastPredB);

        out += output_stride;
      }
    } // m
    return true;
  } // no_bag

  std::array<float, LOCAL_STORAGE_SIZE> local_storage;
  std::unique_ptr<float[]> heap_storage;
  float32x4x2_t* buf;

  if constexpr (!std::is_same<OutType, float>::value) {
    if (static_cast<size_t>(block_size) <= LOCAL_STORAGE_SIZE) {
      buf = reinterpret_cast<float32x4x2_t*>(local_storage.data());
    } else {
      heap_storage.reset(new float[block_size]);
      buf = reinterpret_cast<float32x4x2_t*>(heap_storage.get());
    }
  }

  int64_t current = 0;
  const float* weights_addr = weights;
  int64_t outputSize = output_size;
  for (; outputSize > 0;
       ++offsets_or_lengths, --outputSize, out += output_stride) {
    uint8_t len = use_offsets ? offsets_or_lengths[1] - offsets_or_lengths[0]
                              : offsets_or_lengths[0];
    int64_t end = current + len;
    if (end > index_size) {
      return false;
    }

    if (!normalize_by_lengths) {
      len = 0;
    }

    if constexpr (std::is_same<OutType, float>::value) {
      buf = reinterpret_cast<float32x4x2_t*>(out);
    }

    if (is_weight_positional) {
      weights_addr = weights;
    }
    bool oneIterationDone = false;
    for (; !oneIterationDone && current < end; ++current, ++weights_addr) {
      IndexType idx = indices[current];

      if constexpr (EnablePrefetching) {
        IndexType prefetch_idx =
            indices[std::min(current + prefetch_stride, index_size - 1)];
        const uint8_t* prefetch_addr = input + input_stride * prefetch_idx;
        for (int64_t offset = 0; offset < input_stride;
             offset += CACHE_LINE_SIZE) {
          do_prefetch(prefetch_addr + offset, 1);
        }
      }

      if (idx < 0 || idx >= data_size) {
        if (!scale_bias_last && idx == -1) {
          // When scale_bias_last == false, assume this is for table batched
          // embedding (TBE) that can get -1 for pruned rows.
          continue;
        }
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;

      svfloat32_t scale;
      svfloat32_t bias;
      const float* scale_bias_addr =
          reinterpret_cast<const float*>(input_row_base + scale_bias_offset);
      scale = svdup_n_f32(scale_bias_addr[0]);
      if (scale_bias_last) {
        bias = svdup_n_f32(scale_bias_addr[1]);
      } else {
        bias = svcvtlt_f32_f16_x(fullRowPred, svreinterpret_f16_f32(scale));
        scale = svcvt_f32_f16_x(fullRowPred, svreinterpret_f16_f32(scale));
      }

      if (weights != nullptr) {
        float weight = *weights_addr;
        svfloat32_t weight_v = svdup_n_f32(weight);
        scale = svmul_f32_x(fullRowPred, scale, weight_v);
        bias = svmul_f32_x(fullRowPred, bias, weight_v);
      }

      const uint8_t* input_row = input_row_base + input_offset;
      sve_fma_round<false>(
          input_row,
          buf,
          iters,
          scale,
          bias,
          fullRowPred,
          lastPredA,
          lastPredB);

      oneIterationDone = true;
    }

    for (; current < end; ++current, ++weights_addr) {
      IndexType idx = indices[current];

      if constexpr (EnablePrefetching) {
        IndexType prefetch_idx =
            indices[std::min(current + prefetch_stride, index_size - 1)];
        const uint8_t* prefetch_addr = input + input_stride * prefetch_idx;
        for (int64_t offset = 0; offset < input_stride;
             offset += CACHE_LINE_SIZE) {
          do_prefetch(prefetch_addr + offset, 1);
        }
      }

      if (idx < 0 || idx >= data_size) {
        if (!scale_bias_last && idx == -1) {
          // When scale_bias_last == false, assume this is for table batched
          // embedding (TBE) that can get -1 for pruned rows.
          continue;
        }
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;

      svfloat32_t scale;
      svfloat32_t bias;
      const float* scale_bias_addr =
          reinterpret_cast<const float*>(input_row_base + scale_bias_offset);
      scale = svdup_n_f32(scale_bias_addr[0]);
      if (scale_bias_last) {
        bias = svdup_n_f32(scale_bias_addr[1]);
      } else {
        bias = svcvtlt_f32_f16_x(fullRowPred, svreinterpret_f16_f32(scale));
        scale = svcvt_f32_f16_x(fullRowPred, svreinterpret_f16_f32(scale));
      }

      if (weights != nullptr) {
        float weight = *weights_addr;
        svfloat32_t weight_v = svdup_n_f32(weight);
        scale = svmul_f32_x(fullRowPred, scale, weight_v);
        bias = svmul_f32_x(fullRowPred, bias, weight_v);
      }

      const uint8_t* input_row = input_row_base + input_offset;
      sve_fma_round<true>(
          input_row,
          buf,
          iters,
          scale,
          bias,
          fullRowPred,
          lastPredA,
          lastPredB);
    }
    if (oneIterationDone) {
      if (len) {
        float32x4_t len_v = vdupq_n_f32((float)len);
        float32x4_t normalize_scale = vdupq_n_f32(1.f);
        normalize_scale = vdivq_f32(normalize_scale, len_v);
        fill_output_sve<OutType, true>(
            out,
            buf,
            iters,
            is_bf16_out,
            lastPredA,
            lastPredB,
            lastPredC,
            normalize_scale);
      } else {
        fill_output_sve<OutType, false>(
            out,
            buf,
            iters,
            is_bf16_out,
            lastPredA,
            lastPredB,
            lastPredC,
            vdupq_n_f32(1.f));
      }
    } else {
      memset(out, 0, sizeof(OutType) * block_size);
    }
  }
  return current == index_size;
}

} // namespace internal
} // namespace fbgemm

#endif // #if HAVE_SVE
