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
#include "./EmbeddingStatsTracker.h"
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

template <typename OutType>
static inline EmbeddingStatsTracker::DataType get_output_type(
    const bool is_bf16_out) {
  if constexpr (std::is_same_v<OutType, float>) {
    return EmbeddingStatsTracker::DataType::FP32;

  } else if constexpr (std::is_same_v<OutType, uint16_t>) {
    if (is_bf16_out) {
      return EmbeddingStatsTracker::DataType::BF16;
    }
  }

  return EmbeddingStatsTracker::DataType::FP16;
}

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
  if constexpr (std::is_same_v<OutType, float>) {
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
  } else if constexpr (std::is_same_v<OutType, uint16_t>) {
    if (is_bf16_out) {
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
      float16x4x2_t* outPtr = reinterpret_cast<float16x4x2_t*>(out);

      for (; srcPtr < endPtr;) {
        float32x4_t row_0 = srcPtr->val[0];
        float32x4_t row_1 = srcPtr->val[1];
        srcPtr += 1;

        if constexpr (WithScale) {
          row_0 = vmulq_f32(row_0, scale);
          row_1 = vmulq_f32(row_1, scale);
        }

        float16x4_t converted_row_0 = vcvt_f16_f32(row_0);
        float16x4_t converted_row_1 = vcvt_f16_f32(row_1);

        outPtr->val[0] = converted_row_0;
        outPtr->val[1] = converted_row_1;
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

template <bool FuseWithOutput, typename OutType>
static inline void sve_fma_round(
    const uint8_t* input_row,
    OutType* out,
    uint64_t iters,
    svfloat32_t scale,
    svfloat32_t bias,
    svbool_t fullRowPred,
    svbool_t lastPredA,
    svbool_t lastPredB,
    svbool_t lastPredC,
    const bool is_bf16_out) {
  // If we read from out, they must be float32
  static_assert(!FuseWithOutput || std::is_same_v<OutType, float>);

  float32x4x2_t* buf = reinterpret_cast<float32x4x2_t*>(out);
  float16x4x2_t* outFp16 = reinterpret_cast<float16x4x2_t*>(out);
  uint16_t* outBf16 = reinterpret_cast<uint16_t*>(out);

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

    if constexpr (std::is_same_v<OutType, float>) {
      buf->val[0] = svget_neonq(in_v_0_f);
      buf->val[1] = svget_neonq(in_v_1_f);

      buf += 1;
    } else if constexpr (std::is_same_v<OutType, uint16_t>) {
      if (is_bf16_out) {
        auto svrow_0 = svreinterpret_u32_u16(
            svrshrnb_n_u32(svreinterpret_u32_f32(in_v_0_f), 16));
        auto svrow_1 = svreinterpret_u32_u16(
            svrshrnb_n_u32(svreinterpret_u32_f32(in_v_1_f), 16));

        svst1h_u32(svptrue_b8(), outBf16, svrow_0);
        svst1h_u32(svptrue_b8(), outBf16 + 4, svrow_1);

        outBf16 += 8;
      } else {
        float16x4_t converted_row_0 = vcvt_f16_f32(svget_neonq(in_v_0_f));
        float16x4_t converted_row_1 = vcvt_f16_f32(svget_neonq(in_v_1_f));

        outFp16->val[0] = converted_row_0;
        outFp16->val[1] = converted_row_1;
        outFp16 += 1;
      }
    }
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

    if constexpr (std::is_same_v<OutType, float>) {
      svst1_f32(lastPredA, bufPtr, in_v_0_f);
      svst1_f32(lastPredB, bufPtr + 4, in_v_1_f);
    } else if constexpr (std::is_same_v<OutType, uint16_t>) {
      if (is_bf16_out) {
        auto trailing_row_0_u32 = svreinterpret_u32_u16(
            svrshrnb_n_u32(svreinterpret_u32_f32(in_v_0_f), 16));
        auto trailing_row_1_u32 = svreinterpret_u32_u16(
            svrshrnb_n_u32(svreinterpret_u32_f32(in_v_1_f), 16));

        svst1h_u32(lastPredA, outBf16, trailing_row_0_u32);
        svst1h_u32(lastPredB, outBf16 + 4, trailing_row_1_u32);
      } else {
        float16x4_t converted_trailing_row_0 =
            vcvt_f16_f32(svget_neonq(in_v_0_f));
        float16x8_t combined_trailing_rows =
            vcvt_high_f16_f32(converted_trailing_row_0, svget_neonq(in_v_1_f));

        svst1_f16(
            lastPredC,
            reinterpret_cast<float16_t*>(outFp16),
            svset_neonq_f16(svundef_f16(), combined_trailing_rows));
      }
    }
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
  constexpr bool isOutput8bit = std::is_same_v<OutType, uint8_t>;
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

  constexpr int64_t scale_bias_size = 2 * sizeof(float16);
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

        sve_fma_round<false, OutType>(
            input_row_base + input_offset,
            out,
            iters,
            scale,
            bias,
            fullRowPred,
            lastPredA,
            lastPredB,
            lastPredC,
            is_bf16_out);
      }
      out += output_stride;
    } // m
    // Track every forward pass in the no_bag case
    EmbeddingStatsTracker::getInstance().recordPattern(
        data_size,
        block_size,
        EmbeddingStatsTracker::DataType::INT8,
        isOutput8bit ? EmbeddingStatsTracker::DataType::INT8
                     : get_output_type<OutType>(is_bf16_out),
        output_size,
        1);
    return true;
  } // no_bag

  std::array<float, LOCAL_STORAGE_SIZE> local_storage;
  std::unique_ptr<float[]> heap_storage;
  float32x4x2_t* buf;

  if constexpr (!std::is_same_v<OutType, float>) {
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
    OffsetType len = use_offsets ? offsets_or_lengths[1] - offsets_or_lengths[0]
                                 : offsets_or_lengths[0];
    int64_t end = current + len;
    if (end > index_size) {
      return false;
    }

    // Track every forward inference with the actual bag size (len)
    EmbeddingStatsTracker::getInstance().recordPattern(
        data_size,
        block_size,
        EmbeddingStatsTracker::DataType::INT8,
        get_output_type<OutType>(is_bf16_out),
        output_size,
        len);

    if (!normalize_by_lengths) {
      len = 0;
    }

    if constexpr (std::is_same_v<OutType, float>) {
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
      sve_fma_round<false, float>(
          input_row,
          reinterpret_cast<float*>(buf),
          iters,
          scale,
          bias,
          fullRowPred,
          lastPredA,
          lastPredB,
          lastPredC,
          is_bf16_out);

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
      sve_fma_round<true, float>(
          input_row,
          reinterpret_cast<float*>(buf),
          iters,
          scale,
          bias,
          fullRowPred,
          lastPredA,
          lastPredB,
          lastPredC,
          is_bf16_out);
    }
    if (oneIterationDone) {
      if (len) {
        float32x4_t len_v = vdupq_n_f32(static_cast<float>(len));
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

// Copy fp16 accumulation buffer to fp16 output, optionally scaling.
template <bool WithScale>
static inline void fill_output_sve_fp16(
    uint16_t* out,
    const float16_t* src,
    const uint64_t iters,
    svbool_t lastPredC,
    float16x8_t scale) {
  const uint32_t vals_per_iter = 8;

  const float16_t* srcPtr = src;
  const float16_t* endPtr = srcPtr + iters * vals_per_iter;
  auto ptrOut = reinterpret_cast<float16_t*>(out);

  for (; srcPtr < endPtr; srcPtr += vals_per_iter) {
    float16x8_t row = vld1q_f16(srcPtr);
    if constexpr (WithScale) {
      row = vmulq_f16(row, scale);
    }
    vst1q_f16(ptrOut, row);
    ptrOut += vals_per_iter;
  }

  svfloat16_t trailing = svld1_f16(lastPredC, srcPtr);
  if constexpr (WithScale) {
    trailing =
        svmul_f16_x(lastPredC, trailing, svset_neonq_f16(svundef_f16(), scale));
  }
  svst1_f16(lastPredC, ptrOut, trailing);
}

template <bool FuseWithOutput>
static inline void sve_fma_round_fp16(
    const uint8_t* input_row,
    float16_t* buf,
    uint64_t iters,
    float scale_f32,
    float bias_f32,
    svbool_t fullRowPred,
    svbool_t lastPred) {
  svfloat16_t scale = svdup_n_f16((float16_t)scale_f32);
  svfloat16_t bias = svdup_n_f16((float16_t)bias_f32);

  const uint64_t vec_len_h = svcnth();
  const uint8_t* input_ptr = input_row;
  float16_t* bufPtr = buf;
  for (uint64_t i = 0; i < iters; ++i) {
    svuint16_t in_v = svld1ub_u16(fullRowPred, input_ptr);
    input_ptr += vec_len_h;

    svfloat16_t in_v_f = svcvt_f16_u16_x(fullRowPred, in_v);

    if constexpr (FuseWithOutput) {
      svfloat16_t buf_sv = svld1_f16(fullRowPred, bufPtr);
      buf_sv = svadd_f16_x(fullRowPred, buf_sv, bias);
      in_v_f = svmad_f16_m(fullRowPred, in_v_f, scale, buf_sv);
    } else {
      in_v_f = svmad_f16_m(fullRowPred, in_v_f, scale, bias);
    }

    svst1_f16(fullRowPred, bufPtr, in_v_f);
    bufPtr += vec_len_h;
  }

  {
    svuint16_t in_v = svld1ub_u16(lastPred, input_ptr);

    svfloat16_t in_v_f = svcvt_f16_u16_x(lastPred, in_v);

    if constexpr (FuseWithOutput) {
      svfloat16_t buf_sv = svld1_f16(lastPred, bufPtr);
      buf_sv = svadd_f16_x(lastPred, buf_sv, bias);
      in_v_f = svmad_f16_m(lastPred, in_v_f, scale, buf_sv);
    } else {
      in_v_f = svmad_f16_m(lastPred, in_v_f, scale, bias);
    }

    svst1_f16(lastPred, bufPtr, in_v_f);
  }
}

static inline void load_scale_bias_fp16(
    const uint8_t* row_base,
    int64_t scale_bias_offset,
    bool scale_bias_last,
    float& scale,
    float& bias) {
  if (scale_bias_last) {
    const float* sb =
        reinterpret_cast<const float*>(row_base + scale_bias_offset);
    scale = sb[0];
    bias = sb[1];
  } else {
    const float16_t* sb = reinterpret_cast<const float16_t*>(row_base);
    scale = static_cast<float>(sb[0]);
    bias = static_cast<float>(sb[1]);
  }
}

template <
    typename IndexType,
    typename OffsetType,
    typename OutType,
    bool NoBag,
    bool EnablePrefetching>
bool EmbeddingSpMDM8Bit_Sve_Fp16(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const uint8_t* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights,
    const bool normalize_by_lengths,
    OutType* out,
    const bool is_weight_positional,
    const bool use_offsets,
    const int64_t output_stride,
    const int64_t input_stride,
    const bool scale_bias_last,
    const bool /*is_bf16_out*/) {
  // This kernel is only dispatched for fp16 output (OutType == uint16_t,
  // !is_bf16_out). All paths produce fp16 directly — no fp32 widening.
  if constexpr (!std::is_same_v<OutType, uint16_t>) {
    return false;
  }

  if (data_size < 0) {
    return false;
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

  constexpr int64_t scale_bias_size = 2 * sizeof(float16);
  const int64_t scale_bias_offset = scale_bias_last ? block_size : 0;
  const int64_t input_offset = scale_bias_last ? 0 : scale_bias_size;

  const uint64_t vec_len_h = svcnth();
  uint64_t iters_h = ((uint64_t)std::max<int64_t>(block_size, 0)) / vec_len_h;
  uint64_t iters_8 = ((uint64_t)std::max<int64_t>(block_size, 0)) / 8ull;

  uint64_t block_size_mod_h = block_size % vec_len_h;
  uint64_t block_size_mod_8 = block_size % 8;

  svbool_t fullRowPred16 = svptrue_b16();
  svbool_t lastPredC = svwhilelt_b16_u64(0, block_size_mod_8);
  svbool_t lastPred16 = svwhilelt_b16_u64(0, block_size_mod_h);

  if constexpr (NoBag) {
    std::array<float16_t, LOCAL_STORAGE_SIZE> local_storage;
    std::unique_ptr<float16_t[]> heap_storage;
    float16_t* buf;

    if (static_cast<size_t>(block_size) <= LOCAL_STORAGE_SIZE) {
      buf = reinterpret_cast<float16_t*>(local_storage.data());
    } else {
      heap_storage.reset(new float16_t[block_size]);
      buf = reinterpret_cast<float16_t*>(heap_storage.get());
    }

    for (int64_t m = 0; m < output_size; ++m) {
      const IndexType idx = indices[m];

      if (idx < 0 || idx >= data_size) {
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;
      float scale_val, bias_val;
      load_scale_bias_fp16(
          input_row_base,
          scale_bias_offset,
          scale_bias_last,
          scale_val,
          bias_val);
      if (weights) {
        scale_val *= weights[m];
        bias_val *= weights[m];
      }

      sve_fma_round_fp16<false>(
          input_row_base + input_offset,
          buf,
          iters_h,
          scale_val,
          bias_val,
          fullRowPred16,
          lastPred16);

      fill_output_sve_fp16<false>(
          out, buf, iters_8, lastPredC, vdupq_n_f16(1.0f));
      out += output_stride;
    } // m
    EmbeddingStatsTracker::getInstance().recordPattern(
        data_size,
        block_size,
        EmbeddingStatsTracker::DataType::INT8,
        EmbeddingStatsTracker::DataType::FP16,
        output_size,
        1);
    return true;
  } // no_bag

  // Accumulate entirely in NEON fp16 registers and store fp16 directly —
  // no intermediate buffer, no fp32 widening needed.
  // For ITERS <= 16 (dims <= 128), all accumulators fit in registers.
  // For ITERS > 16, process in tiles of 16 chunks.
  if (iters_8 <= 32) {
    int64_t current = 0;
    const float* weights_addr = weights;
    int64_t outputSize = output_size;

    // Process one output row using ITERS NEON-width (8 fp16) accumulator
    // chunks.  ITERS == iters_8 for this invocation.
    auto process_row = [&]<uint64_t ITERS>() -> bool {
      // For ITERS <= 16: accumulators cover the full dim in registers.
      // For ITERS > 16 (e.g. dim=256): process in tiles of CHUNK=16 chunks,
      // re-iterating all embeddings per tile to keep register pressure at 16.
      constexpr uint64_t CHUNK = ITERS <= 16 ? ITERS : 16;
      constexpr uint64_t NUM_TILES = ITERS <= 16 ? 1 : (ITERS + 15) / 16;

      float16x8_t acc[CHUNK < 1 ? 1 : CHUNK];
      float16x8_t acc_b[ITERS <= 8 ? (CHUNK < 1 ? 1 : CHUNK) : 1];
      float16x8_t acc_tail = vdupq_n_f16(0.0f);
      float16x8_t acc_b_tail = vdupq_n_f16(0.0f);

      const OffsetType raw_len = use_offsets
          ? offsets_or_lengths[1] - offsets_or_lengths[0]
          : offsets_or_lengths[0];
      const int64_t end = current + raw_len;
      if (end > index_size) {
        return false;
      }
      EmbeddingStatsTracker::getInstance().recordPattern(
          data_size,
          block_size,
          EmbeddingStatsTracker::DataType::INT8,
          EmbeddingStatsTracker::DataType::FP16,
          output_size,
          raw_len);

      const int64_t len =
          normalize_by_lengths ? static_cast<int64_t>(raw_len) : 0;
      if (is_weight_positional) {
        weights_addr = weights;
      }

      bool oneIterationDone = false;

      // ---- Tiled path for ITERS > 16 (e.g. dim=256) ----
      // Process the dimension in tiles of CHUNK=16 chunks (128 elements).
      // Each tile iterates all embeddings, accumulating in 16 NEON fp16
      // registers, then optionally normalizes and writes to the output slice.
      if constexpr (ITERS > 16) {
        const int64_t current_start = current;
        const float* weights_start = weights_addr;

        const float16x8_t norm_fp16 = vdupq_n_f16(
            len > 0 ? (float16_t)(1.0f / static_cast<float>(len))
                    : (float16_t)1.0f);
        const bool do_normalize = (len > 0);

        for (uint64_t tile = 0; tile < NUM_TILES; ++tile) {
          const uint64_t tile_offset = tile * CHUNK * 8;
          // Number of full 8-element chunks in this tile
          constexpr uint64_t LAST_TILE_ITERS =
              ITERS % 16 == 0 ? 16 : (ITERS % 16);
          const uint64_t tile_iters =
              (tile == NUM_TILES - 1) ? LAST_TILE_ITERS : CHUNK;

          for (uint64_t j = 0; j < tile_iters; ++j) {
            acc[j] = vdupq_n_f16(0.0f);
          }
          acc_tail = vdupq_n_f16(0.0f);

          // Reset to start of embedding list for each tile
          // (last tile leaves current/weights_addr advanced past end)
          int64_t tile_current =
              (tile == NUM_TILES - 1) ? current : current_start;
          const float* tile_weights = weights_start;

          for (; tile_current < end; ++tile_current) {
            const IndexType idx = indices[tile_current];

            if constexpr (EnablePrefetching) {
              const IndexType pidx = indices[std::min(
                  tile_current + prefetch_stride, index_size - 1)];
              const uint8_t* paddr = input + input_stride * pidx;
              for (int64_t off = 0; off < input_stride;
                   off += CACHE_LINE_SIZE) {
                do_prefetch(paddr + off, 0, 0);
              }
            }

            if (idx < 0 || idx >= data_size) {
              if (!scale_bias_last && idx == -1) {
                if (weights != nullptr) {
                  ++tile_weights;
                }
                continue;
              }
              return false;
            }

            const uint8_t* input_row_base = input + input_stride * idx;
            float scale_val, bias_val;
            load_scale_bias_fp16(
                input_row_base,
                scale_bias_offset,
                scale_bias_last,
                scale_val,
                bias_val);
            if (weights != nullptr) {
              scale_val *= *tile_weights;
              bias_val *= *tile_weights;
              ++tile_weights;
            }

            const float16x8_t scale_neon = vdupq_n_f16((float16_t)scale_val);
            const float16x8_t bias_neon = vdupq_n_f16((float16_t)bias_val);
            const uint8_t* input_row =
                input_row_base + input_offset + tile_offset;

            for (uint64_t j = 0; j < tile_iters; ++j) {
              const float16x8_t in_f16 =
                  vcvtq_f16_u16(vmovl_u8(vld1_u8(input_row + j * 8)));
              acc[j] =
                  vaddq_f16(acc[j], vfmaq_f16(bias_neon, in_f16, scale_neon));
            }
            // Tail only on last tile if block_size % 8 != 0
            if (tile == NUM_TILES - 1 && block_size_mod_8 > 0) {
              const svfloat16_t sc = svset_neonq_f16(svundef_f16(), scale_neon);
              const svfloat16_t bs = svset_neonq_f16(svundef_f16(), bias_neon);
              svuint16_t in_v =
                  svld1ub_u16(lastPredC, input_row + tile_iters * 8);
              svfloat16_t in_sv = svmad_f16_m(
                  lastPredC, svcvt_f16_u16_x(lastPredC, in_v), sc, bs);
              acc_tail = svget_neonq(svadd_f16_m(
                  lastPredC, svset_neonq_f16(svundef_f16(), acc_tail), in_sv));
            }
            oneIterationDone = true;
          }

          // On last tile, advance current/weights_addr past end
          if (tile == NUM_TILES - 1) {
            current = tile_current;
            weights_addr = tile_weights;
          }

          // Store fp16 accumulators directly to output slice
          if (oneIterationDone) {
            float16_t* outPtr = reinterpret_cast<float16_t*>(out) + tile_offset;
            for (uint64_t j = 0; j < tile_iters; ++j) {
              float16x8_t val = acc[j];
              if (do_normalize) {
                val = vmulq_f16(val, norm_fp16);
              }
              vst1q_f16(outPtr, val);
              outPtr += 8;
            }
            if (tile == NUM_TILES - 1 && block_size_mod_8 > 0) {
              float16x8_t tail = acc_tail;
              if (do_normalize) {
                tail = vmulq_f16(tail, norm_fp16);
              }
              svst1_f16(
                  lastPredC, outPtr, svset_neonq_f16(svundef_f16(), tail));
            }
          }
        } // tile loop

        if (!oneIterationDone) {
          memset(out, 0, sizeof(OutType) * block_size);
        }
        return true;
      } // ITERS > 16 tiled path

      // ---- Non-tiled path for ITERS <= 16 ----
      // Initialize accumulators
      for (uint64_t j = 0; j < CHUNK; ++j) {
        acc[j] = vdupq_n_f16(0.0f);
        if constexpr (ITERS <= 8) {
          acc_b[j] = vdupq_n_f16(0.0f);
        }
      }

      // Phase 1: paired loop — process two embeddings per iteration.
      // Only used for ITERS <= 8: 2-way needs 2*ITERS accumulator registers,
      // which fits within 32 Q-regs for ITERS <= 8 (plus temps).
      // For ITERS > 8 (dim=128) Phase 1 is skipped entirely; all embeddings
      // are handled by Phase 2 below.
      if constexpr (ITERS <= 8) {
        while (current + 1 < end) {
          const IndexType idx_a = indices[current];
          const IndexType idx_b = indices[current + 1];

          if (idx_a < 0 || idx_b < 0) {
            break;
          }
          if (idx_a >= data_size || idx_b >= data_size) {
            return false;
          }

          if constexpr (EnablePrefetching) {
            auto prefetch_row = [&](int64_t cur) {
              const IndexType pidx =
                  indices[std::min(cur + prefetch_stride, index_size - 1)];
              const uint8_t* paddr = input + input_stride * pidx;
              for (int64_t off = 0; off < input_stride;
                   off += CACHE_LINE_SIZE) {
                do_prefetch(paddr + off, 0, 0);
              }
            };
            prefetch_row(current);
            prefetch_row(current + 1);
          }

          const uint8_t* row_base_a = input + input_stride * idx_a;
          float scale_a, bias_a;
          load_scale_bias_fp16(
              row_base_a, scale_bias_offset, scale_bias_last, scale_a, bias_a);
          if (weights != nullptr) {
            scale_a *= weights_addr[0];
            bias_a *= weights_addr[0];
          }
          const float16x8_t scale_a_neon = vdupq_n_f16((float16_t)scale_a);
          const float16x8_t bias_a_neon = vdupq_n_f16((float16_t)bias_a);
          const uint8_t* row_a = row_base_a + input_offset;

          const uint8_t* row_base_b = input + input_stride * idx_b;
          float scale_b, bias_b;
          load_scale_bias_fp16(
              row_base_b, scale_bias_offset, scale_bias_last, scale_b, bias_b);
          if (weights != nullptr) {
            scale_b *= weights_addr[1];
            bias_b *= weights_addr[1];
          }
          const float16x8_t scale_b_neon = vdupq_n_f16((float16_t)scale_b);
          const float16x8_t bias_b_neon = vdupq_n_f16((float16_t)bias_b);
          const uint8_t* row_b = row_base_b + input_offset;

          for (uint64_t j = 0; j < ITERS; ++j) {
            const float16x8_t in_a =
                vcvtq_f16_u16(vmovl_u8(vld1_u8(row_a + j * 8)));
            const float16x8_t in_b =
                vcvtq_f16_u16(vmovl_u8(vld1_u8(row_b + j * 8)));
            acc[j] =
                vaddq_f16(acc[j], vfmaq_f16(bias_a_neon, in_a, scale_a_neon));
            acc_b[j] =
                vaddq_f16(acc_b[j], vfmaq_f16(bias_b_neon, in_b, scale_b_neon));
          }
          if (block_size_mod_8 > 0) {
            const svfloat16_t sc_a =
                svset_neonq_f16(svundef_f16(), scale_a_neon);
            const svfloat16_t bs_a =
                svset_neonq_f16(svundef_f16(), bias_a_neon);
            svuint16_t in_v_a = svld1ub_u16(lastPredC, row_a + ITERS * 8);
            svfloat16_t in_sv_a = svmad_f16_m(
                lastPredC, svcvt_f16_u16_x(lastPredC, in_v_a), sc_a, bs_a);
            acc_tail = svget_neonq(svadd_f16_m(
                lastPredC, svset_neonq_f16(svundef_f16(), acc_tail), in_sv_a));

            const svfloat16_t sc_b =
                svset_neonq_f16(svundef_f16(), scale_b_neon);
            const svfloat16_t bs_b =
                svset_neonq_f16(svundef_f16(), bias_b_neon);
            svuint16_t in_v_b = svld1ub_u16(lastPredC, row_b + ITERS * 8);
            svfloat16_t in_sv_b = svmad_f16_m(
                lastPredC, svcvt_f16_u16_x(lastPredC, in_v_b), sc_b, bs_b);
            acc_b_tail = svget_neonq(svadd_f16_m(
                lastPredC,
                svset_neonq_f16(svundef_f16(), acc_b_tail),
                in_sv_b));
          }

          current += 2;
          if (weights != nullptr) {
            weights_addr += 2;
          }
          oneIterationDone = true;
        }
      } // if constexpr (ITERS <= 8)

      // Phase 2: scalar cleanup — handles the last odd embedding and any
      // embedding after a -1 index that caused Phase 1 to break early.
      for (; current < end; ++current) {
        IndexType idx = indices[current];

        if constexpr (EnablePrefetching) {
          IndexType prefetch_idx =
              indices[std::min(current + prefetch_stride, index_size - 1)];
          const uint8_t* prefetch_addr = input + input_stride * prefetch_idx;
          for (int64_t offset = 0; offset < input_stride;
               offset += CACHE_LINE_SIZE) {
            do_prefetch(prefetch_addr + offset, 0, 0);
          }
        }

        if (idx < 0 || idx >= data_size) {
          if (!scale_bias_last && idx == -1) {
            if (weights != nullptr) {
              ++weights_addr;
            }
            continue;
          }
          return false;
        }

        const uint8_t* input_row_base = input + input_stride * idx;
        float scale_val, bias_val;
        load_scale_bias_fp16(
            input_row_base,
            scale_bias_offset,
            scale_bias_last,
            scale_val,
            bias_val);
        if (weights != nullptr) {
          scale_val *= *weights_addr;
          bias_val *= *weights_addr;
          ++weights_addr;
        }

        const float16x8_t scale_neon = vdupq_n_f16((float16_t)scale_val);
        const float16x8_t bias_neon = vdupq_n_f16((float16_t)bias_val);
        const uint8_t* input_row = input_row_base + input_offset;

        for (uint64_t j = 0; j < ITERS; ++j) {
          const float16x8_t in_f16 =
              vcvtq_f16_u16(vmovl_u8(vld1_u8(input_row + j * 8)));
          acc[j] = vaddq_f16(acc[j], vfmaq_f16(bias_neon, in_f16, scale_neon));
        }
        if (block_size_mod_8 > 0) {
          const svfloat16_t sc = svset_neonq_f16(svundef_f16(), scale_neon);
          const svfloat16_t bs = svset_neonq_f16(svundef_f16(), bias_neon);
          svuint16_t in_v = svld1ub_u16(lastPredC, input_row + ITERS * 8);
          svfloat16_t in_sv =
              svmad_f16_m(lastPredC, svcvt_f16_u16_x(lastPredC, in_v), sc, bs);
          acc_tail = svget_neonq(svadd_f16_m(
              lastPredC, svset_neonq_f16(svundef_f16(), acc_tail), in_sv));
        }
        oneIterationDone = true;
      }

      // Store fp16 accumulators directly — no fp32 widening needed
      if (oneIterationDone) {
        const float16x8_t norm_fp16 = vdupq_n_f16(
            len > 0 ? (float16_t)(1.0f / static_cast<float>(len))
                    : (float16_t)1.0f);
        const bool do_normalize = (len > 0);

        float16_t* outPtr = reinterpret_cast<float16_t*>(out);
        for (uint64_t j = 0; j < ITERS; ++j) {
          float16x8_t combined;
          if constexpr (ITERS <= 8) {
            combined = vaddq_f16(acc[j], acc_b[j]);
          } else {
            combined = acc[j];
          }
          if (do_normalize) {
            combined = vmulq_f16(combined, norm_fp16);
          }
          vst1q_f16(outPtr, combined);
          outPtr += 8;
        }
        if (block_size_mod_8 > 0) {
          float16x8_t tail;
          if constexpr (ITERS <= 8) {
            tail = vaddq_f16(acc_tail, acc_b_tail);
          } else {
            tail = acc_tail;
          }
          if (do_normalize) {
            tail = vmulq_f16(tail, norm_fp16);
          }
          svst1_f16(lastPredC, outPtr, svset_neonq_f16(svundef_f16(), tail));
        }
      } else {
        memset(out, 0, sizeof(OutType) * block_size);
      }
      return true;
    };

    for (; outputSize > 0;
         ++offsets_or_lengths, --outputSize, out += output_stride) {
      bool ok;
      switch (iters_8) {
        case 0:
          ok = process_row.template operator()<0>();
          break;
        case 1:
          ok = process_row.template operator()<1>();
          break;
        case 2:
          ok = process_row.template operator()<2>();
          break;
        case 3:
          ok = process_row.template operator()<3>();
          break;
        case 4:
          ok = process_row.template operator()<4>();
          break;
        case 5:
          ok = process_row.template operator()<5>();
          break;
        case 6:
          ok = process_row.template operator()<6>();
          break;
        case 7:
          ok = process_row.template operator()<7>();
          break;
        case 8:
          ok = process_row.template operator()<8>();
          break;
        case 9:
          ok = process_row.template operator()<9>();
          break;
        case 10:
          ok = process_row.template operator()<10>();
          break;
        case 11:
          ok = process_row.template operator()<11>();
          break;
        case 12:
          ok = process_row.template operator()<12>();
          break;
        case 13:
          ok = process_row.template operator()<13>();
          break;
        case 14:
          ok = process_row.template operator()<14>();
          break;
        case 15:
          ok = process_row.template operator()<15>();
          break;
        case 16:
          ok = process_row.template operator()<16>();
          break;
        case 17:
          ok = process_row.template operator()<17>();
          break;
        case 18:
          ok = process_row.template operator()<18>();
          break;
        case 19:
          ok = process_row.template operator()<19>();
          break;
        case 20:
          ok = process_row.template operator()<20>();
          break;
        case 21:
          ok = process_row.template operator()<21>();
          break;
        case 22:
          ok = process_row.template operator()<22>();
          break;
        case 23:
          ok = process_row.template operator()<23>();
          break;
        case 24:
          ok = process_row.template operator()<24>();
          break;
        case 25:
          ok = process_row.template operator()<25>();
          break;
        case 26:
          ok = process_row.template operator()<26>();
          break;
        case 27:
          ok = process_row.template operator()<27>();
          break;
        case 28:
          ok = process_row.template operator()<28>();
          break;
        case 29:
          ok = process_row.template operator()<29>();
          break;
        case 30:
          ok = process_row.template operator()<30>();
          break;
        case 31:
          ok = process_row.template operator()<31>();
          break;
        case 32:
          ok = process_row.template operator()<32>();
          break;
        default:
          ok = false;
          break;
      }
      if (!ok) {
        return false;
      }
    }
    return current == index_size;
  } // register-based fp16 path

  // Fallback: buf-based fp16 accumulation for large dims (iters_8 > 32).
  std::array<float16_t, LOCAL_STORAGE_SIZE> local_storage;
  std::unique_ptr<float16_t[]> heap_storage;
  float16_t* buf;

  if (static_cast<size_t>(block_size) <= LOCAL_STORAGE_SIZE) {
    buf = local_storage.data();
  } else {
    heap_storage.reset(new float16_t[block_size]);
    buf = heap_storage.get();
  }

  int64_t current = 0;
  const float* weights_addr = weights;
  int64_t outputSize = output_size;
  for (; outputSize > 0;
       ++offsets_or_lengths, --outputSize, out += output_stride) {
    OffsetType len = use_offsets ? offsets_or_lengths[1] - offsets_or_lengths[0]
                                 : offsets_or_lengths[0];
    int64_t end = current + len;
    if (end > index_size) {
      return false;
    }

    EmbeddingStatsTracker::getInstance().recordPattern(
        data_size,
        block_size,
        EmbeddingStatsTracker::DataType::INT8,
        EmbeddingStatsTracker::DataType::FP16,
        output_size,
        len);

    if (!normalize_by_lengths) {
      len = 0;
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
          do_prefetch(prefetch_addr + offset, 0, 0);
        }
      }

      if (idx < 0 || idx >= data_size) {
        if (!scale_bias_last && idx == -1) {
          continue;
        }
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;
      float scale_val, bias_val;
      load_scale_bias_fp16(
          input_row_base,
          scale_bias_offset,
          scale_bias_last,
          scale_val,
          bias_val);
      if (weights != nullptr) {
        scale_val *= *weights_addr;
        bias_val *= *weights_addr;
      }

      const uint8_t* input_row = input_row_base + input_offset;
      sve_fma_round_fp16<false>(
          input_row,
          buf,
          iters_h,
          scale_val,
          bias_val,
          fullRowPred16,
          lastPred16);

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
          do_prefetch(prefetch_addr + offset, 0, 0);
        }
      }

      if (idx < 0 || idx >= data_size) {
        if (!scale_bias_last && idx == -1) {
          continue;
        }
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;
      float scale_val, bias_val;
      load_scale_bias_fp16(
          input_row_base,
          scale_bias_offset,
          scale_bias_last,
          scale_val,
          bias_val);
      if (weights != nullptr) {
        scale_val *= *weights_addr;
        bias_val *= *weights_addr;
      }

      const uint8_t* input_row = input_row_base + input_offset;
      sve_fma_round_fp16<true>(
          input_row,
          buf,
          iters_h,
          scale_val,
          bias_val,
          fullRowPred16,
          lastPred16);
    }
    if (oneIterationDone) {
      if (len) {
        float16x8_t normalize_scale =
            vdupq_n_f16((float16_t)(1.0f / static_cast<float>(len)));
        fill_output_sve_fp16<true>(
            out, buf, iters_8, lastPredC, normalize_scale);
      } else {
        fill_output_sve_fp16<false>(
            out, buf, iters_8, lastPredC, vdupq_n_f16(1.0f));
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
