/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "./EmbeddingSpMDMAutovec.h" // @manual
#include <bit>
#include "./EmbeddingStatsTracker.h"
#include "./RefImplementations.h" // @manual
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FloatConversion.h"

#if defined(__clang__) && HAVE_SVE
#include <arm_neon.h>
#include <arm_sve.h>

#include <arm_neon_sve_bridge.h>
#endif

#include <cmath>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

/// @defgroup tbe-cpu-autovec TBE CPU Autovectorization (FP8/16/32)

#ifdef _WIN32
#define do_prefetch(...)
#else
#define do_prefetch(...) __builtin_prefetch(__VA_ARGS__)
#endif

#ifdef __clang__
// https://github.com/llvm/llvm-project/issues/114891 / T206675074
// Work around LLVM loop vectorization not produce optimal code when
// `block_size` is not a multiple of the natural vector size.
#ifdef __AVX512F__
#define FBGEMM_VECTOR_WIDTH 16
#elif __AVX2__
#define FBGEMM_VECTOR_WIDTH 8
#elif __SSE__
#define FBGEMM_VECTOR_WIDTH 4
#endif
#endif // #ifdef __clang__

namespace fbgemm {

static constexpr size_t LOCAL_STORAGE_SIZE = 512;

template <typename OutType>
static inline void fill_output(
    OutType* out,
    const float* src,
    const int64_t block_size,
    const bool is_bf16_out) {
  if constexpr (std::is_same_v<OutType, float>) {
    for (int j = 0; j < block_size; ++j) {
      out[j] = src[j];
    }
  } else if constexpr (std::is_same_v<OutType, uint16_t>) {
    if (is_bf16_out) {
      for (int j = 0; j < block_size; ++j) {
        out[j] = cpu_float2bfloat16(src[j]);
      }
    } else {
      for (int j = 0; j < block_size; ++j) {
        out[j] = cpu_float2half(src[j]);
      }
    }
  }
}

// Provide alterantive to `memset` that can be inlined. Contrary to a `memset`
// call with the default aarch64 calling convention this will only clobber a
// single vector register.
static inline void fillZero(float* ptr, int64_t count) {
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
    memset(ptr, 0, sizeof(float) * count);
#if defined(__clang__) && HAVE_SVE
  }
#endif
}

template <typename OutType>
static constexpr EmbeddingStatsTracker::DataType get_output_type(
    const bool is_bf16_out) {
  if constexpr (std::is_same_v<OutType, float>) {
    return EmbeddingStatsTracker::DataType::FP32;
  } else if (std::is_same_v<OutType, uint16_t> && is_bf16_out) {
    return EmbeddingStatsTracker::DataType::BF16;
  } else {
    return EmbeddingStatsTracker::DataType::FP16;
  }
}

template <typename IndexType, typename OffsetType, typename OutType>
static bool ALWAYS_INLINE EmbeddingSpMDM8Bit_autovec(
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
    const bool no_bag,
    const bool is_bf16_out) {
  constexpr bool isOutput8bit = std::is_same_v<OutType, uint8_t>;
  if (data_size < 0) {
    return false;
  }
  constexpr int64_t CACHE_LINE_SIZE = 64;
  constexpr int64_t MAX_INITIAL_PREFETCH_ROWS = 16;
  const int64_t prefetch_stride =
      std::min(MAX_INITIAL_PREFETCH_ROWS, index_size);
  for (int64_t pf_idx = 0; pf_idx < prefetch_stride; ++pf_idx) {
    const uint8_t* prefetch_addr = input + input_stride * indices[pf_idx];
    for (int64_t offset = 0; offset < input_stride; offset += CACHE_LINE_SIZE) {
      do_prefetch(prefetch_addr + offset, 0, 0);
    }
  }

  const int64_t scale_bias_size = 2 * sizeof(float16);
  const int64_t scale_bias_offset = scale_bias_last ? block_size : 0;
  const int64_t input_offset = scale_bias_last ? 0 : scale_bias_size;

  std::array<float, LOCAL_STORAGE_SIZE> local_storage;
  std::unique_ptr<float[]> heap_storage;
  float* buf = nullptr;
  if (static_cast<size_t>(block_size) <= LOCAL_STORAGE_SIZE) {
    buf = local_storage.data();
  } else {
    heap_storage.reset(new float[block_size]);
    buf = heap_storage.get();
  }

  if (no_bag) {
    const int64_t copy_width = std::min(output_stride, input_stride);
    for (int64_t m = 0; m < output_size; ++m) {
      const IndexType idx = indices[m];

      if (idx < 0 || idx >= data_size) {
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;
      if constexpr (isOutput8bit) {
        memcpy(out, input_row_base, sizeof(uint8_t) * copy_width);
      } else {
        float scale = NAN;
        float bias = NAN;
        const uint8_t* scale_bias_addr = input_row_base + scale_bias_offset;
        if (scale_bias_last) {
          scale = *(reinterpret_cast<const float*>(scale_bias_addr));
          bias = *(
              reinterpret_cast<const float*>(scale_bias_addr + sizeof(float)));
        } else {
          scale = cpu_half2float(
              *reinterpret_cast<const float16*>(scale_bias_addr));
          bias = cpu_half2float(*reinterpret_cast<const float16*>(
              scale_bias_addr + sizeof(float16)));
        }
        if (weights) {
          float weight = weights[m];
          scale *= weight;
          bias *= weight;
        }

        const uint8_t* input_row = input_row_base + input_offset;
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % FBGEMM_VECTOR_WIDTH); ++j) {
          uint8_t value = input_row[j];
          buf[j] = std::fma(scale, static_cast<float>(value), bias);
        }
#endif
        for (; j < block_size; ++j) {
          uint8_t value = input_row[j];
          buf[j] = std::fma(scale, static_cast<float>(value), bias);
        }
        fill_output(out, buf, block_size, is_bf16_out);
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

  int64_t current = 0;
  for (int64_t m = 0; m < output_size; ++m) {
    fillZero(buf, block_size);
    const OffsetType len = use_offsets
        ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
        : offsets_or_lengths[m];
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

    const float* weights_addr = weights != nullptr
        ? (is_weight_positional ? weights : weights + current)
        : nullptr;
    for (; current < end; ++current) {
      IndexType idx = indices[current];

      IndexType prefetch_idx =
          indices[std::min(current + prefetch_stride, index_size - 1)];
      const uint8_t* prefetch_addr = input + input_stride * prefetch_idx;
      for (int64_t offset = 0; offset < input_stride;
           offset += CACHE_LINE_SIZE) {
        do_prefetch(prefetch_addr + offset, 1);
      }
      if (idx < 0 || idx >= data_size) {
        // Skip pruned rows.
        if (idx == -1 && !scale_bias_last) {
          if (weights_addr != nullptr) {
            weights_addr++;
          }
          continue;
        }
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;

      const uint8_t* scale_bias_addr = input_row_base + scale_bias_offset;
      float scale = NAN;
      float bias = NAN;
      if (scale_bias_last) {
        scale = *(reinterpret_cast<const float*>(scale_bias_addr));
        bias =
            *(reinterpret_cast<const float*>(scale_bias_addr + sizeof(float)));
      } else {
        scale =
            cpu_half2float(*reinterpret_cast<const float16*>(scale_bias_addr));
        bias = cpu_half2float(*reinterpret_cast<const float16*>(
            scale_bias_addr + sizeof(float16)));
      }

      if (weights != nullptr) {
        float weight = *weights_addr++;
        scale *= weight;
        bias *= weight;
      }

      const uint8_t* input_row = input_row_base + input_offset;
      int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
      for (; j < block_size - (block_size % FBGEMM_VECTOR_WIDTH); ++j) {
        uint8_t value = input_row[j];
        buf[j] = std::fma(scale, (float)value, buf[j] + bias);
      }
#endif
      for (; j < block_size; ++j) {
        uint8_t value = input_row[j];
        buf[j] = std::fma(scale, (float)value, buf[j] + bias);
      }
    }
    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
      for (int64_t j = 0; j < block_size; ++j) {
        buf[j] *= scale;
      }
    }
    fill_output(out, buf, block_size, is_bf16_out);
    out += output_stride;
  }
  return current == index_size;
}

template <typename IndexType, typename OffsetType, typename OutType>
static bool ALWAYS_INLINE EmbeddingSpMDMNBit_autovec(
    const int input_bit_rate,
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
    const bool is_bf16_out,
    const bool no_bag,
    int output_bit_rate) {
  nbit_embedding_sanity_check<OutType>(input_bit_rate, output_bit_rate, no_bag);
  if (data_size < 0) {
    return false;
  }

  // more prefetch
  // TODO: in the future we should adjust max_prefetch_bytes based on CPU cache
  // size
  constexpr int64_t max_prefetch_bytes = 4096;
  // 16 is manually tuned for Neoverse-V2 for best performance
  constexpr int64_t max_initial_prefetch_rows = 16;
  constexpr int64_t CACHE_LINE_SIZE = 64;
  const int64_t rows_to_prefetch =
      std::min(max_initial_prefetch_rows, max_prefetch_bytes / input_stride);
  const int64_t prefetch_stride = std::min(rows_to_prefetch, index_size);
  const int num_elem_per_byte = 8 / input_bit_rate;
  const int64_t scale_bias_offset =
      scale_bias_last ? div_up(block_size, num_elem_per_byte) : 0;
  const size_t scale_bias_size = 2 * sizeof(float16);
  const int64_t input_row_offset = scale_bias_last ? 0 : scale_bias_size;
  // The following prefetch loop is written in this way for better performance.
  // My understanding is that manually separating the case of input_stride being
  // greater or not greater than cache line size will make the branch predictor
  // work better. Same for line 113-126.
  for (int64_t pf_idx = 0; pf_idx < prefetch_stride; ++pf_idx) {
    const uint8_t* prefetch_addr = input + input_stride * indices[pf_idx];
    for (int64_t offset = 0; offset < input_stride; offset += CACHE_LINE_SIZE) {
      do_prefetch(prefetch_addr + offset, 0, 0);
    }
  }

  if (no_bag) {
    // We currently only support int4 to int4 for sequential TBE in this nbit
    // kernel. Note that assert() will be ignored in release mode, so we check
    // here to double check and also avoid "unused variable" warning
    if (input_bit_rate != 4 || output_bit_rate != 4) {
      WARN_ONCE("no_bag is only supported for int4 to int4");
      return false;
    }

    for (int64_t i = 0; i < output_size; ++i) {
      const auto idx = indices[i];
      if (idx < 0 || idx > data_size) {
        return false;
      }
      const uint8_t* input_row = input + input_stride * idx;
      memcpy(out, input_row, sizeof(uint8_t) * input_stride);
      out += input_stride;
    }

    // Track every forward pass with the actual bag size (len)
    EmbeddingStatsTracker::getInstance().recordPattern(
        data_size,
        block_size,
        EmbeddingStatsTracker::DataType::INT4,
        EmbeddingStatsTracker::DataType::INT4,
        output_size,
        1);
    return true;
  }

  int64_t current = 0;
  const int64_t rounded_block_size = round_up(block_size, num_elem_per_byte);

  std::array<float, LOCAL_STORAGE_SIZE> local_storage;
  std::unique_ptr<float[]> heap_storage;
  float* buf = nullptr;
  if (static_cast<size_t>(rounded_block_size) <= LOCAL_STORAGE_SIZE) {
    buf = local_storage.data();
  } else {
    heap_storage.reset(new float[rounded_block_size]);
    buf = heap_storage.get();
  }

  for (int64_t m = 0; m < output_size; ++m) {
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    int64_t end = current + len;
    if (end > index_size) {
      return false;
    }

    // Track every forward pass with the actual bag size (len)
    EmbeddingStatsTracker::getInstance().recordPattern(
        data_size,
        block_size,
        input_bit_rate == 4 ? EmbeddingStatsTracker::DataType::INT4
                            : EmbeddingStatsTracker::DataType::INT2,
        get_output_type<OutType>(is_bf16_out),
        output_size,
        len);
    fillZero(buf, rounded_block_size);

    const float* weights_addr = weights != nullptr
        ? (is_weight_positional ? weights : weights + current)
        : nullptr;
    for (; current < end; ++current) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        // Skip pruned rows.
        if (idx == -1 && !scale_bias_last) {
          if (weights_addr != nullptr) {
            weights_addr++;
          }
          continue;
        }
        return false;
      }
      int64_t prefetch_idx =
          indices[std::min(current + prefetch_stride, index_size - 1)];

      const uint8_t* input_row_base = input + input_stride * idx;
      const uint8_t* scale_bias_addr = input_row_base + scale_bias_offset;
      const uint8_t* input_row = input_row_base + input_row_offset;

      float16 scale16 = *reinterpret_cast<const float16*>(scale_bias_addr);
      float16 bias16 =
          *reinterpret_cast<const float16*>(scale_bias_addr + sizeof(float16));
      static_assert(sizeof(scale16) + sizeof(bias16) == scale_bias_size);

      float scale = cpu_half2float(scale16);
      float bias = cpu_half2float(bias16);
      if (weights != nullptr) {
        float weight = *weights_addr++;
        scale *= weight;
        bias *= weight;
      }

      if (input_bit_rate == 4) {
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % (FBGEMM_VECTOR_WIDTH * 2));
             j += 2) {
          uint8_t tmp = *input_row++;
          float quantized1 = float(tmp & 0xf);
          float quantized2 = float(tmp >> 4);
          buf[j] = std::fma(scale, quantized1, buf[j] + bias);
          buf[j + 1] = std::fma(scale, quantized2, buf[j + 1] + bias);
        }
#endif
        for (; j < block_size; j += 2) {
          uint8_t tmp = *input_row++;
          float quantized1 = float(tmp & 0xf);
          float quantized2 = float(tmp >> 4);
          buf[j] = std::fma(scale, quantized1, buf[j] + bias);
          buf[j + 1] = std::fma(scale, quantized2, buf[j + 1] + bias);
        }
      } else if (input_bit_rate == 2) {
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % (FBGEMM_VECTOR_WIDTH * 4));
             j += 4) {
          uint8_t tmp = *input_row++;
          float quantized1 = float(tmp & 0x3);
          float quantized2 = float((tmp & 0xC) >> 2);
          float quantized3 = float((tmp & 0x30) >> 4);
          float quantized4 = float(tmp >> 6);
          buf[j] = std::fma(scale, quantized1, buf[j] + bias);
          buf[j + 1] = std::fma(scale, quantized2, buf[j + 1] + bias);
          buf[j + 2] = std::fma(scale, quantized3, buf[j + 2] + bias);
          buf[j + 3] = std::fma(scale, quantized4, buf[j + 3] + bias);
        }
#endif
        for (; j < block_size; j += 4) {
          uint8_t tmp = *input_row++;
          float quantized1 = float(tmp & 0x3);
          float quantized2 = float((tmp & 0xC) >> 2);
          float quantized3 = float((tmp & 0x30) >> 4);
          float quantized4 = float(tmp >> 6);
          buf[j] = std::fma(scale, quantized1, buf[j] + bias);
          buf[j + 1] = std::fma(scale, quantized2, buf[j + 1] + bias);
          buf[j + 2] = std::fma(scale, quantized3, buf[j + 2] + bias);
          buf[j + 3] = std::fma(scale, quantized4, buf[j + 3] + bias);
        }
      }

      const uint8_t* prefetch_addr = input + input_stride * prefetch_idx;
      for (int64_t offset = 0; offset < input_stride;
           offset += CACHE_LINE_SIZE) {
        do_prefetch(prefetch_addr + offset, 0, 0);
      }
    }

    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
      for (int64_t j = 0; j < block_size; ++j) {
        buf[j] *= scale;
      }
    }
    fill_output(out, buf, block_size, is_bf16_out);
    out += output_stride;
  }
  return current == index_size;
}

template <typename IndexType, typename OffsetType>
static bool ALWAYS_INLINE EmbeddingSpMDMNBitRowWiseSparse_autovec(
    const int bit_rate,
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t uncompressed_data_size,
    const uint8_t* input,
    const IndexType* indices,
    const int32_t* compressed_indices_table,
    const OffsetType* offsets_or_lengths,
    const float* weights,
    const bool normalize_by_lengths,
    float* out,
    const bool is_weight_positional,
    const bool use_offsets) {
  if (uncompressed_data_size < 0) {
    return false;
  }

  // block_size is the number of elements and fused_block_size is the size in
  // bytes of an entire row, including scale and bias.
  const int num_elem_per_byte = 8 / bit_rate;
  const int64_t scale_bias_size = 2 * sizeof(float16);
  const uint64_t scale_bias_offset = div_up(block_size, num_elem_per_byte);
  const int64_t fused_block_size = scale_bias_offset + scale_bias_size;

  int64_t current = 0;
  float* buf = out;
  for (int64_t m = 0; m < output_size; ++m) {
    const OffsetType len = use_offsets
        ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
        : offsets_or_lengths[m];
    const int64_t end = current + len;
    if (end > index_size) {
      return false;
    }

    memset(buf, 0, sizeof(float) * block_size);

    const float* weights_addr = weights != nullptr
        ? (is_weight_positional ? weights : weights + current)
        : nullptr;
    for (; current < end; ++current) {
      int64_t uncompressed_idx = indices[current];
      if (uncompressed_idx < 0 || uncompressed_idx >= uncompressed_data_size) {
        return false;
      }
      int64_t idx = compressed_indices_table[uncompressed_idx];
      // Skip pruned rows.
      if (idx == -1) {
        if (weights_addr != nullptr) {
          weights_addr++;
        }
        continue;
      }

      const uint8_t* input_row_base = input + fused_block_size * idx;
      const uint8_t* scale_bias_addr = input_row_base + scale_bias_offset;

      float scale =
          cpu_half2float(*reinterpret_cast<const float16*>(scale_bias_addr));
      float bias = cpu_half2float(
          *reinterpret_cast<const float16*>(scale_bias_addr + sizeof(float16)));

      if (weights != nullptr) {
        float weight = *weights_addr++;
        scale *= weight;
        bias *= weight;
      }

      const uint8_t* input_row = input_row_base;
      if (bit_rate == 4) {
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % (FBGEMM_VECTOR_WIDTH * 2));
             j += 2) {
          uint8_t tmp = *input_row++;
          float quantized1 = float(tmp & 0xf);
          float quantized2 = float(tmp >> 4);
          buf[j] = std::fma(scale, quantized1, buf[j] + bias);
          buf[j + 1] = std::fma(scale, quantized2, buf[j + 1] + bias);
        }
#endif
        for (; j < block_size - (block_size % 2); j += 2) {
          uint8_t tmp = *input_row++;
          float quantized1 = float(tmp & 0xf);
          float quantized2 = float(tmp >> 4);
          buf[j] = std::fma(scale, quantized1, buf[j] + bias);
          buf[j + 1] = std::fma(scale, quantized2, buf[j + 1] + bias);
        }
        for (; j < block_size; ++j) {
          uint8_t quantized = input_row_base[j / num_elem_per_byte] >>
              ((j % num_elem_per_byte) * bit_rate);
          quantized &= (1 << bit_rate) - 1;
          buf[j] = std::fma(scale, float(quantized), buf[j] + bias);
        }
      } else if (bit_rate == 2) {
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % (FBGEMM_VECTOR_WIDTH * 4));
             j += 4) {
          uint8_t tmp = *input_row++;
          float quantized1 = float(tmp & 0x3);
          float quantized2 = float((tmp & 0xC) >> 2);
          float quantized3 = float((tmp & 0x30) >> 4);
          float quantized4 = float(tmp >> 6);
          buf[j] = std::fma(scale, quantized1, buf[j] + bias);
          buf[j + 1] = std::fma(scale, quantized2, buf[j + 1] + bias);
          buf[j + 2] = std::fma(scale, quantized3, buf[j + 2] + bias);
          buf[j + 3] = std::fma(scale, quantized4, buf[j + 3] + bias);
        }
#endif
        for (; j < block_size - (block_size % 4); j += 4) {
          uint8_t tmp = *input_row++;
          float quantized1 = float(tmp & 0x3);
          float quantized2 = float((tmp & 0xC) >> 2);
          float quantized3 = float((tmp & 0x30) >> 4);
          float quantized4 = float(tmp >> 6);
          buf[j] = std::fma(scale, quantized1, buf[j] + bias);
          buf[j + 1] = std::fma(scale, quantized2, buf[j + 1] + bias);
          buf[j + 2] = std::fma(scale, quantized3, buf[j + 2] + bias);
          buf[j + 3] = std::fma(scale, quantized4, buf[j + 3] + bias);
        }
        for (; j < block_size; ++j) {
          uint8_t quantized = input_row_base[j / num_elem_per_byte] >>
              ((j % num_elem_per_byte) * bit_rate);
          quantized &= (1 << bit_rate) - 1;
          buf[j] = std::fma(scale, float(quantized), buf[j] + bias);
        }
      }
    }
    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
      for (int j = 0; j < block_size; ++j) {
        buf[j] *= scale;
      }
    }
    buf += block_size;
  }
  return current == index_size;
}

/// @ingroup tbe-cpu-autovec
///
/// Autovectorized version of method `EmbeddingSpMDM_ref` for FP32 weight type.
///
/// @tparam InType input data type (`uint8_t` is used)
/// @tparam IndexType index data type (`int64_t` is used)
/// @tparam OffsetType offset data type (`int32_t` is used)
/// @tparam OutType output data type (`float` is used)
///
/// @param block_size Number of elements in a block (`int64_t`)
/// @param output_size Number of elements in output (`int64_t`)
/// @param index_size Number of elements in index (`int64_t`)
/// @param data_size Number of elements in data (`int64_t`)
/// @param input Address of input (`InType*`)
/// @param indices Address of index (`IndexType*`)
/// @param offsets_or_lengths Address of offset (`OffsetType*`)
/// @param weights Weights of sum; optional, can be null for non-weighted sum
/// (`float*`)
/// @param normalize_by_lengths Whether or not to normalize by lengths (`bool`)
/// @param out Address of output (`OutType*`)
/// @param is_weight_positional If `true`, weight is positional; set to `false`
/// for FP32 autovec implementation (`bool`)
/// @param use_offsets If `true`, will use offsets instead of lengths; set to
/// `true` for FP32 autovec implementation (`bool`)
/// @param output_stride If -1, output_stride is same as block_size; set to -1
/// for FP32 autovec implementation (`int64_t`)
/// @param input_stride If -1, input_stride is same as block_size; set to -1
/// for FP32 autovec implementation (`int64_t`)
/// @param scale_bias_last If `true`, scale and bias appear at end of each row;
/// set to `true` for FP32 autovec implementation (`bool`)
/// @param no_bag If `true`, no embedding bag; set to `false` for FP32 autovec
/// implementation (`bool`)
/// @param is_bf16_out If `true`, output is `BFLOAT16` type; set to `false` for
/// FP32 autovec implementation (`bool`)
/// @param is_bf16_in If `true`, input is `BFLOAT16` type; set to `false` for
/// FP32 autovec implementation (`bool`)
template <
    typename InType,
    typename IndexType,
    typename OffsetType,
    typename OutType>
static bool ALWAYS_INLINE EmbeddingSpMDM_autovec(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    OutType* out,
    const bool is_weight_positional,
    const bool use_offsets,
    const int64_t output_stride,
    const int64_t input_stride,
    const bool no_bag,
    const bool is_bf16_out,
    const bool is_bf16_in) {
  if (data_size < 0) {
    return false;
  }

  std::array<float, LOCAL_STORAGE_SIZE> local_storage;
  std::unique_ptr<float[]> heap_storage;
  float* buf = nullptr;
  if (static_cast<size_t>(block_size) <= LOCAL_STORAGE_SIZE) {
    buf = local_storage.data();
  } else {
    heap_storage.reset(new float[block_size]);
    buf = heap_storage.get();
  }

  if (no_bag) {
    for (int m = 0; m < output_size; ++m) {
      int64_t idx = indices[m];
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      if (weights != nullptr) {
        float weight = weights[m];
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % FBGEMM_VECTOR_WIDTH); ++j) {
          const InType* inptr = input + input_stride * idx + j;
          buf[j] = weight * convert_to_float_ref(*inptr, is_bf16_in);
        }
#endif
        for (; j < block_size; ++j) {
          const InType* inptr = input + input_stride * idx + j;
          buf[j] = weight * convert_to_float_ref(*inptr, is_bf16_in);
        }
      } else {
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % FBGEMM_VECTOR_WIDTH); ++j) {
          const InType* inptr = input + input_stride * idx + j;
          buf[j] = convert_to_float_ref(*inptr, is_bf16_in);
        }
#endif
        for (; j < block_size; ++j) {
          const InType* inptr = input + input_stride * idx + j;
          buf[j] = convert_to_float_ref(*inptr, is_bf16_in);
        }
      }
      fill_output(out, buf, block_size, is_bf16_out);
      out += output_stride;
    } // m

    EmbeddingStatsTracker::getInstance().recordPattern(
        data_size,
        block_size,
        is_bf16_in ? EmbeddingStatsTracker::DataType::BF16
                   : EmbeddingStatsTracker::DataType::FP32,
        get_output_type<OutType>(is_bf16_out),
        output_size,
        1);

    return true;
  } // no_bag

  // more prefetch
  // TODO: in the future we should adjust max_prefetch_bytes based on CPU
  // cache size
  constexpr int64_t max_prefetch_bytes = 4096;
  // 16 is manually tuned for Neoverse-V2 for best performance
  constexpr int64_t max_initial_prefetch_rows = 8;
  constexpr int64_t CACHE_LINE_SIZE = 64;
  const int64_t rows_to_prefetch =
      std::min(max_initial_prefetch_rows, max_prefetch_bytes / input_stride);
  const int64_t prefetch_stride = std::min(rows_to_prefetch, index_size);
  // The following prefetch loop is written in this way for better
  // performance. My understanding is that manually separating the case of
  // input_stride being greater or not greater than cache line size will make
  // the branch predictor work better. Same for line 113-126.
  for (int pf_idx = 0; pf_idx < prefetch_stride; ++pf_idx) {
    const uint8_t* prefetch_addr = reinterpret_cast<const uint8_t*>(
        input + input_stride * indices[pf_idx]);
    for (int64_t offset = 0; offset < input_stride; offset += CACHE_LINE_SIZE) {
      do_prefetch(prefetch_addr + offset, 0, 0);
    }
  }

  // Reference implementation of FP32 SLS
  int64_t current = 0;
  for (int m = 0; m < output_size; ++m) {
    fillZero(buf, block_size);
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }
    // Track every inference for actual bag size (len)
    EmbeddingStatsTracker::getInstance().recordPattern(
        data_size,
        block_size,
        is_bf16_in ? EmbeddingStatsTracker::DataType::BF16
                   : EmbeddingStatsTracker::DataType::FP32,
        get_output_type<OutType>(is_bf16_out),
        output_size,
        len);

    for (int i = 0; i < len; ++i) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      int64_t prefetch_idx =
          indices[std::min(current + prefetch_stride, index_size - 1)];

      do_prefetch(
          reinterpret_cast<const char*>(input + input_stride * prefetch_idx),
          0,
          0);
      if (input_stride > CACHE_LINE_SIZE) {
        for (int64_t offset = CACHE_LINE_SIZE; offset < input_stride;
             offset += CACHE_LINE_SIZE) {
          do_prefetch(
              reinterpret_cast<const char*>(
                  input + input_stride * prefetch_idx + offset),
              0,
              0);
        }
      }

      float w = 1.f;
      if (weights) {
        w = weights[is_weight_positional ? i : current];
      }

      const InType* input_row = input + input_stride * idx;
      int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
      for (; j < block_size - (block_size % FBGEMM_VECTOR_WIDTH); ++j) {
        InType value = *input_row++;
        buf[j] = std::fma(w, convert_to_float_ref(value, is_bf16_in), buf[j]);
      }
#endif
      for (; j < block_size; ++j) {
        InType value = *input_row++;
        buf[j] = std::fma(w, convert_to_float_ref(value, is_bf16_in), buf[j]);
      }

      ++current;
    }
    if (normalize_by_lengths && len) {
      float scale = 1.f / len;

      for (int64_t j = 0; j < block_size; ++j) {
        buf[j] *= scale;
      }
    }

    fill_output(out, buf, block_size, is_bf16_out);
    out += output_stride;
  }
  return current == index_size;
}

template <typename InType, typename IndexType, typename OffsetType>
static bool ALWAYS_INLINE EmbeddingSpMDMRowWiseSparse_autovec(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t uncompressed_data_size,
    // const int64_t compressed_data_size,
    const InType* input,
    const IndexType* indices,
    const int32_t* compressed_indices_table,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    const bool normalize_by_lengths,
    float* out,
    const bool is_weight_positional,
    const bool use_offsets) {
  constexpr bool is8bit = std::is_same_v<InType, uint8_t>;

  if constexpr (is8bit) {
    // block_size is the number of elements and fused_block_size is the size in
    // bytes of an entire row, including scale and bias.
    const auto scale_bias_offset = 2 * sizeof(float);
    const int64_t fused_block_size = block_size + scale_bias_offset;
    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      fillZero(out, block_size);
      int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                            : offsets_or_lengths[m];
      int64_t end = current + len;
      if (end > index_size) {
        return false;
      }
      EmbeddingStatsTracker::getInstance().recordPattern(
          uncompressed_data_size,
          block_size,
          EmbeddingStatsTracker::DataType::SPARSE_INT8,
          EmbeddingStatsTracker::DataType::FP32,
          output_size,
          len);
      const float* weights_addr = weights != nullptr
          ? (is_weight_positional ? weights : weights + current)
          : nullptr;
      for (; current < end; ++current) {
        IndexType uncompressed_idx = indices[current];
        if (uncompressed_idx < 0 ||
            uncompressed_idx >= uncompressed_data_size) {
          return false;
        }
        IndexType idx = compressed_indices_table[uncompressed_idx];
        // Skip pruned rows.
        if (idx == -1) {
          if (weights_addr != nullptr) {
            weights_addr++;
          }
          continue;
        }
        // if (idx < 0 || idx >= compressed_data_size) {
        //   return false;
        // }

        const uint8_t* scale_bias_addr = reinterpret_cast<const uint8_t*>(
            input + fused_block_size * idx + block_size);

        float scale = *(reinterpret_cast<const float*>(scale_bias_addr));
        float bias =
            *(reinterpret_cast<const float*>(scale_bias_addr + sizeof(float)));
        if (weights != nullptr) {
          float weight = *weights_addr++;
          scale *= weight;
          bias *= weight;
        }

        const InType* input_row = input + fused_block_size * idx;
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % FBGEMM_VECTOR_WIDTH); ++j) {
          InType value = *input_row++;
          out[j] = std::fma(scale, value, out[j] + bias);
        }
#endif
        for (; j < block_size; ++j) {
          InType value = *input_row++;
          out[j] = std::fma(scale, value, out[j] + bias);
        }
      }
      if (normalize_by_lengths && len) {
        float scale = 1.f / len;
        for (int j = 0; j < block_size; ++j) {
          out[j] *= scale;
        }
      }
      out += block_size;
    }
    return current == index_size;
  } else {
    // Reference implementation of FP32 SLS

    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      fillZero(out, block_size);
      int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                            : offsets_or_lengths[m];
      int64_t end = current + len;
      if (end > index_size) {
        return false;
      }

      EmbeddingStatsTracker::getInstance().recordPattern(
          uncompressed_data_size,
          block_size,
          EmbeddingStatsTracker::DataType::SPARSE_FP32,
          EmbeddingStatsTracker::DataType::FP32,
          output_size,
          len);

      const float* weights_addr = weights != nullptr
          ? (is_weight_positional ? weights : weights + current)
          : nullptr;
      for (; current < end; ++current) {
        IndexType uncompressed_idx = indices[current];
        if (uncompressed_idx < 0 ||
            uncompressed_idx >= uncompressed_data_size) {
          return false;
        }
        IndexType idx = compressed_indices_table[uncompressed_idx];
        // Skip pruned rows.
        if (idx == -1) {
          if (weights_addr != nullptr) {
            weights_addr++;
          }
          continue;
        }

        float weight = 1.f;
        if (weights != nullptr) {
          weight = *weights_addr++;
        }

        const InType* input_row = input + block_size * idx;
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % FBGEMM_VECTOR_WIDTH); ++j) {
          const InType* inptr = input_row++;
          out[j] = std::fma(
              weight,
              std::is_same_v<InType, float16> ? cpu_half2float(*inptr) : *inptr,
              out[j]);
        }
#endif
        for (; j < block_size; ++j) {
          const InType* inptr = input_row++;
          out[j] = std::fma(
              weight,
              std::is_same_v<InType, float16> ? cpu_half2float(*inptr) : *inptr,
              out[j]);
        }
      }
      if (normalize_by_lengths && len) {
        float scale = 1.f / len;
        for (int j = 0; j < block_size; ++j) {
          out[j] *= scale;
        }
      }
      out += block_size;
    }
    return current == index_size;
  }
}

namespace {
void Float8ToFloat_ref_batch(
    const uint8_t* input,
    float* output,
    int count,
    int exponent_bits,
    int exponent_bias) {
  for (int i = 0; i < count; ++i) {
    uint32_t val_out = 0, sign = 0, multiplier = 0;
    uint8_t inp = input[i];

    sign = (inp & 0x80) << 24;
    val_out = (inp & 0x7F) << (24 - (8 - exponent_bits));

    multiplier = (127 + (127 - exponent_bias)) << 23; // 2^(127-bias)
    float val_out_f = std::bit_cast<float>(val_out) *
        std::bit_cast<float>(multiplier); // val_out * multiplier
    val_out = std::bit_cast<uint32_t>(val_out_f) | sign;
    output[i] = std::bit_cast<float>(val_out);
  }
}
} // namespace

/// @ingroup tbe-cpu-autovec
///
/// Autovectorized version of method `EmbeddingSpMDM_ref` for FP8 weight type.
///
/// @tparam InType input data type (`uint8_t` is used)
/// @tparam IndexType index data type (`int64_t` is used)
/// @tparam OffsetType offset data type (`int32_t` is used)
/// @tparam OutType output data type (`float` is used)
///
/// @param block_size Number of elements in a block (`int64_t`)
/// @param output_size Number of elements in output (`int64_t`)
/// @param index_size Number of elements in index (`int64_t`)
/// @param data_size Number of elements in data (`int64_t`)
/// @param input Address of input (`InType*`)
/// @param indices Address of index (`IndexType*`)
/// @param offsets_or_lengths Address of offset (`OffsetType*`)
/// @param weights Weights of sum; optional, can be null for non-weighted sum
/// (`float*`)
/// @param normalize_by_lengths Whether or not to normalize by lengths (`bool`)
/// @param out Address of output (`OutType*`)
/// @param is_weight_positional If `true`, weight is positional; set to `false`
/// for FP8 autovec implementation (`bool`)
/// @param use_offsets If `true`, will use offsets instead of lengths; set to
/// `true` for FP8 autovec implementation (`bool`)
/// @param output_stride If -1, output_stride is same as block_size; set to -1
/// for FP8 autovec implementation (`int64_t`)
/// @param exponent_bits Bits to use in exponent
/// @param exponent_bias Bias to use in exponent
/// @param is_bf16_out If `true`, output is `BFLOAT16` type; set to `false` for
/// FP8 autovec implementation (`bool`)
template <typename IndexType, typename OffsetType, typename OutType>
static bool ALWAYS_INLINE EmbeddingSpMDMFP8_autovec(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const uint8_t* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights,
    bool normalize_by_lengths,
    OutType* out,
    const bool is_weight_positional,
    const bool use_offsets,
    const int64_t output_stride,
    const int64_t input_stride,
    const int exponent_bits,
    const int exponent_bias,
    const bool is_bf16_out) {
  if (data_size < 0) {
    return false;
  }

  std::array<float, LOCAL_STORAGE_SIZE> local_storage;
  std::unique_ptr<float[]> heap_storage;
  float* buf = nullptr;
  if (static_cast<size_t>(block_size) <= LOCAL_STORAGE_SIZE) {
    buf = local_storage.data();
  } else {
    heap_storage.reset(new float[block_size]);
    buf = heap_storage.get();
  }

  // more prefetch: prefetch up to 16 rows from the embedding table. Increasing
  // prefetching helps reduce backend stall and therefore enable vectorization
  // reach better of its potential. 16 is tuned for Neoverse-V2.

  // more prefetch
  // TODO: in the future we should adjust max_prefetch_bytes based on CPU cache
  // size
  constexpr int64_t max_prefetch_bytes = 4096;
  // 16 is manually tuned for Neoverse-V2 for best performance
  constexpr int64_t max_initial_prefetch_rows = 16;
  constexpr int64_t CACHE_LINE_SIZE = 64;
  const int64_t rows_to_prefetch =
      std::min(max_initial_prefetch_rows, max_prefetch_bytes / input_stride);
  const int64_t prefetch_stride = std::min(rows_to_prefetch, index_size);
  // The following prefetch loop is written in this way for better performance.
  // My understanding is that manually separating the case of input_stride being
  // greater or not greater than cache line size will make the branch predictor
  // work better. Same for line 113-126.
  for (int pf_idx = 0; pf_idx < prefetch_stride; ++pf_idx) {
    const uint8_t* prefetch_addr = input + input_stride * indices[pf_idx];
    for (int64_t offset = 0; offset < input_stride; offset += CACHE_LINE_SIZE) {
      do_prefetch(prefetch_addr + offset, 0, 0);
    }
  }

  // Reference implementation of FP8 SLS. The algorithm is similar to FP32 SLS
  // except for the FP8->FP32 conversion after reading the embedding weight.
  int64_t current = 0;

  for (int m = 0; m < output_size; ++m) {
    fillZero(buf, block_size);
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    int64_t end = current + len;
    if (end > index_size) {
      return false;
    }

    EmbeddingStatsTracker::getInstance().recordPattern(
        data_size,
        block_size,
        EmbeddingStatsTracker::DataType::FP8,
        get_output_type<OutType>(is_bf16_out),
        output_size,
        len);

    // Adjust these as necessary to reflect actual batch size
    const int batch_size = block_size; // Assuming the entire block is
                                       // processed at once; adjust if needed

    // Temporary buffer to hold the converted floats
    std::unique_ptr<float[]> converted_inputs(new float[batch_size]);

    const float* weights_addr = weights != nullptr
        ? (is_weight_positional ? weights : weights + current)
        : nullptr;
    for (; current < end; ++current) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      int64_t prefetch_idx =
          indices[std::min(current + prefetch_stride, index_size - 1)];

      do_prefetch(
          reinterpret_cast<const char*>(input + input_stride * prefetch_idx),
          0,
          0);
      if (input_stride > CACHE_LINE_SIZE) {
        for (int64_t offset = CACHE_LINE_SIZE; offset < input_stride;
             offset += CACHE_LINE_SIZE) {
          do_prefetch(
              reinterpret_cast<const char*>(
                  input + input_stride * prefetch_idx + offset),
              0,
              0);
        }
      }

      float w = 1.f;
      if (weights != nullptr) {
        w = *weights_addr++;
      }
      // check if each loop interation depends on one another
      //  if not, approach it with parellel,
      //  the code is iterating thru a dimisonals of a embedding vectory

      // Perform the batch conversion
      Float8ToFloat_ref_batch(
          input + input_stride * idx,
          converted_inputs.get(),
          batch_size,
          exponent_bits,
          exponent_bias);

      // Now accumulate the results using vectorized operations if possible
      const float* input_row = converted_inputs.get();
      int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
      for (; j < block_size - (block_size % FBGEMM_VECTOR_WIDTH); ++j) {
        float value = *input_row++;
        buf[j] = std::fma(w, value, buf[j]);
      }
#endif
      for (; j < block_size; ++j) {
        float value = *input_row++;
        buf[j] = std::fma(w, value, buf[j]);
      }
    }
    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
      for (int j = 0; j < block_size; ++j) {
        buf[j] *= scale;
      }
    }

    fill_output(out, buf, block_size, is_bf16_out);
    out += output_stride;
  }
  return current == index_size;
}

template <typename InType>
static constexpr int64_t stride_SpMDMWithStrides(
    int64_t block_size,
    bool scale_bias_last) {
  if constexpr (std::is_same_v<InType, uint8_t>) {
    const size_t scale_bias_offset =
        2 * (scale_bias_last ? sizeof(float) : sizeof(uint16_t));
    return block_size + scale_bias_offset;
  }
  return block_size;
}

namespace {

// Builds the fully generic kernel: every parameter stays a runtime value. This
// is the catch-all that always exists (it replaces the old all-`var`
// SPECIALIZE(...) expansion).
template <
    typename InType,
    typename IndexType,
    typename OffsetType,
    typename OutType>
typename EmbeddingSpMDMKernelSignature<InType, IndexType, OffsetType, OutType>::
    Type
    make_spmdm_generic(
        int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        bool is_weight_positional,
        bool use_offsets,
        int64_t output_stride,
        int64_t input_stride,
        bool scale_bias_last,
        bool no_bag,
        bool is_bf16_out,
        bool is_bf16_in) {
  return [=](int64_t output_size,
             int64_t index_size,
             int64_t data_size,
             const InType* input,
             const IndexType* indices,
             const OffsetType* offsets_or_lengths,
             const float* weights,
             OutType* out) {
    if (has_weight) {
      __builtin_assume(weights != nullptr);
    } else {
      weights = nullptr;
    }
    if constexpr (std::is_same_v<InType, uint8_t>) {
      assert(!is_bf16_in);
      return EmbeddingSpMDM8Bit_autovec(
          block_size,
          output_size,
          index_size,
          data_size,
          reinterpret_cast<const uint8_t*>(input),
          indices,
          offsets_or_lengths,
          weights,
          normalize_by_lengths,
          out,
          is_weight_positional,
          use_offsets,
          output_stride,
          input_stride,
          scale_bias_last,
          no_bag,
          is_bf16_out);
    } else {
      return EmbeddingSpMDM_autovec(
          block_size,
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          normalize_by_lengths,
          out,
          is_weight_positional,
          use_offsets,
          output_stride,
          input_stride,
          no_bag,
          is_bf16_out,
          is_bf16_in);
    }
  };
}

// Builds a kernel with the block size baked in as a compile-time constant
// (NTTP), together with the booleans that every FBGEMM_MORE_SPECIALIZATION
// variant pins. With a known block size the inner loops have a fixed trip
// count, so the autovectorizer can specialize them.
template <
    int64_t BlockSize,
    bool HasWeight,
    bool ScaleBiasLast,
    typename InType,
    typename IndexType,
    typename OffsetType,
    typename OutType>
typename EmbeddingSpMDMKernelSignature<InType, IndexType, OffsetType, OutType>::
    Type
    make_spmdm_fixed_block_size(
        int64_t output_stride,
        bool is_bf16_out,
        bool is_bf16_in) {
  // Pinned by every FBGEMM_MORE_SPECIALIZATION variant.
  constexpr bool kNormalizeByLengths = false;
  constexpr bool kIsWeightPositional = false;
  constexpr bool kUseOffsets = true;
  constexpr bool kNoBag = false;
  // The old macro always matched and passed the stride computed with
  // scale_bias_last == false, independent of ScaleBiasLast.
  constexpr int64_t kInputStride =
      stride_SpMDMWithStrides<InType>(BlockSize, /*scale_bias_last=*/false);
  return [=](int64_t output_size,
             int64_t index_size,
             int64_t data_size,
             const InType* input,
             const IndexType* indices,
             const OffsetType* offsets_or_lengths,
             const float* weights,
             OutType* out) {
    if constexpr (HasWeight) {
      __builtin_assume(weights != nullptr);
    } else {
      weights = nullptr;
    }
    if constexpr (std::is_same_v<InType, uint8_t>) {
      assert(!is_bf16_in);
      return EmbeddingSpMDM8Bit_autovec(
          BlockSize,
          output_size,
          index_size,
          data_size,
          reinterpret_cast<const uint8_t*>(input),
          indices,
          offsets_or_lengths,
          weights,
          kNormalizeByLengths,
          out,
          kIsWeightPositional,
          kUseOffsets,
          output_stride,
          kInputStride,
          ScaleBiasLast,
          kNoBag,
          is_bf16_out);
    } else {
      return EmbeddingSpMDM_autovec(
          BlockSize,
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          kNormalizeByLengths,
          out,
          kIsWeightPositional,
          kUseOffsets,
          output_stride,
          kInputStride,
          kNoBag,
          is_bf16_out,
          is_bf16_in);
    }
  };
}

// Folds over the candidate block sizes for one (HasWeight, ScaleBiasLast)
// combination and returns the matching block-size-specialized kernel, or an
// empty kernel if the runtime block_size / input_stride match none of them.
template <
    bool HasWeight,
    bool ScaleBiasLast,
    typename InType,
    typename IndexType,
    typename OffsetType,
    typename OutType>
typename EmbeddingSpMDMKernelSignature<InType, IndexType, OffsetType, OutType>::
    Type
    try_spmdm_fixed_block_size(
        int64_t block_size,
        int64_t input_stride,
        int64_t output_stride,
        bool is_bf16_out,
        bool is_bf16_in) {
  // Block sizes that get a compile-time-specialized kernel. Listing them as
  // data lets the fold below enumerate them instead of a macro per value.
  static constexpr std::array<int64_t, 18> kBlockSizes{
      4,   24,  32,  36,  64,  72,  96,  124,
      128, 252, 256, 320, 384, 508, 512, 576,
      768, 1024};
  using KernelType = typename EmbeddingSpMDMKernelSignature<
      InType,
      IndexType,
      OffsetType,
      OutType>::Type;
  KernelType kernel = nullptr;
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    ([&] {
      constexpr int64_t kBlockSize = kBlockSizes[Is];
      if (!kernel && block_size == kBlockSize &&
          input_stride == stride_SpMDMWithStrides<InType>(kBlockSize, false)) {
        kernel = make_spmdm_fixed_block_size<
            kBlockSize,
            HasWeight,
            ScaleBiasLast,
            InType,
            IndexType,
            OffsetType,
            OutType>(output_stride, is_bf16_out, is_bf16_in);
      }
    }(),
     ...);
  }(std::make_index_sequence<kBlockSizes.size()>{});
  return kernel;
}

} // namespace

template <
    typename InType,
    typename IndexType,
    typename OffsetType,
    typename OutType>
typename EmbeddingSpMDMKernelSignature<InType, IndexType, OffsetType, OutType>::
    Type
    GenerateEmbeddingSpMDMWithStrides_autovec(
        int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch [[maybe_unused]],
        bool is_weight_positional,
        bool use_offsets,
        int64_t output_stride,
        int64_t input_stride,
        bool scale_bias_last,
        bool no_bag,
        bool is_bf16_out,
        bool is_bf16_in) {
  if (output_stride == -1) {
    output_stride = block_size;
  }
  if (input_stride == -1) {
    input_stride = stride_SpMDMWithStrides<InType>(block_size, scale_bias_last);
  }

#ifdef FBGEMM_MORE_SPECIALIZATION
  // Block-size-specialized fast paths. Every specialized variant pins
  // normalize_by_lengths=false, is_weight_positional=false, use_offsets=true,
  // no_bag=false and varies only (has_weight, scale_bias_last) on top of the
  // block size, so dispatch on that pair and let try_spmdm_fixed_block_size()
  // fold over the candidate block sizes.
  if (!normalize_by_lengths && !is_weight_positional && use_offsets &&
      !no_bag) {
    typename EmbeddingSpMDMKernelSignature<InType, IndexType, OffsetType, OutType>::Type
        kernel = nullptr;
    if (has_weight && !scale_bias_last) {
      kernel = try_spmdm_fixed_block_size<
          true, false, InType, IndexType, OffsetType, OutType>(
          block_size, input_stride, output_stride, is_bf16_out, is_bf16_in);
    } else if (!has_weight && !scale_bias_last) {
      kernel = try_spmdm_fixed_block_size<
          false, false, InType, IndexType, OffsetType, OutType>(
          block_size, input_stride, output_stride, is_bf16_out, is_bf16_in);
    } else if (has_weight && scale_bias_last) {
      kernel = try_spmdm_fixed_block_size<
          true, true, InType, IndexType, OffsetType, OutType>(
          block_size, input_stride, output_stride, is_bf16_out, is_bf16_in);
    } else {
      kernel = try_spmdm_fixed_block_size<
          false, true, InType, IndexType, OffsetType, OutType>(
          block_size, input_stride, output_stride, is_bf16_out, is_bf16_in);
    }
    if (kernel) {
      return kernel;
    }
  }
  WARN_ONCE(
      "fbgemm warning: "
      "using non-specialized EmbeddingSpMDM_autovec (may be slow)\n"
      "    parameters: block_size: %ld has_weight: %d normalize_by_lengths: %d "
      "is_weight_positional: %d use_offsets: %d output_stride: %ld "
      "input_stride: %ld scale_bias_last: %d no_bag: %d\n",
      static_cast<long>(block_size),
      static_cast<int>(has_weight),
      static_cast<int>(normalize_by_lengths),
      static_cast<int>(is_weight_positional),
      static_cast<int>(use_offsets),
      static_cast<long>(output_stride),
      static_cast<long>(input_stride),
      static_cast<int>(scale_bias_last),
      static_cast<int>(no_bag));
#endif

  return make_spmdm_generic<InType, IndexType, OffsetType, OutType>(
      block_size,
      has_weight,
      normalize_by_lengths,
      is_weight_positional,
      use_offsets,
      output_stride,
      input_stride,
      scale_bias_last,
      no_bag,
      is_bf16_out,
      is_bf16_in);
}

static constexpr int64_t stride_SpMDMNBitWith(
    int input_bit_rate,
    int64_t block_size) {
  const int num_elem_per_byte = 8 / input_bit_rate;
  const size_t scale_bias_size = 2 * sizeof(float16);
  return div_up(block_size, num_elem_per_byte) + scale_bias_size;
}

namespace {

// Builds the NBit catch-all kernel: only input_bit_rate is baked in as a
// compile-time constant (the old catch-all always specialized on it); every
// other parameter stays a runtime value.
template <
    int InputBitRate,
    typename IndexType,
    typename OffsetType,
    typename OutType>
typename EmbeddingSpMDMKernelSignature<uint8_t, IndexType, OffsetType,
                                       OutType>::Type
make_nbit_generic(
    int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    bool is_weight_positional,
    bool use_offsets,
    int64_t output_stride,
    int64_t input_stride,
    bool scale_bias_last,
    bool is_bf16_out,
    bool no_bag,
    int output_bit_rate) {
  return [=](int64_t output_size,
             int64_t index_size,
             int64_t data_size,
             const uint8_t* input,
             const IndexType* indices,
             const OffsetType* offsets_or_lengths,
             const float* weights,
             OutType* out) {
    if (has_weight) {
      __builtin_assume(weights != nullptr);
    } else {
      weights = nullptr;
    }
    return EmbeddingSpMDMNBit_autovec(
        InputBitRate,
        block_size,
        output_size,
        index_size,
        data_size,
        input,
        indices,
        offsets_or_lengths,
        weights,
        normalize_by_lengths,
        out,
        is_weight_positional,
        use_offsets,
        output_stride,
        input_stride,
        scale_bias_last,
        is_bf16_out,
        no_bag,
        output_bit_rate);
  };
}

// Builds an NBit kernel with input_bit_rate, block size and the pinned booleans
// baked in as compile-time constants (the old FBGEMM_MORE_SPECIALIZATION
// variants, which exist for input_bit_rate == 4 only).
template <
    int InputBitRate,
    int64_t BlockSize,
    bool HasWeight,
    bool ScaleBiasLast,
    typename IndexType,
    typename OffsetType,
    typename OutType>
typename EmbeddingSpMDMKernelSignature<uint8_t, IndexType, OffsetType,
                                       OutType>::Type
make_nbit_fixed_block_size(int64_t output_stride, bool is_bf16_out) {
  // Pinned by every FBGEMM_MORE_SPECIALIZATION variant.
  constexpr bool kNormalizeByLengths = false;
  constexpr bool kIsWeightPositional = false;
  constexpr bool kUseOffsets = true;
  constexpr bool kNoBag = false;
  constexpr int kOutputBitRate = 8 * static_cast<int>(sizeof(OutType));
  constexpr int64_t kInputStride =
      stride_SpMDMNBitWith(InputBitRate, BlockSize);
  return [=](int64_t output_size,
             int64_t index_size,
             int64_t data_size,
             const uint8_t* input,
             const IndexType* indices,
             const OffsetType* offsets_or_lengths,
             const float* weights,
             OutType* out) {
    if constexpr (HasWeight) {
      __builtin_assume(weights != nullptr);
    } else {
      weights = nullptr;
    }
    return EmbeddingSpMDMNBit_autovec(
        InputBitRate,
        BlockSize,
        output_size,
        index_size,
        data_size,
        input,
        indices,
        offsets_or_lengths,
        weights,
        kNormalizeByLengths,
        out,
        kIsWeightPositional,
        kUseOffsets,
        output_stride,
        kInputStride,
        ScaleBiasLast,
        is_bf16_out,
        kNoBag,
        kOutputBitRate);
  };
}

// Folds over the candidate NBit block sizes for one (HasWeight, ScaleBiasLast)
// combination and returns the matching block-size-specialized kernel, or an
// empty kernel if nothing matches.
template <
    int InputBitRate,
    bool HasWeight,
    bool ScaleBiasLast,
    typename IndexType,
    typename OffsetType,
    typename OutType>
typename EmbeddingSpMDMKernelSignature<uint8_t, IndexType, OffsetType,
                                       OutType>::Type
try_nbit_fixed_block_size(
    int64_t block_size,
    int64_t input_stride,
    int64_t output_stride,
    int output_bit_rate,
    bool is_bf16_out) {
  static constexpr std::array<int64_t, 14> kBlockSizes{
      32, 56, 64, 96, 120, 128, 248, 256, 320, 384, 512, 576, 768, 1024};
  using KernelType = typename EmbeddingSpMDMKernelSignature<
      uint8_t,
      IndexType,
      OffsetType,
      OutType>::Type;
  KernelType kernel = nullptr;
  if (output_bit_rate != 8 * static_cast<int>(sizeof(OutType))) {
    return kernel;
  }
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    ([&] {
      constexpr int64_t kBlockSize = kBlockSizes[Is];
      if (!kernel && block_size == kBlockSize &&
          input_stride == stride_SpMDMNBitWith(InputBitRate, kBlockSize)) {
        kernel = make_nbit_fixed_block_size<
            InputBitRate,
            kBlockSize,
            HasWeight,
            ScaleBiasLast,
            IndexType,
            OffsetType,
            OutType>(output_stride, is_bf16_out);
      }
    }(),
     ...);
  }(std::make_index_sequence<kBlockSizes.size()>{});
  return kernel;
}

} // namespace

template <typename IndexType, typename OffsetType, typename OutType>
FBGEMM_API typename EmbeddingSpMDMKernelSignature<
    uint8_t,
    IndexType,
    OffsetType,
    OutType>::Type
GenerateEmbeddingSpMDMNBitWithStrides_autovec(
    int input_bit_rate,
    int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch [[maybe_unused]],
    bool is_weight_positional,
    bool use_offsets,
    int64_t output_stride,
    int64_t input_stride,
    bool scale_bias_last,
    bool is_bf16_out,
    bool no_bag,
    int output_bit_rate) {
  if (output_bit_rate == -1) {
    output_bit_rate = 8 * sizeof(OutType);
  }
  if (output_stride == -1) {
    output_stride = block_size;
  }

  if (input_stride == -1) {
    input_stride = stride_SpMDMNBitWith(input_bit_rate, block_size);
  }

#ifdef FBGEMM_MORE_SPECIALIZATION
  // Block-size-specialized fast paths, which exist for input_bit_rate == 4
  // only. Every variant pins normalize_by_lengths=false,
  // is_weight_positional=false, use_offsets=true, no_bag=false and
  // output_bit_rate=8*sizeof(OutType), and varies only (has_weight,
  // scale_bias_last) on top of the block size.
  if (input_bit_rate == 4 && !normalize_by_lengths && !is_weight_positional &&
      use_offsets && !no_bag) {
    typename EmbeddingSpMDMKernelSignature<uint8_t, IndexType, OffsetType,
                                           OutType>::Type kernel = nullptr;
    if (has_weight && !scale_bias_last) {
      kernel = try_nbit_fixed_block_size<4, true, false, IndexType, OffsetType,
                                         OutType>(
          block_size, input_stride, output_stride, output_bit_rate,
          is_bf16_out);
    } else if (!has_weight && !scale_bias_last) {
      kernel = try_nbit_fixed_block_size<4, false, false, IndexType, OffsetType,
                                         OutType>(
          block_size, input_stride, output_stride, output_bit_rate,
          is_bf16_out);
    } else if (has_weight && scale_bias_last) {
      kernel = try_nbit_fixed_block_size<4, true, true, IndexType, OffsetType,
                                         OutType>(
          block_size, input_stride, output_stride, output_bit_rate,
          is_bf16_out);
    } else {
      kernel = try_nbit_fixed_block_size<4, false, true, IndexType, OffsetType,
                                         OutType>(
          block_size, input_stride, output_stride, output_bit_rate,
          is_bf16_out);
    }
    if (kernel) {
      return kernel;
    }
  }
  WARN_ONCE(
      "fbgemm warning: "
      "using non-specialized EmbeddingSpMDMNBit_autovec (may be slow)\n"
      "    parameters: input_bit_rate: %d block_size: %ld has_weight: %d "
      "normalize_by_lengths: %d is_weight_positional: %d use_offsets: %d "
      "output_stride: %ld input_stride: %ld scale_bias_last: %d no_bag: %d "
      "output_bit_rate: %d\n",
      input_bit_rate,
      static_cast<long>(block_size),
      static_cast<int>(has_weight),
      static_cast<int>(normalize_by_lengths),
      static_cast<int>(is_weight_positional),
      static_cast<int>(use_offsets),
      static_cast<long>(output_stride),
      static_cast<long>(input_stride),
      static_cast<int>(scale_bias_last),
      static_cast<int>(no_bag),
      output_bit_rate);
#endif

  // Catch-all: specialize only on input_bit_rate (2 or 4).
  if (input_bit_rate == 2) {
    return make_nbit_generic<2, IndexType, OffsetType, OutType>(
        block_size, has_weight, normalize_by_lengths, is_weight_positional,
        use_offsets, output_stride, input_stride, scale_bias_last, is_bf16_out,
        no_bag, output_bit_rate);
  }
  if (input_bit_rate == 4) {
    return make_nbit_generic<4, IndexType, OffsetType, OutType>(
        block_size, has_weight, normalize_by_lengths, is_weight_positional,
        use_offsets, output_stride, input_stride, scale_bias_last, is_bf16_out,
        no_bag, output_bit_rate);
  }
  abort(); // should not get here
}


template <typename IndexType, typename OffsetType, typename OutType>
typename EmbeddingSpMDMKernelSignature<
    uint8_t,
    IndexType,
    OffsetType,
    OutType>::Type
GenerateEmbeddingSpMDMFP8WithStrides_autovec(
    int64_t block_size,
    bool normalize_by_lengths,
    bool is_weight_positional,
    bool use_offsets,
    int64_t output_stride,
    int64_t input_stride,
    int exponent_bits,
    int exponent_bias,
    bool is_bf16_out) {
  if (output_stride == -1) {
    output_stride = block_size;
  }
  if (input_stride == -1) {
    input_stride = block_size;
  }
  return [=](int64_t output_size,
             int64_t index_size,
             int64_t data_size,
             const uint8_t* input,
             const IndexType* indices,
             const OffsetType* offsets_or_lengths,
             const float* weights,
             OutType* out) {
    return EmbeddingSpMDMFP8_autovec(
        /*block_size=*/block_size,
        /*output_size=*/output_size,
        /*index_size=*/index_size,
        /*data_size=*/data_size,
        /*input=*/input,
        /*indices=*/indices,
        /*offsets_or_lengths=*/offsets_or_lengths,
        /*weights=*/weights,
        /*normalize_by_lengths=*/normalize_by_lengths,
        /*out=*/out,
        /*is_weight_positional=*/is_weight_positional,
        /*use_offsets=*/use_offsets,
        /*output_stride=*/output_stride,
        /*input_stride=*/input_stride,
        /*exponent_bits=*/exponent_bits,
        /*exponent_bias=*/exponent_bias,
        /*is_bf16_out=*/is_bf16_out);
  };
}

template <typename InType, typename IndexType, typename OffsetType>
typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    InType,
    IndexType,
    OffsetType>::Type
GenerateEmbeddingSpMDMRowWiseSparse_autovec(
    int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch [[maybe_unused]],
    bool is_weight_positional,
    bool use_offsets) {
  return [=](int64_t output_size,
             int64_t index_size,
             int64_t uncompressed_data_size,
             const InType* input,
             const IndexType* indices,
             const OffsetType* offsets_or_lengths,
             const float* weights,
             float* out,
             const int32_t* compressed_indices_table) {
    if (!has_weight) {
      weights = nullptr;
    }
    return EmbeddingSpMDMRowWiseSparse_autovec(
        /*block_size=*/block_size,
        /*output_size=*/output_size,
        /*index_size=*/index_size,
        /*uncompressed_data_size=*/uncompressed_data_size,
        /*input=*/input,
        /*indices=*/indices,
        /*compressed_indices_table=*/compressed_indices_table,
        /*offsets_or_lengths=*/offsets_or_lengths,
        /*weights=*/weights,
        /*normalize_by_lengths=*/normalize_by_lengths,
        /*out=*/out,
        /*is_weight_positional=*/is_weight_positional,
        /*use_offsets=*/use_offsets);
  };
}

#define INSTANTIATE_SPMDM_NBIT_WITH_STRIDES(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  template typename EmbeddingSpMDMKernelSignature<                             \
      uint8_t,                                                                 \
      INDEX_TYPE,                                                              \
      OFFSET_TYPE,                                                             \
      OUT_TYPE>::Type FBGEMM_API                                               \
  GenerateEmbeddingSpMDMNBitWithStrides_autovec<                               \
      INDEX_TYPE,                                                              \
      OFFSET_TYPE,                                                             \
      OUT_TYPE>(                                                               \
      int input_bit_rate,                                                      \
      int64_t block_size,                                                      \
      bool has_weight,                                                         \
      bool normalize_by_lengths,                                               \
      int prefetch,                                                            \
      bool is_weight_positional,                                               \
      bool use_offsets,                                                        \
      int64_t output_stride,                                                   \
      int64_t input_stride,                                                    \
      bool scale_bias_last,                                                    \
      bool is_bf16_out,                                                        \
      bool no_bag,                                                             \
      int output_bit_rate);

#define INSTANTIATE_SPMDM_FP8(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  template typename EmbeddingSpMDMKernelSignature<               \
      uint8_t,                                                   \
      INDEX_TYPE,                                                \
      OFFSET_TYPE,                                               \
      OUT_TYPE>::Type                                            \
  GenerateEmbeddingSpMDMFP8WithStrides_autovec<                  \
      INDEX_TYPE,                                                \
      OFFSET_TYPE,                                               \
      OUT_TYPE>(                                                 \
      int64_t block_size,                                        \
      bool normalize_by_lengths,                                 \
      bool is_weight_positional,                                 \
      bool use_offsets,                                          \
      int64_t output_stride,                                     \
      int64_t input_stride,                                      \
      int exponent_bits,                                         \
      int exponent_bias,                                         \
      bool is_bf16_out);

#define INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE)        \
  INSTANTIATE_SPMDM_NBIT_WITH_STRIDES(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  INSTANTIATE_SPMDM_FP8(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE)

#define INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, OFFSET_TYPE)   \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, float)   \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, float16) \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, uint8_t)

#define INSTANTIATE_SPMDM_OFFSET_T(INDEX_TYPE) \
  INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, int32_t) \
  INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, int64_t)

INSTANTIATE_SPMDM_OFFSET_T(int32_t)
INSTANTIATE_SPMDM_OFFSET_T(int64_t)

#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_OUT_T
#undef INSTANTIATE_SPMDM_BASE

#define INSTANTIATE_SPMDM_ROWWISE(IN_TYPE, OFFSET_TYPE, OUT_TYPE)              \
  template typename EmbeddingSpMDMRowWiseSparseKernelSignature<                \
      IN_TYPE,                                                                 \
      OFFSET_TYPE,                                                             \
      OUT_TYPE>::Type                                                          \
  GenerateEmbeddingSpMDMRowWiseSparse_autovec<IN_TYPE, OFFSET_TYPE, OUT_TYPE>( \
      int64_t block_size,                                                      \
      bool has_weight,                                                         \
      bool normalize_by_lengths,                                               \
      int prefetch,                                                            \
      bool is_weight_positional,                                               \
      bool use_offsets);

#define INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  template typename EmbeddingSpMDMKernelSignature<                         \
      IN_TYPE,                                                             \
      INDEX_TYPE,                                                          \
      OFFSET_TYPE,                                                         \
      OUT_TYPE>::Type                                                      \
  GenerateEmbeddingSpMDMWithStrides_autovec<                               \
      IN_TYPE,                                                             \
      INDEX_TYPE,                                                          \
      OFFSET_TYPE,                                                         \
      OUT_TYPE>(                                                           \
      int64_t block_size,                                                  \
      bool has_weight,                                                     \
      bool normalize_by_lengths,                                           \
      int prefetch,                                                        \
      bool is_weight_positional,                                           \
      bool use_offsets,                                                    \
      int64_t output_stride,                                               \
      int64_t input_stride,                                                \
      bool scale_bias_last,                                                \
      bool no_bag,                                                         \
      bool is_bf16_out,                                                    \
      bool is_bf16_in);

#define INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, OFFSET_TYPE)        \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, float)        \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, float16)      \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, std::uint8_t) \
  INSTANTIATE_SPMDM_ROWWISE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE)

#define INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, INDEX_TYPE)      \
  INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, std::int64_t)

#define INSTANTIATE_SPMDM_INDEX_T(IN_TYPE)          \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int64_t)

INSTANTIATE_SPMDM_INDEX_T(float)
INSTANTIATE_SPMDM_INDEX_T(float16)
INSTANTIATE_SPMDM_INDEX_T(std::uint8_t)

#undef INSTANTIATE_SPMDM_ROWWISE
#undef INSTANTIATE_SPMDM_INDEX_T
#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_OUT_T
#undef INSTANTIATE_SPMDM_BASE

namespace {

// Builds the NBit row-wise-sparse kernel with bit_rate baked in as a
// compile-time constant (the old dispatch specialized only on bit_rate);
// every other parameter stays a runtime value.
template <int BitRate, typename IndexType, typename OffsetType>
typename EmbeddingSpMDMRowWiseSparseKernelSignature<uint8_t, IndexType,
                                                    OffsetType>::Type
make_nbit_rowwise_sparse_generic(
    int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    bool is_weight_positional,
    bool use_offsets) {
  return [=](int64_t output_size,
             int64_t index_size,
             int64_t uncompressed_data_size,
             const uint8_t* input,
             const IndexType* indices,
             const OffsetType* offsets_or_lengths,
             const float* weights,
             float* out,
             const int32_t* compressed_indices_table) {
    if (has_weight) {
      __builtin_assume(weights != nullptr);
    } else {
      weights = nullptr;
    }
    return EmbeddingSpMDMNBitRowWiseSparse_autovec(
        BitRate,
        block_size,
        output_size,
        index_size,
        uncompressed_data_size,
        input,
        indices,
        compressed_indices_table,
        offsets_or_lengths,
        weights,
        normalize_by_lengths,
        out,
        is_weight_positional,
        use_offsets);
  };
}

} // namespace

template <typename IndexType, typename OffsetType>
typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    uint8_t,
    IndexType,
    OffsetType>::Type
GenerateEmbeddingSpMDMNBitRowWiseSparse_autovec(
    int bit_rate,
    int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    [[maybe_unused]] int prefetch,
    bool is_weight_positional,
    bool use_offsets) {
  assert((bit_rate == 2 || bit_rate == 4) && "bit_rate must be 2 or 4");
  if (bit_rate == 4) {
    return make_nbit_rowwise_sparse_generic<4, IndexType, OffsetType>(
        block_size, has_weight, normalize_by_lengths, is_weight_positional,
        use_offsets);
  }
  if (bit_rate == 2) {
    return make_nbit_rowwise_sparse_generic<2, IndexType, OffsetType>(
        block_size, has_weight, normalize_by_lengths, is_weight_positional,
        use_offsets);
  }
  abort(); // should not get here
}


#define INSTANTIATE_SPMDM_NBIT_ROWWISE_SPARSE(INDEX_TYPE, OFFSET_TYPE)      \
  template typename EmbeddingSpMDMRowWiseSparseKernelSignature<             \
      uint8_t,                                                              \
      INDEX_TYPE,                                                           \
      OFFSET_TYPE>::Type                                                    \
  GenerateEmbeddingSpMDMNBitRowWiseSparse_autovec<INDEX_TYPE, OFFSET_TYPE>( \
      int bit_rate,                                                         \
      int64_t block_size,                                                   \
      bool has_weight,                                                      \
      bool normalize_by_lengths,                                            \
      int prefetch,                                                         \
      bool is_weight_positional,                                            \
      bool use_offsets);

INSTANTIATE_SPMDM_NBIT_ROWWISE_SPARSE(int32_t, int32_t)
INSTANTIATE_SPMDM_NBIT_ROWWISE_SPARSE(int32_t, int64_t)
INSTANTIATE_SPMDM_NBIT_ROWWISE_SPARSE(int64_t, int32_t)
INSTANTIATE_SPMDM_NBIT_ROWWISE_SPARSE(int64_t, int64_t)

#undef INSTANTIATE_SPMDM_NBIT_ROWWISE_SPARSE

} // namespace fbgemm
