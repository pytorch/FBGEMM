/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __linux__

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
  if (std::is_same<OutType, float>::value) {
    for (int j = 0; j < block_size; ++j) {
      out[j] = src[j];
    }
  } else if (std::is_same<OutType, uint16_t>::value && is_bf16_out) {
    for (int j = 0; j < block_size; ++j) {
      out[j] = cpu_float2bfloat16(src[j]);
    }
  } else {
    for (int j = 0; j < block_size; ++j) {
      out[j] = cpu_float2half(src[j]);
    }
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
  float* buf;
  if (static_cast<size_t>(block_size) <= LOCAL_STORAGE_SIZE) {
    buf = local_storage.data();
  } else {
    heap_storage.reset(new float[block_size]);
    buf = heap_storage.get();
  }

  if (no_bag) {
    for (int64_t m = 0; m < output_size; ++m) {
      const IndexType idx = indices[m];

      if (idx < 0 || idx >= data_size) {
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;
      if constexpr (isOutput8bit) {
        memcpy(out, input_row_base, sizeof(uint8_t) * input_stride);
      } else {
        memset(buf, 0, sizeof(float) * block_size);

        float scale;
        float bias;
        const uint8_t* scale_bias_addr = input_row_base + scale_bias_offset;
        if (scale_bias_last) {
          memcpy(&scale, scale_bias_addr, sizeof(float));
          memcpy(&bias, scale_bias_addr + sizeof(float), sizeof(float));
        } else {
          float16 scale16;
          float16 bias16;
          memcpy(&scale16, scale_bias_addr, sizeof(float16));
          memcpy(&bias16, scale_bias_addr + sizeof(float16), sizeof(float16));
          scale = cpu_half2float(scale16);
          bias = cpu_half2float(bias16);
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
          buf[j] = std::fma(scale, (float)value, buf[j] + bias);
        }
#endif
        for (; j < block_size; ++j) {
          uint8_t value = input_row[j];
          buf[j] = std::fma(scale, (float)value, buf[j] + bias);
        }
        fill_output(out, buf, block_size, is_bf16_out);
      }
      out += output_stride;
    } // m
    return true;
  } // no_bag

  int64_t current = 0;
  for (int64_t m = 0; m < output_size; ++m) {
    memset(buf, 0, sizeof(float) * block_size);
    const OffsetType len = use_offsets
        ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
        : offsets_or_lengths[m];
    int64_t end = current + len;
    if (end > index_size) {
      return false;
    }

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
        if (!scale_bias_last && idx == -1) {
          // When scale_bias_last == false, assume this is for table batched
          // embedding (TBE) that can get -1 for pruned rows.
          continue;
        }
        return false;
      }

      const uint8_t* input_row_base = input + input_stride * idx;

      const uint8_t* scale_bias_addr = input_row_base + scale_bias_offset;
      float scale;
      float bias;
      if (scale_bias_last) {
        memcpy(&scale, scale_bias_addr, sizeof(float));
        memcpy(&bias, scale_bias_addr + sizeof(float), sizeof(float));
      } else {
        float16 scale16;
        float16 bias16;
        memcpy(&scale16, scale_bias_addr, sizeof(float16));
        memcpy(&bias16, scale_bias_addr + sizeof(float16), sizeof(float16));
        scale = cpu_half2float(scale16);
        bias = cpu_half2float(bias16);
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
    if (!(input_bit_rate == 4 && output_bit_rate == 4)) {
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
    return true;
  }

  int64_t current = 0;
  const int64_t rounded_block_size = round_up(block_size, num_elem_per_byte);

  std::array<float, LOCAL_STORAGE_SIZE> local_storage;
  std::unique_ptr<float[]> heap_storage;
  float* buf;
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
    memset(buf, 0, sizeof(float) * rounded_block_size);

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

      const uint8_t* input_row_base = input + input_stride * idx;
      const uint8_t* scale_bias_addr = input_row_base + scale_bias_offset;
      const uint8_t* input_row = input_row_base + input_row_offset;

      float16 scale16;
      float16 bias16;
      memcpy(&scale16, scale_bias_addr, sizeof(float16));
      memcpy(&bias16, scale_bias_addr + sizeof(float16), sizeof(float16));
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
  float* buf;
  if (static_cast<size_t>(block_size) <= LOCAL_STORAGE_SIZE) {
    buf = local_storage.data();
  } else {
    heap_storage.reset(new float[block_size]);
    buf = heap_storage.get();
  }

  if (no_bag) {
    for (int m = 0; m < output_size; ++m) {
      memset(buf, 0, sizeof(float) * block_size);
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
          buf[j] = std::fma(
              weight, convert_to_float_ref(*inptr, is_bf16_in), buf[j]);
        }
#endif
        for (; j < block_size; ++j) {
          const InType* inptr = input + input_stride * idx + j;
          buf[j] = std::fma(
              weight, convert_to_float_ref(*inptr, is_bf16_in), buf[j]);
        }
      } else {
        int64_t j = 0;
#ifdef FBGEMM_VECTOR_WIDTH
        for (; j < block_size - (block_size % FBGEMM_VECTOR_WIDTH); ++j) {
          const InType* inptr = input + input_stride * idx + j;
          buf[j] += convert_to_float_ref(*inptr, is_bf16_in);
        }
#endif
        for (; j < block_size; ++j) {
          const InType* inptr = input + input_stride * idx + j;
          buf[j] += convert_to_float_ref(*inptr, is_bf16_in);
        }
      }
      fill_output(out, buf, block_size, is_bf16_out);
      out += output_stride;
    } // m
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
    memset(buf, 0, sizeof(float) * block_size);
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }

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
  bool is8bit = std::is_same<InType, uint8_t>::value;

  if (is8bit) {
    // block_size is the number of elements and fused_block_size is the size
    // of an entire row, including scale and bias.
    const auto scale_bias_offset = 2 * sizeof(float);
    const int64_t fused_block_size = block_size + scale_bias_offset;
    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      memset(out, 0, sizeof(float) * block_size);
      int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                            : offsets_or_lengths[m];
      int64_t end = current + len;
      if (end > index_size) {
        return false;
      }
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
        if (idx == -1) {
          continue;
        }
        // if (idx < 0 || idx >= compressed_data_size) {
        //   return false;
        // }

        const uint8_t* scale_bias_addr = reinterpret_cast<const uint8_t*>(
            input + fused_block_size * idx + block_size);

        float scale;
        float bias;
        memcpy(&scale, scale_bias_addr, sizeof(float));
        memcpy(&bias, scale_bias_addr + sizeof(float), sizeof(float));
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
      memset(out, 0, sizeof(float) * block_size);
      int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                            : offsets_or_lengths[m];
      int64_t end = current + len;
      if (end > index_size) {
        return false;
      }

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
        if (idx == -1) {
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
              std::is_same<InType, float16>::value ? cpu_half2float(*inptr)
                                                   : *inptr,
              out[j]);
        }
#endif
        for (; j < block_size; ++j) {
          const InType* inptr = input_row++;
          out[j] = std::fma(
              weight,
              std::is_same<InType, float16>::value ? cpu_half2float(*inptr)
                                                   : *inptr,
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
    uint32_t val_out, sign, multiplier;
    uint8_t inp = input[i];

    sign = (inp & 0x80) << 24;
    val_out = (inp & 0x7F) << (24 - (8 - exponent_bits));

    multiplier = (127 + (127 - exponent_bias)) << 23; // 2^(127-bias)
    float val_out_f = *reinterpret_cast<float*>(&val_out) *
        *reinterpret_cast<float*>(&multiplier); // val_out * multiplier
    val_out = *reinterpret_cast<uint32_t*>(&val_out_f) | sign;
    output[i] = *reinterpret_cast<float*>(&val_out);
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
  float* buf;
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
    memset(buf, 0, sizeof(float) * block_size);
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    int64_t end = current + len;
    if (end > index_size) {
      return false;
    }

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

namespace {
namespace specialization_helper {

/// The idea with the specialization helper is to create a copy of a given
/// algorithm with some parameters set to fixed values (specialized) so the
/// compiler can perform additional optimization for the specific variant.
/// This is achieved by marking the generic functions `ALWAYS_INLINE` inline
/// and defining a macro invoking match/specialize so you can choose between
/// fixed and variable values for each parameter.

template <typename T>
struct FixedParameter {
  T value;
};
struct VariableParameter {};

template <typename T>
ALWAYS_INLINE constexpr FixedParameter<T> fixed(T value) {
  return FixedParameter<T>{value};
}
static constexpr VariableParameter var = VariableParameter();

template <typename T>
ALWAYS_INLINE bool match(VariableParameter, T) {
  return true;
}
template <typename T>
ALWAYS_INLINE bool match(FixedParameter<T> fixed_parameter, T value) {
  return fixed_parameter.value == value;
}

template <typename T>
ALWAYS_INLINE T specialize(VariableParameter, T value) {
  return value;
}
template <typename T>
ALWAYS_INLINE T specialize(FixedParameter<T> fixed_parameter, T) {
  return fixed_parameter.value;
}
} // namespace specialization_helper
} // namespace

template <typename InType>
static int64_t stride_SpMDMWithStrides(
    int64_t block_size,
    bool scale_bias_last) {
  if (std::is_same<InType, uint8_t>::value) {
    const size_t scale_bias_offset =
        2 * (scale_bias_last ? sizeof(float) : sizeof(uint16_t));
    return block_size + scale_bias_offset;
  }
  return block_size;
}

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
        [[maybe_unused]] int prefetch,
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

  using specialization_helper::fixed;
  using specialization_helper::match;
  using specialization_helper::specialize;
  using specialization_helper::var;

#define SPECIALIZE(                                                       \
    BLOCK_SIZE,                                                           \
    HAS_WEIGHT,                                                           \
    NORMALIZE_BY_LENGTHS,                                                 \
    PREFETCH,                                                             \
    IS_WEIGHT_POSITIONAL,                                                 \
    USE_OFFSETS,                                                          \
    OUTPUT_STRIDE,                                                        \
    INPUT_STRIDE,                                                         \
    SCALE_BIAS_LAST,                                                      \
    NO_BAG,                                                               \
    IS_BF16_OUT,                                                          \
    IS_BF16_IN)                                                           \
  if (match(BLOCK_SIZE, block_size) && match(HAS_WEIGHT, has_weight) &&   \
      match(NORMALIZE_BY_LENGTHS, normalize_by_lengths) &&                \
      match(PREFETCH, prefetch) &&                                        \
      match(IS_WEIGHT_POSITIONAL, is_weight_positional) &&                \
      match(USE_OFFSETS, use_offsets) &&                                  \
      match(OUTPUT_STRIDE, output_stride) &&                              \
      match(INPUT_STRIDE, input_stride) &&                                \
      match(SCALE_BIAS_LAST, scale_bias_last) && match(NO_BAG, no_bag) && \
      match(IS_BF16_OUT, is_bf16_out) && match(IS_BF16_IN, is_bf16_in)) { \
    return [=](int64_t output_size,                                       \
               int64_t index_size,                                        \
               int64_t data_size,                                         \
               const InType* input,                                       \
               const IndexType* indices,                                  \
               const OffsetType* offsets_or_lengths,                      \
               const float* weights,                                      \
               OutType* out) {                                            \
      const uint8_t* input_u8 = reinterpret_cast<const uint8_t*>(input);  \
      if (specialize(HAS_WEIGHT, has_weight)) {                           \
        __builtin_assume(weights != nullptr);                             \
      } else {                                                            \
        weights = nullptr;                                                \
      }                                                                   \
      if (std::is_same<InType, uint8_t>::value) {                         \
        assert(!specialize(IS_BF16_IN, is_bf16_in));                      \
        return EmbeddingSpMDM8Bit_autovec(                                \
            specialize(BLOCK_SIZE, block_size),                           \
            output_size,                                                  \
            index_size,                                                   \
            data_size,                                                    \
            input_u8,                                                     \
            indices,                                                      \
            offsets_or_lengths,                                           \
            weights,                                                      \
            specialize(NORMALIZE_BY_LENGTHS, normalize_by_lengths),       \
            out,                                                          \
            specialize(IS_WEIGHT_POSITIONAL, is_weight_positional),       \
            specialize(USE_OFFSETS, use_offsets),                         \
            specialize(OUTPUT_STRIDE, output_stride),                     \
            specialize(INPUT_STRIDE, input_stride),                       \
            specialize(SCALE_BIAS_LAST, scale_bias_last),                 \
            specialize(NO_BAG, no_bag),                                   \
            specialize(IS_BF16_OUT, is_bf16_out));                        \
      } else {                                                            \
        return EmbeddingSpMDM_autovec(                                    \
            /*block_size=*/specialize(BLOCK_SIZE, block_size),            \
            /*output_size=*/output_size,                                  \
            /*index_size=*/index_size,                                    \
            /*data_size=*/data_size,                                      \
            /*input=*/input,                                              \
            /*indices=*/indices,                                          \
            /*offsets_or_lengths=*/offsets_or_lengths,                    \
            /*weights=*/weights, /*normalize_by_lengths=*/                \
            specialize(NORMALIZE_BY_LENGTHS, normalize_by_lengths),       \
            /*out=*/out, /*is_weight_positional=*/                        \
            specialize(IS_WEIGHT_POSITIONAL, is_weight_positional),       \
            /*use_offsets=*/specialize(USE_OFFSETS, use_offsets),         \
            /*output_stride=*/specialize(OUTPUT_STRIDE, output_stride),   \
            /*input_stride=*/specialize(INPUT_STRIDE, input_stride),      \
            /*no_bag=*/specialize(NO_BAG, no_bag),                        \
            /*is_bf16_out=*/specialize(IS_BF16_OUT, is_bf16_out),         \
            /*is_bf16_in=*/specialize(IS_BF16_IN, is_bf16_in));           \
      }                                                                   \
    };                                                                    \
  }

#define SPECIALIZE_BLOCK_SIZE(                                             \
    HAS_WEIGHT,                                                            \
    NORMALIZE_BY_LENGTHS,                                                  \
    PREFETCH,                                                              \
    IS_WEIGHT_POSITIONAL,                                                  \
    USE_OFFSETS,                                                           \
    NO_BAG,                                                                \
    IS_BF16_OUT,                                                           \
    IS_BF16_IN)                                                            \
  SPECIALIZE(                                                              \
      /*BLOCK_SIZE*/ fixed(int64_t{32}),                                   \
      HAS_WEIGHT,                                                          \
      NORMALIZE_BY_LENGTHS,                                                \
      PREFETCH,                                                            \
      IS_WEIGHT_POSITIONAL,                                                \
      USE_OFFSETS,                                                         \
      /*OUTPUT_STRIDE*/ var,                                               \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMWithStrides<InType>(32, false)),  \
      /*SCALE_BIAS_LAST*/ fixed(false),                                    \
      NO_BAG,                                                              \
      IS_BF16_OUT,                                                         \
      IS_BF16_IN)                                                          \
  SPECIALIZE(                                                              \
      /*BLOCK_SIZE*/ fixed(int64_t{64}),                                   \
      HAS_WEIGHT,                                                          \
      NORMALIZE_BY_LENGTHS,                                                \
      PREFETCH,                                                            \
      IS_WEIGHT_POSITIONAL,                                                \
      USE_OFFSETS,                                                         \
      /*OUTPUT_STRIDE*/ var,                                               \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMWithStrides<InType>(64, false)),  \
      /*SCALE_BIAS_LAST*/ fixed(false),                                    \
      NO_BAG,                                                              \
      IS_BF16_OUT,                                                         \
      IS_BF16_IN)                                                          \
  SPECIALIZE(                                                              \
      /*BLOCK_SIZE*/ fixed(int64_t{124}),                                  \
      HAS_WEIGHT,                                                          \
      NORMALIZE_BY_LENGTHS,                                                \
      PREFETCH,                                                            \
      IS_WEIGHT_POSITIONAL,                                                \
      USE_OFFSETS,                                                         \
      /*OUTPUT_STRIDE*/ var,                                               \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMWithStrides<InType>(124, false)), \
      /*SCALE_BIAS_LAST*/ fixed(false),                                    \
      NO_BAG,                                                              \
      IS_BF16_OUT,                                                         \
      IS_BF16_IN)                                                          \
  SPECIALIZE(                                                              \
      /*BLOCK_SIZE*/ fixed(int64_t{128}),                                  \
      HAS_WEIGHT,                                                          \
      NORMALIZE_BY_LENGTHS,                                                \
      PREFETCH,                                                            \
      IS_WEIGHT_POSITIONAL,                                                \
      USE_OFFSETS,                                                         \
      /*OUTPUT_STRIDE*/ var,                                               \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMWithStrides<InType>(128, false)), \
      /*SCALE_BIAS_LAST*/ fixed(false),                                    \
      NO_BAG,                                                              \
      IS_BF16_OUT,                                                         \
      IS_BF16_IN)                                                          \
  SPECIALIZE(                                                              \
      /*BLOCK_SIZE*/ fixed(int64_t{252}),                                  \
      HAS_WEIGHT,                                                          \
      NORMALIZE_BY_LENGTHS,                                                \
      PREFETCH,                                                            \
      IS_WEIGHT_POSITIONAL,                                                \
      USE_OFFSETS,                                                         \
      /*OUTPUT_STRIDE*/ var,                                               \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMWithStrides<InType>(252, false)), \
      /*SCALE_BIAS_LAST*/ fixed(false),                                    \
      NO_BAG,                                                              \
      IS_BF16_OUT,                                                         \
      IS_BF16_IN)                                                          \
  SPECIALIZE(                                                              \
      /*BLOCK_SIZE*/ fixed(int64_t{256}),                                  \
      HAS_WEIGHT,                                                          \
      NORMALIZE_BY_LENGTHS,                                                \
      PREFETCH,                                                            \
      IS_WEIGHT_POSITIONAL,                                                \
      USE_OFFSETS,                                                         \
      /*OUTPUT_STRIDE*/ var,                                               \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMWithStrides<InType>(256, false)), \
      /*SCALE_BIAS_LAST*/ fixed(false),                                    \
      NO_BAG,                                                              \
      IS_BF16_OUT,                                                         \
      IS_BF16_IN)                                                          \
  SPECIALIZE(                                                              \
      /*BLOCK_SIZE*/ fixed(int64_t{508}),                                  \
      HAS_WEIGHT,                                                          \
      NORMALIZE_BY_LENGTHS,                                                \
      PREFETCH,                                                            \
      IS_WEIGHT_POSITIONAL,                                                \
      USE_OFFSETS,                                                         \
      /*OUTPUT_STRIDE*/ var,                                               \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMWithStrides<InType>(508, false)), \
      /*SCALE_BIAS_LAST*/ fixed(false),                                    \
      NO_BAG,                                                              \
      IS_BF16_OUT,                                                         \
      IS_BF16_IN)                                                          \
  SPECIALIZE(                                                              \
      /*BLOCK_SIZE*/ fixed(int64_t{512}),                                  \
      HAS_WEIGHT,                                                          \
      NORMALIZE_BY_LENGTHS,                                                \
      PREFETCH,                                                            \
      IS_WEIGHT_POSITIONAL,                                                \
      USE_OFFSETS,                                                         \
      /*OUTPUT_STRIDE*/ var,                                               \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMWithStrides<InType>(512, false)), \
      /*SCALE_BIAS_LAST*/ fixed(false),                                    \
      NO_BAG,                                                              \
      IS_BF16_OUT,                                                         \
      IS_BF16_IN)

#ifdef FBGEMM_MORE_SPECIALIZATION
  SPECIALIZE_BLOCK_SIZE(
      /*HAS_WEIGHT*/ fixed(true),
      /*NORMALIZE_BY_LENGTHS*/ fixed(false),
      /*PREFETCH*/ var,
      /*IS_WEIGHT_POSITIONAL*/ fixed(false),
      /*USE_OFFSETS*/ fixed(true),
      /*NO_BAG*/ fixed(false),
      /*IS_BF16_OUT*/ var,
      /*IS_BF16_IN*/ var)
  SPECIALIZE_BLOCK_SIZE(
      /*HAS_WEIGHT*/ fixed(false),
      /*NORMALIZE_BY_LENGTHS*/ fixed(false),
      /*PREFETCH*/ var,
      /*IS_WEIGHT_POSITIONAL*/ fixed(false),
      /*USE_OFFSETS*/ fixed(true),
      /*NO_BAG*/ fixed(false),
      /*IS_BF16_OUT*/ var,
      /*IS_BF16_IN*/ var)
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

  // Catch-all case.
  SPECIALIZE(var, var, var, var, var, var, var, var, var, var, var, var)
  abort(); // should not get here

#undef SPECIALIZE_BLOCK_SIZE
#undef SPECIALIZE
}

static int64_t stride_SpMDMNBitWith(int input_bit_rate, int64_t block_size) {
  const int num_elem_per_byte = 8 / input_bit_rate;
  const size_t scale_bias_size = 2 * sizeof(float16);
  return div_up(block_size, num_elem_per_byte) + scale_bias_size;
}

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
    [[maybe_unused]] int prefetch,
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

  using specialization_helper::fixed;
  using specialization_helper::match;
  using specialization_helper::specialize;
  using specialization_helper::var;

#define SPECIALIZE(                                                          \
    INPUT_BIT_RATE,                                                          \
    BLOCK_SIZE,                                                              \
    HAS_WEIGHT,                                                              \
    NORMALIZE_BY_LENGTHS,                                                    \
    IS_WEIGHT_POSITIONAL,                                                    \
    USE_OFFSETS,                                                             \
    OUTPUT_STRIDE,                                                           \
    INPUT_STRIDE,                                                            \
    SCALE_BIAS_LAST,                                                         \
    IS_BF16_OUT,                                                             \
    NO_BAG,                                                                  \
    OUTPUT_BIT_RATE)                                                         \
  if (match(INPUT_BIT_RATE, input_bit_rate) &&                               \
      match(BLOCK_SIZE, block_size) && match(HAS_WEIGHT, has_weight) &&      \
      match(NORMALIZE_BY_LENGTHS, normalize_by_lengths) &&                   \
      match(IS_WEIGHT_POSITIONAL, is_weight_positional) &&                   \
      match(USE_OFFSETS, use_offsets) &&                                     \
      match(OUTPUT_STRIDE, output_stride) &&                                 \
      match(INPUT_STRIDE, input_stride) &&                                   \
      match(SCALE_BIAS_LAST, scale_bias_last) &&                             \
      match(IS_BF16_OUT, is_bf16_out) && match(NO_BAG, no_bag) &&            \
      match(OUTPUT_BIT_RATE, output_bit_rate)) {                             \
    return [=](int64_t output_size,                                          \
               int64_t index_size,                                           \
               int64_t data_size,                                            \
               const uint8_t* input,                                         \
               const IndexType* indices,                                     \
               const OffsetType* offsets_or_lengths,                         \
               const float* weights,                                         \
               OutType* out) {                                               \
      if (specialize(HAS_WEIGHT, has_weight)) {                              \
        __builtin_assume(weights != nullptr);                                \
      } else {                                                               \
        weights = nullptr;                                                   \
      }                                                                      \
      return EmbeddingSpMDMNBit_autovec(                                     \
          /*input_bit_rate=*/specialize(INPUT_BIT_RATE, input_bit_rate),     \
          /*block_size=*/specialize(BLOCK_SIZE, block_size),                 \
          /*output_size=*/output_size,                                       \
          /*index_size=*/index_size,                                         \
          /*data_size=*/data_size,                                           \
          /*input=*/input,                                                   \
          /*indices=*/indices,                                               \
          /*offsets_or_lengths=*/offsets_or_lengths,                         \
          /*weights=*/weights, /*normalize_by_lengths=*/                     \
          specialize(NORMALIZE_BY_LENGTHS, normalize_by_lengths),            \
          /*out=*/out, /*is_weight_positional=*/                             \
          specialize(IS_WEIGHT_POSITIONAL, is_weight_positional),            \
          /*use_offsets=*/specialize(USE_OFFSETS, use_offsets),              \
          /*output_stride=*/specialize(OUTPUT_STRIDE, output_stride),        \
          /*input_stride=*/specialize(INPUT_STRIDE, input_stride),           \
          /*scale_bias_last=*/specialize(SCALE_BIAS_LAST, scale_bias_last),  \
          /*is_bf16_out=*/specialize(IS_BF16_OUT, is_bf16_out),              \
          /*no_bag=*/specialize(NO_BAG, no_bag),                             \
          /*output_bit_rate=*/specialize(OUTPUT_BIT_RATE, output_bit_rate)); \
    };                                                                       \
  }

#define SPECIALIZE_BLOCK_SIZE(                                                 \
    INPUT_BIT_RATE,                                                            \
    HAS_WEIGHT,                                                                \
    NORMALIZE_BY_LENGTHS,                                                      \
    IS_WEIGHT_POSITIONAL,                                                      \
    USE_OFFSETS,                                                               \
    SCALE_BIAS_LAST,                                                           \
    IS_BF16_OUT,                                                               \
    NO_BAG,                                                                    \
    OUTPUT_BIT_RATE)                                                           \
  SPECIALIZE(                                                                  \
      INPUT_BIT_RATE,                                                          \
      /*BLOCK_SIZE*/ fixed(int64_t{32}),                                       \
      HAS_WEIGHT,                                                              \
      NORMALIZE_BY_LENGTHS,                                                    \
      IS_WEIGHT_POSITIONAL,                                                    \
      USE_OFFSETS,                                                             \
      /*OUTPUT_STRIDE*/ var,                                                   \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMNBitWith(INPUT_BIT_RATE.value, 32)),  \
      SCALE_BIAS_LAST,                                                         \
      IS_BF16_OUT,                                                             \
      NO_BAG,                                                                  \
      OUTPUT_BIT_RATE)                                                         \
  SPECIALIZE(                                                                  \
      INPUT_BIT_RATE,                                                          \
      /*BLOCK_SIZE*/ fixed(int64_t{56}),                                       \
      HAS_WEIGHT,                                                              \
      NORMALIZE_BY_LENGTHS,                                                    \
      IS_WEIGHT_POSITIONAL,                                                    \
      USE_OFFSETS,                                                             \
      /*OUTPUT_STRIDE*/ var,                                                   \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMNBitWith(INPUT_BIT_RATE.value, 56)),  \
      SCALE_BIAS_LAST,                                                         \
      IS_BF16_OUT,                                                             \
      NO_BAG,                                                                  \
      OUTPUT_BIT_RATE)                                                         \
  SPECIALIZE(                                                                  \
      INPUT_BIT_RATE,                                                          \
      /*BLOCK_SIZE*/ fixed(int64_t{64}),                                       \
      HAS_WEIGHT,                                                              \
      NORMALIZE_BY_LENGTHS,                                                    \
      IS_WEIGHT_POSITIONAL,                                                    \
      USE_OFFSETS,                                                             \
      /*OUTPUT_STRIDE*/ var,                                                   \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMNBitWith(INPUT_BIT_RATE.value, 64)),  \
      SCALE_BIAS_LAST,                                                         \
      IS_BF16_OUT,                                                             \
      NO_BAG,                                                                  \
      OUTPUT_BIT_RATE)                                                         \
  SPECIALIZE(                                                                  \
      INPUT_BIT_RATE,                                                          \
      /*BLOCK_SIZE*/ fixed(int64_t{120}),                                      \
      HAS_WEIGHT,                                                              \
      NORMALIZE_BY_LENGTHS,                                                    \
      IS_WEIGHT_POSITIONAL,                                                    \
      USE_OFFSETS,                                                             \
      /*OUTPUT_STRIDE*/ var,                                                   \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMNBitWith(INPUT_BIT_RATE.value, 120)), \
      SCALE_BIAS_LAST,                                                         \
      IS_BF16_OUT,                                                             \
      NO_BAG,                                                                  \
      OUTPUT_BIT_RATE)                                                         \
  SPECIALIZE(                                                                  \
      INPUT_BIT_RATE,                                                          \
      /*BLOCK_SIZE*/ fixed(int64_t{128}),                                      \
      HAS_WEIGHT,                                                              \
      NORMALIZE_BY_LENGTHS,                                                    \
      IS_WEIGHT_POSITIONAL,                                                    \
      USE_OFFSETS,                                                             \
      /*OUTPUT_STRIDE*/ var,                                                   \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMNBitWith(INPUT_BIT_RATE.value, 128)), \
      SCALE_BIAS_LAST,                                                         \
      IS_BF16_OUT,                                                             \
      NO_BAG,                                                                  \
      OUTPUT_BIT_RATE)                                                         \
  SPECIALIZE(                                                                  \
      INPUT_BIT_RATE,                                                          \
      /*BLOCK_SIZE*/ fixed(int64_t{248}),                                      \
      HAS_WEIGHT,                                                              \
      NORMALIZE_BY_LENGTHS,                                                    \
      IS_WEIGHT_POSITIONAL,                                                    \
      USE_OFFSETS,                                                             \
      /*OUTPUT_STRIDE*/ var,                                                   \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMNBitWith(INPUT_BIT_RATE.value, 248)), \
      SCALE_BIAS_LAST,                                                         \
      IS_BF16_OUT,                                                             \
      NO_BAG,                                                                  \
      OUTPUT_BIT_RATE)                                                         \
  SPECIALIZE(                                                                  \
      INPUT_BIT_RATE,                                                          \
      /*BLOCK_SIZE*/ fixed(int64_t{256}),                                      \
      HAS_WEIGHT,                                                              \
      NORMALIZE_BY_LENGTHS,                                                    \
      IS_WEIGHT_POSITIONAL,                                                    \
      USE_OFFSETS,                                                             \
      /*OUTPUT_STRIDE*/ var,                                                   \
      /*INPUT_STRIDE*/ fixed(stride_SpMDMNBitWith(INPUT_BIT_RATE.value, 256)), \
      SCALE_BIAS_LAST,                                                         \
      IS_BF16_OUT,                                                             \
      NO_BAG,                                                                  \
      OUTPUT_BIT_RATE)

#define SPECIALIZE_INPUT_RATE(     \
    HAS_WEIGHT,                    \
    NORMALIZE_BY_LENGTHS,          \
    IS_WEIGHT_POSITIONAL,          \
    USE_OFFSETS,                   \
    SCALE_BIAS_LAST,               \
    IS_BF16_OUT,                   \
    NO_BAG)                        \
  SPECIALIZE_BLOCK_SIZE(           \
      /*INPUT_BIT_RATE*/ fixed(4), \
      HAS_WEIGHT,                  \
      NORMALIZE_BY_LENGTHS,        \
      IS_WEIGHT_POSITIONAL,        \
      USE_OFFSETS,                 \
      SCALE_BIAS_LAST,             \
      IS_BF16_OUT,                 \
      NO_BAG,                      \
      /*OUTPUT_BIT_RATE*/ fixed(int{8 * sizeof(OutType)}))

#ifdef FBGEMM_MORE_SPECIALIZATION
  SPECIALIZE_INPUT_RATE(
      /*HAS_WEIGHT*/ fixed(true),
      /*NORMALIZE_BY_LENGTHS*/ fixed(false),
      /*IS_WEIGHT_POSITIONAL*/ fixed(false),
      /*USE_OFFSETS*/ fixed(true),
      /*SCALE_BIAS_LAST*/ fixed(false),
      /*IS_BF16_OUT*/ var,
      /*NO_BAG*/ fixed(false))
  SPECIALIZE_INPUT_RATE(
      /*HAS_WEIGHT*/ fixed(false),
      /*NORMALIZE_BY_LENGTHS*/ fixed(false),
      /*IS_WEIGHT_POSITIONAL*/ fixed(false),
      /*USE_OFFSETS*/ fixed(true),
      /*SCALE_BIAS_LAST*/ fixed(false),
      /*IS_BF16_OUT*/ var,
      /*NO_BAG*/ fixed(false))
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

  // Catch-all case: Just specialize for input_bit_rate.
  SPECIALIZE(
      /*INPUT_BIT_RATE*/ fixed(2),
      var,
      var,
      var,
      var,
      var,
      var,
      var,
      var,
      var,
      var,
      var);
  SPECIALIZE(
      /*INPUT_BIT_RATE*/ fixed(4),
      var,
      var,
      var,
      var,
      var,
      var,
      var,
      var,
      var,
      var,
      var);
  abort(); // should not get here

#undef SPECIALIZE_BLOCK_SIZE
#undef SPECIALIZE_INPUT_RATE
#undef SPECIALIZE
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
    [[maybe_unused]] int prefetch,
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

} // namespace fbgemm

#endif // #ifdef __linux__
