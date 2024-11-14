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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <new>
#include <numeric>
#include <thread>

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
bool EmbeddingSpMDM8Bit_autovec(
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
    const bool is_weight_positional /*=false*/,
    const bool use_offsets /*=true*/,
    int64_t output_stride /*=-1*/,
    int64_t input_stride /*=-1*/,
    const bool scale_bias_last /*=true*/,
    const bool no_bag /*=false*/,
    const bool is_bf16_out /*=false*/) {
  constexpr bool isOutput8bit = std::is_same<OutType, uint8_t>::value;
  if (data_size < 0) {
    return false;
  }
  if (output_stride == -1) {
    output_stride = block_size;
  }
  if constexpr (isOutput8bit) {
    assert(input_stride == output_stride);
  }

  // block_size is the number of elements and fused_block_size is the size of
  // an entire row, including scale and bias.
  if (input_stride == -1) {
    // scale_bias_last == false is for table batched embedding that stores
    // scale and bias in float16
    const auto scale_bias_offset =
        2 * (scale_bias_last ? sizeof(float) : sizeof(float16));
    input_stride = block_size + scale_bias_offset;
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
        for (int64_t j = 0; j < block_size; ++j) {
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
      if (block_size <= 64) {
        for (int64_t j = 0; j < block_size; ++j) {
          uint8_t value = input_row[j];
          buf[j] = std::fma(scale, (float)value, buf[j] + bias);
        }
      } else {
        for (int64_t j = 0; j < block_size; ++j) {
          uint8_t value = input_row[j];
          buf[j] = std::fma(scale, (float)value, buf[j] + bias);
        }
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

#define INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  template FBGEMM_API bool EmbeddingSpMDM8Bit_autovec(            \
      const int64_t block_size,                                   \
      const int64_t output_size,                                  \
      const int64_t index_size,                                   \
      const int64_t data_size,                                    \
      const uint8_t* input,                                       \
      const INDEX_TYPE* indices,                                  \
      const OFFSET_TYPE* offsets_or_lengths,                      \
      const float* weights,                                       \
      const bool normalize_by_lengths,                            \
      OUT_TYPE* out,                                              \
      const bool is_weight_positional,                            \
      const bool use_offsets,                                     \
      int64_t input_stride,                                       \
      int64_t output_stride,                                      \
      const bool scale_bias_last,                                 \
      const bool no_bag,                                          \
      const bool is_bf16_out);

#define INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, OFFSET_TYPE)   \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, float)   \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, float16) \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, std::uint8_t)

#define INSTANTIATE_SPMDM_OFFSET_T(INDEX_TYPE)      \
  INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, std::int64_t)

#define INSTANTIATE_SPMDM_INDEX_T()        \
  INSTANTIATE_SPMDM_OFFSET_T(std::int32_t) \
  INSTANTIATE_SPMDM_OFFSET_T(std::int64_t)

INSTANTIATE_SPMDM_INDEX_T()

#undef INSTANTIATE_SPMDM_INDEX_T
#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_OUT_T
#undef INSTANTIATE_SPMDM_BASE

template <typename IndexType, typename OffsetType, typename OutType>
bool EmbeddingSpMDMNBit_autovec(
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
    const bool is_weight_positional /*=false*/,
    const bool use_offsets /*=true*/,
    int64_t output_stride /*=-1*/,
    int64_t input_stride /*=-1*/,
    const bool scale_bias_last /*=true*/,
    const bool is_bf16_out /*=false*/,
    const bool no_bag /*=false*/,
    int output_bit_rate /*=-1*/) {
  if (output_bit_rate == -1) {
    output_bit_rate = 8 * sizeof(OutType);
  }
  nbit_embedding_sanity_check<OutType>(input_bit_rate, output_bit_rate, no_bag);
  const int num_elem_per_byte = 8 / input_bit_rate;
  if (data_size < 0) {
    return false;
  }
  if (output_stride == -1) {
    output_stride = block_size;
  }

  // block_size is the number of elements and fused_block_size is the size of
  // an entire row, including scale and bias.
  const size_t scale_bias_size = 2 * sizeof(float16);
  if (input_stride == -1) {
    input_stride = div_up(block_size, num_elem_per_byte) + scale_bias_size;
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
  const int64_t scale_bias_offset =
      scale_bias_last ? div_up(block_size, num_elem_per_byte) : 0;
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
        for (int64_t j = 0, k = 0; j < block_size; j += 2) {
          uint8_t tmp = input_row[k++];
          float quantized1 = float(tmp & 0xf);
          float quantized2 = float(tmp >> 4);
          buf[j] = std::fma(scale, quantized1, buf[j] + bias);
          buf[j + 1] = std::fma(scale, quantized2, buf[j + 1] + bias);
        }
      } else if (input_bit_rate == 2) {
        for (int64_t j = 0, k = 0; j < block_size; j += 4) {
          uint8_t tmp = input_row[k++];
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

template <
    typename InType,
    typename IndexType,
    typename OffsetType,
    typename OutType>
bool EmbeddingSpMDM_autovec(
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
    bool is_weight_positional /*=false*/,
    bool use_offsets /*=true*/,
    int64_t output_stride /*=-1*/,
    int64_t input_stride /*=-1*/,
    bool scale_bias_last /*=true*/,
    bool no_bag /*=false*/,
    bool is_bf16_out /*=false*/,
    bool is_bf16_in /*=false*/) {
  if (std::is_same<InType, std::uint8_t>::value) {
    const uint8_t* input_u8 = reinterpret_cast<const uint8_t*>(input);
    return EmbeddingSpMDM8Bit_autovec(
        block_size,
        output_size,
        index_size,
        data_size,
        input_u8,
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
  }
  if (data_size < 0) {
    return false;
  }
  if (output_stride == -1) {
    output_stride = block_size;
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

  if (input_stride == -1) {
    input_stride = block_size;
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
        for (int64_t j = 0; j < block_size; ++j) {
          const InType* inptr = input + input_stride * idx + j;
          buf[j] = std::fma(
              weight, convert_to_float_ref(*inptr, is_bf16_in), buf[j]);
        }
      } else {
        for (int64_t j = 0; j < block_size; ++j) {
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

      for (int64_t j = 0; j < block_size; ++j) {
        const InType* inptr = input + input_stride * idx + j;
        buf[j] = std::fma(w, convert_to_float_ref(*inptr, is_bf16_in), buf[j]);
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
bool EmbeddingSpMDMRowWiseSparse_autovec(
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
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional,
    bool use_offsets) {
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

        for (int j = 0; j < block_size; ++j) {
          out[j] =
              std::fma(scale, input[fused_block_size * idx + j], out[j] + bias);
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

        for (int j = 0; j < block_size; ++j) {
          const InType* inptr = input + block_size * idx + j;
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

#define INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  template FBGEMM_API bool EmbeddingSpMDM_autovec(                         \
      const int64_t block_size,                                            \
      const int64_t output_size,                                           \
      const int64_t index_size,                                            \
      const int64_t data_size,                                             \
      const IN_TYPE* input,                                                \
      const INDEX_TYPE* indices,                                           \
      const OFFSET_TYPE* offsets_or_lengths,                               \
      const float* weights,                                                \
      bool normalize_by_lengths,                                           \
      OUT_TYPE* out,                                                       \
      bool is_weight_positional,                                           \
      bool use_offsets,                                                    \
      int64_t input_stride,                                                \
      int64_t output_stride,                                               \
      bool scale_bias_last,                                                \
      bool no_bag,                                                         \
      bool is_bf16_out,                                                    \
      bool is_bf16_in);

#define INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, OFFSET_TYPE)        \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, float)        \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, float16)      \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, std::uint8_t) \
  template FBGEMM_API bool EmbeddingSpMDMRowWiseSparse_autovec(          \
      const int64_t block_size,                                          \
      const int64_t output_size,                                         \
      const int64_t index_size,                                          \
      const int64_t uncompressed_data_size,                              \
      const IN_TYPE* input,                                              \
      const INDEX_TYPE* indices,                                         \
      const int32_t* compressed_indices_table,                           \
      const OFFSET_TYPE* offsets_or_lengths,                             \
      const float* weights,                                              \
      bool normalize_by_lengths,                                         \
      float* out,                                                        \
      bool is_weight_positional,                                         \
      bool use_offsets);

#define INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, INDEX_TYPE)      \
  INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, std::int64_t)

#define INSTANTIATE_SPMDM_INDEX_T(IN_TYPE)          \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int64_t)

INSTANTIATE_SPMDM_INDEX_T(float)
INSTANTIATE_SPMDM_INDEX_T(float16)
INSTANTIATE_SPMDM_INDEX_T(std::uint8_t)

#undef INSTANTIATE_SPMDM_INDEX_T
#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_OUT_T
#undef INSTANTIATE_SPMDM_BASE

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

template <typename IndexType, typename OffsetType, typename OutType>
bool EmbeddingSpMDMFP8_autovec(
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
    bool is_weight_positional,
    bool use_offsets,
    int64_t output_stride,
    int64_t input_stride,
    int exponent_bits,
    int exponent_bias,
    bool is_bf16_out /*=false*/) {
  if (data_size < 0) {
    return false;
  }
  if (output_stride == -1) {
    output_stride = block_size;
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

  if (input_stride == -1) {
    input_stride = block_size;
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
      for (int j = 0; j < block_size; ++j) {
        buf[j] = std::fma(w, converted_inputs[j], buf[j]);
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

#define INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  template FBGEMM_API bool EmbeddingSpMDMNBit_autovec(            \
      const int input_bit_rate,                                   \
      const int64_t block_size,                                   \
      const int64_t output_size,                                  \
      const int64_t index_size,                                   \
      const int64_t data_size,                                    \
      const uint8_t* input,                                       \
      const INDEX_TYPE* indices,                                  \
      const OFFSET_TYPE* offsets_or_lengths,                      \
      const float* weights,                                       \
      const bool normalize_by_lengths,                            \
      OUT_TYPE* out,                                              \
      const bool is_weight_positional,                            \
      const bool use_offsets,                                     \
      int64_t output_stride,                                      \
      int64_t input_stride,                                       \
      const bool scale_bias_last,                                 \
      const bool is_bf16_out,                                     \
      const bool no_bag,                                          \
      int output_bit_rate);                                       \
  template FBGEMM_API bool EmbeddingSpMDMFP8_autovec(             \
      const int64_t block_size,                                   \
      const int64_t output_size,                                  \
      const int64_t index_size,                                   \
      const int64_t data_size,                                    \
      const uint8_t* input,                                       \
      const INDEX_TYPE* indices,                                  \
      const OFFSET_TYPE* offsets_or_lengths,                      \
      const float* weights,                                       \
      bool normalize_by_lengths,                                  \
      OUT_TYPE* out,                                              \
      bool is_weight_positional,                                  \
      bool use_offsets,                                           \
      int64_t output_stride,                                      \
      int64_t input_stride,                                       \
      int exponent_bits,                                          \
      int exponent_bias,                                          \
      bool is_bf16_out);

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

} // namespace fbgemm

#endif // #ifdef __linux__
