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
#include "fbgemm/FbgemmConvert.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <new>
#include <numeric>
#include <thread>

using std::vector;

namespace fbgemm {

template <typename OutType>
static inline void fill_output(
    OutType* out,
    const float* src,
    const int64_t block_size,
    const bool is_bf16_out) {
  if (std::is_same<OutType, float>::value) {
#pragma omp simd
    for (int j = 0; j < block_size; ++j) {
      out[j] = src[j];
    }
  } else if (std::is_same<OutType, uint16_t>::value && is_bf16_out) {
#pragma omp simd
    for (int j = 0; j < block_size; ++j) {
      out[j] = cpu_float2bfloat16(src[j]);
    }
  } else {
#pragma omp simd
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
  if (output_stride == -1) {
    output_stride = block_size;
  }
  if constexpr (isOutput8bit) {
    assert(input_stride == output_stride);
  }
  vector<float> buf(block_size);

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
  for (int pf_idx = 0; pf_idx < prefetch_stride; ++pf_idx) {
    for (int col = 0; col < input_stride; col += CACHE_LINE_SIZE) {
      do_prefetch(
          reinterpret_cast<const char*>(
              input + input_stride * indices[pf_idx] + col),
          0,
          0);
    }
  }
  IndexType current = 0;

  if (no_bag) {
    for (int m = 0; m < output_size; ++m) {
      const auto idx = indices[m];

      if (idx < 0 || idx >= data_size) {
        return false;
      }
      if constexpr (isOutput8bit) {
        const uint8_t* input_row_ptr = input + input_stride * idx;
        memcpy(out, input_row_ptr, sizeof(uint8_t) * input_stride);
      } else {
        memset(buf.data(), 0, sizeof(float) * block_size);
        const float* scale_bias = reinterpret_cast<const float*>(
            input + input_stride * idx + (scale_bias_last ? block_size : 0));

        const auto weight = weights ? weights[m] : 1.0f;

        float scale;
        float bias;
        if (scale_bias_last) {
          scale = weight * scale_bias[0];
          bias = weight * scale_bias[1];
        } else {
          scale = weight *
              cpu_half2float(reinterpret_cast<const float16*>(scale_bias)[0]);
          bias = weight *
              cpu_half2float(reinterpret_cast<const float16*>(scale_bias)[1]);
        }

        const size_t input_offset =
            input_stride * idx + (scale_bias_last ? 0 : (2 * sizeof(float16)));
#pragma omp simd
        for (int j = 0; j < block_size; ++j) {
          buf[j] =
              std::fma(scale, (float)input[input_offset + j], buf[j] + bias);
        }
        fill_output(out, buf.data(), block_size, is_bf16_out);
        out += output_stride;
      }
    } // m
    return true;
  } // no_bag

  for (int m = 0; m < output_size; ++m) {
    memset(buf.data(), 0, sizeof(float) * block_size);
    const auto len = use_offsets
        ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
        : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }

    for (OffsetType i = 0; i < len; ++i) {
      const auto idx = indices[current];
      const auto prefetch_idx =
          indices[std::min(current + prefetch_stride, index_size - 1)];
      for (int64_t col = 0; col < input_stride; col += CACHE_LINE_SIZE) {
        do_prefetch(
            reinterpret_cast<const char*>(
                input + input_stride * prefetch_idx + col),
            1);
      }
      if (!scale_bias_last && idx == -1) {
        // When scale_bias_last == false, assume this is for table batched
        // embedding (TBE) that can get -1 for pruned rows.
        continue;
      }
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      const float* scale_bias = reinterpret_cast<const float*>(
          input + input_stride * idx + (scale_bias_last ? block_size : 0));

      const auto weight =
          weights ? weights[is_weight_positional ? i : current] : 1.0f;
      float scale;
      float bias;
      if (scale_bias_last) {
        scale = weight * scale_bias[0];
        bias = weight * scale_bias[1];
      } else {
        scale = weight *
            cpu_half2float(reinterpret_cast<const float16*>(scale_bias)[0]);
        bias = weight *
            cpu_half2float(reinterpret_cast<const float16*>(scale_bias)[1]);
      }

      size_t input_offset =
          input_stride * idx + (scale_bias_last ? 0 : 2 * sizeof(float16));
      if (block_size <= 64) {
#ifdef __clang__
#pragma clang loop vectorize_width(4) interleave_count(8)
#endif
        for (int j = 0; j < block_size; ++j) {
          buf[j] =
              std::fma(scale, (float)input[input_offset + j], buf[j] + bias);
        }
      } else {
#ifdef __clang__
#pragma clang loop vectorize_width(4) interleave_count(16)
#endif
        for (int j = 0; j < block_size; ++j) {
          buf[j] =
              std::fma(scale, (float)input[input_offset + j], buf[j] + bias);
        }
      }

      ++current;
    }
    if (normalize_by_lengths && len) {
      const float scale = 1.f / len;
#pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        buf[j] *= scale;
      }
    }
    fill_output(out, buf.data(), block_size, is_bf16_out);
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
    const int bit_rate,
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
    const bool is_bf16_out /*=false*/) {
  assert((bit_rate == 2 || bit_rate == 4) && "bit_rate must be 2 or 4");
  const int num_elem_per_byte = 8 / bit_rate;

  if (output_stride == -1) {
    output_stride = block_size;
  }

  // block_size is the number of elements and fused_block_size is the size of
  // an entire row, including scale and bias.
  const auto scale_bias_offset = 2 * sizeof(float16);
  if (input_stride == -1) {
    input_stride = div_up(block_size, num_elem_per_byte) + scale_bias_offset;
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
  // The following prefetch loop is written in this way for better performance.
  // My understanding is that manually separating the case of input_stride being
  // greater or not greater than cache line size will make the branch predictor
  // work better. Same for line 113-126.
  for (int pf_idx = 0; pf_idx < prefetch_stride; ++pf_idx) {
    do_prefetch(
        reinterpret_cast<const char*>(input + input_stride * indices[pf_idx]),
        0,
        0);
    if (input_stride > CACHE_LINE_SIZE) {
      for (int64_t offset = CACHE_LINE_SIZE; offset < input_stride;
           offset += CACHE_LINE_SIZE) {
        do_prefetch(
            reinterpret_cast<const char*>(
                input + input_stride * indices[pf_idx] + offset),
            0,
            0);
      }
    }
  }

  int64_t current = 0;
  const int64_t rounded_bs = round_up(block_size, num_elem_per_byte);
  vector<float> buf(rounded_bs);
  for (int m = 0; m < output_size; ++m) {
    memset(buf.data(), 0, sizeof(float) * rounded_bs);
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }
#if _OPENMP >= 202011
    constexpr int tile_size = 4;
#pragma omp tile sizes(tile_size)
#endif
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

      const float16* scale_bias = reinterpret_cast<const float16*>(
          input + input_stride * idx +
          (scale_bias_last ? div_up(block_size, num_elem_per_byte) : 0));

      float scale = cpu_half2float(scale_bias[0]);
      float bias = cpu_half2float(scale_bias[1]);
      if (weights) {
        float weight = weights[is_weight_positional ? i : current];
        scale *= weight;
        bias *= weight;
      }

      const int64_t offset =
          input_stride * idx + (scale_bias_last ? 0 : scale_bias_offset);
      const uint8_t* input_row = input + offset;
      if (bit_rate == 4) {
        const size_t halfbufsz = (block_size + 1) / 2;
        for (size_t j = 0; j < halfbufsz; ++j) {
          float quantized1 = float(input_row[j] & 0xf);
          float quantized2 = float(input_row[j] >> 4);
          buf[j * 2] = std::fma(scale, quantized1, buf[j * 2] + bias);
          buf[j * 2 + 1] = std::fma(scale, quantized2, buf[j * 2 + 1] + bias);
        }
      } else if (bit_rate == 2) {
        size_t qbufsz = (block_size + 3) / 4;
        const uint8_t mask1 = 0x3;
        const uint8_t mask2 = 0xC;
        const uint8_t mask3 = 0x30;
        for (size_t j = 0; j < qbufsz; ++j) {
          uint8_t tmp = input[offset + j];
          float quantized1 = float(tmp & mask1);
          buf[j * 4] = std::fma(scale, quantized1, buf[j * 4] + bias);
          float quantized2 = float((tmp & mask2) >> 2);
          buf[j * 4 + 1] = std::fma(scale, quantized2, buf[j * 4 + 1] + bias);
          float quantized3 = float((tmp & mask3) >> 4);
          buf[j * 4 + 2] = std::fma(scale, quantized3, buf[j * 4 + 2] + bias);
          float quantized4 = float(tmp >> 6);
          buf[j * 4 + 3] = std::fma(scale, quantized4, buf[j * 4 + 3] + bias);
        }
      }
      ++current;
    }

    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
#pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        buf[j] *= scale;
      }
    }
    if (std::is_same<OutType, float>::value) {
#pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        out[j] = buf[j];
      }
    } else if (std::is_same<OutType, uint16_t>::value && is_bf16_out) {
#pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        out[j] = convert_from_float_ref<uint16_t>(buf[j], true);
          //cpu_bf162float(buf[j]);
      }
    } else {
#pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        out[j] = convert_from_float_ref<uint16_t>(buf[j], false);
      }
    }
    out += output_stride;
  }
  return current == index_size;
}
namespace{
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
    float val_out_f = *(float*)&val_out * *(float*)&multiplier; // val_out * multiplier
    val_out = *(uint32_t*)&val_out_f | sign;
    output[i] = *(float*)&val_out;
  }
}
}
/// @ingroup tbe-cpu-autovec
///
/// Autovectorized version of method `EmbeddingSpMDM_ref` for FP8 weight type.
///
/// @tparam InType input data type (`uint8_t` is used)
///
/// @tparam IndexType index data type (`int64_t` is used)
///
/// @tparam OffsetType offset data type (`int32_t` is used)
///
/// @tparam OutType output data type (`float` is used)
///
///  @param block_size Number of elements in a block (`int64_t`)
///  @param output_size Number of elements in output (`int64_t`)
///  @param index_size Number of elements in index (`int64_t`)
///  @param data_size Number of elements in data (`int64_t`)
///  @param input Address of input (`InType*`)
///  @param indices Address of index (`IndexType*`)
///  @param offsets_or_lengths Address of offset (`OffsetType*`)
///  @param weights Weights of sum; optional, can be null for non-weighted sum (`float*`)
///  @param normalize_by_lengths Whether or not to normalize by lengths (`bool`)
///  @param out Address of output (`OutType*`)
///  @param is_weight_positional If `true`, weight is positional; set to `false` for FP8 autovec implementation (`bool`)
///  @param use_offsets If `true`, will use offsets instead of lengths; set to `true` for FP8 autovec implementation (`bool`)
///  @param output_stride If -1, output_stride is same as block_size; set to -1 for FP8 autovec implementation (`int64_t`)
///  @param exponent_bits Bits to use in exponent
///  @param exponent_bias Bias to use in exponent
///  @param is_bf16_out If `true`, output is `BFLOAT16` type; set to `false` for FP8 autovec implementation (`bool`)

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
  if (output_stride == -1) {
    output_stride = block_size;
  }

  vector<float> buf(block_size);

  if (input_stride == -1) {
    input_stride = block_size;
  }
  // had some weird issue so i plot it here to make sure it get referenced
 
  // more prefetch: prefetch up to 16 rows from the embedding table. Increasing
  // prefetching helps reduce backend stall and therefore enable vectorization
  // reach better of its potential. 16 is tuned for Neoverse-V2.

  // print to test
  // printf("testing autovec PEPPA PIG");

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
    do_prefetch(
        reinterpret_cast<const char*>(input + input_stride * indices[pf_idx]),
        0,
        0);
    if (input_stride > CACHE_LINE_SIZE) {
      for (int64_t offset = CACHE_LINE_SIZE; offset < input_stride;
           offset += CACHE_LINE_SIZE) {
        do_prefetch(
            reinterpret_cast<const char*>(
                input + input_stride * indices[pf_idx] + offset),
            0,
            0);
      }
    }
  }


  // Reference implementation of FP8 SLS. The algorithm is similar to FP32 SLS
  // except for the FP8->FP32 conversion after reading the embedding weight.
  int64_t current = 0;

  for (int m = 0; m < output_size; ++m) {
    memset(buf.data(), 0, sizeof(float) * block_size);
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }

    constexpr int tile_size = 4;
    #if _OPENMP >= 202011
    #pragma omp tile sizes(tile_size)
    #endif
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
      //check if each loop interation depends on one another
      // if not, approach it with parellel,
      // the code is iterating thru a dimisonals of a embedding vectory

      
      // original
      
      // #pragma omp simd
      // for (int j = 0; j < block_size; ++j) {
      //   // input stride equals the stride between different embeddings
      //   //idx is what vector is being process
      //   //j is each element of the specfic vector
      //   //input is start
      //   const uint8_t* inptr = input + input_stride * idx + j;
      //   float input_f;
      //   // Dequantize FP8 to FP32 before compute
      //   //vector time
      //   //maybe need to check if we call this function differently
      //   Float8ToFloat_ref(*inptr, &input_f, exponent_bits, exponent_bias);
        
      //   buf[j] = std::fma(w, input_f, buf[j]);
      // }

      // test 1
      // Adjust these as necessary to reflect actual batch size
      const int batch_size = block_size; // Assuming the entire block is processed at once; adjust if needed

        // Temporary buffer to hold the converted floats
      std::vector<float> converted_inputs(batch_size);

        // Perform the batch conversion
      Float8ToFloat_ref_batch(input + input_stride * idx, converted_inputs.data(), batch_size, exponent_bits, exponent_bias);

        // Now accumulate the results using vectorized operations if possible
      #pragma omp simd
        for (int j = 0; j < block_size; ++j) {
            buf[j] = std::fma(w, converted_inputs[j], buf[j]);
        } 



      ++current;
    }
    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
      //#pragma omp unroll
      #pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        buf[j] *= scale;
      }
    }

    if (std::is_same<OutType, float>::value) {
      //#pragma omp unroll
      #pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        out[j] = buf[j];
      }
    } else if (std::is_same<OutType, uint16_t>::value && is_bf16_out) {
      //#pragma omp unroll
      #pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        // out[j] = cpu_bf162float(buf[j]);
        out[j] = convert_from_float_ref<uint16_t>(buf[j], true);
      }
    } else {
      //#pragma omp unroll
      #pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        // out[j] = cpu_half2float(buf[j]);
        out[j] = convert_from_float_ref<uint16_t>(buf[j], false);
      }
    }

    out += output_stride;
  }
  return current == index_size;
}

#define INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  template FBGEMM_API bool EmbeddingSpMDMNBit_autovec(            \
      const int bit_rate,                                         \
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
      const bool is_bf16_out);                                    \
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

#define INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, OFFSET_TYPE) \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, float) \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, float16)

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
