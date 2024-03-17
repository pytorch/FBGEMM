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
//libaray check if works

#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FbgemmConvert.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <thread>

using std::vector;

namespace fbgemm {

// void Float8ToFloat_Autovec(
//         const unit8_t* input, //a pointer for mutiple inputs
//         float* output,
//         int exponent_bits,
//         int exponent_bias, 
//         size_t num_elements) { //number of elements to work on
//         simd_vector<unit8_t> vInput;
//         simd_vector<uint32_t> vValOut, Vsign, vMultipler;
//         simd_vector<float> vOutput;

//         vMultiplier = simd_set1_uint32((127 + (127 - exponent_bias)) << 23);

//         for (size_t i = 0; i < num_elements; i += SIMD_WIDTH) {
//           //load the vector of inputs
//           vInput = simd_loadu_uint8(input + i);

//           //opertation but vectorized
//           vSign = simd_s11_uint32(simd_and_uint8(vInput, 0*80), 24);
//           vValOut = simd_s11_uint32(simd_and_uint8(vInput, 0*7f), (24 - (8 - exponent_bias)));

//           //apply the multiplier and adjust sign 
//           vOutput = simd_mul_float(simd_to_float(vValOut), simd_to_float(vMultipler));
//           vOutput = simd_or_uint32(vOutput, vSign);

//           //store result back to output array
//           simd_storeu_float(output + i, vOutput)
//         } 
//         }

  // void Float8ToFloat_Autovec(
  //     const vector<uint8_t>& input,
  //     float* output,
  //     int exponent_bits,
  //     int exponent_bias,
  //     int chunk_size) {
        
        
  //    for (int i = 0; i < chunk_size; ++i) {
  //       fint32 val_out, sign, multiplier;

  //       sign.I = (input[i] & 0x80) << 24;
  //       val_out.I = (input[i] & 0x7F) << (24 - (8 - exponent_bits));
  //       // so that the mantissa bits start at the mantissa bit positions of FP32
  //       // encoding

  //       // Let the hfp8 mantissa bits correspond to the value frac, 0 <= frac < 1
  //       // So if the hfp8 value is a normal number, it's value is 2^e x (1+frac)
  //       // where e is its (true, unbiased) exponent
  //       // If the hfp8 value is denormal, the value is 2^(1-bias) x frac

  //       // However, the bit pattern in the 8-bit exponent field of val_out.F
  //       // is bias+e when hfp8 is normal, and 0 when hfp8 is subnormal.
  //       // So, as an FP32 value, when hfp8 is normal, val_out.F represents the value
  //       // of 2^(bias+e-127) * (1+frac)
  //       // And when hfp8 is subnormal, val_out.F is also subnormal, and represents the
  //       // value of 2^(-126) * frac In either case, val_out.F corresponds to
  //       // 2^(bias-127) * (value of hfp8 input) Thus, if we multiply val_out.F by
  //       // 2^(127-bias), we obtain the hfp8 value as an FP32 number

  //       multiplier.I = (127 + (127 - exponent_bias))
  //           << 23; // multiplier.F is 2^(127-bias)
  //       val_out.F *= multiplier.F;
  //       val_out.I |= sign.I;
  //       output[i] = val_out.F;
  //   }
  // }


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

  printf("test autovec int2 int4");

  // block_size is the number of elements and fused_block_size is the size of
  // an entire row, including scale and bias.
  const auto scale_bias_offset = 2 * sizeof(float16);
  if (input_stride == -1) {
    input_stride = div_up(block_size, num_elem_per_byte) + scale_bias_offset;
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
    constexpr int tile_size = 4;
#if _OPENMP >= 202011
#pragma omp tile sizes(tile_size)
#endif
    for (int i = 0; i < len; ++i) {
      int64_t idx = indices[current];
      int64_t prefetch_idx =
          indices[std::min(current + tile_size, index_size - 1)];
      do_prefetch(
          reinterpret_cast<const char*>(input + input_stride * prefetch_idx),
          1);
      if (idx < 0 || idx >= data_size) {
        return false;
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
      if (bit_rate == 4) {
        const size_t halfbufsz = (block_size + 1) / 2;
        assert(halfbufsz > 0);
        for (size_t j = 0; j < halfbufsz; ++j) {
          uint8_t tmp = input[offset + j];
          float quantized1 = float(tmp & 0xf);
          float quantized2 = float(tmp >> 4);
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
        out[j] = cpu_bf162float(buf[j]);
      }
    } else {
#pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        out[j] = cpu_half2float(buf[j]);
      }
    }
    out += output_stride;
  }
  return current == index_size;
}


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

  // more prefetch: prefetch up to 16 rows from the embedding table. Increasing
  // prefetching helps reduce backend stall and therefore enable vectorization
  // reach better of its potential. 16 is tuned for Neoverse-V2.

  // print to test
  printf("testing autovec");

  constexpr int64_t max_initial_prefetch_rows = 16;
  const int64_t prefetch_stride = std::min(max_initial_prefetch_rows, index_size);
  for (int pf_idx = 0; pf_idx < prefetch_stride; ++pf_idx) {
    do_prefetch(
        reinterpret_cast<const char*>(input + input_stride * indices[pf_idx]),
        0,
        0);
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

      float w = 1.f;
      if (weights) {
        w = weights[is_weight_positional ? i : current];
      }
      //check if each loop interation depends on one another
      // if not, approach it with parellel,
      // the code is iterating thru a dimisonals of a embedding vectory

      // original
      
      #pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        // input stride equals the stride between different embeddings
        //idx is what vector is being process
        //j is each element of the specfic vector
        //input is start
        const uint8_t* inptr = input + input_stride * idx + j;
        float input_f;
        // Dequantize FP8 to FP32 before compute
        //vector time
        //maybe need to check if we call this function differently
        Float8ToFloat_ref(*inptr, &input_f, exponent_bits, exponent_bias);
        
        buf[j] = std::fma(w, input_f, buf[j]);
      }
      
      // int block_width;
      // if (block_size % 1 == 0) {
      //   block_width = 1

      // } else if (block_size % 2 == 0) { 
      //   block_width = 2
      // }
      // else if (block_size % 3 == 0) { 
      //   block_width = 3
      
      // }
      
      // for (int j = 0; j < block_size; j+=block_width) {
      //   // input stride equals the stride between different embeddings
      //   //idx is what vector is being process
      //   //j is each element of the specfic vector
      //   //input is start
      //   const uint8_t* inptr1 = input + input_stride * idx + j;
      //   const uint8_t* inptr2 = input + input_stride * idx + (j+1);
      //   const uint8_t* inptr3 = input + input_stride * idx + (j+2);
      //   const uint8_t* inptr4 = input + input_stride * idx + (j+3);
      //   vector<uint8_t> inptr;
        
        
      //   float input_f;
      //   // Dequantize FP8 to FP32 before compute
      //   //vector time
      //   //maybe need to check if we call this function differently
      //   Float8ToFloat_Autovec(inptr, &input_f, exponent_bits, exponent_bias, block_width);
        
      //   buf[j] = std::fma(w, input_f, buf[j]);
      // }

      
      

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
        out[j] = cpu_bf162float(buf[j]);
      }
    } else {
      #pragma omp simd
      for (int j = 0; j < block_size; ++j) {
        out[j] = cpu_half2float(buf[j]);
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
      const bool is_bf16_out);

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
