/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_MX_CUH
#define PYT_MX_MX_CUH

#include "mx_common.cuh"

//-----------------------------------------------------------------------
// quantize implementation float to fp4
// For MX4, input float to the function must have already been scaled by
// the shared exponent. Output will be 4-bit stored in 8-bit data type
// The 4-bit output is mapped to 16 values for dequantization
//-----------------------------------------------------------------------

__device__ __forceinline__ uint8_t quantize_elemwise_4bit(
    const float input,
    const int bits, // bits = mantissa bits + sign bit
    const int exp_bits, // exp_bits == 0 indicates integer dtype
    const float max_norm,
    const RoundingMode rounding_mode = rd_away,
    const bool saturate_normals = false,
    const bool allow_denorm = true) {
  u_float_int input_;
  input_.f = input;

  // TODO: Refactor to return unsigned data
  int biased_exp = get_biased_exponent(input_);
  int sign = get_sign(input_);
  int tmant = get_trailing_mantissa(input_);

  // Mantissa bits to quantize to (remove sign)
  const int mbits = bits - 1;
  const bool is_int = exp_bits == 0;

  // Integers can be treated has having exp bias of 1
  const int new_bias = is_int ? 1 : (1 << (exp_bits - 1)) - 1;
  int new_biased_exp = biased_exp - FLOAT32_EXP_BIAS + new_bias;

  // Skip denorms
  if ((!is_int) && (!allow_denorm) && (new_biased_exp < 1)) {
    return 0.0;
  }

  // Use exp_diff to truncate additional bits for subnorms
  // mbits includes implicit 1, so when new_biased_exp==0
  // we want exp_diff = 1 to truncate away 1 bit
  int exp_diff = (new_biased_exp <= 0) ? 1 - new_biased_exp : 0;
  exp_diff = (exp_diff > FLOAT32_FULL_MBITS) ? FLOAT32_FULL_MBITS : exp_diff;

  // Shift down and round mantissa, allow overflow except for integers
  // This converts tmant into a full mantissa
  shift_right_round_mantissa(
      tmant, biased_exp == 0, mbits, exp_diff, rounding_mode, !is_int);

  if (tmant == 0) {
    return 0.0;
  }

  // Shift back up to restore mantissa
  // This converts back to a trailing mantissa
  const bool overflow =
      shift_left_mantissa(tmant, biased_exp == 0, mbits, exp_diff);
  if (overflow) {
    biased_exp = biased_exp + 1;
    new_biased_exp = new_biased_exp + 1;
  }

  // Reconstruct float number
  const float output = construct_float(sign, biased_exp, tmant);

  /* Convert float to MX4 encodings:
    bits  FP4     [int4 lookup]
                    +  - (sign)
    S000 = 0    <=> 0  8
    S001 = 0.5  <=> 1  9
    S010 = 1    <=> 2  10
    S011 = 1.5  <=> 3  11
    S100 = 2.0  <=> 4  12
    S101 = 3.0  <=> 5  13
    S110 = 4.0  <=> 6  14
    S111 = 6.0  <=> 7  15
  */

  // construct the 4 bit using 1-bit sign, 2-bit new_exp 1-bit tmant
  // |0.5f| is the exception since it has tmant of 0 instead of 1
  // return the lookup value
  if (output == 0.5f) {
    return 1; // bits 0001
  } else if (output == -0.5f) {
    return 9; // bits 1001
  }

  // Return Inf if rounded value is out of bounds,
  // unless target format is integer or saturate_normals==True
  if (abs(output) > max_norm) {
    if (is_int || saturate_normals) {
      // max norm = 6.0f => bias=3, tmant = 1, sign remains the same
      new_biased_exp = 3;
      tmant = 4194304; // bit 10000000000000000000000
    } else {
      // TODO: set Inf for 4 bit for other patterns
      new_biased_exp = 0xFF;
      tmant = 0;
      // e2m1 has no inf
      CUDA_KERNEL_ASSERT(false);
    }
  }
  CUDA_KERNEL_ASSERT(new_biased_exp >= 0 && new_biased_exp <= 3);
  return construct_fp4(sign, new_biased_exp, tmant);
}

//-----------------------------------------------------------------------
// quantize float to mx4 kernel
//-----------------------------------------------------------------------

template <typename T>
__global__ void quantize_float_to_mx4_kernel(
    const pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> input,
    const int group_size, // can change to Blockdim.x
    const uint32_t total_elems,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> output) {
  const auto linear_group_id = (blockIdx.x * blockDim.y) + threadIdx.y;
  const auto linear_tid = linear_group_id * group_size + threadIdx.x;
  if (linear_tid >= total_elems)
    return;

  // MX4 values
  constexpr int scale_bits = 8;
  constexpr int elem_ebits = 2;
  constexpr int elem_mbits = 3;
  constexpr float elem_max_norm = 6.0;
  constexpr int elem_emax = 2;

  const T elem = input[linear_tid];

  extern __shared__ __align__(16) float smem[];
  const uint32_t group_offset_in_block = threadIdx.y * group_size;
  // set smem base address for each group
  int* smem_base = reinterpret_cast<int*>(smem + group_offset_in_block);

  // // allreduce to get the max value in each group size
  int shared_exp = get_biased_exponent(elem);
  smem_base[threadIdx.x] = shared_exp;
  __syncthreads();

  const uint32_t half_group_size = group_size / 2;

  for (uint32_t s = half_group_size; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smem_base[threadIdx.x] =
          max(smem_base[threadIdx.x], smem_base[threadIdx.x + s]);
    }
    __syncthreads();
  }
  // get shared_exponent stored at tid = 0
  shared_exp = smem_base[0];

  // Offset shared exponent by elem_emax, preserve NaNs
  shared_exp =
      (shared_exp != FLOAT32_EXP_MAX) ? shared_exp - elem_emax : shared_exp;

  // Clamp to scale_bits range
  const uint8_t clamped_shared_exp = clamp_shared_exp(shared_exp, scale_bits);

  const bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

  // Divided by 2^shared_exp
  const T scaled_in =
      flush_tile ? 0 : elem / pow(2.0f, clamped_shared_exp - FLOAT32_EXP_BIAS);

  // convert to FP4 -> there is 16 possible values
  // bit pattern of the uint8 result is `0000 xxxx`
  const uint8_t quantized_val = quantize_elemwise_4bit(
      scaled_in,
      elem_mbits,
      elem_ebits,
      elem_max_norm,
      rounding_mode,
      true,
      true);

  // Store 2 `quantized_val` in one uint8
  // let even threads store their 4-bit quantized_val on the left (position 4-7)
  // and odd threads store on right (position 0-3)
  uint8_t* stored_8bit =
      reinterpret_cast<uint8_t*>(smem_base) + (threadIdx.x / 2);
  const auto is_even_tid = threadIdx.x % 2 == 0;

  // even thread work on the left side
  if (is_even_tid) {
    // the 4 bits are on the rightmost, need to shift 4 bit
    // this becomes `xxxx 0000`
    *stored_8bit = (quantized_val << 4);
  }
  __syncthreads();

  // odd threads work on the right side (position 0-3)
  if (!is_even_tid) {
    // 4 bits are already on the rightmost, so just `bitwise OR` to combine
    *stored_8bit |= quantized_val;
  }
  __syncthreads();

  const uint32_t data_size_per_group = half_group_size + 1;

  // Let each thread write 1 byte of output data
  if (threadIdx.x < half_group_size) {
    // write data output using uint8_t (1 bytes)

    uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(smem_base);
    const uint32_t start_output_idx = (data_size_per_group)*linear_group_id;
    uint8_t* output_base = &output[start_output_idx];

    output_base[threadIdx.x] = smem_ptr[threadIdx.x];

    // write share exp
    if (threadIdx.x == 0) {
      // shared_exp_idx is stored after data
      // need to offset with start_output + output data
      output_base[half_group_size] = clamped_shared_exp;
    }
  }
}

//-----------------------------------------------------------------------
// quantize mx4 to float kernel
//-----------------------------------------------------------------------

template <typename T>
__global__ void dequantize_mx4_to_float_kernel(
    const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> input,
    const int group_size,
    const int64_t total_elems,
    pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> output) {
  const auto linear_group_id = (blockIdx.x * blockDim.y) + threadIdx.y;
  const auto linear_tid = linear_group_id * group_size + threadIdx.x;
  if (linear_tid >= total_elems)
    return;

  const uint32_t half_group_size = group_size / 2;
  const uint32_t data_size_per_group = half_group_size + 1;

  const uint32_t start_output_idx = (data_size_per_group)*linear_group_id;
  uint8_t elem = input[start_output_idx + (threadIdx.x / 2)];
  const uint32_t shared_exp_idx = start_output_idx + half_group_size;
  const uint8_t shared_exp = input[shared_exp_idx];

  constexpr uint8_t upper_4bit_mask = 0xF0;
  constexpr uint8_t lower_4bit_mask = 0x0F;
  // even threads take care of 4 bit on the left
  // odd threads, the right
  if (uint32_t(threadIdx.x % 2) == 0) {
    // shift the 4-bit to the end for even threads
    elem = (elem & upper_4bit_mask);
    elem = (elem >> 4);
  } else {
    elem = (elem & lower_4bit_mask);
  }
  CUDA_KERNEL_ASSERT(elem < 16);

  output[linear_tid] = MX4_values[elem] * pow(2, shared_exp - FLOAT32_EXP_BIAS);
}

#endif
