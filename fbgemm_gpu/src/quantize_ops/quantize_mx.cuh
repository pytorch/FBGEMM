/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
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

__device__ __forceinline__ uint8_t quantize_elemwise_mx4(const float input) {
  // bits = mantissa bits + sign bit = 3
  constexpr int exp_bits = 2;
  // const RoundingMode rounding_mode = rd_away,
  // const bool saturate_normals = true;
  // const bool allow_denorm = true;
  // input won't be integers => is_int = false => remove any int decisions

  u_float_int input_;
  input_.f = input;

  // TODO: Refactor to return unsigned data
  int biased_exp = get_biased_exponent(input_);
  int sign = get_sign(input_);
  int tmant = get_trailing_mantissa(input_);

  // Mantissa bits to quantize to (remove sign)
  // const int mbits = bits - 1;
  constexpr int mbits = 2;

  constexpr int new_bias = (1 << (exp_bits - 1)) - 1;
  int new_biased_exp = biased_exp - FLOAT32_EXP_BIAS + new_bias;

  // Use exp_diff to truncate additional bits for subnorms
  // mbits includes implicit 1, so when new_biased_exp==0
  // we want exp_diff = 1 to truncate away 1 bit
  int exp_diff = (new_biased_exp <= 0) ? 1 - new_biased_exp : 0;
  exp_diff = (exp_diff > FLOAT32_FULL_MBITS) ? FLOAT32_FULL_MBITS : exp_diff;
  bool is_subnorm = biased_exp == 0;

  // Shift down and round mantissa, allow overflow except for integers
  // This converts tmant into a full mantissa
  shift_right_round_mantissa(
      tmant,
      is_subnorm,
      mbits,
      exp_diff,
      rd_away,
      /*!is_int=*/true);

  if (tmant == 0) {
    return 0.0;
  }

  // Check for overflow.
  constexpr int overflow_threshold = 4;
  bool overflow = tmant >= overflow_threshold;
  // Allow subnorms to overflow into normals, otherwise shift away overflow.
  if (overflow && (!is_subnorm)) {
    tmant = tmant >> 1;
  }
  // Special case where a value is subnormal and has a large mantissa, overflow.
  constexpr int subnormal_round = 2;
  if (new_biased_exp <= 0 && tmant == subnormal_round) {
    new_biased_exp = 1;
  }
  // Remove implicit 1.
  constexpr int implicit_mask = 0x1;
  tmant = tmant & implicit_mask;
  // Add overflow to exponent.
  if (overflow) {
    new_biased_exp = new_biased_exp + 1;
  }
  // If exponent overflows, set mantissa to max value (equivalent to clamp).
  if (new_biased_exp >= overflow_threshold) {
    tmant = 1;
  }
  // Explicitly clamp new exponent.
  constexpr int max_mx4_exp = 3;
  new_biased_exp = std::max(0, std::min(new_biased_exp, max_mx4_exp));
  return construct_fp4(sign, new_biased_exp, tmant);
}

//-----------------------------------------------------------------------
// quantize float to mx4 kernel
//-----------------------------------------------------------------------

// TO DO: make flush_fp32_subnorms template argument
template <typename T, bool has_multiple_warps_in_group>
__global__ void quantize_float_to_mx4_kernel(
    const pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> input,
    const int group_size,
    const uint32_t total_elems,
    // const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> output,
    const uint32_t num_warps_in_group,
    const uint32_t smem_stride) {
  const auto linear_group_id = (blockIdx.x * blockDim.y) + threadIdx.y;
  const auto linear_tid = linear_group_id * group_size + threadIdx.x;
  if (linear_tid >= total_elems)
    return;

  // MX4 values
  constexpr int scale_bits = 8;
  constexpr int elem_emax = 2;

  const T elem = input[linear_tid];

  extern __shared__ __align__(16) float smem[];
  const uint32_t group_offset_in_block = threadIdx.y * smem_stride;
  // set smem base address for each group
  int* smem_base = reinterpret_cast<int*>(smem + group_offset_in_block);

  // // allreduce to get the max value in each group size
  int shared_exp = get_biased_exponent(elem);

  const uint32_t half_group_size = group_size / 2;

  // find max shared_exp for each warp
  for (uint32_t mask = half_group_size; mask > 0; mask /= 2) {
    int temp_shared_exp = fbgemm_gpu::shfl_xor(shared_exp, mask);
    shared_exp = max(temp_shared_exp, shared_exp);
  }

  // find max shared_exp between warps in the group
  if (has_multiple_warps_in_group) {
    const uint32_t rep_tid = threadIdx.x / 32;
    const bool is_rep_tid = (threadIdx.x % 32 == 0);
    // put the max shared exp of each warp in shared memory
    if (is_rep_tid) {
      smem_base[rep_tid] = shared_exp;
    }
    __syncthreads();

    // find max shared_exp across warps in the group
    // let thread `i` store max shared_exp of warp `i`
    if (threadIdx.x < num_warps_in_group) {
      shared_exp = smem_base[threadIdx.x];
    }
    fbgemm_gpu::syncwarp();
    // find max shared_exp
    for (uint32_t s = num_warps_in_group; s > 0; s /= 2) {
      int temp_shared_exp = fbgemm_gpu::shfl_xor(shared_exp, s);
      shared_exp = max(temp_shared_exp, shared_exp);
    }

    // strore shared_exp in shared_mem
    if (threadIdx.x == 0) {
      *smem_base = shared_exp;
    }
    __syncthreads();

    // representative thread in each warp in the group reads the max
    // shared_memory
    if (is_rep_tid) {
      shared_exp = *smem_base;
    }
    // broadcast max shared_exp to every thread
    shared_exp =
        fbgemm_gpu::shfl_sync(shared_exp, 0, WARP_SIZE, FULL_WARP_MASK);
  }

  // Offset shared exponent by elem_emax, preserve NaNs
  shared_exp =
      (shared_exp != FLOAT32_EXP_MAX) ? shared_exp - elem_emax : shared_exp;

  // Clamp to scale_bits range
  const uint8_t clamped_shared_exp = clamp_shared_exp(shared_exp, scale_bits);

  // TODO: Update flush_fp32_subnorms to be template argument
  // const bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

  // Divided by 2^shared_exp
  // const T scaled_in = flush_tile
  //     ? 0
  //     : scalbn(elem, (clamped_shared_exp - FLOAT32_EXP_BIAS) * -1);
  const T scaled_in =
      scalbn(elem, (clamped_shared_exp - FLOAT32_EXP_BIAS) * -1);

  // convert to FP4 -> there is 16 possible values
  // bit pattern of the uint8 result is `0000 xxxx`
  const uint8_t quantized_val = quantize_elemwise_mx4(scaled_in);

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

  // Let each thread write 1 byte of output data
  if (threadIdx.x < half_group_size) {
    // write data output using uint8_t (1 bytes)

    uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(smem_base);
    const uint32_t start_output_idx = (half_group_size + 1) * linear_group_id;
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
    const int64_t total_quant_elems,
    pta::PackedTensorAccessor64<T, 1, at::RestrictPtrTraits> output) {
  const auto linear_group_id = (blockIdx.x * blockDim.y) + threadIdx.y;
  const auto linear_tid = linear_group_id * group_size + threadIdx.x;
  if (linear_tid >= total_quant_elems)
    return;

  const uint32_t sub_group_size = group_size / 4;
  const uint32_t half_group_size = group_size / 2;
  const uint32_t output_idx = linear_tid * 4;
  const uint32_t group_id = uint32_t(linear_tid / sub_group_size);
  const uint32_t start_offset = linear_tid * 2 + group_id;

  const uint8_t elem = input[start_offset];
  const uint8_t elem2 = input[start_offset + 1];
  // shared_exp is stored at [half_group_size + offset]
  // e.g., if group_size=32, data size per group is 16+1=17
  // shared_exp_indices are 16, 16+17=33, 16+34=50, ...
  const uint32_t shared_exp_idx =
      half_group_size + ((half_group_size + 1) * group_id);
  const uint8_t shared_exp = input[shared_exp_idx];

  constexpr uint8_t upper_4bit_mask = 0xF0;
  constexpr uint8_t lower_4bit_mask = 0x0F;

  // each threads takes care of 8 bits
  // first 4 bits
  const uint8_t high_1 = (elem & lower_4bit_mask);
  const uint8_t low_1 = (elem & upper_4bit_mask) >> 4;
  const uint8_t high_2 = (elem2 & lower_4bit_mask);
  const uint8_t low_2 = (elem2 & upper_4bit_mask) >> 4;

  // last 4 bits

  // CUDA_KERNEL_ASSERT(low < 16 && elem < 16 && low_2 < 16 && elem2 < 16);

  float4* output_ptr = reinterpret_cast<float4*>(&output[output_idx]);
  const int exp = shared_exp - FLOAT32_EXP_BIAS;
  float4 deq;
  deq.x = scalbn(MX4_values[low_1], exp);
  deq.y = scalbn(MX4_values[high_1], exp);
  deq.z = scalbn(MX4_values[low_2], exp);
  deq.w = scalbn(MX4_values[high_2], exp);
  *output_ptr = deq;
}

#endif
