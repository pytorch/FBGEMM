/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define QUANTIZE_OPS_MAX(a, b) ((a) > (b) ? (a) : (b))
#define QUANTIZE_OPS_MIN(a, b) ((a) < (b) ? (a) : (b))

__global__ void _get_8bit_qparam_cuda_kernel(
    const float* __restrict__ input,
    int nrows,
    int ncols,
    uint8_t* __restrict__ output,
    float* __restrict__ range_list) {
  int row = (int)blockIdx.x * blockDim.x + threadIdx.x;
  int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  int output_columns = ncols_aligned + 2 * sizeof(float);

  if (row < nrows) {
    const float* input_row = input + row * ncols;
    float* output_row_qparams =
        reinterpret_cast<float*>(output + row * output_columns + ncols_aligned);

    // Option 1: CUB:
    // https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html, search
    // max-reduction
    // Option 2: thrust
    // TODO: Benchmark CUB vs. thrust
    float minimum_element =
        *thrust::min_element(thrust::device, input_row, input_row + ncols);
    float maximum_element =
        *thrust::max_element(thrust::device, input_row, input_row + ncols);
    float range = maximum_element - minimum_element;

    output_row_qparams[0] = range / 255.0f;
    output_row_qparams[1] = minimum_element;
    range_list[row] = range;
  }
}

__global__ void _compute_8bit_quantize_cuda_kernel(
    const float* const __restrict__ input,
    const float* const __restrict__ range_list,
    const int nrows,
    const int ncols,
    std::uint8_t* const __restrict__ output) {
  constexpr float kEpsilon = 1e-8f;

  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (col < ncols) {
      // load scale, bias
      float* row_qparams = reinterpret_cast<float*>(
          output + row * output_columns + ncols_aligned);
      float bias = row_qparams[1];

      int input_idx = row * ncols + col;
      uint8_t* output_addr = output + row * output_columns + col;
      // TODO: lift range_list into shared memory. However, when nrows is large,
      // it might exceed the size of shared memory.
      const auto inverse_scale = 255.0f / (range_list[row] + kEpsilon);
      output_addr[0] = std::lrintf((input[input_idx] - bias) * inverse_scale);
    }
  }
}

// FP32 -> Fused 8-bit rowwise kernel
__global__ void _float_to_fused8bitrowwise_cuda_kernel(
    const float* __restrict__ input,
    int nrows,
    int ncols,
    std::uint8_t* __restrict__ output) {
  constexpr float kEpsilon = 1e-8f;

  int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  int output_columns = ncols_aligned + 2 * sizeof(float);

  int64_t row = (int)blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nrows) {
    const float* input_row = input + row * ncols;
    std::uint8_t* output_row = output + row * output_columns;
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + ncols_aligned);

    float minimum_element =
        *thrust::min_element(thrust::device, input_row, input_row + ncols);
    float maximum_element =
        *thrust::max_element(thrust::device, input_row, input_row + ncols);
    float range = maximum_element - minimum_element;

    output_row_scale_bias[0] = range / 255.0f;
    output_row_scale_bias[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    for (std::size_t col = 0; col < ncols; ++col) {
      output_row[col] =
          std::lrintf((input_row[col] - minimum_element) * inverse_scale);
    }
  }
}

// Fused 8-bit rowwise -> FP32 kernel
__global__ void _fused8bitrowwise_to_float_cuda_kernel(
    const std::uint8_t* const __restrict__ input,
    const int nrows,
    const int ncols,
    float* const __restrict__ output) {
  const int output_columns = ncols - 2 * sizeof(float);

  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (col < output_columns) {
      const std::uint8_t* input_row = input + row * ncols;
      const float* input_row_scale_bias =
          reinterpret_cast<const float*>(input_row + output_columns);
      float* output_row = output + row * output_columns;

      output_row[col] =
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
    }
  }
}

// Fake 8-bit quantize kernel: FP32 -> UINT8 rowwise -> FP32
__global__ void _fake_8bit_quantize_cuda_kernel(
    const float* __restrict__ input,
    int nrows,
    int ncols,
    float* __restrict__ output) {
  constexpr float kEpsilon = 1e-8f;
  int row = (int)blockIdx.x * blockDim.x + threadIdx.x;
  int col = (int)blockIdx.y * blockDim.y + threadIdx.y;

  if (row < nrows && col < ncols) {
    const float* input_row = input + row * ncols;
    float* output_row = output + row * ncols;
    float minimum_element =
        *thrust::min_element(thrust::device, input_row, input_row + ncols);
    float maximum_element =
        *thrust::max_element(thrust::device, input_row, input_row + ncols);
    float range = maximum_element - minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    std::uint8_t quantized_val =
        std::lrintf((input_row[col] - minimum_element) * inverse_scale);
    output_row[col] = quantized_val * (range / 255.0f) + minimum_element;
  }
}

// FP32 -> Fused 4/2-bit rowwise kernel
__global__ void _float_to_fusednbitrowwise_cuda_kernel(
    int bit_rate,
    const float* __restrict__ input,
    int nrows,
    int ncols,
    std::uint8_t* __restrict__ output) {
  int num_elem_per_byte = 8 / bit_rate;
  int output_columns =
      (ncols + num_elem_per_byte - 1) / num_elem_per_byte + 2 * sizeof(__half);

  int row = (int)blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nrows) {
    const float* input_row = input + row * ncols;
    std::uint8_t* output_row = output + row * output_columns;
    __half* output_row_scale_bias = reinterpret_cast<__half*>(
        output_row + (ncols + num_elem_per_byte - 1) / num_elem_per_byte);

    float minimum_element =
        *thrust::min_element(thrust::device, input_row, input_row + ncols);
    float maximum_element =
        *thrust::max_element(thrust::device, input_row, input_row + ncols);

    minimum_element = __half2float(__float2half(minimum_element));
    const float range = maximum_element - minimum_element;

    float scale = __half2float(
        __float2half(range == 0 ? 1.0f : range / ((1 << bit_rate) - 1)));
    if (scale == 0) {
      // Corner case handling when maximum_element == minimum_element
      // Any scale would work because X - minimum_element will be 0 for all X
      scale = 1.0f;
    }
    float inverse_scale = 1.0f / scale;
    if (std::isinf(inverse_scale)) {
      scale = 1.0f;
      inverse_scale = 1.0f;
    }

    output_row_scale_bias[0] = __float2half(scale);
    output_row_scale_bias[1] = __float2half(minimum_element);
    for (std::size_t col = 0; col < ncols; ++col) {
      float X = input_row[col];

      std::uint8_t quantized = QUANTIZE_OPS_MAX(
          0,
          QUANTIZE_OPS_MIN(
              static_cast<int>(
                  std::lrintf((X - minimum_element) * inverse_scale)),
              static_cast<int>((1 << bit_rate) - 1)));

      if (col % num_elem_per_byte == 0) {
        output_row[col / num_elem_per_byte] = quantized;
      } else {
        output_row[col / num_elem_per_byte] |=
            (quantized << ((col & (num_elem_per_byte - 1)) * bit_rate));
      }
    }
  }
}

// Fused 4/2-bit rowwise -> FP32 kernel
__global__ void _fusednbitrowwise_to_float_cuda_kernel(
    const int bit_rate,
    const std::uint8_t* input,
    const int nrows,
    const int ncols,
    float* const output) {
  const int num_elem_per_byte = 8 / bit_rate;
  const int output_columns = (ncols - 2 * sizeof(__half)) * num_elem_per_byte;

  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (row < nrows && col < output_columns) {
      const std::uint8_t* input_row = input + row * ncols;
      const __half* input_row_scale_bias = reinterpret_cast<const __half*>(
          input_row +
          (output_columns + num_elem_per_byte - 1) / num_elem_per_byte);
      float scale = __half2float(input_row_scale_bias[0]);
      float bias = __half2float(input_row_scale_bias[1]);
      float* output_row = output + row * output_columns;

      std::uint8_t quantized = input_row[col / num_elem_per_byte];
      quantized >>= (col % num_elem_per_byte) * bit_rate;
      quantized &= (1 << bit_rate) - 1;
      output_row[col] = scale * quantized + bias;
    }
  }
}

// FP32 -> BF16 kernel
__global__ void _float_to_bfloat16_cuda_kernel(
    const float* __restrict__ input,
    const int nrows,
    const int ncols,
    uint16_t* __restrict__ output) {
  int row = (int)blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nrows) {
    const float* input_row = input + row * ncols;
    uint16_t* output_row = output + row * ncols;
    for (std::size_t col = 0; col < ncols; ++col) {
      // Add 2^15 and right shift 16 to do round-nearest
      output_row[col] =
          (*reinterpret_cast<const uint32_t*>(input_row + col) + (1 << 15)) >>
          16;
    }
  }
}

// BF16 -> FP32 kernel
__global__ void _bfloat16_to_float_cuda_kernel(
    const uint16_t* __restrict__ input,
    const int nrows,
    const int ncols,
    float* __restrict__ output) {
  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (col < ncols) {
      const uint16_t* input_row = input + row * ncols;
      float* output_row = output + row * ncols;
      uint32_t val_fp32 = static_cast<uint32_t>(
                              reinterpret_cast<const uint16_t*>(input_row)[col])
          << 16;
      reinterpret_cast<uint32_t*>(output_row)[col] = val_fp32;
    }
  }
}

#undef QUANTIZE_OPS_MAX
#undef QUANTIZE_OPS_MIN
