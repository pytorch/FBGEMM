/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/TensorIterator.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#ifndef __HIP_PLATFORM_HCC__
#include <math_constants.h>
#endif

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/quantize_ops.cuh"
#include "fbgemm_gpu/quantize_ops_utils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

using Tensor = at::Tensor;

/// @defgroup quantize-data-cuda Quantization Data CUDA Operators
/// The following are CUDA Operators

namespace fbgemm_gpu {
///@ingroup quantize-data-cuda
at::Tensor _float_to_bfloat16_gpu(const at::Tensor& input) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  // TODO: replace Half by BFloat16, after BFloat16 is supported by Nvidia
  // NCCL input.options().dtype(at::kBFloat16)); // at::kBFloat16
  auto output = at::empty({}, input.options().dtype(at::kHalf));
  output.resize_(0);

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();
  at::native::gpu_kernel(iter, [] GPU_LAMBDA(float in) -> at::Half {
    fbgemm_gpu::fint32 temp;
    temp.F = in;
    return at::Half((temp.I + (1 << 15)) >> 16, at::Half::from_bits());
  });

  return output;
}

///@ingroup quantize-data-cuda
at::Tensor _bfloat16_to_float_gpu(const at::Tensor& input) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  auto output = at::empty({}, input.options().dtype(at::kFloat));
  output.resize_(0);
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();

  at::native::gpu_kernel(iter, [] GPU_LAMBDA(at::Half in) -> float {
    fbgemm_gpu::fint32 temp;
    temp.I = in.x << 16;
    return temp.F;
  });
  return output;
}

namespace {

// FP32/FP16 -> Fused 8-bit rowwise kernel
template <typename input_t>
__global__ inline void _float_to_fused8bitrowwise_cuda_kernel(
    const input_t* __restrict__ input,
    const int nrows,
    const int ncols,
    std::uint8_t* __restrict__ output) {
  constexpr float kEpsilon = 1e-20f;

  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  const int64_t row = (int)blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nrows) {
    const input_t* input_row = input + row * ncols;
    std::uint8_t* output_row = output + row * output_columns;
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + ncols_aligned);

    const float minimum_element = fbgemm_gpu::min(input_row, input_row + ncols);
    const float maximum_element = fbgemm_gpu::max(input_row, input_row + ncols);
    const float range = maximum_element - minimum_element;

    output_row_scale_bias[0] = range / 255.0f;
    output_row_scale_bias[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    for (std::size_t col = 0; col < ncols; ++col) {
      output_row[col] =
          lrintf((input_row[col] - minimum_element) * inverse_scale);
    }
  }
}

template <typename T>
__device__ inline __attribute__((always_inline)) T
quantize_ops_shfl_xor(const T val, int laneMask, int width) {
#if defined(__HIP_PLATFORM_HCC__) || CUDA_VERSION < 9000
  return __shfl_xor(val, laneMask, width);
#else
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#endif
}

// FP32/FP16 -> FP8 rowwise kernel
template <typename input_t>
__global__ inline void _float_to_FP8rowwise_cuda_kernel(
    const input_t* __restrict__ input,
    const int nrows,
    const int ncols,
    std::uint8_t* __restrict__ output,
    const bool forward) {
  constexpr float kEpsilon = 1e-20f;
  const int ebit = forward ? 4 : 5;
  const int bias = forward ? 15 : 31;
  const float max_pos = forward ? 0.9375 : 0.875;

  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  const int64_t row = (int)blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nrows) {
    const input_t* input_row = input + row * ncols;
    std::uint8_t* output_row = output + row * output_columns;
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + ncols_aligned);

    const float minimum_element = fbgemm_gpu::min(input_row, input_row + ncols);
    const float maximum_element = fbgemm_gpu::max(input_row, input_row + ncols);

    const auto scale =
        max_pos / (kEpsilon + fmaxf(maximum_element, -minimum_element));
    output_row_scale_bias[0] = scale;
    for (std::size_t col = 0; col < ncols; ++col) {
      output_row[col] =
          float_to_hfp8(input_row[col] * scale, ebit, bias, max_pos);
    }
  }
}
// FP32/FP16 -> FP8 rowwise kernel
template <typename input_t>
__global__ inline void _float_to_paddedFP8rowwise_cuda_kernel(
    const input_t* __restrict__ input,
    const int nrows,
    const int ncols,
    std::uint8_t* __restrict__ output,
    const bool forward,
    const int row_dim) {
  constexpr float kEpsilon = 1e-20f;
  const int ebit = forward ? 4 : 5;
  const int bias = forward ? 15 : 31;
  const float max_pos = forward ? 0.9375 : 0.875;

  const int ncols_aligned = (ncols + row_dim - 1) / row_dim * row_dim;
  int pad = ncols_aligned - ncols;
  const int output_columns =
      ncols_aligned + (ncols + row_dim - 1) / row_dim * 8;

  const int64_t row = (int)blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nrows) {
    const input_t* input_row = input + row * ncols;
    std::uint8_t* output_row = output + row * output_columns;
    for (int col = 0; col < ncols; col += row_dim) {
      int col_offset = col / row_dim * 8;
      int last_buc_idx = (ncols - col) / row_dim *
          -1; // negative suggest it's an indice offset
      float* output_row_scale =
          reinterpret_cast<float*>(output_row + col + col_offset + row_dim);
      int buc_end = (row_dim < ncols - col) ? row_dim : ncols - col;
      float minimum_element =
          fbgemm_gpu::min(input_row + col, input_row + buc_end + col);
      float maximum_element =
          fbgemm_gpu::max(input_row + col, input_row + buc_end + col);
      auto scale =
          max_pos / (kEpsilon + fmaxf(maximum_element, -minimum_element));
      output_row_scale[0] = scale;
      output_row_scale[1] = *reinterpret_cast<float*>(
          (ncols - col > row_dim) ? &last_buc_idx : &pad);
      for (int bi = 0; bi < std::min(row_dim, (int)(ncols - col)); ++bi) {
        output_row[col + bi + col_offset] =
            float_to_hfp8(input_row[col + bi] * scale, ebit, bias, max_pos);
      }
    }
  }
}

template <typename input_t>
__global__ inline void _get_8bit_qparam_cuda_kernel(
    const input_t* __restrict__ input,
    const int nrows,
    const int ncols,
    uint8_t* __restrict__ output,
    float* __restrict__ range_list) {
  const int row = (int)blockIdx.x * blockDim.y + threadIdx.y;

  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  // starting values for future reductions
#ifdef __HIP_PLATFORM_HCC__
#define HIPRT_INF_F __int_as_float(0x7f800000)
  float minimum_element = HIPRT_INF_F;
  float maximum_element = -HIPRT_INF_F;
#undef HIPRT_INF_F
#else
  float minimum_element = CUDART_INF_F;
  float maximum_element = -CUDART_INF_F;
#endif

  // always a power of 2 up to size 32. Multiple rows can share the same warp
  // when smaller than 32.
  const int lane_width = blockDim.x;

  // March warp-wise through the row, doing thread local min and max reductions.
  // This loop will only execute once when ncol <= 32
  if (row < nrows) {
    const input_t* const input_row = input + row * ncols;

    for (int col = threadIdx.x; col < ncols; col += lane_width) {
      // Get thread-local minmax. These are the smallest min and max ever seen
      // by this thread.
      minimum_element = fminf(minimum_element, input_row[col]);
      maximum_element = fmaxf(maximum_element, input_row[col]);
    }
  }

  // Perform warp-wide min and max reductions. All threads in the warp
  // participate, even if they aren't assigned to a row, since we can't assume
  // the existence of the `*_sync` warp primitives with support for masking.
  for (int offset = lane_width >> 1; offset > 0; offset >>= 1) {
    minimum_element = fminf(
        minimum_element,
        quantize_ops_shfl_xor(minimum_element, offset, lane_width));
    maximum_element = fmaxf(
        maximum_element,
        quantize_ops_shfl_xor(maximum_element, offset, lane_width));
  }

  // only the leading thread in the warp is needed to return the final result in
  // output. Additionally, threads mapped to non-existent rows do not write to
  // the output array.
  if (threadIdx.x != 0 || row >= nrows) {
    return;
  }

  const float range = maximum_element - minimum_element;
  float* const output_row_qparams =
      reinterpret_cast<float*>(output + row * output_columns + ncols_aligned);

  output_row_qparams[0] = range / 255.0f;
  output_row_qparams[1] = minimum_element;
  range_list[row] = range;
}

template <typename input_t>
__global__ inline void _get_FP8_qparam_cuda_kernel(
    const input_t* __restrict__ input,
    const int nrows,
    const int ncols,
    uint8_t* __restrict__ output,
    float* __restrict__ range_list,
    const bool forward) {
  const int row = (int)blockIdx.x * blockDim.y + threadIdx.y;

  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);
  float max_pos;
  if (forward) {
    max_pos = 0.9375;
  } else {
    max_pos = 0.875;
  }
  // starting values for future reductions
  constexpr float kEpsilon = 1e-20f;
  float maximum_element = kEpsilon;
  // always a power of 2 up to size 32. Multiple rows can share the same warp
  // when smaller than 32.
  const int lane_width = blockDim.x;

  // March warp-wise through the row, doing thread local min and max reductions.
  // This loop will only execute once when ncol <= 32
  if (row < nrows) {
    const input_t* const input_row = input + row * ncols;

    for (int col = threadIdx.x; col < ncols; col += lane_width) {
      // Get thread-local minmax. These are the smallest min and max ever seen
      // by this thread.
      maximum_element = fmaxf(maximum_element, fabs(input_row[col]));
    }
  }

  // Perform warp-wide min and max reductions. All threads in the warp
  // participate, even if they aren't assigned to a row, since we can't assume
  // the existence of the `*_sync` warp primitives with support for masking.
  for (int offset = lane_width >> 1; offset > 0; offset >>= 1) {
    maximum_element = fmaxf(
        maximum_element,
        quantize_ops_shfl_xor(maximum_element, offset, lane_width));
  }

  // only the leading thread in the warp is needed to return the final result in
  // output. Additionally, threads mapped to non-existent rows do not write to
  // the output array.
  if (threadIdx.x != 0 || row >= nrows) {
    return;
  }
  float* const output_row_qparams =
      reinterpret_cast<float*>(output + row * output_columns + ncols_aligned);

  output_row_qparams[0] = max_pos / (kEpsilon + maximum_element);
}

template <typename input_t>
__global__ inline void _compute_8bit_quantize_cuda_kernel(
    const input_t* const __restrict__ input,
    const float* const __restrict__ range_list,
    const int nrows,
    const int ncols,
    std::uint8_t* const __restrict__ output) {
  constexpr float kEpsilon = 1e-20f;

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
      const float bias = row_qparams[1];

      const int input_idx = row * ncols + col;
      uint8_t* output_addr = output + row * output_columns + col;
      // TODO: lift range_list into shared memory. However, when nrows is large,
      // it might exceed the size of shared memory.
      const auto inverse_scale = 255.0f / (range_list[row] + kEpsilon);
      output_addr[0] = lrintf((input[input_idx] - bias) * inverse_scale);
    }
  }
}

template <typename input_t>
__global__ inline void _compute_FP8_quantize_cuda_kernel(
    const input_t* const __restrict__ input,
    const float* const __restrict__ range_list,
    const int nrows,
    const int ncols,
    std::uint8_t* const __restrict__ output,
    const bool forward) {
  int ebit;
  int bias;
  float max_pos;
  if (forward) {
    ebit = 4;
    bias = 15;
    max_pos = 0.9375;
  } else {
    ebit = 5;
    bias = 31;
    max_pos = 0.875;
  }

  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (col < ncols) {
      float* row_qparams = reinterpret_cast<float*>(
          output + row * output_columns + ncols_aligned);
      const float scale = row_qparams[0];
      const int input_idx = row * ncols + col;
      uint8_t* output_addr = output + row * output_columns + col;
      // TODO: lift range_list into shared memory. However, when nrows is large,
      // it might exceed the size of shared memory.
      // output_addr[0] = lrintf((input[input_idx] - bias) * inverse_scale);
      output_addr[0] =
          float_to_hfp8(input[input_idx] * scale, ebit, bias, max_pos);
    }
  }
}

// Fused 8-bit rowwise -> FP32/FP16 kernel
template <typename output_t>
__global__ inline void _fused8bitrowwise_to_float_cuda_kernel(
    const std::uint8_t* const __restrict__ input,
    const int nrows,
    const int ncols,
    output_t* const __restrict__ output) {
  const int output_columns = ncols - 2 * sizeof(float);

  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (col < output_columns) {
      const std::uint8_t* input_row = input + row * ncols;
      const float* input_row_scale_bias =
          reinterpret_cast<const float*>(input_row + output_columns);
      output_t* output_row = output + row * output_columns;

      output_row[col] =
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
    }
  }
}
template <typename output_t>
__global__ inline void _PaddedFP8rowwise_to_float_cuda_kernel(
    const std::uint8_t* const __restrict__ input,
    const int nrows,
    const int ncols,
    const int output_columns,
    output_t* const __restrict__ output,
    const bool forward,
    const int row_dim) {
  const int row_ext = row_dim + 8;
  const int ebit = forward ? 4 : 5;
  const int bias = forward ? 15 : 31;

  const int64_t row = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) {
    return;
  }
  const std::uint8_t* const input_row = input + row * ncols;
  output_t* output_row = output + row * output_columns;
  int col_offset = 0;
  for (int col = 0; col < ncols; col = col + row_ext) {
    const float* input_row_scale =
        reinterpret_cast<const float*>(input_row + col + row_ext - 8);
    int pad = *reinterpret_cast<const int*>(&input_row_scale[1]);
    // if pad is negative it's used to indidate indices of the next padded
    // bucket
    pad = (pad > 0) ? pad : 0;
    for (int bi = 0; bi < row_dim - pad; ++bi) {
      output_row[col + bi - col_offset] =
          hfp8_to_float(input_row[col + bi], ebit, bias) / input_row_scale[0];
    }
    col_offset = col_offset + 8 + pad;
  }
}

template <typename output_t>
__global__ inline void _FP8rowwise_to_float_cuda_kernel(
    const std::uint8_t* const __restrict__ input,
    const int nrows,
    const int ncols,
    output_t* const __restrict__ output,
    const bool forward) {
  const int output_columns = ncols - 2 * sizeof(float);
  const int ebit = forward ? 4 : 5;
  const int bias = forward ? 15 : 31;

  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (col < output_columns) {
      const std::uint8_t* input_row = input + row * ncols;
      const float* input_row_scale_bias =
          reinterpret_cast<const float*>(input_row + output_columns);
      output_t* output_row = output + row * output_columns;

      output_row[col] =
          hfp8_to_float(input_row[col], ebit, bias) / input_row_scale_bias[0];
    }
  }
}

// Fused 8-bit rowwise -> FP32/FP16 kernel
template <typename output_t>
__global__ inline void _fused8bitrowwise_to_float_mixed_dim_cuda_kernel(
    const at::PackedTensorAccessor32<uint8_t, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    at::PackedTensorAccessor32<output_t, 2, at::RestrictPtrTraits> output) {
  const int batch_size = input.size(0);

  const int thread_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_tables = D_offsets.size(0) - 1;
  const int qparam_size = 8;

  if (batch_size == 0 || num_tables == 0) {
    return;
  }

  // num_table * batch_size = total warps
  // warp_id = num_tables * batch_idx + table_idx
  const int table_idx = thread_idx % num_tables;
  const int batch_idx = thread_idx / num_tables;
  if (table_idx >= num_tables || batch_idx >= batch_size) {
    return;
  }
  const int table_qparam_offset = D_offsets[table_idx + 1] - qparam_size;
  const int table_D =
      D_offsets[table_idx + 1] - D_offsets[table_idx] - qparam_size;

  // int total_D = input.size(1);
  // CUDA_KERNEL_ASSERT(table_qparam_offset <= total_D && "table_idx <
  // total_D");

  const float2 qparams =
      *reinterpret_cast<const float2*>(&input[batch_idx][table_qparam_offset]);
  const int64_t input_offset = D_offsets[table_idx];
  const int64_t output_offset = input_offset - table_idx * qparam_size;
  for (int i = threadIdx.x; i < table_D; i += kWarpSize) {
    output[batch_idx][i + output_offset] =
        input[batch_idx][i + input_offset] * qparams.x + qparams.y;
  }
}

#define QUANTIZE_OPS_MAX(a, b) ((a) > (b) ? (a) : (b))
#define QUANTIZE_OPS_MIN(a, b) ((a) < (b) ? (a) : (b))

// FP32/FP16 -> Fused 4/2-bit rowwise kernel
template <typename input_t>
__global__ inline void _float_to_fusednbitrowwise_cuda_kernel(
    const int bit_rate,
    const input_t* __restrict__ input,
    const int nrows,
    const int ncols,
    std::uint8_t* __restrict__ output) {
  const int num_elem_per_byte = 8 / bit_rate;
  const int output_columns =
      (ncols + num_elem_per_byte - 1) / num_elem_per_byte + 2 * sizeof(__half);

  int row = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.x * gridDim.x;
  for (/*row*/; row < nrows; row += row_incre) {
    const input_t* input_row = input + row * ncols;
    std::uint8_t* output_row = output + row * output_columns;
    __half* output_row_scale_bias = reinterpret_cast<__half*>(
        output_row + (ncols + num_elem_per_byte - 1) / num_elem_per_byte);

    float minimum_element = fbgemm_gpu::min(input_row, input_row + ncols);
    float maximum_element = fbgemm_gpu::max(input_row, input_row + ncols);
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
      const float X = input_row[col];

      std::uint8_t quantized = QUANTIZE_OPS_MAX(
          0,
          QUANTIZE_OPS_MIN(
              static_cast<int>(lrintf((X - minimum_element) * inverse_scale)),
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

// Fused 4/2-bit rowwise -> FP32/FP16 kernel
template <typename output_t>
__global__ inline void _fusednbitrowwise_to_float_cuda_kernel(
    const int bit_rate,
    const std::uint8_t* input,
    const int nrows,
    const int ncols,
    output_t* const output) {
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
      output_t* output_row = output + row * output_columns;

      std::uint8_t quantized = input_row[col / num_elem_per_byte];
      quantized >>= (col % num_elem_per_byte) * bit_rate;
      quantized &= (1 << bit_rate) - 1;
      output_row[col] = scale * quantized + bias;
    }
  }
}
} // namespace

template <typename input_t>
Tensor _float_to_fused8bitrowwise_gpu_t(const Tensor& input) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  auto output = at::empty(
      output_dims, // 4 = sizeof(float)
      input.options().dtype(at::kByte));

  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(nrows, threads_per_block);
  // think unsigned as we use 0, 255

  if (nrows <= 20) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "_float_to_fused8bitrowwise_cuda_kernel", [&] {
          _float_to_fused8bitrowwise_cuda_kernel<scalar_t>
              <<<num_blocks,
                 threads_per_block,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  input.data_ptr<scalar_t>(),
                  nrows,
                  ncols,
                  output.data_ptr<std::uint8_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // range_tensor is used to store the range for each embedding row.
    // We save range/255.0f as row scale, and use 255.0f / (range + kEpsilon) to
    // quantize. This will guarantee the numerical match but bring some perf
    // regression.
    auto range_tensor = at::empty({nrows}, input.options().dtype(at::kFloat));

    {
      // we need a blockDim.x that is a power of 2 no larger than the warp size
      // of 32

      int blockDim_x = 1;
      if (ncols > 16) {
        // max warp size
        blockDim_x = 32;
      } else {
        while (blockDim_x < ncols) {
          blockDim_x <<= 1;
        }
      }

      const int rows_per_block = threads_per_block / blockDim_x;
      const auto num_blocks_warp =
          cuda_calc_xblock_count(nrows, rows_per_block);

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(), "_get_8bit_qparam_cuda_kernel", [&] {
            _get_8bit_qparam_cuda_kernel<scalar_t>
                <<<num_blocks_warp,
                   dim3(blockDim_x, rows_per_block),
                   0,
                   at::cuda::getCurrentCUDAStream()>>>(
                    input.data_ptr<scalar_t>(),
                    nrows,
                    ncols,
                    output.data_ptr<std::uint8_t>(),
                    range_tensor.data_ptr<float>());
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    {
      const int blockDim_x = std::min(ncols, threads_per_block);
      dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
      const auto gridDim_x = cuda_calc_xblock_count(ncols, blockDim.x);
      const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
      dim3 gridDim(gridDim_x, gridDim_y);

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(), "_compute_8bit_quantize_cuda_kernel", [&] {
            _compute_8bit_quantize_cuda_kernel<scalar_t>
                <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                    input.data_ptr<scalar_t>(),
                    range_tensor.data_ptr<float>(),
                    nrows,
                    ncols,
                    output.data_ptr<std::uint8_t>());
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  return output;
}

///@ingroup quantize-data-cuda
Tensor _float_to_fused8bitrowwise_gpu(const Tensor& input) {
  return _float_to_fused8bitrowwise_gpu_t<float>(input);
}

Tensor _half_to_fused8bitrowwise_gpu(const Tensor& input) {
  return _float_to_fused8bitrowwise_gpu_t<at::Half>(input);
}

///@ingroup quantize-data-cuda
Tensor _float_or_half_to_fused8bitrowwise_gpu(const Tensor& input) {
  Tensor output;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(),
      "float_or_half_to_fused8bitrowwise_cuda_kernel",
      [&] { output = _float_to_fused8bitrowwise_gpu_t<scalar_t>(input); });
  return output;
}

// revising INT8 rowwise template for FP8 rowwise quantization
template <typename input_t>
Tensor _float_to_FP8rowwise_gpu_t(const Tensor& input, const bool forward) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  auto output = at::empty(
      output_dims, // 4 = sizeof(float)
      input.options().dtype(at::kByte));

  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(nrows, threads_per_block);
  // think unsigned as we use 0, 255

  if (nrows <= 20) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "_float_to_FP8rowwise_cuda_kernel", [&] {
          _float_to_FP8rowwise_cuda_kernel<scalar_t>
              <<<num_blocks,
                 threads_per_block,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  input.data_ptr<scalar_t>(),
                  nrows,
                  ncols,
                  output.data_ptr<std::uint8_t>(),
                  forward);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // range_tensor is used to store the range for each embedding row.
    // We save max_pos/max_val(rowwise) as row scale to quantize
    // unlike INT8, FP8 does not have zero shift
    // This will guarantee the numerical match but bring some perf
    // regression.
    auto range_tensor = at::empty({nrows}, input.options().dtype(at::kFloat));

    {
      // we need a blockDim.x that is a power of 2 no larger than the warp size
      // of 32

      int blockDim_x = 1;
      if (ncols > 16) {
        // max warp size
        blockDim_x = 32;
      } else {
        while (blockDim_x < ncols) {
          blockDim_x <<= 1;
        }
      }

      const int rows_per_block = threads_per_block / blockDim_x;
      const auto num_blocks_warp =
          cuda_calc_xblock_count(nrows, rows_per_block);

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(), "_get_FP8_qparam_cuda_kernel", [&] {
            _get_FP8_qparam_cuda_kernel<scalar_t>
                <<<num_blocks_warp,
                   dim3(blockDim_x, rows_per_block),
                   0,
                   at::cuda::getCurrentCUDAStream()>>>(
                    input.data_ptr<scalar_t>(),
                    nrows,
                    ncols,
                    output.data_ptr<std::uint8_t>(),
                    range_tensor.data_ptr<float>(),
                    forward);
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    {
      const int blockDim_x = std::min(ncols, threads_per_block);
      dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
      const auto gridDim_x = cuda_calc_xblock_count(ncols, blockDim.x);
      const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
      dim3 gridDim(gridDim_x, gridDim_y);

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(), "_compute_FP8_quantize_cuda_kernel", [&] {
            _compute_FP8_quantize_cuda_kernel<scalar_t>
                <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                    input.data_ptr<scalar_t>(),
                    range_tensor.data_ptr<float>(),
                    nrows,
                    ncols,
                    output.data_ptr<std::uint8_t>(),
                    forward);
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  return output;
}

// revising INT8 rowwise template for FP8 rowwise quantization
template <typename input_t>
Tensor _float_to_paddedFP8rowwise_gpu_t(
    const Tensor& input,
    const bool forward,
    const int64_t row_dim) {
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(input);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const auto nrows = c10::size_to_dim_(last_dim, input_sizes);
  const auto ncols = input_sizes[last_dim];
  const int output_columns = (ncols + row_dim - 1) / row_dim * (row_dim + 8);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;

  // auto output = at::empty(
  auto output = at::zeros(
      output_dims, // 4 = sizeof(float)
      input.options().dtype(at::kByte));

  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(nrows, threads_per_block);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "_float_to_FP8rowwise_cuda_kernel", [&] {
        _float_to_paddedFP8rowwise_cuda_kernel<scalar_t>
            <<<num_blocks,
               threads_per_block,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<scalar_t>(),
                nrows,
                ncols,
                output.data_ptr<std::uint8_t>(),
                forward,
                row_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

///@ingroup quantize-data-cuda
Tensor _float_to_FP8rowwise_gpu(const Tensor& input, const bool forward) {
  return _float_to_FP8rowwise_gpu_t<float>(input, forward);
}

///@ingroup quantize-data-cuda
Tensor _float_to_paddedFP8rowwise_gpu(
    const Tensor& input,
    const bool forward,
    const int64_t row_dim) {
  return _float_to_paddedFP8rowwise_gpu_t<float>(input, forward, row_dim);
}

template <typename output_t>
Tensor _fused8bitrowwise_to_float_gpu_t(const Tensor& input) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned - 2 * sizeof(float);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  Tensor output;
  if constexpr (std::is_same_v<output_t, float>) {
    output = at::empty(
        output_dims, // 4 = sizeof(float)
        input.options().dtype(at::kFloat));
  } else { // T = at::Half
    output = at::empty(
        output_dims, // 4 = sizeof(float)
        input.options().dtype(at::kHalf));
  }

  if (nrows == 0 || output_columns == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;

  const int blockDim_x = std::min(threads_per_block, output_columns);
  const dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);

  const auto gridDim_x = cuda_calc_xblock_count(output_columns, blockDim.x);
  const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
  const dim3 gridDim(gridDim_x, gridDim_y);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      output.scalar_type(), "fused8bitrowwise_to_float_cuda_kernel", [&] {
        _fused8bitrowwise_to_float_cuda_kernel<scalar_t>
            <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<std::uint8_t>(),
                nrows,
                ncols,
                output.data_ptr<scalar_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

at::Tensor _fused8bitrowwise_to_float_gpu(const at::Tensor& input) {
  return _fused8bitrowwise_to_float_gpu_t<float>(input);
}

at::Tensor _fused8bitrowwise_to_half_gpu(const at::Tensor& input) {
  return _fused8bitrowwise_to_float_gpu_t<at::Half>(input);
}

///@ingroup quantize-data-cuda
at::Tensor _fused8bitrowwise_to_float_or_half_gpu(
    const at::Tensor& input,
    const int64_t output_dtype) {
  Tensor output;

  SparseType output_sparse_dtype = static_cast<SparseType>(output_dtype);
  switch (output_sparse_dtype) {
    case SparseType::FP32:
      output = _fused8bitrowwise_to_float_gpu_t<float>(input);
      break;
    case SparseType::FP16:
      output = _fused8bitrowwise_to_float_gpu_t<at::Half>(input);
      break;
    default:
      TORCH_CHECK(false);
  }

  return output;
}

template <typename output_t>
Tensor _FP8rowwise_to_float_gpu_t(const Tensor& input, bool forward) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned - 2 * sizeof(float);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  Tensor output;
  if constexpr (std::is_same_v<output_t, float>) {
    output = at::empty(
        output_dims, // 4 = sizeof(float)
        input.options().dtype(at::kFloat));
  } else { // T = at::Half
    output = at::empty(
        output_dims, // 4 = sizeof(float)
        input.options().dtype(at::kHalf));
  }

  if (nrows == 0 || output_columns == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;

  const int blockDim_x = std::min(threads_per_block, output_columns);
  const dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);

  const auto gridDim_x = cuda_calc_xblock_count(output_columns, blockDim.x);
  const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
  const dim3 gridDim(gridDim_x, gridDim_y);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      output.scalar_type(), "FP8rowwise_to_float_cuda_kernel", [&] {
        _FP8rowwise_to_float_cuda_kernel<scalar_t>
            <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<std::uint8_t>(),
                nrows,
                ncols,
                output.data_ptr<scalar_t>(),
                forward);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}
template <typename output_t>
Tensor _paddedFP8rowwise_to_float_gpu_t(
    const Tensor& input,
    const bool forward,
    const int64_t row_dim) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int row_ext = row_dim + 8;
  int output_columns = ncols - (ncols + row_ext - 1) / row_ext * 8;
  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output_dims = input_sizes.vec();

  std::uint8_t pad[4];
  int total_pad = 0;
  int col = 0;
  while (col < ncols) {
    for (int i = 0; i < 4; i++) {
      pad[i] = input[0][col + row_ext - 4 + i].item<uint8_t>();
    }
    // rule: if pad value is less than zero, its abs is the offset to the
    // nearest padding value's address
    int pad_int = *reinterpret_cast<int*>(pad);
    if (pad_int < 0) {
      col -= pad_int * row_ext;
    } else {
      total_pad += pad_int;
      col += row_ext;
    }
  }
  output_columns -= total_pad;
  output_dims[last_dim] = output_columns;
  Tensor output;
  if constexpr (std::is_same_v<output_t, float>) {
    output = at::empty(
        output_dims, // 4 = sizeof(float)
        input.options().dtype(at::kFloat));
  } else { // T = at::Half
    output = at::empty(
        output_dims, // 4 = sizeof(float)
        input.options().dtype(at::kHalf));
  }

  if (nrows == 0 || output_columns == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(nrows, threads_per_block);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      output.scalar_type(), "PaddedFP8rowwise_to_float_cuda_kernel", [&] {
        _PaddedFP8rowwise_to_float_cuda_kernel<scalar_t>
            <<<num_blocks,
               threads_per_block,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<std::uint8_t>(),
                nrows,
                ncols,
                output_columns,
                output.data_ptr<scalar_t>(),
                forward,
                row_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

at::Tensor _FP8rowwise_to_float_gpu(const at::Tensor& input, bool forward) {
  return _FP8rowwise_to_float_gpu_t<float>(input, forward);
}

at::Tensor _paddedFP8rowwise_to_float_gpu(
    const at::Tensor& input,
    const bool forward,
    const int64_t row_dim) {
  return _paddedFP8rowwise_to_float_gpu_t<float>(input, forward, row_dim);
}

///@ingroup quantize-data-cuda
at::Tensor _fused8bitrowwise_to_float_mixed_dim_gpu(
    const at::Tensor& input,
    const at::Tensor& D_offsets,
    const int64_t output_dtype) {
  // assumes input is 2D with [B x sum(D)] format.
  // D_offsets is a 1D tensor that marks the boundary between quantized output
  // row of each table
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(input);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(D_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int64_t batch_size = input.size(0);
  const int qparam_size = 8;
  // allocate a warp for each output row
  const int num_tables = D_offsets.size(0) - 1;
  const int64_t output_dim =
      input.size(1) - static_cast<int64_t>(qparam_size * num_tables);
  at::Tensor output;
  SparseType output_sparse_dtype = static_cast<SparseType>(output_dtype);
  switch (output_sparse_dtype) {
    case SparseType::FP32:
      output = at::zeros(
          {batch_size, output_dim}, input.options().dtype(at::kFloat));
      break;
    case SparseType::FP16:
      output =
          at::zeros({batch_size, output_dim}, input.options().dtype(at::kHalf));
      break;
    default:
      TORCH_CHECK(false);
  }
  if (batch_size == 0) {
    return output;
  }
  constexpr int threads_per_block = 256;
  const dim3 blockDim(kWarpSize, threads_per_block / kWarpSize);
  const dim3 gridDim(
      cuda_calc_xblock_count(num_tables * batch_size, blockDim.y));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      output.scalar_type(),
      "_fused8bitrowwise_to_float_mixed_dim_cuda_kernel",
      [&] {
        _fused8bitrowwise_to_float_mixed_dim_cuda_kernel<scalar_t>
            <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.packed_accessor32<uint8_t, 2, at::RestrictPtrTraits>(),
                D_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

///@ingroup quantize-data-cuda
template <typename input_t>
Tensor _float_to_fusednbitrowwise_gpu_t(
    const Tensor& input,
    const int64_t bit_rate) {
  TENSOR_ON_CUDA_GPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int nrows = input.size(0);
  const int ncols = input.size(1);
  const int num_elem_per_byte = 8 / bit_rate;
  TORCH_CHECK(
      ncols % (2 * num_elem_per_byte) == 0,
      "ncols needs to be multiple of 2 Bytes (half type size) to make the address aligned");
  const int output_columns =
      (ncols + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(at::Half);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output = at::empty(
      {nrows, output_columns},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t

  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr auto threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(nrows, threads_per_block);
  // think unsigned as we use 0, 255

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "_float_to_fusednbitrowwise_cuda_kernel", [&] {
        _float_to_fusednbitrowwise_cuda_kernel<scalar_t>
            <<<num_blocks,
               threads_per_block,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                bit_rate,
                input.data_ptr<scalar_t>(),
                nrows,
                ncols,
                output.data_ptr<std::uint8_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

///@ingroup quantize-data-cuda
Tensor _float_to_fusednbitrowwise_gpu(
    const Tensor& input,
    const int64_t bit_rate) {
  return _float_to_fusednbitrowwise_gpu_t<float>(input, bit_rate);
}

///@ingroup quantize-data-cuda
at::Tensor _half_to_fusednbitrowwise_gpu(
    const at::Tensor& input,
    const int64_t bit_rate) {
  return _float_to_fusednbitrowwise_gpu_t<at::Half>(input, bit_rate);
}

///@ingroup sparse-data-cuda
Tensor _float_or_half_to_fusednbitrowwise_gpu(
    const Tensor& input,
    const int64_t bit_rate) {
  Tensor output;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(),
      "float_or_half_to_fusednbitrowwise_cuda_kernel",
      [&] {
        output = _float_to_fusednbitrowwise_gpu_t<scalar_t>(input, bit_rate);
      });
  return output;
}

///@ingroup quantize-data-cuda
template <typename output_t>
Tensor _fusednbitrowwise_to_float_gpu_t(
    const Tensor& input,
    const int64_t bit_rate) {
  TENSOR_ON_CUDA_GPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int nrows = input.size(0);
  const int ncols = input.size(1);
  const int num_elem_per_byte = 8 / bit_rate;
  const int output_columns = (ncols - 2 * sizeof(at::Half)) * num_elem_per_byte;

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  Tensor output;
  if constexpr (std::is_same_v<output_t, float>) {
    output = at::empty(
        {nrows, output_columns}, // 4 = sizeof(float)
        input.options().dtype(at::kFloat));
  } else { // T = at::Half
    output = at::empty(
        {nrows, output_columns}, // 4 = sizeof(float)
        input.options().dtype(at::kHalf));
  }

  if (nrows == 0 || output_columns == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;

  const int blockDim_x = std::min(output_columns, threads_per_block);
  const dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
  const auto gridDim_x = cuda_calc_xblock_count(output_columns, blockDim.x);
  const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
  const dim3 gridDim(gridDim_x, gridDim_y);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      output.scalar_type(), "fusednbitrowwise_to_float_cuda_kernel", [&] {
        _fusednbitrowwise_to_float_cuda_kernel<scalar_t>
            <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                bit_rate,
                input.data_ptr<uint8_t>(),
                nrows,
                ncols,
                output.data_ptr<scalar_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

at::Tensor _fusednbitrowwise_to_float_gpu(
    const at::Tensor& input,
    const int64_t bit_rate) {
  return _fusednbitrowwise_to_float_gpu_t<float>(input, bit_rate);
}

///@ingroup quantize-data-cuda
at::Tensor _fusednbitrowwise_to_half_gpu(
    const at::Tensor& input,
    const int64_t bit_rate) {
  return _fusednbitrowwise_to_float_gpu_t<at::Half>(input, bit_rate);
}

///@ingroup quantize-data-cuda
at::Tensor _fusednbitrowwise_to_float_or_half_gpu(
    const at::Tensor& input,
    const int64_t bit_rate,
    const int64_t output_dtype) {
  Tensor output;

  SparseType output_sparse_dtype = static_cast<SparseType>(output_dtype);
  switch (output_sparse_dtype) {
    case SparseType::FP32:
      output = _fusednbitrowwise_to_float_gpu_t<float>(input, bit_rate);
      break;
    case SparseType::FP16:
      output = _fusednbitrowwise_to_float_gpu_t<at::Half>(input, bit_rate);
      break;
    default:
      TORCH_CHECK(false);
  }

  return output;
}

at::Tensor _float_to_hfp8_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias,
    const double max_pos) {
  TORCH_CHECK(ebits > 0);
  TORCH_CHECK(exponent_bias > 0);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  auto output = at::empty({}, input.options().dtype(at::kByte));
  output.resize_(0);

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();

  at::native::gpu_kernel(iter, [=] GPU_LAMBDA(float in) -> uint8_t {
    return float_to_hfp8(in, ebits, exponent_bias, max_pos);
  });

  return output;
}

at::Tensor _hfp8_to_float_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias) {
  TORCH_CHECK(ebits > 0);
  TORCH_CHECK(exponent_bias > 0);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  auto output = at::empty({}, input.options().dtype(at::kFloat));
  output.resize_(0);

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();

  at::native::gpu_kernel(iter, [=] GPU_LAMBDA(uint8_t in) -> float {
    return hfp8_to_float(in, ebits, exponent_bias);
  });

  return output;
}

__host__ __device__ inline float float_to_msfp(
    const float val_fp,
    const int shared_expo,
    const int mbits,
    const int bias,
    const float max_pos) {
  fbgemm_gpu::fint32 X, bouncer, scale, inv_scale;
  int32_t expo, emin, delta_E, nbits2round;

  X.F = val_fp;
  const uint32_t sign_bit = X.I & 0x80000000;
  X.I = X.I & 0x7FFFFFFF; // 31 bits

  emin = 1 - bias;

  // Because the input value can be of extreme magnitude
  // We scale them into less extreme to avoid potential exception during
  // manipulation
  const int32_t E = ((X.I & 0x7F800000) >> 23) - 127;
  if (E >= 0) {
    scale.I = 0X2F800000;
    inv_scale.I = 0X4F800000; // scale is 2^-32, inv_scale is 2^32
    delta_E = -32;
  } else {
    scale.I = 0x4F800000;
    inv_scale.I = 0x2F800000;
    delta_E = 32;
  }
  X.F *= scale.F; // at this point X is never close to over/underflow
  expo = ((X.I & 0x7F800000) >> 23) - 127 - delta_E;

  // If expo >= emin
  // We round to mbits explicit mantissa bits
  // That is, we want to round off 23-mbits of the trailing bits in X
  nbits2round = 23 - mbits;
  // However, if expo < emin, we need to round more bits off
  nbits2round += ::max(emin - expo, 0); // max(emin - expo, 0);
  // also need to right shift mantissa with the shared expoennt
  nbits2round += ::max(shared_expo - expo, 0);

  bouncer.I = (nbits2round << 23) + (X.I & 0x7F800000);
  X.F = X.F + bouncer.F; // Because bouncer is exactly 2^nbits2round bigger
                         // this addition forces the rounding off of nbits2round
  X.F = X.F - bouncer.F; // X.F is the original X with nbits2round rounded off

  // restore the true magnitude by undoing the previous scale
  X.F *= inv_scale.F;
  // clip on the large end of the domain
  X.F = ::min(X.F, max_pos);
  // restores the original sign
  X.I |= sign_bit;

  const float val_msfp = X.F;
  return val_msfp;
}

__global__ inline void _compute_msfp_shared_exponent_cuda_kernel(
    const float* __restrict__ input,
    const int nrows,
    const int ncols,
    const int bounding_box_size,
    int* __restrict__ shared_exponents) {
  const int tidy = blockIdx.y * blockDim.y +
      threadIdx.y; // to get the threadid-y dimension of this thread
  const int tidx = blockIdx.x * blockDim.x +
      threadIdx.x; // to get the threadid-x dimension of this thread

  const int row_incre = blockDim.y * gridDim.y;
  const int col_incre = blockDim.x * gridDim.x;

  for (int row = tidy; row < nrows; row += row_incre) {
    const float* input_row = input + row * ncols;
    int* shared_expo_row = shared_exponents + row * ncols;
    for (int col = tidx; col < ncols; col += col_incre) {
      const int boundingbox_start = col / bounding_box_size * bounding_box_size;
      const int boundingbox_end =
          ::min(boundingbox_start + bounding_box_size, ncols);

      int32_t max_exponent = 0;
      for (int i = boundingbox_start; i < boundingbox_end; i++) {
        // update the max_exponent
        fbgemm_gpu::fint32 org_data;
        org_data.F = input_row[i];
        org_data.I = org_data.I & 0x7FFFFFFF; // 31 bits
        const int32_t exponent = ((org_data.I & 0x7F800000) >> 23);
        max_exponent = ::max(max_exponent, exponent);
      }
      shared_expo_row[col] = static_cast<int>(max_exponent) - 127;
    }
  }
}

at::Tensor _float_to_msfp_gpu(
    const at::Tensor& input,
    const int64_t bounding_box_size,
    const int64_t ebits,
    const int64_t mbits,
    const int64_t bias,
    const double min_pos,
    const double max_pos) {
  TENSOR_ON_CUDA_GPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  TORCH_CHECK(ebits <= 8);
  TORCH_CHECK(mbits <= 23);
  TORCH_CHECK(ebits > 0 && mbits > 0);
  TORCH_CHECK(min_pos > 0 && max_pos > 0 && max_pos > min_pos);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int nrows = input.size(0);
  const int ncols = input.size(1);

  auto output = at::empty({nrows, ncols}, input.options().dtype(at::kFloat));
  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;

  const int blockDim_x = std::min(ncols, threads_per_block);
  const dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
  const int gridDim_x = (ncols + blockDim.x - 1) / blockDim.x;
  const int gridDim_y = std::min((nrows + blockDim.y - 1) / blockDim.y, 65535u);
  const dim3 gridDim(gridDim_x, gridDim_y);

  auto shared_exponents =
      at::empty({nrows, ncols}, input.options().dtype(at::kInt));

  _compute_msfp_shared_exponent_cuda_kernel<<<
      gridDim,
      blockDim,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      input.contiguous().data_ptr<float>(),
      nrows,
      ncols,
      bounding_box_size,
      shared_exponents.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .add_input(shared_exponents)
                  .build();

  at::native::gpu_kernel(
      iter, [=] GPU_LAMBDA(float in, int shared_expo) -> float {
        return float_to_msfp(in, shared_expo, mbits, bias, max_pos);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

///@ingroup quantize-data-cuda
at::Tensor _msfp_to_float_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t mbits,
    const int64_t bias) {
  TENSOR_ON_CUDA_GPU(input);

  // Because float_to_msfp is a fakequant operator,
  // the input msfp number is already a FP32 number
  // with limited precision.
  // Thus this msfp_to_float is really a no-op
  return input.clone();
}
} // namespace fbgemm_gpu
