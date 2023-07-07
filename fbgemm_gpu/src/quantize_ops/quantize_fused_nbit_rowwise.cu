/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

/// @defgroup quantize-data-cuda Quantization Data CUDA Operators
/// The following are CUDA Operators

namespace fbgemm_gpu {

namespace {

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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output;
}

///@ingroup quantize-data-cuda
DLL_PUBLIC Tensor
_float_to_fusednbitrowwise_gpu(const Tensor& input, const int64_t bit_rate) {
  return _float_to_fusednbitrowwise_gpu_t<float>(input, bit_rate);
}

///@ingroup quantize-data-cuda
DLL_PUBLIC at::Tensor _half_to_fusednbitrowwise_gpu(
    const at::Tensor& input,
    const int64_t bit_rate) {
  return _float_to_fusednbitrowwise_gpu_t<at::Half>(input, bit_rate);
}

///@ingroup sparse-data-cuda
DLL_PUBLIC Tensor _float_or_half_to_fusednbitrowwise_gpu(
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output;
}

DLL_PUBLIC at::Tensor _fusednbitrowwise_to_float_gpu(
    const at::Tensor& input,
    const int64_t bit_rate) {
  return _fusednbitrowwise_to_float_gpu_t<float>(input, bit_rate);
}

///@ingroup quantize-data-cuda
DLL_PUBLIC at::Tensor _fusednbitrowwise_to_half_gpu(
    const at::Tensor& input,
    const int64_t bit_rate) {
  return _fusednbitrowwise_to_float_gpu_t<at::Half>(input, bit_rate);
}

///@ingroup quantize-data-cuda
DLL_PUBLIC at::Tensor _fusednbitrowwise_to_float_or_half_gpu(
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

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "FloatToFusedNBitRowwiseQuantizedSBHalf",
    fbgemm_gpu::_float_to_fusednbitrowwise_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "HalfToFusedNBitRowwiseQuantizedSBHalf",
    fbgemm_gpu::_half_to_fusednbitrowwise_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf",
    fbgemm_gpu::_float_or_half_to_fusednbitrowwise_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "FusedNBitRowwiseQuantizedSBHalfToFloat",
    fbgemm_gpu::_fusednbitrowwise_to_float_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "FusedNBitRowwiseQuantizedSBHalfToHalf",
    fbgemm_gpu::_fusednbitrowwise_to_half_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf",
    fbgemm_gpu::_fusednbitrowwise_to_float_or_half_gpu);
