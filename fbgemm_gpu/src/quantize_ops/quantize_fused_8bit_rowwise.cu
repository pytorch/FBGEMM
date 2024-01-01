/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

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
#ifdef USE_ROCM
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
    minimum_element =
        fminf(minimum_element, shfl_xor(minimum_element, offset, lane_width));
    maximum_element =
        fmaxf(maximum_element, shfl_xor(maximum_element, offset, lane_width));
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
    FBGEMM_DISPATCH_FLOAT_HALF_AND_BFLOAT16(
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
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
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

      FBGEMM_DISPATCH_FLOAT_HALF_AND_BFLOAT16(
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
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    }

    {
      const int blockDim_x = std::min(ncols, threads_per_block);
      dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
      const auto gridDim_x = cuda_calc_xblock_count(ncols, blockDim.x);
      const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
      dim3 gridDim(gridDim_x, gridDim_y);

      FBGEMM_DISPATCH_FLOAT_HALF_AND_BFLOAT16(
          input.scalar_type(), "_compute_8bit_quantize_cuda_kernel", [&] {
            _compute_8bit_quantize_cuda_kernel<scalar_t>
                <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                    input.data_ptr<scalar_t>(),
                    range_tensor.data_ptr<float>(),
                    nrows,
                    ncols,
                    output.data_ptr<std::uint8_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    }
  }

  return output;
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of `float` values into a tensor of fused 8-bit rowwise
/// values.
///
/// @param input A tensor of `float` values
///
/// @return A new tensor with values from the input tensor converted to
/// fused 8-bit rowwise.
DLL_PUBLIC Tensor _float_to_fused8bitrowwise_gpu(const Tensor& input) {
  return _float_to_fused8bitrowwise_gpu_t<float>(input);
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of `at::Half` values into a tensor of fused 8-bit rowwise
/// values.
///
/// @param input A tensor of `at::Half` values
///
/// @return A new tensor with values from the input tensor converted to
/// fused 8-bit rowwise.
DLL_PUBLIC Tensor _half_to_fused8bitrowwise_gpu(const Tensor& input) {
  return _float_to_fused8bitrowwise_gpu_t<at::Half>(input);
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of `at::Single` or `at::Half` values into a tensor of
/// fused 8-bit rowwise values.
///
/// @param input A tensor of `at::Single` or `at::Half` values
///
/// @return A new tensor with values from the input tensor converted to
/// fused 8-bit rowwise.
DLL_PUBLIC Tensor
_single_or_half_precision_to_fused8bitrowwise_gpu(const Tensor& input) {
  Tensor output;
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BFLOAT16(
      input.scalar_type(),
      "float_or_half_to_fused8bitrowwise_cuda_kernel",
      [&] { output = _float_to_fused8bitrowwise_gpu_t<scalar_t>(input); });
  return output;
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
  } else if constexpr (std::is_same_v<output_t, at::Half>) { // T = at::Half
    output = at::empty(
        output_dims, // 2 = sizeof(half)
        input.options().dtype(at::kHalf));
  } else if constexpr (std::is_same_v<output_t, at::BFloat16>) {
    output = at::empty(
        output_dims, // 2 = sizeof(bfloat16)
        input.options().dtype(at::kBFloat16));
  } else {
    TORCH_CHECK(false);
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

  FBGEMM_DISPATCH_FLOAT_HALF_AND_BFLOAT16(
      output.scalar_type(), "fused8bitrowwise_to_float_cuda_kernel", [&] {
        _fused8bitrowwise_to_float_cuda_kernel<scalar_t>
            <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<std::uint8_t>(),
                nrows,
                ncols,
                output.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output;
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of fused 8-bit rowwise values into a tensor of `float`
/// values.
///
/// @param input A tensor of fused 8-bit rowwise values
///
/// @return A new tensor with values from the input tensor converted to `float`.
DLL_PUBLIC at::Tensor _fused8bitrowwise_to_float_gpu(const at::Tensor& input) {
  return _fused8bitrowwise_to_float_gpu_t<float>(input);
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of fused 8-bit rowwise values into a tensor of `at::Half`
/// values.
///
/// @param input A tensor of fused 8-bit rowwise values
///
/// @return A new tensor with values from the input tensor converted to
/// `at::Half`.
DLL_PUBLIC at::Tensor _fused8bitrowwise_to_half_gpu(const at::Tensor& input) {
  return _fused8bitrowwise_to_float_gpu_t<at::Half>(input);
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of fused 8-bit rowwise values into a tensor of `float`,
/// `at::Half`, or `at::BFloat16` values.
///
/// @param input A tensor of fused 8-bit rowwise values
/// @param output_dtype The target floating point type, specified as integer
///                     representation of `SparseType` enum
///
/// @return A new tensor with values from the input tensor converted to `float`,
/// `at::Half`, or `at::BFloat16`.
///
/// @throw c10::Error if `output_dtype` is not one of (`SparseType::FP32`,
/// `SparseType::FP16`, or `SparseType::BF16`).
DLL_PUBLIC at::Tensor _fused8bitrowwise_to_single_or_half_precision_gpu(
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
    case SparseType::BF16:
      output = _fused8bitrowwise_to_float_gpu_t<at::BFloat16>(input);
      break;
    default:
      TORCH_CHECK(false);
  }

  return output;
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of fused 8-bit rowwise values into a tensor of
/// `at::kFloat` or `at::kHalf` values.
///
/// @param input A tensor of fused 8-bit rowwise values
/// @param D_offsets
/// @param output_dtype The target floating point type, specified as integer
///                     representation of `SparseType` enum
///
/// @return A new tensor with values from the input tensor converted to
/// `at::kFloat` or `at::kHalf`.
///
/// @throw c10::Error if `output_dtype` is not one of (`SparseType::FP32`,
/// `SparseType::FP16`)
DLL_PUBLIC at::Tensor _fused8bitrowwise_to_float_mixed_dim_gpu(
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
  FBGEMM_DISPATCH_FLOAT_AND_HALF(
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

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "FloatToFused8BitRowwiseQuantized",
    fbgemm_gpu::_float_to_fused8bitrowwise_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "HalfToFused8BitRowwiseQuantized",
    fbgemm_gpu::_half_to_fused8bitrowwise_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "FloatOrHalfToFused8BitRowwiseQuantized",
    fbgemm_gpu::_single_or_half_precision_to_fused8bitrowwise_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "Fused8BitRowwiseQuantizedToFloat",
    fbgemm_gpu::_fused8bitrowwise_to_float_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "Fused8BitRowwiseQuantizedToHalf",
    fbgemm_gpu::_fused8bitrowwise_to_half_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "Fused8BitRowwiseQuantizedToFloatOrHalf",
    fbgemm_gpu::_fused8bitrowwise_to_single_or_half_precision_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "Fused8BitRowwiseQuantizedToFloatMixedDim",
    fbgemm_gpu::_fused8bitrowwise_to_float_mixed_dim_gpu);
