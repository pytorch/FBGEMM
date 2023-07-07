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

} // namespace

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
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
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
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
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
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    }
  }

  return output;
}

///@ingroup quantize-data-cuda
DLL_PUBLIC Tensor
_float_to_FP8rowwise_gpu(const Tensor& input, const bool forward) {
  return _float_to_FP8rowwise_gpu_t<float>(input, forward);
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output;
}

DLL_PUBLIC at::Tensor _FP8rowwise_to_float_gpu(
    const at::Tensor& input,
    bool forward) {
  return _FP8rowwise_to_float_gpu_t<float>(input, forward);
}

} // namespace fbgemm_gpu
