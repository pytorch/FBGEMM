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
  const auto row_ext =
      row_dim + 8; // 8 bytes = float (scale) + int32 (pad value)
  const int output_columns =
      ncols_aligned + (ncols + row_dim - 1) / row_dim * 8;

  const int64_t row = (int)blockIdx.x * blockDim.x + threadIdx.x;
  // for 1D case, unsqueezing needed
  if (nrows == 1) {
    const auto threads = (ncols + row_dim - 1) / row_dim;
    if (row >= threads) {
      return;
    }
    const input_t* const input_row = input + row * row_dim;
    std::uint8_t* output_row = output + row * row_ext;
    int last_buc_idx = row - (threads - 1);
    float* output_row_scale = reinterpret_cast<float*>(output_row + row_dim);
    const auto range = (row == threads - 1) ? row_dim - pad : row_dim;
    float minimum_element = fbgemm_gpu::min(input_row, input_row + range);
    float maximum_element = fbgemm_gpu::max(input_row, input_row + range);
    auto scale =
        max_pos / (kEpsilon + fmaxf(maximum_element, -minimum_element));
    output_row_scale[0] = scale;
    // if no padding, the pad value is negative to indicate where the next
    // non-zero pad value is for output size counting in host
    output_row_scale[1] =
        *reinterpret_cast<float*>((row == threads - 1) ? &pad : &last_buc_idx);
    for (int col = 0; col < range; col += 1) {
      output_row[col] =
          float_to_hfp8(to_float(input_row[col]) * scale, ebit, bias, max_pos);
    }
    return;
  }
  // for 2D case

  if (row >= nrows) {
    return;
  }
  const input_t* input_row = input + row * ncols;
  std::uint8_t* output_row = output + row * output_columns;
  for (int col = 0; col < ncols; col += row_dim) {
    int col_offset = col / row_dim * 8;
    int last_buc_idx = (ncols - col) / row_dim * -1;
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
      output_row[col + bi + col_offset] = float_to_hfp8(
          to_float(input_row[col + bi]) * scale, ebit, bias, max_pos);
    }
  }
}

__global__ inline void _get_padding_value_kernel(
    const int ncols,
    const int row_dim,
    const std::uint8_t* const __restrict__ input,
    int* const __restrict__ offsets) {
  const int64_t row = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_ext = row_dim + 8;
  const auto threads = (ncols + row_ext - 1) / row_ext;
  if (row > threads)
    return;
  const std::uint8_t* const input_row = input + row * row_ext;
  int pad = *reinterpret_cast<const int*>(input_row + row_dim + 4);
  pad = (pad > 0) ? pad : 0;
  offsets[row] = pad;
}

__global__ inline void _single_thread_sum_padding_kernel(
    const int ncols,
    const int row_dim,
    const std::uint8_t* const __restrict__ input,
    int* __restrict__ total_pad) {
  // this is to count the sum of padding in the first row of 2D input
  // in one kernel launch to remove multiple H to D Syncs.
  const auto tid = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid != 0) {
    return;
  }
  const int row_ext = row_dim + 8;
  int offset = row_dim + 4;
  int pad = 0;
  total_pad[0] = 0;
  while (offset + 4 <= ncols) {
    pad = *reinterpret_cast<const int*>(input + offset);
    if (pad < 0) {
      offset += -pad * row_ext;
    } else {
      total_pad[0] += pad;
      offset += row_ext;
    }
  }
}

template <typename output_t>
__global__ inline void _PaddedFP8rowwise_to_float_1d_cuda_kernel(
    output_t* const __restrict__ output,
    const std::uint8_t* const __restrict__ input,
    const int output_columns,
    const int row_dim,
    const int* const __restrict__ offsets,
    const int row_ext,
    const int ebit,
    const int bias) {
  const int64_t row = blockIdx.x;
  // gridDim.x is num_rows
  if (row >= gridDim.x) {
    return;
  }
  const std::uint8_t* const input_row = input + row * row_ext;
  const float* input_row_scale =
      reinterpret_cast<const float*>(input_row + row_dim);
  const auto scale = input_row_scale[0];
  int pad = *reinterpret_cast<const int*>(&input_row_scale[1]);
  pad = (pad > 0) ? pad : 0;
  const auto pad_offset = offsets[row];
  output_t* output_row = output + row * row_dim - pad_offset;
  for (int col = threadIdx.x; col < row_dim - pad; col += blockDim.x) {
    const auto output_ = hfp8_to_float(input_row[col], ebit, bias) / scale;
    quantize_float_store(&output_row[col], output_);
  }
}

template <typename output_t>
__global__ inline void _PaddedFP8rowwise_to_float_2d_cuda_kernel(
    const std::uint8_t* const __restrict__ input,
    const int nrows,
    const int ncols,
    const int output_columns,
    output_t* const __restrict__ output,
    const bool forward,
    const int row_dim,
    int* const __restrict__ offsets) {
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
      const auto output_ =
          hfp8_to_float(input_row[col + bi], ebit, bias) / input_row_scale[0];
      quantize_float_store(&output_row[col + bi - col_offset], output_);
    }
    col_offset = col_offset + 8 + pad;
  }
}

} // namespace

// revising INT8 rowwise template for FP8 rowwise quantization
Tensor _float_to_paddedFP8rowwise_gpu_t(
    const Tensor& input,
    const bool forward,
    const int64_t row_dim) {
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(input);
  CUDA_DEVICE_GUARD(input);

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
  const auto num_blocks = cuda_calc_xblock_count(
      nrows == 1 ? (ncols + row_dim - 1) / row_dim : nrows, threads_per_block);

  FBGEMM_DISPATCH_FLOATING_TYPES(
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

Tensor _paddedFP8rowwise_to_float_gpu_t(
    const Tensor& input,
    const bool forward,
    const int64_t row_dim,
    const int64_t output_last_dim,
    const int64_t output_dtype) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  CUDA_DEVICE_GUARD(input);

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

  constexpr int threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(
      (nrows == 1) ? (ncols + row_ext - 1) / row_ext + 1 : nrows,
      threads_per_block);
  Tensor offsets = at::empty(
      (nrows == 1) ? num_blocks * threads_per_block + 1
                   : 0, // 4 = sizeof(float)
      input.options().dtype(at::kInt));
  int total_pad = 0;
  if (nrows == 1) {
    _get_padding_value_kernel<<<
        num_blocks,
        threads_per_block,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        ncols,
        row_dim,
        input.data_ptr<std::uint8_t>(),
        offsets.data_ptr<int>());
    offsets = asynchronous_complete_cumsum_gpu(offsets);
  }
  if (output_last_dim < 0) {
    if (nrows != 1) {
      Tensor total_pad_tensor = at::empty(1, input.options().dtype(at::kInt));
      _single_thread_sum_padding_kernel<<<
          1,
          1,
          0,
          at::cuda::getCurrentCUDAStream()>>>(
          ncols,
          row_dim,
          input.data_ptr<std::uint8_t>(),
          total_pad_tensor.data_ptr<int>());
      total_pad = total_pad_tensor[0].item<int>();
    } else {
      total_pad = offsets[((ncols + row_ext - 1) / row_ext)].item<int>();
    }
    output_columns -= total_pad;
  } else {
    output_columns = output_last_dim;
  }

  output_dims[last_dim] = output_columns;

  const auto output_sdtype = static_cast<SparseType>(output_dtype);
  TORCH_CHECK(
      output_sdtype == SparseType::FP32 || output_sdtype == SparseType::FP16 ||
      output_sdtype == SparseType::BF16);

  Tensor output = at::empty(
      output_dims, // 4 = sizeof(float)
      input.options().dtype(getScalarType(output_sdtype)));

  if (nrows == 0 || output_columns == 0) {
    return output;
  }

  if (nrows == 1) {
    // Use one thread block to work on 1 row for nrows == 1
    TORCH_CHECK(
        ncols % row_ext == 0,
        "ncols (",
        ncols,
        ") must be multiple of ",
        row_ext)
    const int num_rows = ncols / row_ext;
    const int ebit = forward ? 4 : 5;
    const int bias = forward ? 15 : 31;
    constexpr int kMaxThreads = 1024;
    const auto threads_per_block =
        kMaxThreads < row_dim ? kMaxThreads : row_dim;
    FBGEMM_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "PaddedFP8rowwise_to_float_1d_cuda_kernel", [&] {
          _PaddedFP8rowwise_to_float_1d_cuda_kernel<scalar_t>
              <<<num_rows,
                 threads_per_block,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  output.data_ptr<scalar_t>(),
                  input.data_ptr<std::uint8_t>(),
                  output_columns,
                  row_dim,
                  offsets.data_ptr<int>(),
                  row_ext,
                  ebit,
                  bias);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    FBGEMM_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "PaddedFP8rowwise_to_float_2d_cuda_kernel", [&] {
          _PaddedFP8rowwise_to_float_2d_cuda_kernel<scalar_t>
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
                  row_dim,
                  offsets.data_ptr<int>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return output;
}

/// @ingroup quantize-ops-cuda
///
/// Converts a tensor of `float` values into a tensor of padded `fp8` rowwise
/// values.
///
/// @param input A tensor of `float` values.  The dtype can be either
///              `SparseType::FP32`, `SparseType::FP16`, or `SparseType::BF16`
/// @param forward
/// @param row_dim
///
/// @return A new tensor with values from the input tensor converted to padded
/// `fp8` rowwise.
DLL_PUBLIC Tensor _float_to_paddedFP8rowwise_gpu(
    const Tensor& input,
    const bool forward,
    const int64_t row_dim) {
  return _float_to_paddedFP8rowwise_gpu_t(input, forward, row_dim);
}

/// @ingroup quantize-ops-cuda
///
/// Converts a tensor of padded `fp8` rowwise values into a tensor of `float
/// values`.
///
/// @param input A tensor of `float` values.  The dtype can be either
///              `SparseType::FP32`, `SparseType::FP16`, or `SparseType::BF16`
/// @param forward
/// @param row_dim
/// @param output_last_dim
/// @param output_dtype The target floating point type, specified as integer
///                     representation of `SparseType` enum
///
/// @return A new tensor with values from the input tensor converted to `float`.
///
/// @throw c10::Error if `output_dtype` is not one of (`SparseType::FP32`,
/// `SparseType::FP16`, `SparseType::BF16`).
DLL_PUBLIC at::Tensor _paddedFP8rowwise_to_float_gpu(
    const at::Tensor& input,
    const bool forward,
    const int64_t row_dim,
    const int64_t output_last_dim,
    const int64_t output_dtype) {
  return _paddedFP8rowwise_to_float_gpu_t(
      input, forward, row_dim, output_last_dim, output_dtype);
}

} // namespace fbgemm_gpu
