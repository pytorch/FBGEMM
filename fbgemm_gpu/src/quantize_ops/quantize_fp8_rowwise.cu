/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"
#include "fbgemm_gpu/config/feature_gates.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {

// FP32/FP16 -> FP8 rowwise kernel
template <typename input_t>
__global__ inline void _float_to_FP8rowwise_cuda_kernel(
    const pta::PackedTensorAccessor64<input_t, 1, at::RestrictPtrTraits> input,
    const int64_t nrows,
    const int64_t ncols,
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> output,
    const bool forward) {
  // Assert if index is out of bound
  CUDA_KERNEL_ASSERT(nrows * ncols >= 0);

  constexpr float kEpsilon = 1e-20f;
  const int ebit = forward ? 4 : 5;
  const int bias = forward ? 15 : 31;
  const float max_pos = forward ? 0.9375 : 0.875;

  const int64_t ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int64_t output_columns = ncols_aligned + 2 * sizeof(float);

  const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nrows) {
    const input_t* input_row = &input[row * ncols];
    std::uint8_t* output_row = &output[row * output_columns];
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + ncols_aligned);

    const float minimum_element = fbgemm_gpu::min(input_row, input_row + ncols);
    const float maximum_element = fbgemm_gpu::max(input_row, input_row + ncols);
    const auto scale =
        max_pos / (kEpsilon + fmaxf(maximum_element, -minimum_element));
    output_row_scale_bias[0] = scale;
    // 8 bytes are allocated for scale but only 4 bytes are used
    // value of the unassigned 4 bytes are hence indeterministic
    // Initialize it to make the output deterministic for PT2 compliance
    output_row_scale_bias[1] = 0.0;
    for (int64_t col = 0; col < ncols; ++col) {
      output_row[col] =
          float_to_hfp8(to_float(input_row[col]) * scale, ebit, bias, max_pos);
    }
  }
}

template <typename input_t>
__global__ inline void _get_FP8_qparam_cuda_kernel(
    const pta::PackedTensorAccessor64<input_t, 1, at::RestrictPtrTraits> input,
    const int64_t nrows,
    const int64_t ncols,
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> output,
    const bool forward) {
  // Assert if index is out of bound
  CUDA_KERNEL_ASSERT(nrows * ncols >= 0);
  const int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int64_t output_columns = ncols_aligned + 2 * sizeof(float);

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
  const auto lane_width = blockDim.x;

  // March warp-wise through the row, doing thread local min and max reductions.
  // This loop will only execute once when ncol <= 32
  if (row < nrows) {
    const input_t* input_row = &input[row * ncols];
    for (int64_t col = threadIdx.x; col < ncols; col += lane_width) {
      // Get thread-local minmax. These are the smallest min and max ever seen
      // by this thread.
      maximum_element = fmaxf(maximum_element, fabs(to_float(input_row[col])));
    }
  }

  // Perform warp-wide min and max reductions. All threads in the warp
  // participate, even if they aren't assigned to a row, since we can't assume
  // the existence of the `*_sync` warp primitives with support for masking.
  for (int offset = lane_width >> 1; offset > 0; offset >>= 1) {
    maximum_element =
        fmaxf(maximum_element, shfl_xor(maximum_element, offset, lane_width));
  }

  // only the leading thread in the warp is needed to return the final result in
  // output. Additionally, threads mapped to non-existent rows do not write to
  // the output array.
  if (threadIdx.x != 0 || row >= nrows) {
    return;
  }
  float* const output_row_qparams =
      reinterpret_cast<float*>(&output[row * output_columns + ncols_aligned]);

  output_row_qparams[0] = max_pos / (kEpsilon + maximum_element);
  // Initialize it to make the output deterministic for PT2 compliance
  output_row_qparams[1] = 0.0;
}

template <typename scalar_t>
struct VectorSizeTraits {
  // Default to 4 elements for most types (16 bytes for float)
  static constexpr int value = 4;
};

// Specialization for half (float16)
template <>
struct VectorSizeTraits<c10::Half> {
  // 8 elements for half precision (16 bytes total)
  static constexpr int value = 8;
};

// Specialization for __nv_bfloat16
template <>
struct VectorSizeTraits<c10::BFloat16> {
  // 8 elements for bfloat16 precision (16 bytes total)
  static constexpr int value = 8;
};

// aligned vector generates vectorized load/store on CUDA (copy-pasted from
// MemoryAccess.cuh)
template <typename scalar_t, int vec_size = VectorSizeTraits<scalar_t>::value>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <typename input_t>
__global__ inline void _compute_FP8_quantize_cuda_kernel(
    const pta::PackedTensorAccessor64<input_t, 1, at::RestrictPtrTraits> input,
    const int64_t nrows,
    const int64_t ncols,
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> output,
    const bool forward) {
  // Assert if index is out of bound
  CUDA_KERNEL_ASSERT(nrows * ncols >= 0);

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

  const int64_t ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int64_t output_columns = ncols_aligned + 2 * sizeof(float);

  int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    std::uint8_t* output_row = &output[row * output_columns];
    if (col < ncols) {
      float* row_qparams = reinterpret_cast<float*>(output_row + ncols_aligned);
      const float scale = row_qparams[0];
      output_row[col] = float_to_hfp8(
          to_float(input[row * ncols + col]) * scale, ebit, bias, max_pos);
    }
  }
}

template <typename input_t>
__global__ inline void _compute_FP8_quantize_cuda_vectorized_kernel(
    const pta::PackedTensorAccessor64<input_t, 1, at::RestrictPtrTraits> input,
    const int64_t nrows,
    const int64_t ncols,
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> output,
    const bool forward) {
  CUDA_KERNEL_ASSERT(nrows * ncols >= 0);

  // Calculate global row index with 2D thread blocks
  const int64_t gy = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  static constexpr int vec_size = VectorSizeTraits<input_t>::value;
  // Early return if row is out of bounds
  if (gy >= nrows || (thread_idx * vec_size) >= ncols) {
    return;
  }

  int ebit = forward ? 4 : 5;
  int bias = forward ? 15 : 31;
  float max_pos = forward ? 0.9375 : 0.875;

  // Calculate output width
  const auto ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const auto output_columns = ncols_aligned + 2 * sizeof(float);

  // Calculate base offsets for the current row
  const int64_t input_row_offset = gy * ncols;
  const int64_t output_row_offset = gy * output_columns;

  // Calculate the position where the scale values are stored
  const int64_t scale_offset = output_row_offset + ncols_aligned;
  const float scale_value = reinterpret_cast<float*>(&output[scale_offset])[0];

  const int64_t vector_blocks = ncols / vec_size;

  using vec_t = aligned_vector<input_t, vec_size>;
  using vec_i = aligned_vector<uint8_t, vec_size>;

  const int64_t col_idx = thread_idx * vec_size;

  // Don't access beyond the valid input data
  if (col_idx + (vec_size - 1) < ncols) {
    // Load vec_size elements - handle both aligned and unaligned cases
    // correctly
    const vec_t* input_row =
        reinterpret_cast<const vec_t*>(&input[input_row_offset + col_idx]);

    vec_i* output_row =
        reinterpret_cast<vec_i*>(&output[output_row_offset + col_idx]);

#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      output_row->val[i] = float_to_hfp8(
          to_float(input_row->val[i]) * scale_value, ebit, bias, max_pos);
    }
  }

  // 2. Process any remaining elements (less than vec_size) with scalar
  // operations
  const int64_t remaining_start = vector_blocks * vec_size;
  for (int64_t col = remaining_start + threadIdx.x; col < ncols;
       col += blockDim.x) {
    output[output_row_offset + col] = float_to_hfp8(
        to_float(input[input_row_offset + col]) * scale_value,
        ebit,
        bias,
        max_pos);
  }
}

template <typename output_t>
__global__ inline void _FP8rowwise_to_float_cuda_kernel(
    pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> input,
    const int64_t nrows,
    const int64_t ncols,
    pta::PackedTensorAccessor64<output_t, 1, at::RestrictPtrTraits> output,
    const bool forward) {
  // Assert if index is out of bound
  CUDA_KERNEL_ASSERT(nrows * ncols >= 0);

  const int64_t output_columns = ncols - 2 * sizeof(float);
  const int ebit = forward ? 4 : 5;
  const int bias = forward ? 15 : 31;

  int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const int64_t col =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row_incre = blockDim.y * gridDim.y;

  for (/*row*/; row < nrows; row += row_incre) {
    if (col < output_columns) {
      const std::uint8_t* input_row = &input[row * ncols];
      output_t* output_row = &output[row * output_columns];
      const float* input_row_scale_bias =
          reinterpret_cast<const float*>(input_row + output_columns);
      const float output_ =
          hfp8_to_float(input_row[col], ebit, bias) / input_row_scale_bias[0];
      quantize_float_store(&output_row[col], output_);
    }
  }
}

} // namespace

// revising INT8 rowwise template for FP8 rowwise quantization
template <typename input_t>
Tensor _float_to_FP8rowwise_gpu_t(const Tensor& input, const bool forward) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  CUDA_DEVICE_GUARD(input);

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int64_t nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int64_t ncols = input_sizes[last_dim];
  const int64_t ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int64_t output_columns = ncols_aligned + 2 * sizeof(float);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;

  if (nrows == 0 || ncols == 0) {
    return at::zeros(
        output_dims, // 4 = sizeof(float)
        input.options().dtype(at::kByte));
  }

  auto output = at::empty(
      output_dims, // 4 = sizeof(float)
      input.options().dtype(at::kByte));

  constexpr int threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(nrows, threads_per_block);
  // think unsigned as we use 0, 255

  const auto input_1D = input.flatten();
  const auto output_1D = output.flatten();

  if (nrows <= 20) {
    FBGEMM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "_float_to_FP8rowwise_cuda_kernel", [&] {
#ifdef FBGEMM_GPU_MEMCHECK
          const auto func_name = "_float_to_FP8rowwise_cuda_kernel";
#endif
          _float_to_FP8rowwise_cuda_kernel<scalar_t>
              <<<num_blocks,
                 threads_per_block,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  MAKE_PTA_WITH_NAME(func_name, input_1D, scalar_t, 1, 64),
                  nrows,
                  ncols,
                  MAKE_PTA_WITH_NAME(func_name, output_1D, uint8_t, 1, 64),
                  forward);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  } else {
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

      const auto rows_per_block = threads_per_block / blockDim_x;
      const auto num_blocks_warp =
          cuda_calc_xblock_count(nrows, rows_per_block);

      FBGEMM_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "_get_FP8_qparam_cuda_kernel", [&] {
#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name = "_get_FP8_qparam_cuda_kernel";
#endif
            _get_FP8_qparam_cuda_kernel<scalar_t>
                <<<num_blocks_warp,
                   dim3(blockDim_x, rows_per_block),
                   0,
                   at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, input_1D, scalar_t, 1, 64),
                    nrows,
                    ncols,
                    MAKE_PTA_WITH_NAME(func_name, output_1D, uint8_t, 1, 64),
                    forward);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    }

    const uintptr_t addr = reinterpret_cast<uintptr_t>(&input);
    const static bool use_vectorization =
        ((addr % 16) == 0) &&
        !config::is_feature_enabled(
            config::FeatureGateName::DISABLE_FP8_QUANT_VECTORIZATION);

    if (use_vectorization) {
      const constexpr int vec_size = VectorSizeTraits<input_t>::value;
      const int blockDim_x = std::min(
          ncols > vec_size ? ncols / vec_size : 1,
          static_cast<int64_t>(threads_per_block));
      dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
      const auto gridDim_x = cuda_calc_xblock_count(ncols, blockDim.x);
      const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
      dim3 gridDim(gridDim_x, gridDim_y);

      FBGEMM_DISPATCH_FLOATING_TYPES(
          input.scalar_type(),
          "_compute_FP8_quantize_cuda_vectorized_kernel",
          [&] {
#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name =
                "_compute_FP8_quantize_cuda_vectorized_kernel";
#endif
            _compute_FP8_quantize_cuda_vectorized_kernel<scalar_t>
                <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, input_1D, scalar_t, 1, 64),
                    nrows,
                    ncols,
                    MAKE_PTA_WITH_NAME(func_name, output_1D, uint8_t, 1, 64),
                    forward);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    } else {
      const int blockDim_x =
          std::min(ncols, static_cast<int64_t>(threads_per_block));
      dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
      const auto gridDim_x = cuda_calc_xblock_count(ncols, blockDim.x);
      const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
      dim3 gridDim(gridDim_x, gridDim_y);

      FBGEMM_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "_compute_FP8_quantize_cuda_kernel", [&] {
#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name = "_compute_FP8_quantize_cuda_kernel";
#endif
            _compute_FP8_quantize_cuda_kernel<scalar_t>
                <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA_WITH_NAME(func_name, input_1D, scalar_t, 1, 64),
                    nrows,
                    ncols,
                    MAKE_PTA_WITH_NAME(func_name, output_1D, uint8_t, 1, 64),
                    forward);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    }
  }

  return output;
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of `float` values into a tensor of `fp8` values.
///
/// @param input A tensor of `float` values.  The dtype can be either
///              `SparseType::FP32`, `SparseType::FP16`, or `SparseType::BF16`
/// @param forward
///
/// @return A new tensor with values from the input tensor converted to `fp8`.
///
/// @throw c10::Error if `input.dtype` is not one of (`SparseType::FP32`,
/// `SparseType::FP16`, or `SparseType::BF16`).
DLL_PUBLIC Tensor
_float_to_FP8rowwise_gpu(const Tensor& input, const bool forward) {
  auto input_type = input.dtype();
  if (input_type == at::kHalf) {
    return _float_to_FP8rowwise_gpu_t<half>(input, forward);
  } else if (input_type == at::kBFloat16) {
    return _float_to_FP8rowwise_gpu_t<__nv_bfloat16>(input, forward);
  } else {
    return _float_to_FP8rowwise_gpu_t<float>(input, forward);
  }
}

Tensor _FP8rowwise_to_float_gpu_t(
    const Tensor& input,
    bool forward,
    const int64_t output_dtype) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  CUDA_DEVICE_GUARD(input);

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int64_t nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int64_t ncols = input_sizes[last_dim];
  const int64_t ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int64_t output_columns = ncols_aligned - 2 * sizeof(float);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple
  // of that size).
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  const auto output_sdtype = static_cast<SparseType>(output_dtype);
  TORCH_CHECK(
      output_sdtype == SparseType::FP32 || output_sdtype == SparseType::FP16 ||
      output_sdtype == SparseType::BF16);

  if (nrows == 0 || output_columns == 0) {
    return at::zeros(
        output_dims, // 4 = sizeof(float)
        input.options().dtype(getScalarType(output_sdtype)));
  }

  Tensor output = at::empty(
      output_dims, // 4 = sizeof(float)
      input.options().dtype(getScalarType(output_sdtype)));

  constexpr int threads_per_block = 256;

  const int blockDim_x =
      std::min(static_cast<int64_t>(threads_per_block), output_columns);
  const dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);

  const auto gridDim_x = cuda_calc_xblock_count(output_columns, blockDim.x);
  const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
  const dim3 gridDim(gridDim_x, gridDim_y);

  const auto input_1D = input.flatten();
  const auto output_1D = output.flatten();

  FBGEMM_DISPATCH_FLOATING_TYPES(
      output.scalar_type(), "FP8rowwise_to_float_cuda_kernel", [&] {
#ifdef FBGEMM_GPU_MEMCHECK
        const auto func_name = "_FP8rowwise_to_float_cuda_kernel";
#endif
        _FP8rowwise_to_float_cuda_kernel<scalar_t>
            <<<gridDim, blockDim, 0, at::cuda::getCurrentCUDAStream()>>>(
                MAKE_PTA_WITH_NAME(func_name, input_1D, uint8_t, 1, 64),
                nrows,
                ncols,
                MAKE_PTA_WITH_NAME(func_name, output_1D, scalar_t, 1, 64),
                forward);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output;
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of `fp8` values into a tensor of `float` values.
///
/// @param input A tensor of `fp8` values
/// @param forward
/// @param output_dtype The target floating point type, specified as integer
///                     representation of `SparseType` enum
///
/// @return A new tensor with values from the input tensor converted to
/// `float` (with `dtype` of either `SparseType::FP32`, `SparseType::FP16`, or
/// `SparseType::BF16`).
///
/// @throw c10::Error if `output_dtype` is not one of (`SparseType::FP32`,
/// `SparseType::FP16`, or `SparseType::BF16`).
DLL_PUBLIC at::Tensor _FP8rowwise_to_float_gpu(
    const at::Tensor& input,
    bool forward,
    const int64_t output_dtype) {
  return _FP8rowwise_to_float_gpu_t(input, forward, output_dtype);
}

} // namespace fbgemm_gpu
