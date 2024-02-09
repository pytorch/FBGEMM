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

std::tuple<uint32_t, uint32_t, uint32_t> calc_offsets_range_thread_block(
    const int64_t output_size,
    const int64_t num_seq) {
  uint32_t threads_per_block;
  uint32_t vector_size;
  if (output_size / num_seq < 2) {
    threads_per_block = 512;
    vector_size = 2;
  } else if (output_size / num_seq < 4) {
    threads_per_block = 512;
    vector_size = 4;
  } else if (output_size / num_seq < 64) {
    threads_per_block = 512;
    vector_size = 8;
  } else if (output_size / num_seq < 128) {
    threads_per_block = 512;
    vector_size = 16;
  } else {
    threads_per_block = 512;
    vector_size = 32;
  }
  uint32_t rows_per_block = threads_per_block / vector_size;
  const auto num_blocks = cuda_calc_xblock_count(num_seq, rows_per_block);

  return std::make_tuple(num_blocks, rows_per_block, vector_size);
}

// Kernel for calculating the offsets ranges
template <typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void _offsets_range_cuda_kernel(
    int64_t N,
    int64_t range_size,
    const scalar_t* __restrict__ offsets_data,
    scalar_t* __restrict__ range_data) {
  int start_row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int row_idx = start_row_idx; row_idx < N; row_idx += stride) {
    scalar_t row_start = offsets_data[row_idx];
    scalar_t row_end =
        (row_idx < N - 1 ? offsets_data[row_idx + 1] : range_size);
    if (blockDim.x == 32) {
      scalar_t i = row_start - (row_start & 31) + threadIdx.x;
      // unaligned part
      if (i >= row_start && i < row_end) {
        range_data[i] = i - row_start;
      }
      // aligned part
      for (i += 32; i < row_end; i += 32) {
        range_data[i] = i - row_start;
      }
    } else {
      for (scalar_t i = row_start + threadIdx.x; i < row_end; i += blockDim.x) {
        range_data[i] = i - row_start;
      }
    }
  }
}

DLL_PUBLIC Tensor
offsets_range_cuda(const Tensor& offsets, int64_t range_size) {
  TENSOR_ON_CUDA_GPU(offsets);
  TENSOR_NDIM_EQUALS(offsets, 1);

  CUDA_DEVICE_GUARD(offsets);

  auto offsets_arg = at::TensorArg(offsets, "offsets", 1);
  checkScalarTypes("_offsets_range_cuda", offsets_arg, {at::kLong, at::kInt});
  auto range = at::empty(range_size, offsets.options());
  if (range_size == 0) {
    return range;
  }
  auto offsets_contig = offsets.contiguous();
  int64_t N = offsets_contig.numel();

  uint32_t num_blocks, rows_per_block, vector_size;
  std::tie(num_blocks, rows_per_block, vector_size) =
      calc_offsets_range_thread_block(range_size, N);

  dim3 threads(vector_size, rows_per_block);

  AT_DISPATCH_INDEX_TYPES(
      offsets_contig.scalar_type(), "offsets_range_kernel", [&] {
        _offsets_range_cuda_kernel<index_t>
            <<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                range_size,
                offsets_contig.data_ptr<index_t>(),
                range.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return range;
}

DLL_PUBLIC Tensor lengths_range_cuda(
    const Tensor& t_in,
    const c10::optional<std::vector<int64_t>>& shape) {
  TENSOR_ON_CUDA_GPU(t_in);
  TENSOR_NDIM_EQUALS(t_in, 1);

  CUDA_DEVICE_GUARD(t_in);

  const auto t_in_contig = t_in.contiguous();
  const auto num_seq = t_in_contig.numel();

  Tensor offsets;
  int64_t output_size = 1;

  if (shape.has_value()) {
    offsets = fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(t_in_contig);
    output_size = c10::multiply_integers(shape.value());
  } else {
    // if we don't provide the the shape info, this is a slow path
    // we need to transfer the size of the output from GPU to CPU
    offsets = fbgemm_gpu::asynchronous_complete_cumsum_gpu(t_in_contig);
    AT_DISPATCH_INDEX_TYPES(
        t_in_contig.scalar_type(), "lengths_range_output_size", [&] {
          output_size = *(offsets[num_seq].cpu().data_ptr<index_t>());
        });
  }

  auto output = at::empty({output_size}, t_in.options());

  uint32_t num_blocks, rows_per_block, vector_size;
  std::tie(num_blocks, rows_per_block, vector_size) =
      calc_offsets_range_thread_block(output_size, num_seq);

  dim3 threads(vector_size, rows_per_block);

  AT_DISPATCH_INDEX_TYPES(
      t_in_contig.scalar_type(), "lengths_range_compute", [&] {
        fbgemm_gpu::_offsets_range_cuda_kernel<index_t>
            <<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_seq,
                output_size,
                offsets.data_ptr<index_t>(),
                output.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(CUDA, "offsets_range", fbgemm_gpu::offsets_range_cuda);
FBGEMM_OP_DISPATCH(CUDA, "lengths_range", fbgemm_gpu::lengths_range_cuda);
