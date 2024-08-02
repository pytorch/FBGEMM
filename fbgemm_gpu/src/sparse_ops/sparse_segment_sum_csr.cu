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

// Kernel for calculating the segmented sum for sparse matrix with CSR format.
// See https://moderngpu.github.io/segreduce.html
template <typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void _segment_sum_csr_cuda_kernel(
    int num_segments,
    int batch_size,
    const int* csr_seg_data,
    const scalar_t* values_data,
    scalar_t* output_data) {
  typedef FBGEMM_GPU_CUB_NS_PREFIX cub::BlockReduce<scalar_t, 256> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;
  int seg_start = csr_seg_data[blockIdx.x] * batch_size;
  int seg_end = csr_seg_data[blockIdx.x + 1] * batch_size;
  scalar_t sum = 0;

  for (int i = seg_start; i < seg_end; i += blockDim.x) {
    scalar_t thread_data;
    if (threadIdx.x < seg_end - i) {
      thread_data = values_data[i + threadIdx.x];
    }

    scalar_t aggregate =
        BlockReduce(temp_storage).Sum(thread_data, seg_end - i);

    __syncthreads();

    if (threadIdx.x == 0) {
      sum += aggregate;
    }
  }

  if (threadIdx.x == 0) {
    output_data[blockIdx.x] = sum;
  }
}

DLL_PUBLIC Tensor segment_sum_csr_cuda(
    const int64_t batch_size,
    const Tensor& csr_seg,
    const Tensor& values) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(csr_seg, values);

  CUDA_DEVICE_GUARD(values);

  TORCH_CHECK(csr_seg.numel() >= 1, "The csr_seg tensor should not be empty")

  auto output = at::empty(csr_seg.numel() - 1, values.options());

  if (csr_seg.numel() == 1) {
    return output;
  }

  constexpr uint32_t threads_per_block = 256;
  const uint32_t num_blocks = csr_seg.numel() - 1;

  FBGEMM_DISPATCH_ALL_TYPES(values.scalar_type(), "_segment_sum_csr_cuda", [&] {
    _segment_sum_csr_cuda_kernel<scalar_t>
        <<<num_blocks,
           threads_per_block,
           0,
           at::cuda::getCurrentCUDAStream()>>>(
            csr_seg.numel() - 1,
            batch_size,
            csr_seg.data_ptr<int>(),
            values.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(CUDA, "segment_sum_csr", fbgemm_gpu::segment_sum_csr_cuda);
