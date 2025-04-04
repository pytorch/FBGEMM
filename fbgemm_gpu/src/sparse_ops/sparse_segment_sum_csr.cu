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
template <typename values_t, typename index_t>
__global__ __launch_bounds__(kMaxThreads) void _segment_sum_csr_cuda_kernel(
    int num_segments,
    int batch_size,
    const index_t* csr_seg_data,
    const values_t* values_data,
    values_t* output_data) {
  typedef FBGEMM_GPU_CUB_NS_PREFIX cub::BlockReduce<values_t, 256> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;
  index_t seg_start = csr_seg_data[blockIdx.x] * batch_size;
  index_t seg_end = csr_seg_data[blockIdx.x + 1] * batch_size;
  values_t sum = 0;

  for (index_t i = seg_start; i < seg_end; i += blockDim.x) {
    values_t thread_data;
    if (threadIdx.x < seg_end - i) {
      thread_data = values_data[i + threadIdx.x];
    }

    values_t aggregate =
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

  FBGEMM_DISPATCH_ALL_TYPES(
      values.scalar_type(), "_segment_sum_csr_cuda_1", [&] {
        using values_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(
            csr_seg.scalar_type(), "_segment_sum_csr_cuda_2", [&] {
              _segment_sum_csr_cuda_kernel<values_t, index_t>
                  <<<num_blocks,
                     threads_per_block,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      csr_seg.numel() - 1,
                      batch_size,
                      csr_seg.data_ptr<index_t>(),
                      values.data_ptr<values_t>(),
                      output.data_ptr<values_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(CUDA, "segment_sum_csr", fbgemm_gpu::segment_sum_csr_cuda);
