/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <limits>

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Kernel for calculating the segmented sum for sparse matrix with CSR format.
// See https://moderngpu.github.io/segreduce.html
template <typename values_t, typename index_t>
__global__ __launch_bounds__(kMaxThreads) void _segment_sum_csr_cuda_kernel(
    int num_segments,
    int64_t batch_size,
    const index_t* csr_seg_data,
    const values_t* values_data,
    values_t* output_data) {
  typedef FBGEMM_GPU_CUB_NS_PREFIX cub::BlockReduce<values_t, 256> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Grid-stride over segments so a capped grid (used on ROCm to avoid the
  // 2^32 launch-side limit) still covers all segments.
  for (auto seg = blockIdx.x; seg < num_segments; seg += gridDim.x) {
    // 64-bit segment offsets: csr_seg_data[seg] (index_t) * batch_size
    // (int64_t) can exceed INT_MAX, so accumulate in int64_t to avoid
    // truncation.
    int64_t seg_start = csr_seg_data[seg] * batch_size;
    int64_t seg_end = csr_seg_data[seg + 1] * batch_size;
    values_t sum = 0;

    for (int64_t i = seg_start; i < seg_end; i += blockDim.x) {
      values_t thread_data;
      if (threadIdx.x < seg_end - i) {
        thread_data = values_data[i + threadIdx.x];
      }

      // cap at blockDim.x to fit cub's int num_valid without truncation
      const int num_valid =
          static_cast<int>(seg_end - i < blockDim.x ? seg_end - i : blockDim.x);
      values_t aggregate =
          BlockReduce(temp_storage).Sum(thread_data, num_valid);

      __syncthreads();

      if (threadIdx.x == 0) {
        sum += aggregate;
      }
    }

    if (threadIdx.x == 0) {
      output_data[seg] = sum;
    }

    // Ensure all threads have finished using temp_storage before the next
    // outer-iteration's BlockReduce overwrites it.
    __syncthreads();
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
  const int64_t num_segments = csr_seg.numel() - 1;
  TORCH_CHECK(
      num_segments <= std::numeric_limits<int32_t>::max(),
      "segment_sum_csr: number of segments (",
      num_segments,
      ") exceeds the maximum CUDA grid dimension");
  // HIP enforces a hard limit of 2^32 total threads per launch (unlike CUDA,
  // which silently wraps). _segment_sum_csr_cuda_kernel grid-strides over
  // segments, so capping the launch grid is correctness-preserving.
  // See: https://github.com/ROCm/hip/issues/2253
  const uint32_t num_blocks = utils::cuda::cap_grid_dim_x(
      static_cast<uint32_t>(num_segments),
      threads_per_block,
      at::cuda::getCurrentCUDAStream());

  FBGEMM_DISPATCH_ALL_TYPES(
      values.scalar_type(), "_segment_sum_csr_cuda_1", [&] {
        using values_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(
            csr_seg.scalar_type(), "_segment_sum_csr_cuda_2", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (_segment_sum_csr_cuda_kernel<values_t, index_t>),
                  num_blocks,
                  threads_per_block,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  static_cast<int32_t>(num_segments),
                  batch_size,
                  csr_seg.data_ptr<index_t>(),
                  values.data_ptr<values_t>(),
                  output.mutable_data_ptr<values_t>());
            });
      });

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(CUDA, "segment_sum_csr", fbgemm_gpu::segment_sum_csr_cuda);
