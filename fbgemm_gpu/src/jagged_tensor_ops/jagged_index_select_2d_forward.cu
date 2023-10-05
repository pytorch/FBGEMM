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

template <typename index_t, typename offset_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void jagged_index_select_2d_kernel(
    at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> output,
    const at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<offset_t, 1, at::RestrictPtrTraits>
        input_offsets,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<offset_t, 1, at::RestrictPtrTraits>
        output_offsets,
    const int64_t num_dense_output_rows) {
  __shared__ int smem[1];
  for (offset_t dense_output_offset = blockIdx.x;
       dense_output_offset < num_dense_output_rows;
       dense_output_offset += gridDim.x) {
    // Binary search
    // TODO: use multiple threads to do bin search to reduce number of steps
    if (threadIdx.x == 0) {
      const auto num_output_rows = indices.size(0);
      binary_search_range(
          smem, &output_offsets[0], dense_output_offset, num_output_rows);
    }
    __syncthreads();

    // All threads load index_pos from shared memory and return if the index_pos
    // is invalid
    int index_pos = smem[0];

    // TODO: Can also be obtained during the binary search
    // Relative index position
    const offset_t rel_index = dense_output_offset -
        (index_pos == 0 ? 0 : output_offsets[index_pos - 1]);
    const index_t index = indices[index_pos];
    const offset_t input_offset =
        (index == 0 ? 0 : input_offsets[index - 1]) + rel_index;

    const auto num_cols = input.size(1);
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
      output[dense_output_offset][i] = input[input_offset][i];
    }
  }
}

/// Copy sequences from input jagged tensor based on indices specified in the
/// indices tensor to an output jagged tensor (host function for dispatching
/// jagged_index_select_2d_kernel to GPU)
/// @param values                2D dense value tensor of input jagged tensor
/// @param indices               1D tensor that contains indices to be selected
///                              from output jagged tensor
/// @param input_offsets         1D tensor that contains offsets of input
///                              jagged tensor
/// @param output_offsets        1D tensor that contains offsets of output
///                              jagged tensor
/// @param num_dense_output_rows The total number of rows in the 2D dense value
///                              tensor of output jagged tensor
Tensor jagged_index_select_2d_forward_cuda(
    const Tensor& values,
    const Tensor& indices,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    const int64_t num_dense_output_rows) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      values, indices, input_offsets, output_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  auto num_cols = values.size(1);

  const int64_t max_num_blocks = 1024; // Arbitrarily set to this number of now
  const int64_t max_num_threads = kMaxThreads;
  const int64_t num_blocks = std::min(max_num_blocks, num_dense_output_rows);
  const int64_t num_threads = std::min(max_num_threads, num_cols);
  Tensor output =
      at::empty({num_dense_output_rows, num_cols}, values.options());

  if (num_blocks > 0) {
    // output_offsets has to be contiguous since it is passed to
    // binary_search_range which accepts raw pointers
    const auto output_offsets_contig = output_offsets.expect_contiguous();
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        values.scalar_type(),
        "jagged_index_select_2d_kernel_wrapper_1",
        [&] {
          AT_DISPATCH_INDEX_TYPES(
              indices.scalar_type(),
              "jagged_index_select_2d_kernel_wrapper_2",
              [&] {
                jagged_index_select_2d_kernel<<<
                    dim3(num_blocks),
                    dim3(num_cols),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    output.packed_accessor64<
                        scalar_t,
                        2,
                        at::RestrictPtrTraits>(),
                    values.packed_accessor64<
                        scalar_t,
                        2,
                        at::RestrictPtrTraits>(),
                    input_offsets
                        .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    indices
                        .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                    output_offsets_contig->packed_accessor32<
                        int64_t,
                        1,
                        at::RestrictPtrTraits>(),
                    num_dense_output_rows);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_index_select_2d_forward",
    fbgemm_gpu::jagged_index_select_2d_forward_cuda);
