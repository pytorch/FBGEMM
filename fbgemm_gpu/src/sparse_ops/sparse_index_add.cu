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

template <typename index_t, typename scalar_t, int UNROLL_FACTOR>
__global__
__launch_bounds__(kMaxThreads) void index_add_2d_with_unique_indices_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        out_grad,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        orig_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor64<scalar_t, 2> in_deduped_grad,
    const int stride_D,
    const int rounded_D,
    const int remaining_D,
    const bool consecutive_indices,
    const int consecutive_range_start) {
  const int start_offset = blockIdx.x == 0 ? 0 : offsets[blockIdx.x - 1];
  const int end_offset = offsets[blockIdx.x];
  index_t dst_idx = consecutive_indices ? blockIdx.x + consecutive_range_start
                                        : unique_indices[blockIdx.x];
  const bool has_remainder = blockIdx.y == blockDim.y - 1 && remaining_D > 0 &&
      threadIdx.x < remaining_D;

  // Buffer for storing temporary results
  scalar_t sum[MAX_ELEMENTS_PER_THREAD];
  for (int i = 0; i < MAX_ELEMENTS_PER_THREAD; i++) {
    sum[i] = 0;
  }

  scalar_t sum_remainder = 0;

  // Each thread block processes max of stride_D elements
  int start_D = (blockIdx.y * stride_D) + (threadIdx.x * UNROLL_FACTOR);

  // For each row
  for (int row = start_offset; row < end_offset; row++) {
    int64_t src_idx = orig_indices[row];
    int col, i;
    for (col = start_D, i = 0; col < start_D + stride_D && col < rounded_D;
         col += blockDim.x * UNROLL_FACTOR, i += UNROLL_FACTOR) {
#pragma unroll
      for (int j = 0; j < UNROLL_FACTOR; j++) {
        sum[i + j] += LDG(&out_grad[src_idx][col + j]);
      }
    }
    if (has_remainder) {
      sum_remainder += LDG(&out_grad[src_idx][rounded_D + threadIdx.x]);
    }
  } // for each row

  // Write results to global memory
  int col, i;
  for (col = start_D, i = 0; col < start_D + stride_D && col < rounded_D;
       col += blockDim.x * UNROLL_FACTOR, i += UNROLL_FACTOR) {
#pragma unroll
    for (int j = 0; j < UNROLL_FACTOR; j++) {
      in_deduped_grad[dst_idx][col + j] = sum[i + j];
    }
  }
  if (has_remainder) {
    in_deduped_grad[dst_idx][rounded_D + threadIdx.x] += sum_remainder;
  }
}

DLL_PUBLIC Tensor index_add_with_unique_indices_cuda(
    const Tensor& grad_output,
    const Tensor& sorted_indices,
    const Tensor& orig_indices,
    std::vector<int64_t>& input_shape,
    const int consecutive_range_start,
    const int consecutive_range_length) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const int N = grad_output.size(0);

  if (grad_output.numel() == 0) {
    return at::zeros(input_shape, grad_output.options());
  }

  const Tensor grad_output_reshaped = grad_output.reshape({N, -1});
  const int D = grad_output_reshaped.size(1);

  TORCH_CHECK(sorted_indices.size(0) == N);

  Tensor input_grad = at::zeros({input_shape[0], D}, grad_output.options());
  bool consecutive_indices =
      consecutive_range_start >= 0 && consecutive_range_length > 0;

  AT_DISPATCH_INDEX_TYPES(
      sorted_indices.scalar_type(), "index_add_2d_kernel_1", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_output.scalar_type(), "index_add_2d_kernel_2", [&] {
              // UNROLL_FACTOR is determined based on the empirical study
              const int UNROLL_FACTOR = std::is_same<scalar_t, float>() ? 4 : 2;
              const int rounded_D = D / UNROLL_FACTOR * UNROLL_FACTOR;
              const int remaining_D = D - rounded_D;
              int block_size =
                  std::min(div_round_up(D, UNROLL_FACTOR), kMaxThreads);
              block_size = std::max(remaining_D, block_size);
              // Number of elements per block
              const int stride_D = MAX_ELEMENTS_PER_THREAD * block_size;

              int num_unique_indices;
              Tensor unique_indices, offsets;
              if (consecutive_indices) {
                TORCH_CHECK(
                    consecutive_range_start < input_shape[0] &&
                    consecutive_range_start + consecutive_range_length - 1 <
                        input_shape[0]);

                // Since indices are selected from consecutive range, we can
                // infer the number of unique indices from
                // consecutive_range_length
                num_unique_indices = consecutive_range_length;
                compute_frequency_sequence(
                    sorted_indices,
                    offsets,
                    consecutive_range_start,
                    num_unique_indices);
                offsets = offsets.cumsum(0);
              } else {
                Tensor unique_count;
                // Unique consecutive does D->H transfer internally
                // (enforcing synchronization between host and device)
                std::tie(unique_indices, std::ignore, unique_count) =
                    at::unique_consecutive(sorted_indices, false, true, 0);

                // This does D->H transfer
                num_unique_indices = unique_indices.numel();
                offsets = unique_count.cumsum(0);
              }

              const dim3 grid_size(
                  cuda_calc_xblock_count(num_unique_indices, 1),
                  (D + stride_D - 1) / stride_D,
                  1);

              index_add_2d_with_unique_indices_kernel<
                  index_t,
                  scalar_t,
                  UNROLL_FACTOR><<<
                  grid_size,
                  block_size,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  grad_output_reshaped
                      .packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                  consecutive_indices ? dummy_packed_accessor32<
                                            index_t,
                                            1,
                                            at::RestrictPtrTraits>()
                                      : unique_indices.packed_accessor32<
                                            index_t,
                                            1,
                                            at::RestrictPtrTraits>(),
                  orig_indices
                      .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                  offsets
                      .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                  input_grad.packed_accessor64<scalar_t, 2>(),
                  stride_D, // Pass constants as kernel args
                  rounded_D,
                  remaining_D,
                  consecutive_indices,
                  consecutive_range_start);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return input_grad.reshape(input_shape);
}

} // namespace fbgemm_gpu
