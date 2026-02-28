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

// Kernel to compute output lengths from prefix sum differences
template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void masked_select_jagged_1d_lengths_kernel(
    const index_t* __restrict__ input_offsets,
    const int32_t* __restrict__ mask_prefix_sum,
    index_t* __restrict__ output_lengths,
    const index_t batch_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= batch_size) {
    return;
  }

  const index_t start = input_offsets[idx];
  const index_t end = input_offsets[idx + 1];
  output_lengths[idx] =
      static_cast<index_t>(mask_prefix_sum[end] - mask_prefix_sum[start]);
}

// Kernel to copy masked values in parallel using prefix sum for O(1) index
// lookups. Output offset per batch is derived directly from mask_prefix_sum.
template <typename index_t, typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void masked_select_jagged_1d_values_kernel(
    const scalar_t* __restrict__ values,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ masked_values,
    const index_t* __restrict__ input_offsets,
    const int32_t* __restrict__ mask_prefix_sum,
    const index_t batch_size) {
  const index_t batch_idx = blockIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  const index_t input_offset = input_offsets[batch_idx];
  // Output offset derived directly from prefix sum — no separate cumsum needed
  const int32_t output_offset = mask_prefix_sum[input_offset];
  const index_t input_len =
      input_offsets[batch_idx + 1] - input_offsets[batch_idx];

  for (index_t i = threadIdx.x; i < input_len; i += blockDim.x) {
    const index_t input_idx = input_offset + i;
    if (mask[input_idx]) {
      const int32_t out_idx = mask_prefix_sum[input_idx] - output_offset;
      masked_values[output_offset + out_idx] = values[input_idx];
    }
  }
}

std::tuple<Tensor, Tensor> masked_select_jagged_1d_cuda(
    const Tensor& values,
    const Tensor& lengths,
    const Tensor& mask,
    const std::optional<bool> check_length) {
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(values);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(lengths);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(mask);

  CUDA_DEVICE_GUARD(values);

  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(lengths.dim() == 1);
  TORCH_CHECK(mask.dim() == 1);

  if (check_length.has_value() && check_length.value()) {
    TORCH_CHECK(
        mask.numel() == values.numel(),
        "mask and values should have the same numel, but got mask numel: ",
        mask.numel(),
        " values numel: ",
        values.numel());
  }

  const auto batch_size = lengths.numel();
  Tensor masked_lengths = at::empty_like(lengths);

  if (batch_size == 0) {
    Tensor masked_values = at::empty({0}, values.options());
    return {masked_values, masked_lengths};
  }

  // Compute input offsets (complete cumsum, size B+1)
  Tensor input_offsets = asynchronous_complete_cumsum_gpu(lengths);

  // Compute mask prefix sum (complete cumsum, size N+1)
  // mask_prefix_sum[i] = count of true values in mask[0..i-1]
  Tensor mask_int = mask.to(at::kInt);
  Tensor mask_cumsum = asynchronous_complete_cumsum_gpu(mask_int);
  const int32_t num_outputs = mask_cumsum[-1].item<int32_t>();
  Tensor masked_values = at::empty({num_outputs}, values.options());

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "masked_select_jagged_1d", [&] {
        // Compute output lengths from prefix sum differences
        {
          constexpr int threads = 256;
          const int blocks = (batch_size + threads - 1) / threads;
          FBGEMM_LAUNCH_KERNEL(
              (masked_select_jagged_1d_lengths_kernel<index_t>),
              blocks,
              threads,
              0,
              at::cuda::getCurrentCUDAStream(),
              input_offsets.data_ptr<index_t>(),
              mask_cumsum.data_ptr<int32_t>(),
              masked_lengths.data_ptr<index_t>(),
              static_cast<index_t>(batch_size));
        }

        // Copy masked values using prefix sum for parallel writes
        FBGEMM_DISPATCH_ALL_TYPES(
            values.scalar_type(), "masked_select_jagged_1d_values", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (masked_select_jagged_1d_values_kernel<index_t, scalar_t>),
                  batch_size,
                  kMaxThreads,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  values.data_ptr<scalar_t>(),
                  mask.data_ptr<bool>(),
                  masked_values.data_ptr<scalar_t>(),
                  input_offsets.data_ptr<index_t>(),
                  mask_cumsum.data_ptr<int32_t>(),
                  static_cast<index_t>(batch_size));
            });
      });

  return {masked_values, masked_lengths};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "masked_select_jagged_1d",
    fbgemm_gpu::masked_select_jagged_1d_cuda);
