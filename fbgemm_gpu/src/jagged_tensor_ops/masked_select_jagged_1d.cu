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

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void masked_select_jagged_1d_lengths_kernel(
    const index_t* __restrict__ lengths,
    const bool* __restrict__ mask,
    index_t* __restrict__ masked_lengths,
    const index_t* __restrict__ input_offsets,
    const index_t batch_size) {
  const index_t batch_idx = blockIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  const index_t input_offset = input_offsets[batch_idx];
  const index_t input_len = lengths[batch_idx];

  int32_t local_count = 0;
  for (index_t i = threadIdx.x; i < input_len; i += blockDim.x) {
    const index_t input_idx = input_offset + i;

    if (mask[input_idx]) {
      local_count++;
    }
  }

  __shared__ int32_t shared_counts[kMaxThreads];
  shared_counts[threadIdx.x] = local_count;
  __syncthreads();

  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_counts[threadIdx.x] += shared_counts[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    masked_lengths[batch_idx] = static_cast<index_t>(shared_counts[0]);
  }
}

template <typename index_t, typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void masked_select_jagged_1d_values_kernel(
    const scalar_t* __restrict__ values,
    const index_t* __restrict__ lengths,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ masked_values,
    const index_t* __restrict__ input_offsets,
    const index_t* __restrict__ output_offsets,
    const index_t batch_size) {
  const index_t batch_idx = blockIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  const index_t input_offset = input_offsets[batch_idx];
  const index_t output_offset = output_offsets[batch_idx];
  const index_t input_len = lengths[batch_idx];

  int32_t write_pos = 0;

  for (index_t i = 0; i < input_len; i++) {
    const index_t input_idx = input_offset + i;

    const bool is_masked = mask[input_idx];

    if (threadIdx.x == 0 && is_masked) {
      const index_t output_idx = output_offset + write_pos;

      masked_values[output_idx] = values[input_idx];
      write_pos++;
    }
  }
}

std::tuple<Tensor, Tensor> masked_select_jagged_1d_cuda(
    const Tensor& values,
    const Tensor& lengths,
    const Tensor& mask,
    const std::optional<bool> check_length) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(mask);

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

  Tensor input_offsets = asynchronous_complete_cumsum_gpu(lengths);

  TORCH_CHECK(
      input_offsets.numel() == batch_size + 1,
      "input_offsets should have size batch_size+1, got ",
      input_offsets.numel(),
      " expected ",
      batch_size + 1);

  Tensor mask_int = mask.to(at::kInt);
  Tensor mask_cumsum = asynchronous_complete_cumsum_gpu(mask_int);
  const int32_t num_outputs = mask_cumsum[-1].item<int32_t>();
  Tensor masked_values = at::empty({num_outputs}, values.options());

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "masked_select_jagged_1d_lengths", [&] {
        const int num_blocks = batch_size;
        // First pass: compute masked lengths
        FBGEMM_LAUNCH_KERNEL(
            (masked_select_jagged_1d_lengths_kernel<index_t>),
            num_blocks,
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream(),
            lengths.data_ptr<index_t>(),
            mask.data_ptr<bool>(),
            masked_lengths.data_ptr<index_t>(),
            input_offsets.data_ptr<index_t>(),
            static_cast<index_t>(batch_size));

        Tensor output_offsets =
            asynchronous_complete_cumsum_gpu(masked_lengths);

        TORCH_CHECK(
            output_offsets.numel() == batch_size + 1,
            "output_offsets should have size batch_size+1, got ",
            output_offsets.numel(),
            " expected ",
            batch_size + 1);

        // Second pass: write masked values
        FBGEMM_DISPATCH_ALL_TYPES(
            values.scalar_type(), "masked_select_jagged_1d_values", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (masked_select_jagged_1d_values_kernel<index_t, scalar_t>),
                  num_blocks,
                  1, // Use single thread per block for simplicity
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  values.data_ptr<scalar_t>(),
                  lengths.data_ptr<index_t>(),
                  mask.data_ptr<bool>(),
                  masked_values.data_ptr<scalar_t>(),
                  input_offsets.data_ptr<index_t>(),
                  output_offsets.data_ptr<index_t>(),
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
