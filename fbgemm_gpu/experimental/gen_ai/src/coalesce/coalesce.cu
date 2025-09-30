/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fbgemm_gpu/utils/kernel_launcher.cuh>

#include <ATen/cuda/Exceptions.h> // @manual
#include <c10/cuda/CUDAStream.h>

#include "coalesce.h"

template <typename BATCH_T, int step = 4>
__global__ void coalesce_batches_kernel(
    int64_t* input_ptrs,
    int64_t* output_ptrs,
    const int64_t* element_sizes,
    const int64_t* strides,
    const BATCH_T* old_bids,
    const BATCH_T* new_bids,
    const int64_t num_inputs,
    const int64_t num_bids) {
  const int64_t block_size = blockDim.x;
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  // (input_id, bid, block_id)
  const int64_t block_id = (tid / block_size) / (num_inputs * num_bids);
  const int64_t input_id_bid = (tid / block_size) % (num_inputs * num_bids);
  const int64_t input_id = input_id_bid / num_bids;
  const int64_t bid = input_id_bid % num_bids;
  const int64_t in_block_id = tid % block_size;

  if (input_id >= num_inputs) {
    return;
  }

  const int64_t element_size = element_sizes[input_id];

  const int64_t element_offset = (block_id * block_size + in_block_id) * step;
  const int64_t stride = strides[input_id];

  if (element_offset >= element_size) {
    return;
  }

  const int64_t input_offset = old_bids[bid] * stride;
  int64_t output_offset = new_bids[bid] * stride;
  const uint8_t* input_ptr = reinterpret_cast<uint8_t*>(input_ptrs[input_id]);
  uint8_t* output_ptr = reinterpret_cast<uint8_t*>(output_ptrs[input_id]);

  if (element_offset + step > element_size) {
    for (auto i = 0; i < element_size - element_offset; ++i) {
      output_ptr[output_offset + element_offset + i] =
          input_ptr[input_offset + element_offset + i];
    }
  } else {
#pragma unroll
    for (auto i = 0; i < step; ++i) {
      output_ptr[output_offset + element_offset + i] =
          input_ptr[input_offset + element_offset + i];
    }
  }
}

namespace fbgemm_gpu {
std::vector<at::Tensor> coalesce_batches_gpu(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& output,
    const at::Tensor& old_bids,
    const at::Tensor& new_bids) {
  static_assert(sizeof(int64_t) == sizeof(uint8_t*));

  TORCH_CHECK_EQ(old_bids.numel(), new_bids.numel());
  TORCH_CHECK_EQ(input.size(), output.size());

  if (input.size() == 0) {
    return output;
  }

  std::vector<int64_t> input_ptrs;
  std::vector<int64_t> output_ptrs;
  std::vector<int64_t> element_sizes;
  std::vector<int64_t> strides;

  input_ptrs.reserve(input.size());
  output_ptrs.reserve(input.size());
  element_sizes.reserve(input.size());
  strides.reserve(input.size());

  int64_t max_element_size = 0;

  for (const auto i : c10::irange(input.size())) {
    auto& src = input[i];
    auto& dst = output[i];
    input_ptrs.push_back(reinterpret_cast<int64_t>(src.data_ptr()));
    output_ptrs.push_back(reinterpret_cast<int64_t>(dst.data_ptr()));
    TORCH_CHECK_EQ(src.stride(0), dst.stride(0));
    strides.push_back(src.stride(0) * src.element_size());
    auto element_size = src.numel() * src.element_size() / src.size(0);
    max_element_size = std::max(element_size, max_element_size);
    element_sizes.push_back(element_size);
  }

  TORCH_CHECK_GT(max_element_size, 0);

  auto input_ptrs_tensor =
      at::tensor(
          input_ptrs, at::TensorOptions().dtype(at::kLong).pinned_memory(true))
          .to(at::kCUDA, /*non_blocking=*/true);
  auto output_ptrs_tensor =
      at::tensor(
          output_ptrs, at::TensorOptions().dtype(at::kLong).pinned_memory(true))
          .to(at::kCUDA, /*non_blocking=*/true);
  auto element_sizes_tensor =
      at::tensor(
          element_sizes,
          at::TensorOptions().dtype(at::kLong).pinned_memory(true))
          .to(at::kCUDA, /*non_blocking=*/true);
  auto strides_tensor =
      at::tensor(
          strides, at::TensorOptions().dtype(at::kLong).pinned_memory(true))
          .to(at::kCUDA, /*non_blocking=*/true);

  constexpr int block_size = 128;
  constexpr int step = 4;
  const int64_t num_blocks = input_ptrs_tensor.numel() * old_bids.numel() *
      ((max_element_size - 1) / (block_size * step) + 1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (old_bids.dtype() == at::kInt) {
    FBGEMM_LAUNCH_KERNEL(
        (coalesce_batches_kernel<int32_t, step>),
        num_blocks,
        block_size,
        0,
        stream,
        input_ptrs_tensor.data_ptr<int64_t>(),
        output_ptrs_tensor.data_ptr<int64_t>(),
        element_sizes_tensor.data_ptr<int64_t>(),
        strides_tensor.data_ptr<int64_t>(),
        old_bids.data_ptr<int32_t>(),
        new_bids.data_ptr<int32_t>(),
        input_ptrs_tensor.numel(),
        old_bids.numel());
  } else {
    FBGEMM_LAUNCH_KERNEL(
        (coalesce_batches_kernel<int64_t, step>),
        num_blocks,
        block_size,
        0,
        stream,
        input_ptrs_tensor.data_ptr<int64_t>(),
        output_ptrs_tensor.data_ptr<int64_t>(),
        element_sizes_tensor.data_ptr<int64_t>(),
        strides_tensor.data_ptr<int64_t>(),
        old_bids.data_ptr<int64_t>(),
        new_bids.data_ptr<int64_t>(),
        input_ptrs_tensor.numel(),
        old_bids.numel());
  }

  return output;
}

} // namespace fbgemm_gpu
