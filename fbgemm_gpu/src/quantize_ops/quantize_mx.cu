/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "c10/core/ScalarType.h"
#include "fbgemm_gpu/ops_utils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include "quantize_mx.cuh"

namespace fbgemm_gpu {

//-----------------------------------------------------------------------
// quantize_mx_cuda
//-----------------------------------------------------------------------
DLL_PUBLIC at::Tensor quantize_mx_cuda(
    const at::Tensor& input,
    const std::vector<int64_t>& split_sizes,
    const int64_t scale_bits,
    const int64_t elem_ebits,
    const int64_t elem_mbits,
    const double elem_max_norm,
    const int64_t mx_group_size,
    const bool flush_fp32_subnorms = false,
    const int64_t rounding_mode = 0) {
  TORCH_CHECK((split_sizes.size() > 0), "Input split sizes cannot be empty");
  TORCH_CHECK((mx_group_size % 32 == 0), "Group size needs to be power of 2");
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input);

  at::Device device = input.device();
  const at::cuda::CUDAGuard device_guard{device};
  const uint32_t total_elems = input.numel();
  const uint32_t total_num_groups = input.numel() / mx_group_size;

  // Compute offsets to be passed to kernel
  auto start_output_cumsum =
      at::empty(split_sizes.size() + 1, at::TensorOptions().dtype(at::kUInt32));
  auto group_ids =
      at::empty(total_num_groups, at::TensorOptions().dtype(at::kUInt32));
  auto num_groups_cumsum =
      at::empty(split_sizes.size() + 1, at::TensorOptions().dtype(at::kUInt32));
  uint32_t offset = 0;
  start_output_cumsum[0] = 0;
  num_groups_cumsum[0] = 0;
  uint32_t num_groups_cumsum_ = 0;
  int64_t start_idx = 0;
  int64_t end_idx = 0;
  for (int i = 0; i < split_sizes.size(); i++) {
    const uint32_t split_size = split_sizes[i];

    TORCH_CHECK(
        split_size % mx_group_size == 0,
        " Number of inputs needs to be a multiple of group size");
    const uint32_t num_groups = split_size / mx_group_size;
    end_idx += num_groups;
    offset += align((split_size / 2) + num_groups, 16);
    start_output_cumsum[i + 1] = offset;
    num_groups_cumsum_ += num_groups;
    num_groups_cumsum[i + 1] = num_groups_cumsum_;
    group_ids.index_put_({at::indexing::Slice(start_idx, end_idx)}, i);
    start_idx = end_idx;
  }

  // TODO: Search in the kernel
  start_output_cumsum = start_output_cumsum.to(device, /*non_blocking=*/true);
  group_ids = group_ids.to(device, /*non_blocking=*/true);
  num_groups_cumsum = num_groups_cumsum.to(device, /*non_blocking=*/true);

  RoundingMode rd = static_cast<RoundingMode>(rounding_mode);

  const int num_groups_per_block = MAX_THREADS / mx_group_size;
  const auto gridDim_x = round_up(total_num_groups, num_groups_per_block);

  const dim3 gridDim(gridDim_x);
  const dim3 blockDim(mx_group_size, num_groups_per_block);

  // Use shmem to find max exponent (int) and temporarily store output (unint8)
  // max(num_elem_in_block * sizeof(int), num_elem_in_block /2 * sizeof(uint8))
  const int smem_size = num_groups_per_block * mx_group_size * (sizeof(int));
  auto output = at::empty(
      offset, // 4 = sizeof(float)
      input.options().dtype(at::kByte));
  // Call CUDA kernel
  if (input.dtype() == torch::ScalarType::Half) {
    AT_ASSERTM(0, " fp16 not supported for MX");
  } else {
    quantize_float_to_mx4_kernel<<<gridDim, blockDim, smem_size>>>(
        input.data_ptr<float>(),
        mx_group_size,
        group_ids.data_ptr<uint32_t>(),
        start_output_cumsum.data_ptr<uint32_t>(),
        num_groups_cumsum.data_ptr<uint32_t>(),
        total_elems,
        flush_fp32_subnorms,
        rd,
        output.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  gpuErrchk(cudaPeekAtLastError());
  return output;
}

DLL_PUBLIC at::Tensor dequantize_mx_cuda(
    const at::Tensor& input,
    const std::vector<int64_t>& split_sizes,
    const int64_t mx_group_size) {
  TORCH_CHECK((split_sizes.size() > 0), "Input sizes cannot be empty");
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input);
  at::Device device = input.device();
  const at::cuda::CUDAGuard device_guard{device};
  const int64_t total_elems =
      std::accumulate(split_sizes.begin(), split_sizes.end(), 0);
  const uint32_t total_num_groups = total_elems / mx_group_size;

  auto start_output_cumsum =
      at::empty(split_sizes.size() + 1, at::TensorOptions().dtype(at::kUInt32));
  auto group_ids =
      at::empty(total_num_groups, at::TensorOptions().dtype(at::kUInt32));
  auto num_groups_cumsum =
      at::empty(split_sizes.size() + 1, at::TensorOptions().dtype(at::kUInt32));
  uint32_t offset = 0;
  start_output_cumsum[0] = 0;
  num_groups_cumsum[0] = 0;
  uint32_t num_groups_cumsum_ = 0;
  int64_t start_idx = 0;
  int64_t end_idx = 0;

  for (int i = 0; i < split_sizes.size(); i++) {
    const uint32_t split_size = split_sizes[i];

    TORCH_CHECK(
        split_size % mx_group_size == 0,
        " Number of inputs needs to be a multiple of group size");
    const uint32_t num_groups = split_size / mx_group_size;
    end_idx += num_groups;
    offset += align((split_size / 2) + num_groups, 16);
    start_output_cumsum[i + 1] = offset;
    num_groups_cumsum_ += num_groups;
    num_groups_cumsum[i + 1] = num_groups_cumsum_;
    group_ids.index_put_({at::indexing::Slice(start_idx, end_idx)}, i);
    start_idx = end_idx;
  }
  start_output_cumsum = start_output_cumsum.to(device, /*non_blocking=*/true);
  group_ids = group_ids.to(device, /*non_blocking=*/true);
  num_groups_cumsum = num_groups_cumsum.to(device, /*non_blocking=*/true);

  auto output = at::empty(
      total_elems, // 4 = sizeof(float)
      input.options().dtype(at::kFloat));
  const int num_groups_per_block = MAX_THREADS / mx_group_size;
  const auto gridDim_x = round_up(total_num_groups, num_groups_per_block);

  const dim3 gridDim(gridDim_x);
  const dim3 blockDim(mx_group_size, num_groups_per_block);

  // Call CUDA kernel
  if (input.dtype() == torch::ScalarType::Half) {
    AT_ASSERTM(0, " fp16 not supported for MX");
  } else {
    dequantize_mx4_to_float_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<uint8_t>(),
        mx_group_size,
        total_elems,
        group_ids.data_ptr<uint32_t>(),
        start_output_cumsum.data_ptr<uint32_t>(),
        num_groups_cumsum.data_ptr<uint32_t>(),
        output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  gpuErrchk(cudaPeekAtLastError());
  return output;
}

} // namespace fbgemm_gpu
