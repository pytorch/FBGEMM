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

#include <ATen/core/TensorAccessor.h>
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "quantize_mx.cuh"

namespace fbgemm_gpu {

//-----------------------------------------------------------------------
// quantize_mx_cuda
//-----------------------------------------------------------------------
DLL_PUBLIC at::Tensor quantize_mx_cuda(
    const at::Tensor& input,
    const int64_t scale_bits,
    const int64_t elem_ebits,
    const int64_t elem_mbits,
    const double elem_max_norm,
    const int64_t mx_group_size,
    const bool flush_fp32_subnorms = false,
    const int64_t rounding_mode = 0) {
  TORCH_CHECK((mx_group_size % 32 == 0), "Group size needs to be power of 2");
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input);

  at::Device device = input.device();
  const at::cuda::CUDAGuard device_guard{device};
  const uint32_t total_elems = input.numel();
  const uint32_t total_num_groups = input.numel() / mx_group_size;

  RoundingMode rd = static_cast<RoundingMode>(rounding_mode);

  const int num_groups_per_block = MAX_THREADS / mx_group_size;
  const auto gridDim_x = round_up(total_num_groups, num_groups_per_block);

  const dim3 gridDim(gridDim_x);
  const dim3 blockDim(mx_group_size, num_groups_per_block);

  // Use shmem to find max exponent (int) and temporarily store output (unint8)
  // max(num_elem_in_block * sizeof(int), num_elem_in_block /2 * sizeof(uint8))
  const int smem_size = num_groups_per_block * mx_group_size * (sizeof(int));
  auto output = at::empty(
      (total_elems / 2) + total_num_groups, input.options().dtype(at::kByte));
  // Call CUDA kernel
  if (input.dtype() == torch::ScalarType::Half) {
    TORCH_CHECK(0, " fp16 not supported for MX");
  } else {
#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name = "quantize_float_to_mx4_kernel";
#endif
    quantize_float_to_mx4_kernel<<<gridDim, blockDim, smem_size>>>(
        MAKE_PTA_WITH_NAME(func_name, input, float, 1, 64),
        mx_group_size,
        total_elems,
        flush_fp32_subnorms,
        rd,
        MAKE_PTA_WITH_NAME(func_name, output, uint8_t, 1, 64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  gpuErrchk(cudaPeekAtLastError());
  return output;
}

DLL_PUBLIC at::Tensor dequantize_mx_cuda(
    const at::Tensor& input,
    const int64_t mx_group_size) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input);
  at::Device device = input.device();
  const at::cuda::CUDAGuard device_guard{device};
  // num quantized elems = half of the total float elms + total number of groups
  // so, quantized input size = (total_num_elems/2)+(total_num_elems/group_size)
  // Note that this formula won't work if there's padding to quantized output
  // and total_elems need to be passed.
  const int64_t total_elems =
      (2 * mx_group_size * input.numel()) / (mx_group_size + 2);
  const uint32_t total_num_groups = total_elems / mx_group_size;

  auto output = at::empty(
      total_elems, // 4 = sizeof(float)
      input.options().dtype(at::kFloat));
  const int num_groups_per_block = MAX_THREADS / mx_group_size;
  const auto gridDim_x = round_up(total_num_groups, num_groups_per_block);

  const dim3 gridDim(gridDim_x);
  const dim3 blockDim(mx_group_size, num_groups_per_block);

  // Call CUDA kernel
  if (input.dtype() == torch::ScalarType::Half) {
    TORCH_CHECK(0, " fp16 not supported for MX");
  } else {
#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name = "dequantize_mx4_to_float_kernel";
#endif
    dequantize_mx4_to_float_kernel<<<gridDim, blockDim>>>(
        MAKE_PTA_WITH_NAME(func_name, input, uint8_t, 1, 64),
        mx_group_size,
        total_elems,
        MAKE_PTA_WITH_NAME(func_name, output, float, 1, 64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  gpuErrchk(cudaPeekAtLastError());
  return output;
}

} // namespace fbgemm_gpu
