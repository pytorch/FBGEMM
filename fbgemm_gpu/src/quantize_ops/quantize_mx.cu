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

// from codegen/training/backward/embedding_backward_split_template.cu
template <typename func_t>
int32_t compute_num_groups_and_dynamic_smem_bytes(
    uint32_t* num_groups_per_block,
    const int64_t mx_group_size,
    const int device,
    const func_t kernel_func_name) {
  int32_t smem_bytes = 0;
  // V100: 96 KB; A100: 160 KB; H100: 228 KB.
  int max_shared_bytes = 0;
#ifndef USE_ROCM
  cudaDeviceGetAttribute(
      &max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
#else
  // MI100 has 64 KB local memory (shared memory) per workgroup
  max_shared_bytes = 64 << 10;
#endif
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int shared_kb = max_shared_bytes >> 10;
  // V100: 64 KB; A100: 96 KB; H100: 144 KB
#ifndef USE_ROCM
  // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
  int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
  TORCH_CHECK_GT(used_shared_kb, 0);
#else
  // MI100 has independent shared mem and L1
  int used_shared_kb = shared_kb;
#endif
  const int used_shared_bytes = used_shared_kb << 10;
  // Stay under used_shared_kb of shared memory (V100: 64 KB;
  // A100: 96 KB; H100: 144 KB), num_groups must be a power
  // of two.
  // max(num_elem_in_block * sizeof(int), num_elem_in_block /2 * sizeof(uint8))
  while ((smem_bytes = *num_groups_per_block * mx_group_size * (sizeof(int))) >=
         used_shared_bytes) {
    *num_groups_per_block /= 2;
  }
  TORCH_CHECK_GE(*num_groups_per_block, 1);

  // Check
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // "Compute capability 7.x devices allow a single thread block to
  // address the full capacity of shared memory: 96 KB on Volta,
  // 64 KB on Turing. Kernels relying on shared memory allocations
  // over 48 KB per block are architecture-specific, as such they
  // must use dynamic shared memory (rather than statically sized
  // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".
#ifndef USE_ROCM
  cudaFuncSetAttribute(
      kernel_func_name,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
  return smem_bytes;
}

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
  // Currently we only support MX4 E2M1, for other MX types, we will dispatch
  // different kernels
  TORCH_CHECK(
      scale_bits == 8 && elem_ebits == 2 && elem_mbits == 3 &&
          elem_max_norm == 6.0,
      "FBGEMM currently only supports MX4 E2M1.");

  at::Device device = input.device();
  const at::cuda::CUDAGuard device_guard{device};
  const uint32_t total_elems = input.numel();
  const uint32_t total_num_groups = input.numel() / mx_group_size;

  RoundingMode rd = static_cast<RoundingMode>(rounding_mode);

  uint32_t num_groups_per_block = MAX_THREADS / mx_group_size;
  const auto kernel_func = quantize_float_to_mx4_kernel<float>;

  // Use shmem to find max exponent (int) and temporarily store output (unint8)
  const int32_t smem_size = compute_num_groups_and_dynamic_smem_bytes(
      &num_groups_per_block, mx_group_size, input.get_device(), kernel_func);

  const auto gridDim_x = div_round_up(total_num_groups, num_groups_per_block);

  const dim3 gridDim(gridDim_x);
  const dim3 blockDim(mx_group_size, num_groups_per_block);

  auto output = at::empty(
      (total_elems / 2) + total_num_groups, input.options().dtype(at::kByte));

  // Call CUDA kernel
  if (input.dtype() == torch::ScalarType::Half) {
    TORCH_CHECK(0, " fp16 not supported for MX");
  } else {
#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name = "quantize_float_to_mx4_kernel";
#endif
    kernel_func<<<
        gridDim,
        blockDim,
        smem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        MAKE_PTA_WITH_NAME(func_name, input, float, 1, 64),
        mx_group_size,
        total_elems,
        flush_fp32_subnorms,
        rd,
        MAKE_PTA_WITH_NAME(func_name, output, uint8_t, 1, 64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
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
  const auto gridDim_x = div_round_up(total_num_groups, num_groups_per_block);

  const dim3 gridDim(gridDim_x);
  const dim3 blockDim(mx_group_size, num_groups_per_block);

  // Call CUDA kernel
  if (input.dtype() == torch::ScalarType::Half) {
    TORCH_CHECK(0, " fp16 not supported for MX");
  } else {
#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name = "dequantize_mx4_to_float_kernel";
#endif
    dequantize_mx4_to_float_kernel<<<
        gridDim,
        blockDim,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        MAKE_PTA_WITH_NAME(func_name, input, uint8_t, 1, 64),
        mx_group_size,
        total_elems,
        MAKE_PTA_WITH_NAME(func_name, output, float, 1, 64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return output;
}

} // namespace fbgemm_gpu
