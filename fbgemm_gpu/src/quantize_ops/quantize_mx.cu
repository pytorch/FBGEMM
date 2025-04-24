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
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

#include <ATen/core/TensorAccessor.h>
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "quantize_mx.cuh"

namespace fbgemm_gpu {

int32_t compute_smem_bytes(
    const uint32_t num_warps_in_group,
    const uint32_t num_groups_per_block,
    const int64_t mx_group_size) {
  const auto smem_size =
      (num_groups_per_block * mx_group_size) * sizeof(uint8_t);

  if (num_warps_in_group > 1) {
    return max(
        num_warps_in_group * (num_groups_per_block) * sizeof(int), smem_size);
  }
  return smem_size;
}

// from codegen/training/backward/embedding_backward_split_template.cu
template <typename func_t>
int32_t compute_num_groups_and_dynamic_smem_bytes(
    uint32_t* num_groups_per_block,
    const int64_t mx_group_size,
    const int device,
    const func_t kernel_func_name,
    const uint32_t num_warps_in_group) {
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
  // max(num_warps_in_group * num_groups_per_block * sizeof(int),
  // num_elem_in_block * sizeof(uint8))
  while ((smem_bytes = compute_smem_bytes(
              num_warps_in_group, *num_groups_per_block, mx_group_size)) >=
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
  TORCH_CHECK(mx_group_size > 0, "Group size needs to be > 0");
  TORCH_CHECK(
      mx_group_size % 32 == 0,
      "Group size needs to be multiply of 32 but is found to be ",
      mx_group_size);
  TORCH_CHECK(!flush_fp32_subnorms, "flush_fp32_subnorms is not yet supported");
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input);

  const uint32_t total_elems = input.numel();
  if (total_elems == 0) {
    return at::empty(0, input.options().dtype(at::kByte));
  }
  TORCH_CHECK(
      total_elems > mx_group_size,
      "Input needs to be > mx_group_size of ",
      mx_group_size,
      " but is found to be ",
      total_elems);
  TORCH_CHECK(
      total_elems % mx_group_size == 0,
      "Input needs to be multiply of ",
      mx_group_size,
      "but is found to be ",
      total_elems);

  // Currently we only support MX4 E2M1, for other MX types, we will dispatch
  // different kernels
  TORCH_CHECK(
      scale_bits == 8 && elem_ebits == 2 && elem_mbits == 3 &&
          elem_max_norm == 6.0,
      "FBGEMM currently only supports MX4 E2M1.");

  at::Device device = input.device();
  const at::cuda::CUDAGuard device_guard{device};
  const uint32_t total_num_groups = input.numel() / mx_group_size;

  RoundingMode rd = static_cast<RoundingMode>(rounding_mode);

  const uint32_t num_warps_in_group = mx_group_size / WARP_SIZE;
  CUDA_KERNEL_ASSERT(num_warps_in_group <= WARP_SIZE);

  uint32_t num_groups_per_block = MAX_THREADS / mx_group_size;
  const auto kernel_func = (num_warps_in_group > 1)
      ? quantize_float_to_mx4_kernel<float, true>
      : quantize_float_to_mx4_kernel<float, false>;

  int device_id = input.get_device();

  // Use shmem to find max exponent (int) and temporarily store output (unint8)
  const int32_t smem_size = compute_num_groups_and_dynamic_smem_bytes(
      &num_groups_per_block,
      mx_group_size,
      device_id,
      kernel_func,
      num_warps_in_group);

  const auto gridDim_x =
      max(1, div_round_up(total_num_groups, num_groups_per_block));

  const dim3 gridDim(gridDim_x);
  const dim3 blockDim(mx_group_size, num_groups_per_block);

  auto output = at::empty(
      (total_elems / 2) + total_num_groups, input.options().dtype(at::kByte));

  int max_grid_size = 0;
  cudaDeviceGetAttribute(&max_grid_size, cudaDevAttrMaxGridDimX, device_id);

  TORCH_CHECK(
      gridDim_x > 0 && gridDim_x <= max_grid_size,
      "gridDim_x is of bound with value ",
      gridDim_x,
      ". MaxGridDimX is ",
      max_grid_size);
  TORCH_CHECK(
      smem_size >= 0,
      "shared memory size needs to be >= 0 but found to be ",
      smem_size);

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
        // flush_fp32_subnorms, // TODO: Update as template argument
        rd,
        MAKE_PTA_WITH_NAME(func_name, output, uint8_t, 1, 64),
        num_warps_in_group,
        max(num_warps_in_group, int(mx_group_size / 4))); // smem_stride
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

DLL_PUBLIC at::Tensor dequantize_mx_cuda(
    const at::Tensor& input,
    const int64_t mx_group_size) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input);
  TORCH_CHECK(mx_group_size > 0, "Group size needs to be > 0");
  TORCH_CHECK(
      mx_group_size % 32 == 0,
      "Group size needs to be multiply of 32 but is found to be ",
      mx_group_size);
  if (input.numel() == 0) {
    return at::empty(0, input.options().dtype(at::kFloat));
  }

  at::Device device = input.device();
  const at::cuda::CUDAGuard device_guard{device};
  // input size = half of the total (float) elms + total number of groups
  // i.e., input.numel() = (total_num_elems/2)+(total_num_elems/group_size)
  // so, total_elems = (mx_group_size*input.numel())/(mx_group_size + 2)
  // Note that this formula won't work if there's padding to quantized output
  // and total_elems need to be passed to the function as an argument.
  const int64_t total_elems =
      (2 * mx_group_size * input.numel()) / (mx_group_size + 2);

  // one thread works on 2 uint8 quantized input (4 mx4).
  const int64_t total_quant_elems = total_elems / 4;
  TORCH_CHECK(
      total_elems > mx_group_size,
      "Input needs to be > mx_group_size of ",
      mx_group_size,
      " but is found to be ",
      total_elems);
  TORCH_CHECK(
      total_elems % mx_group_size == 0,
      "Input needs to be multiply of ",
      mx_group_size,
      " but is found to be ",
      total_elems);
  const uint32_t total_num_groups = total_quant_elems / mx_group_size;

  auto output = at::empty(
      total_elems, // 4 = sizeof(float)
      input.options().dtype(at::kFloat));
  const int num_groups_per_block = MAX_THREADS / mx_group_size;
  const auto gridDim_x =
      max(1, div_round_up(total_num_groups, num_groups_per_block));

  const dim3 gridDim(gridDim_x);
  const dim3 blockDim(mx_group_size, num_groups_per_block);

  int max_grid_size = 0;
  cudaDeviceGetAttribute(
      &max_grid_size, cudaDevAttrMaxGridDimX, input.get_device());

  TORCH_CHECK(
      gridDim_x > 0 && gridDim_x <= max_grid_size,
      "gridDim_x is of bound with value ",
      gridDim_x,
      ". MaxGridDimX is ",
      max_grid_size);

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
        total_quant_elems,
        MAKE_PTA_WITH_NAME(func_name, output, float, 1, 64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return output;
}

} // namespace fbgemm_gpu
