/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/layout_transform_ops.cuh"
#include "fbgemm_gpu/permute_pooled_embedding_ops_split.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor permute_pooled_embs_split_gpu(
    const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list) {
  // inv_permute_list is not being used so it's not checked here.
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      pooled_embs, offset_dim_list, permute_list, inv_offset_dim_list);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(pooled_embs.get_device());
  // We couldn't pass the "pooled_embs.is_contiguous()" check in the backward
  // passs after D22767058. TODO: optimize and make sure pooled_embs is
  // contiguous.
  auto pooled_embs_contiguous = pooled_embs.contiguous();
  const int64_t B = pooled_embs_contiguous.size(0);
  const int64_t T = permute_list.numel();
  const int64_t dim_sum = pooled_embs_contiguous.size(1);
  // inv_permute_list is not being used so it's not checked here.
  TENSORS_ON_SAME_DEVICE(pooled_embs_contiguous, offset_dim_list);
  TENSORS_ON_SAME_DEVICE(pooled_embs_contiguous, permute_list);
  TENSORS_ON_SAME_DEVICE(pooled_embs_contiguous, inv_offset_dim_list);
  TORCH_CHECK(offset_dim_list.numel() == permute_list.numel() + 1);
  TORCH_CHECK(offset_dim_list.numel() == inv_offset_dim_list.numel());
  Tensor permuted_pooled_embs = at::empty_like(pooled_embs_contiguous);

  // This kernel is moving D elements per warp.
  // We are launching ( div_round_up(T, warp_per_block), B ) blocks.
  // The grid z dimension is also used by B in case it's greater than 65535.
  const int32_t warp_per_block =
      fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize;
  const int32_t max_grid_dim_y =
      32768; // The CUDA maximum is 65535, not a power of 2.
  const dim3 threads(fbgemm_gpu::kMaxThreads);
  const dim3 blocks(
      fbgemm_gpu::div_round_up(T, warp_per_block),
      std::min(static_cast<int32_t>(B), max_grid_dim_y),
      (B + max_grid_dim_y - 1) / max_grid_dim_y);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      pooled_embs_contiguous.scalar_type(),
      "permute_pooled_embeddings",
      [&] {
        permute_pooled_embs_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                pooled_embs_contiguous.data_ptr<scalar_t>(),
                offset_dim_list.data_ptr<int64_t>(),
                permute_list.data_ptr<int64_t>(),
                inv_offset_dim_list.data_ptr<int64_t>(),
                permuted_pooled_embs.data_ptr<scalar_t>(),
                B,
                T,
                dim_sum);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return permuted_pooled_embs;
}
} // namespace fbgemm_gpu
