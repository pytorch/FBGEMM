/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "ATen/Parallel.h"
#include "fbgemm_gpu/layout_transform_ops.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

///@defgroup layout-transform-cuda Layout Transformation CUDA Operators
///

namespace fbgemm_gpu {

///@ingroup layout-transform-cuda
Tensor recat_embedding_grad_output_cuda(
    Tensor grad_output, // [B_local][T_global][D]
    const std::vector<int64_t>& num_features_per_rank) {
  TENSOR_ON_CUDA_GPU(grad_output);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  TORCH_CHECK(grad_output.is_contiguous());
  const auto B_local = grad_output.size(0);
  const auto T_global = grad_output.size(1);
  const auto D = grad_output.size(2);

  Tensor sharded_grad_output =
      at::empty({grad_output.numel()}, grad_output.options());
  FBGEMM_DISPATCH_FLOAT_AND_HALF(
      grad_output.scalar_type(), "recat_embedding_gradients", [&] {
        const auto go = grad_output.accessor<scalar_t, 3>();
        auto sgo = sharded_grad_output.accessor<scalar_t, 1>();
        int64_t feature_offset = 0;
        int64_t sgo_offset = 0;
        for (auto num_features : num_features_per_rank) {
          if (num_features == 0) {
            continue;
          }
          AT_CUDA_CHECK(cudaMemcpy2DAsync(
              &sgo[sgo_offset],
              num_features * D * sizeof(scalar_t),
              &go[0][feature_offset][0],
              T_global * D * sizeof(scalar_t),
              num_features * D * sizeof(scalar_t),
              B_local,
              cudaMemcpyDeviceToDevice,
              at::cuda::getCurrentCUDAStream()));
          feature_offset += num_features;
          sgo_offset += B_local * num_features * D;
        }
        TORCH_CHECK(sgo_offset == grad_output.numel());
        TORCH_CHECK(feature_offset == T_global);
      });
  return sharded_grad_output;
}

///@ingroup layout-transform-cuda
Tensor recat_embedding_grad_output_mixed_D_cuda(
    const Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank) {
  TENSOR_ON_CUDA_GPU(grad_output);
  TORCH_CHECK(grad_output.is_contiguous());

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const auto B_local = grad_output.size(0);
  const auto global_dim_sum = at::sum_integers(dim_sum_per_rank);

  Tensor sharded_grad_output =
      at::empty({grad_output.numel()}, grad_output.options());

  FBGEMM_DISPATCH_FLOAT_AND_HALF(
      grad_output.scalar_type(), "recat_embedding_gradients", [&] {
        const auto go = grad_output.accessor<scalar_t, 2>();
        auto sgo = sharded_grad_output.accessor<scalar_t, 1>();
        int64_t sgo_offset = 0;
        int64_t accum_dim_sum = 0;
        for (auto dim_sum : dim_sum_per_rank) {
          if (dim_sum == 0) {
            continue;
          }
          AT_CUDA_CHECK(cudaMemcpy2DAsync(
              &sgo[sgo_offset],
              dim_sum * sizeof(scalar_t),
              &go[0][accum_dim_sum],
              global_dim_sum * sizeof(scalar_t),
              dim_sum * sizeof(scalar_t),
              B_local,
              cudaMemcpyDeviceToDevice,
              at::cuda::getCurrentCUDAStream()));
          sgo_offset += B_local * dim_sum;
          accum_dim_sum += dim_sum;
        }
        TORCH_CHECK(sgo_offset == grad_output.numel());
        TORCH_CHECK(accum_dim_sum == global_dim_sum);
      });

  return sharded_grad_output;
}

///@ingroup layout-transform-cuda
Tensor recat_embedding_grad_output_mixed_D_batch_cuda(
    const Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const Tensor& dim_sum_per_rank,
    const Tensor& cumsum_dim_sum_per_rank) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      grad_output, dim_sum_per_rank, cumsum_dim_sum_per_rank);
  TORCH_CHECK(grad_output.is_contiguous());

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const auto B_local = grad_output.size(0);
  Tensor sharded_grad_output =
      at::empty({grad_output.numel()}, grad_output.options());
  const auto dim_num = dim_sum_per_rank.size(0);
  const auto dim_sum = grad_output.size(1);

  const dim3 threads(
      fbgemm_gpu::kWarpSize, fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize);
  const dim3 blocks(fbgemm_gpu::div_round_up(
      (B_local * dim_num), fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize));

  FBGEMM_DISPATCH_FLOAT_AND_HALF(
      grad_output.scalar_type(), "recat_embedding_gradients", [&] {
        recat_copy_async_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dim_sum_per_rank.data_ptr<int64_t>(),
                cumsum_dim_sum_per_rank.data_ptr<int64_t>(),
                grad_output.data_ptr<scalar_t>(),
                sharded_grad_output.data_ptr<scalar_t>(),
                dim_num,
                B_local,
                dim_sum);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return sharded_grad_output;
}

} // namespace fbgemm_gpu
