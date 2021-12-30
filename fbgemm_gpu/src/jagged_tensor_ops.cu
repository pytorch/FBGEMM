/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <typename index_t, typename scalar_t>
__global__ void jagged_2d_to_dense_forward_kernel(
    int32_t B,
    int32_t max_L,
    int32_t D,
    at::PackedTensorAccessor32<index_t, 1> offsets,
    at::PackedTensorAccessor64<scalar_t, 2> values,
    at::PackedTensorAccessor64<scalar_t, 3> padded_values) {
  int32_t b_l = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t l = b_l / B;
  int32_t b = b_l % B;
  if (b_l >= B * max_L) {
    return;
  }
  int32_t row_start = offsets[b];
  int32_t row_end = offsets[b + 1];
  int32_t length = row_end - row_start;
  if (l < length) {
    for (int32_t d = 0; d < D; d += fbgemm_gpu::kWarpSize) {
      if (d + threadIdx.x < D) {
        padded_values[b][l][d + threadIdx.x] =
            values[row_start + l][d + threadIdx.x];
      }
    }
  } else {
    for (int32_t d = 0; d < D; d += fbgemm_gpu::kWarpSize) {
      if (d + threadIdx.x < D) {
        padded_values[b][l][d + threadIdx.x] = 0.0;
      }
    }
  }
}

Tensor
jagged_2d_to_dense_forward_cuda(Tensor values, Tensor offsets, int32_t max_L) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(offsets);

  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  int32_t D = values.size(1);
  int32_t B = offsets.numel() - 1;
  auto padded_values = at::empty({B, max_L, D}, values.options());
  const auto values_contig = values.contiguous();
  const auto offsets_contig = offsets.contiguous();

  AT_DISPATCH_INDEX_TYPES(
      offsets.scalar_type(), "jagged_2d_to_dense_forward_kernel_1", ([&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            values.scalar_type(),
            "jagged_2d_to_dense_forward_kernel_2",
            ([&]() {
              jagged_2d_to_dense_forward_kernel<index_t, scalar_t>
                  <<<fbgemm_gpu::div_round_up(
                         (B * max_L),
                         fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize),
                     dim3(
                         fbgemm_gpu::kWarpSize,
                         fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize),
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      B,
                      max_L,
                      D,
                      offsets_contig.packed_accessor32<index_t, 1>(),
                      values_contig.packed_accessor64<scalar_t, 2>(),
                      padded_values.packed_accessor64<scalar_t, 3>());
            }));
      }));

  return padded_values;
}

template <typename index_t, typename scalar_t>
__global__ void jagged_2d_to_dense_backward_kernel(
    int32_t B,
    int32_t max_L,
    int32_t D,
    at::PackedTensorAccessor32<index_t, 1> offsets,
    at::PackedTensorAccessor64<scalar_t, 3> grad_padded_values,
    at::PackedTensorAccessor64<scalar_t, 2> grad_values) {
  int32_t b_l = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t l = b_l / B;
  int32_t b = b_l % B;
  if (b_l >= B * max_L) {
    return;
  }
  int32_t row_start = offsets[b];
  int32_t row_end = offsets[b + 1];
  int32_t length = row_end - row_start;
  if (l < length) {
    for (int32_t d = 0; d < D; d += fbgemm_gpu::kWarpSize) {
      if (d + threadIdx.x < D) {
        grad_values[row_start + l][d + threadIdx.x] =
            grad_padded_values[b][l][d + threadIdx.x];
      }
    }
  }
}

Tensor jagged_2d_to_dense_backward_cuda(
    Tensor grad_padded_values,
    Tensor offsets,
    int32_t total_L) {
  TENSOR_ON_CUDA_GPU(grad_padded_values);
  TENSOR_ON_CUDA_GPU(offsets);

  TORCH_CHECK(grad_padded_values.dim() == 3);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(total_L >= 0);
  TORCH_CHECK(offsets.numel() == grad_padded_values.size(0) + 1);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_padded_values.get_device());

  int32_t B = grad_padded_values.size(0);
  int32_t max_L = grad_padded_values.size(1);
  int32_t D = grad_padded_values.size(2);
  auto grad_values = at::zeros({total_L, D}, grad_padded_values.options());
  const auto grad_padded_values_config = grad_padded_values.contiguous();
  const auto offsets_contig = offsets.contiguous();

  AT_DISPATCH_INDEX_TYPES(
      offsets.scalar_type(), "jagged_2d_to_dense_backward_kernel_1", ([&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_padded_values.scalar_type(),
            "jagged_2d_to_dense_backward_kernel_2",
            ([&]() {
              jagged_2d_to_dense_backward_kernel<index_t, scalar_t>
                  <<<fbgemm_gpu::div_round_up(
                         (B * max_L),
                         fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize),
                     dim3(
                         fbgemm_gpu::kWarpSize,
                         fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize),
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      B,
                      max_L,
                      D,
                      offsets_contig.packed_accessor32<index_t, 1>(),
                      grad_padded_values_config
                          .packed_accessor64<scalar_t, 3>(),
                      grad_values.packed_accessor64<scalar_t, 2>());
            }));
      }));

  return grad_values;
}

template <typename index_t, typename data_t>
__global__ void jagged_1d_to_dense_kernel(
    int32_t B,
    int32_t max_L,
    data_t padding_value,
    at::PackedTensorAccessor32<index_t, 1> offsets,
    at::PackedTensorAccessor64<data_t, 1> values,
    at::PackedTensorAccessor64<data_t, 2> padded_values) {
  const int32_t b_l = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_l >= B * max_L) {
    return;
  }
  int32_t b = b_l / max_L;
  int32_t l = b_l % max_L;
  int32_t row_start = offsets[b];
  int32_t row_end = offsets[b + 1];
  int32_t length = row_end - row_start;
  if (l < length) {
    padded_values[b][l] = values[row_start + l];
  } else {
    padded_values[b][l] = padding_value;
  }
}

Tensor jagged_1d_to_dense_gpu(
    Tensor values,
    Tensor offsets,
    int64_t max_L,
    int64_t padding_value) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(offsets);

  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  int32_t B = offsets.numel() - 1;
  auto padded_values = at::empty({B, max_L}, values.options());
  const auto values_contig = values.contiguous();
  const auto offsets_contig = offsets.contiguous();
  const int32_t num_threads = 512; // 256~1024 per xingl
  AT_DISPATCH_INDEX_TYPES(
      offsets.scalar_type(), "jagged_1d_to_dense_kernel_1", ([&]() {
        AT_DISPATCH_ALL_TYPES(
            values.scalar_type(), "jagged_1d_to_dense_kernel_2", ([&]() {
              jagged_1d_to_dense_kernel<index_t, scalar_t>
                  <<<div_round_up(B * max_L, num_threads),
                     num_threads,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      B,
                      max_L,
                      padding_value,
                      offsets_contig.packed_accessor32<index_t, 1>(),
                      values_contig.packed_accessor64<scalar_t, 1>(),
                      padded_values.packed_accessor64<scalar_t, 2>());
            }));
      }));

  return padded_values;
}

} // namespace fbgemm_gpu
