/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include "cub/device/device_scan.cuh"
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <typename index_t, typename scalar_t>
__global__ void jagged_2d_to_dense_forward_kernel(
    int32_t B,
    int32_t max_L,
    int32_t D,
    index_t* offsets,
    scalar_t* values,
    scalar_t* padded_values) {
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
    for (int32_t d = threadIdx.x; d < D; d += kWarpSize) {
      padded_values[b * max_L * D + l * D + d] =
          values[(row_start + l) * D + d];
    }
  } else {
    for (int32_t d = threadIdx.x; d < D; d += kWarpSize) {
      padded_values[b * max_L * D + l * D + d] = 0.0;
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
                  <<<div_round_up((B * max_L), kMaxThreads / kWarpSize),
                     dim3(kWarpSize, kMaxThreads / kWarpSize),
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      B,
                      max_L,
                      D,
                      offsets_contig.data_ptr<index_t>(),
                      values_contig.data_ptr<scalar_t>(),
                      padded_values.data_ptr<scalar_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
      }));

  return padded_values;
}

template <typename index_t, typename scalar_t>
__global__ void jagged_2d_to_dense_backward_kernel(
    int32_t B,
    int32_t max_L,
    int32_t D,
    index_t* offsets,
    scalar_t* grad_padded_values,
    scalar_t* grad_values) {
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
    for (int32_t d = threadIdx.x; d < D; d += kWarpSize) {
      grad_values[(row_start + l) * D + d] =
          grad_padded_values[b * max_L * D + l * D + d];
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
                  <<<div_round_up((B * max_L), kMaxThreads / kWarpSize),
                     dim3(kWarpSize, kMaxThreads / kWarpSize),
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      B,
                      max_L,
                      D,
                      offsets_contig.data_ptr<index_t>(),
                      grad_padded_values_config.data_ptr<scalar_t>(),
                      grad_values.data_ptr<scalar_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
      }));

  return grad_values;
}

template <typename index_t, typename scalar_t>
__global__ void jagged_1d_to_dense_kernel(
    int32_t B,
    int32_t max_L,
    scalar_t padding_value,
    index_t* offsets,
    scalar_t* values,
    scalar_t* padded_values) {
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
    padded_values[b * max_L + l] = values[row_start + l];
  } else {
    padded_values[b * max_L + l] = padding_value;
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
                      static_cast<scalar_t>(padding_value),
                      offsets_contig.data_ptr<index_t>(),
                      values_contig.data_ptr<scalar_t>(),
                      padded_values.data_ptr<scalar_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
      }));

  return padded_values;
}

// stacked ops
std::tuple<std::vector<Tensor>, std::vector<Tensor>>
stacked_jagged_2d_to_dense_forward_cuda(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(lengths.dim() == 2);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const auto values_contig = values.contiguous();
  const auto lengths_contig = lengths.contiguous();
  int32_t D = values.size(1);
  int32_t B = lengths.size(1);
  int32_t T = lengths.size(0);
  std::vector<Tensor> padded_values_per_key;
  std::vector<Tensor> offsets_tensor_per_key;
  for (int32_t t = 0; t < T; t++) {
    int64_t max_L = max_lengths_per_key[t];
    size_t temp_storage_bytes = 0;
    auto offsets = at::empty({B + 1}, lengths.options());
    offsets[0].zero_();
    AT_DISPATCH_INTEGRAL_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper1", ([&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              nullptr,
              temp_storage_bytes,
              &(lengths_contig.data_ptr<scalar_t>()[t * B]),
              offsets.data_ptr<scalar_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        }));
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        lengths.options().dtype(at::kByte));
    AT_DISPATCH_INTEGRAL_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper2", ([&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              temp_storage.data_ptr(),
              temp_storage_bytes,
              &(lengths_contig.data_ptr<scalar_t>()[t * B]),
              offsets.data_ptr<scalar_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        }));
    offsets_tensor_per_key.push_back(offsets);
    auto padded_values = at::empty({B, max_L, D}, values.options());
    padded_values_per_key.push_back(padded_values);
    int64_t start = offset_per_key[t] * D;
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
                        offsets.data_ptr<index_t>(),
                        &(values_contig.data_ptr<scalar_t>()[start]),
                        padded_values.data_ptr<scalar_t>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }));
        }));
  }

  return std::make_tuple(padded_values_per_key, offsets_tensor_per_key);
}

Tensor stacked_jagged_2d_to_dense_backward_cuda(
    int64_t B,
    int64_t D,
    int64_t total_L,
    const std::vector<Tensor>& grad_padded_values_per_key,
    const std::vector<Tensor>& offsets_tensor_per_key,
    const std::vector<int64_t>& offset_per_key) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_padded_values_per_key[0].get_device());

  auto grad_values =
      at::zeros({total_L, D}, grad_padded_values_per_key[0].options());
  int32_t T = grad_padded_values_per_key.size();
  for (int32_t t = 0; t < T; t++) {
    const auto grad_padded_values_config =
        grad_padded_values_per_key[t].contiguous();
    int64_t start = offset_per_key[t] * D;
    const auto offsets_config = offsets_tensor_per_key[t].contiguous();
    TORCH_CHECK(grad_padded_values_config.dim() == 3);
    TORCH_CHECK(grad_padded_values_config.size(0) == B);
    TORCH_CHECK(grad_padded_values_config.size(2) == D);
    int32_t max_L = grad_padded_values_config.size(1);
    AT_DISPATCH_INDEX_TYPES(
        offsets_config.scalar_type(),
        "jagged_2d_to_dense_backward_kernel_1",
        ([&]() {
          AT_DISPATCH_FLOATING_TYPES_AND_HALF(
              grad_padded_values_config.scalar_type(),
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
                        offsets_config.data_ptr<index_t>(),
                        grad_padded_values_config.data_ptr<scalar_t>(),
                        &(grad_values.data_ptr<scalar_t>()[start]));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }));
        }));
  }

  return grad_values;
}

std::vector<Tensor> stacked_jagged_1d_to_dense_gpu(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(lengths.dim() == 2);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const auto values_contig = values.contiguous();
  const auto lengths_contig = lengths.contiguous();
  int32_t B = lengths.size(1);
  int32_t T = lengths.size(0);
  auto offsets = at::empty({B + 1}, lengths.options());
  offsets[0].zero_();
  std::vector<Tensor> padded_values_per_key;
  for (int32_t t = 0; t < T; t++) {
    int64_t max_L = max_lengths_per_key[t];
    size_t temp_storage_bytes = 0;
    AT_DISPATCH_INTEGRAL_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper1", ([&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              nullptr,
              temp_storage_bytes,
              &(lengths_contig.data_ptr<scalar_t>()[t * B]),
              offsets.data_ptr<scalar_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        }));
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        lengths.options().dtype(at::kByte));
    AT_DISPATCH_INTEGRAL_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper2", ([&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              temp_storage.data_ptr(),
              temp_storage_bytes,
              &(lengths_contig.data_ptr<scalar_t>()[t * B]),
              offsets.data_ptr<scalar_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        }));
    auto padded_values = at::empty({B, max_L}, values.options());
    padded_values_per_key.push_back(padded_values);
    int64_t start = offset_per_key[t];
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
                        static_cast<scalar_t>(padding_value),
                        offsets.data_ptr<index_t>(),
                        &(values_contig.data_ptr<scalar_t>()[start]),
                        padded_values.data_ptr<scalar_t>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }));
        }));
  }

  return padded_values_per_key;
}

} // namespace fbgemm_gpu
