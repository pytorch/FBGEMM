/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm_gpu/sparse_ops.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/library.h>

#include "ATen/Parallel.h"
#include "cub/device/device_scan.cuh"

namespace at {

Tensor asynchronous_exclusive_cumsum(const Tensor& t_in) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == kInt || t_in.dtype() == kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  auto t_out = at::empty_like(t_in);
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_exclusive_sum_wrapper1", ([&] {
        AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)}, t_in.options().dtype(kByte));
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_exclusive_sum_wrapper2", ([&] {
        AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  return t_out;
}

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights) {
  TENSOR_ON_CUDA_GPU(permute);
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(weights);

  TENSORS_ON_SAME_DEVICE(permute, lengths);
  TENSORS_ON_SAME_DEVICE(permute, indices);
  TENSORS_ON_SAME_DEVICE(permute, weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the data to permute over can be less or more with or without
  // repetitions
  const auto T = permute.numel();
  const auto T_ = lengths.size(0);
  const auto B = lengths.view({ lengths.sizes()[0], -1 }).sizes()[1];

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;

  permuted_lengths = at::empty({T, B}, lengths.options());

  constexpr int32_t threads_1 = 256;
  const auto blocks_1 =
      cuda_calc_xblock_count(B * T, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_lengths_kernel", ([&] {
        permute_lengths_kernel<index_t>
            <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
                T,
                B,
                lengths_contig.data_ptr<index_t>(),
                permute.data_ptr<int32_t>(),
                permuted_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));

  // convert lengths to offsets
  const auto input_offsets = asynchronous_exclusive_cumsum(lengths_contig);
  const auto output_offsets = asynchronous_exclusive_cumsum(permuted_lengths);
  int64_t permuted_lengths_sum = indices.numel();

  /* TODO: Remove the condition protecting the slow path because even when the
   * condition below is true permuted_lengths.sum() could still be needed. For
   * instance if there are three features with indices `[0, 1, 2]`, `permute`
   * can be `[0, 1, 1]` for which permuted lengths sum would be needed to create
   * permuted_{indices, weights} and `permuted_lengths_sum = indices.numel() or
   * weights.numdel() would be incorrect.
   */
  if (T_ != T) {
    permuted_lengths_sum = permuted_lengths.sum().item<int64_t>();
  }

  constexpr int32_t BT_blocks = 32;
  dim3 threads_2(32, BT_blocks);
  const auto blocks_2 =
      cuda_calc_xblock_count(B * T, BT_blocks);
  permuted_indices = at::empty(permuted_lengths_sum, indices.options());

  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_data_kernel_1", ([&] {
        using offsets_t = index_t;
        AT_DISPATCH_ALL_TYPES(
          indices.scalar_type(), "permute_data_kernel_2", ([&] {
            using indices_t = scalar_t;
            if (weights.has_value()) {
              const Tensor weights_value = weights.value();
              const auto weights_value_contig = weights_value.contiguous();
              permuted_weights = at::empty(permuted_lengths_sum, weights_value.options());
              AT_DISPATCH_FLOATING_TYPES(
                weights_value.scalar_type(), "permute_data_kernel_3", ([&] {
                  using weights_t = scalar_t;
                  permute_data_kernel<true, offsets_t, indices_t, weights_t>
                      <<<blocks_2,
                        threads_2,
                        0,
                        at::cuda::getCurrentCUDAStream()>>>(
                          permuted_lengths_sum,
                          T,
                          B,
                          indices_contig.data_ptr<indices_t>(),
                          weights_value_contig.data_ptr<weights_t>(),
                          permute_contig.data_ptr<int32_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets.data_ptr<offsets_t>(),
                          permuted_indices.data_ptr<indices_t>(),
                          permuted_weights.data_ptr<weights_t>());
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                })); // for each weights_t
              } else {
                permute_data_kernel<false, offsets_t, indices_t, std::nullptr_t>
                    <<<blocks_2,
                      threads_2,
                      0,
                      at::cuda::getCurrentCUDAStream()>>>(
                        permuted_lengths_sum,
                        T,
                        B,
                        indices_contig.data_ptr<indices_t>(),
                        nullptr,
                        permute_contig.data_ptr<int32_t>(),
                        input_offsets.data_ptr<offsets_t>(),
                        output_offsets.data_ptr<offsets_t>(),
                        permuted_indices.data_ptr<indices_t>(),
                        nullptr);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
          })); // for each indices_t
        })); // for each offsets_t
  return {permuted_lengths, permuted_indices, permuted_weights};
}

} // namespace at
