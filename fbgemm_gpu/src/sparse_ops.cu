/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm_gpu/batched_unary_embedding_ops.cuh"
#include "fbgemm_gpu/quantize_ops.cuh"
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

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include "cub/device/device_scan.cuh"
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

std::tuple<uint32_t, uint32_t, uint32_t> calc_offsets_range_thread_block(
    const int64_t output_size,
    const int64_t num_seq) {
  uint32_t threads_per_block;
  uint32_t vector_size;
  if (output_size / num_seq < 2) {
    threads_per_block = 512;
    vector_size = 2;
  } else if (output_size / num_seq < 4) {
    threads_per_block = 512;
    vector_size = 4;
  } else if (output_size / num_seq < 64) {
    threads_per_block = 512;
    vector_size = 8;
  } else if (output_size / num_seq < 128) {
    threads_per_block = 512;
    vector_size = 16;
  } else {
    threads_per_block = 512;
    vector_size = 32;
  }
  uint32_t rows_per_block = threads_per_block / vector_size;
  const auto num_blocks = cuda_calc_xblock_count(num_seq, rows_per_block);

  return std::make_tuple(num_blocks, rows_per_block, vector_size);
}

Tensor offsets_range_cuda(const Tensor& offsets, int64_t range_size) {
  TENSOR_ON_CUDA_GPU(offsets);
  TENSOR_NDIM_EQUALS(offsets, 1);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(offsets.get_device());

  auto offsets_arg = at::TensorArg(offsets, "offsets", 1);
  checkScalarTypes("_offsets_range_cuda", offsets_arg, {at::kLong, at::kInt});
  auto range = at::empty(range_size, offsets.options());
  if (range_size == 0) {
    return range;
  }
  auto offsets_contig = offsets.contiguous();
  int64_t N = offsets_contig.numel();

  uint32_t vector_size;
  uint32_t rows_per_block;
  uint32_t num_blocks;
  std::tie(num_blocks, rows_per_block, vector_size) =
      calc_offsets_range_thread_block(range_size, N);
  dim3 threads(vector_size, rows_per_block);
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig.scalar_type(), "offsets_range_kernel", [&]() {
        _offsets_range_cuda_kernel<index_t>
            <<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                range_size,
                offsets_contig.data_ptr<index_t>(),
                range.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return range;
}

Tensor segment_sum_csr_cuda(
    const int64_t batch_size,
    const Tensor& csr_seg,
    const Tensor& values) {
  TENSOR_ON_CUDA_GPU(csr_seg);
  TENSOR_ON_CUDA_GPU(values);

  TENSORS_ON_SAME_DEVICE(csr_seg, values);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  auto output = at::empty(csr_seg.numel() - 1, values.options());
  constexpr uint32_t threads_per_block = 256;
  const uint32_t num_blocks = csr_seg.numel() - 1;
  AT_DISPATCH_ALL_TYPES(values.type(), "_segment_sum_csr_cuda", [&]() {
    _segment_sum_csr_cuda_kernel<scalar_t>
        <<<num_blocks,
           threads_per_block,
           0,
           at::cuda::getCurrentCUDAStream()>>>(
            csr_seg.numel() - 1,
            batch_size,
            csr_seg.data_ptr<int>(),
            values.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return output;
}

Tensor asynchronous_inclusive_cumsum_gpu(const Tensor& t_in) {
  TENSOR_ON_CUDA_GPU(t_in);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  auto t_out = at::empty_like(t_in);
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper1", ([&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      t_in.options().dtype(at::kByte));
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper2", ([&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  return t_out;
}

Tensor asynchronous_exclusive_cumsum_gpu(const Tensor& t_in) {
  TENSOR_ON_CUDA_GPU(t_in);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  auto t_out = at::empty_like(t_in);
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_exclusive_sum_wrapper1", ([&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::ExclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      t_in.options().dtype(at::kByte));
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_exclusive_sum_wrapper2", ([&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::ExclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  return t_out;
}

Tensor asynchronous_complete_cumsum_gpu(const Tensor& t_in) {
  TENSOR_ON_CUDA_GPU(t_in);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(t_in.dim() == 1);
  auto t_out = at::empty({t_in.numel() + 1}, t_in.options());
  t_out[0].zero_();
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper1", ([&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      t_in.options().dtype(at::kByte));
  AT_DISPATCH_INTEGRAL_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper2", ([&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<scalar_t>(),
            t_out.data_ptr<scalar_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      }));
  return t_out;
}

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum) {
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
  const auto B = lengths.view({lengths.sizes()[0], -1}).sizes()[1];

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;

  permuted_lengths = at::empty({T, B}, lengths.options());

  constexpr int32_t threads_1 = 256;
  const auto blocks_1 = cuda_calc_xblock_count(B * T, threads_1);
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
  const auto input_offsets = asynchronous_exclusive_cumsum_gpu(lengths_contig);
  const auto output_offsets =
      asynchronous_exclusive_cumsum_gpu(permuted_lengths);
  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size = permuted_lengths.sum().item<int64_t>();
  }

  constexpr int32_t BT_blocks = 32;
  dim3 threads_2(32, BT_blocks);
  const auto blocks_2 = cuda_calc_xblock_count(B * T, BT_blocks);
  permuted_indices = at::empty(permuted_indices_size, indices.options());

  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_data_kernel_1", ([&] {
        using offsets_t = index_t;
        AT_DISPATCH_ALL_TYPES_AND(
            at::ScalarType::Half,
            indices.scalar_type(),
            "permute_data_kernel_2",
            ([&] {
              using indices_t = scalar_t;
              if (weights.has_value()) {
                const Tensor weights_value = weights.value();
                const auto weights_value_contig = weights_value.contiguous();
                permuted_weights =
                    at::empty(permuted_indices_size, weights_value.options());
                AT_DISPATCH_ALL_TYPES_AND(
                    at::ScalarType::Half,
                    weights_value.scalar_type(),
                    "permute_data_kernel_3",
                    ([&] {
                      using weights_t = scalar_t;
                      permute_data_kernel<true, offsets_t, indices_t, weights_t>
                          <<<blocks_2,
                             threads_2,
                             0,
                             at::cuda::getCurrentCUDAStream()>>>(
                              permuted_indices_size,
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
                        permuted_indices_size,
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

// This function partitions sparse features
// continuously along the sparse dimension into my_size blocks
std::tuple<
    Tensor,
    Tensor,
    c10::optional<Tensor>,
    c10::optional<Tensor>,
    c10::optional<Tensor>>
block_bucketize_sparse_features_cuda(
    Tensor lengths,
    Tensor indices,
    bool bucketize_pos,
    bool sequence,
    Tensor block_sizes,
    int64_t my_size,
    c10::optional<Tensor> weights) {
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(indices);
  TENSORS_ON_SAME_DEVICE(lengths, indices);
  TENSOR_ON_CUDA_GPU(weights);
  TENSORS_ON_SAME_DEVICE(lengths, weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lengths.get_device());
  // allocate tensors and buffers
  const int lengths_size = lengths.numel();
  const int T = block_sizes.numel();
  const int B = lengths_size / T;
  const int new_lengths_size = lengths_size * my_size;
  auto offsets = at::empty({lengths_size}, lengths.options());
  auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
  auto new_offsets = at::empty({new_lengths_size}, lengths.options());
  auto new_indices = at::empty_like(indices);
  auto lengths_contig = lengths.contiguous();
  auto indices_contig = indices.contiguous();
  auto offsets_contig = offsets.contiguous();
  Tensor new_weights;
  Tensor new_pos;
  Tensor unbucketize_permute;
  // count nonzeros
  offsets_contig = asynchronous_inclusive_cumsum_gpu(lengths);
  int threads_per_block = 256;
  int num_blocks = (lengths_size + threads_per_block - 1) / threads_per_block;
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig.scalar_type(),
      "_block_bucketize_sparse_features_cuda_kernel1",
      ([&] {
        using offset_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            indices_contig.scalar_type(),
            "_block_bucketize_sparse_features_cuda_kernel2",
            ([&] {
              _block_bucketize_sparse_features_cuda_kernel1<<<
                  num_blocks,
                  threads_per_block,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  lengths_size,
                  B,
                  block_sizes.data_ptr<index_t>(),
                  my_size,
                  offsets_contig.data_ptr<offset_t>(),
                  indices_contig.data_ptr<index_t>(),
                  new_lengths.data_ptr<offset_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
      }));

  // bucketize nonzeros
  new_offsets = asynchronous_exclusive_cumsum_gpu(new_lengths);
  if (sequence) {
    const auto lengths_sum = indices.numel();
    unbucketize_permute = at::empty({lengths_sum}, indices.options());
    if (weights.has_value() & bucketize_pos) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                ([&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      ([&] {
                        _block_bucketize_sparse_features_cuda_kernel2<
                            true,
                            true,
                            true,
                            offset_t,
                            index_t,
                            scalar_t>
                            <<<num_blocks,
                               threads_per_block,
                               0,
                               at::cuda::getCurrentCUDAStream()>>>(
                                lengths_size,
                                B,
                                block_sizes.data_ptr<index_t>(),
                                my_size,
                                offsets_contig.data_ptr<offset_t>(),
                                indices_contig.data_ptr<index_t>(),
                                weights_value_contig.data_ptr<scalar_t>(),
                                new_offsets.data_ptr<offset_t>(),
                                new_indices.data_ptr<index_t>(),
                                new_weights.data_ptr<scalar_t>(),
                                new_pos.data_ptr<index_t>(),
                                unbucketize_permute.data_ptr<index_t>());
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      }));
                }));
          }));
    } else if (weights.has_value()) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                ([&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      ([&] {
                        _block_bucketize_sparse_features_cuda_kernel2<
                            true,
                            true,
                            false,
                            offset_t,
                            index_t,
                            scalar_t>
                            <<<num_blocks,
                               threads_per_block,
                               0,
                               at::cuda::getCurrentCUDAStream()>>>(
                                lengths_size,
                                B,
                                block_sizes.data_ptr<index_t>(),
                                my_size,
                                offsets_contig.data_ptr<offset_t>(),
                                indices_contig.data_ptr<index_t>(),
                                weights_value_contig.data_ptr<scalar_t>(),
                                new_offsets.data_ptr<offset_t>(),
                                new_indices.data_ptr<index_t>(),
                                new_weights.data_ptr<scalar_t>(),
                                nullptr,
                                unbucketize_permute.data_ptr<index_t>());
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      }));
                }));
          }));

    } else if (bucketize_pos) {
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                ([&] {
                  _block_bucketize_sparse_features_cuda_kernel2<
                      true,
                      false,
                      true,
                      offset_t,
                      index_t,
                      std::nullptr_t>
                      <<<num_blocks,
                         threads_per_block,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          lengths_size,
                          B,
                          block_sizes.data_ptr<index_t>(),
                          my_size,
                          offsets_contig.data_ptr<offset_t>(),
                          indices_contig.data_ptr<index_t>(),
                          nullptr,
                          new_offsets.data_ptr<offset_t>(),
                          new_indices.data_ptr<index_t>(),
                          nullptr,
                          new_pos.data_ptr<index_t>(),
                          unbucketize_permute.data_ptr<index_t>());
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                }));
          }));

    } else {
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                ([&] {
                  _block_bucketize_sparse_features_cuda_kernel2<
                      true,
                      false,
                      false,
                      offset_t,
                      index_t,
                      std::nullptr_t>
                      <<<num_blocks,
                         threads_per_block,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          lengths_size,
                          B,
                          block_sizes.data_ptr<index_t>(),
                          my_size,
                          offsets_contig.data_ptr<offset_t>(),
                          indices_contig.data_ptr<index_t>(),
                          nullptr,
                          new_offsets.data_ptr<offset_t>(),
                          new_indices.data_ptr<index_t>(),
                          nullptr,
                          nullptr,
                          unbucketize_permute.data_ptr<index_t>());
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                }));
          }));
    }
  } else {
    if (weights.has_value() & bucketize_pos) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                ([&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      ([&] {
                        _block_bucketize_sparse_features_cuda_kernel2<
                            false,
                            true,
                            true,
                            offset_t,
                            index_t,
                            scalar_t>
                            <<<num_blocks,
                               threads_per_block,
                               0,
                               at::cuda::getCurrentCUDAStream()>>>(
                                lengths_size,
                                B,
                                block_sizes.data_ptr<index_t>(),
                                my_size,
                                offsets_contig.data_ptr<offset_t>(),
                                indices_contig.data_ptr<index_t>(),
                                weights_value_contig.data_ptr<scalar_t>(),
                                new_offsets.data_ptr<offset_t>(),
                                new_indices.data_ptr<index_t>(),
                                new_weights.data_ptr<scalar_t>(),
                                new_pos.data_ptr<index_t>(),
                                nullptr);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      }));
                }));
          }));

    } else if (weights.has_value()) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                ([&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      ([&] {
                        _block_bucketize_sparse_features_cuda_kernel2<
                            false,
                            true,
                            false,
                            offset_t,
                            index_t,
                            scalar_t>
                            <<<num_blocks,
                               threads_per_block,
                               0,
                               at::cuda::getCurrentCUDAStream()>>>(
                                lengths_size,
                                B,
                                block_sizes.data_ptr<index_t>(),
                                my_size,
                                offsets_contig.data_ptr<offset_t>(),
                                indices_contig.data_ptr<index_t>(),
                                weights_value_contig.data_ptr<scalar_t>(),
                                new_offsets.data_ptr<offset_t>(),
                                new_indices.data_ptr<index_t>(),
                                new_weights.data_ptr<scalar_t>(),
                                nullptr,
                                nullptr);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      }));
                }));
          }));

    } else if (bucketize_pos) {
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                ([&] {
                  _block_bucketize_sparse_features_cuda_kernel2<
                      false,
                      false,
                      true,
                      offset_t,
                      index_t,
                      std::nullptr_t>
                      <<<num_blocks,
                         threads_per_block,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          lengths_size,
                          B,
                          block_sizes.data_ptr<index_t>(),
                          my_size,
                          offsets_contig.data_ptr<offset_t>(),
                          indices_contig.data_ptr<index_t>(),
                          nullptr,
                          new_offsets.data_ptr<offset_t>(),
                          new_indices.data_ptr<index_t>(),
                          nullptr,
                          new_pos.data_ptr<index_t>(),
                          nullptr);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                }));
          }));

    } else {
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                ([&] {
                  _block_bucketize_sparse_features_cuda_kernel2<
                      false,
                      false,
                      false,
                      offset_t,
                      index_t,
                      std::nullptr_t>
                      <<<num_blocks,
                         threads_per_block,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          lengths_size,
                          B,
                          block_sizes.data_ptr<index_t>(),
                          my_size,
                          offsets_contig.data_ptr<offset_t>(),
                          indices_contig.data_ptr<index_t>(),
                          nullptr,
                          new_offsets.data_ptr<offset_t>(),
                          new_indices.data_ptr<index_t>(),
                          nullptr,
                          nullptr,
                          nullptr);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                }));
          }));
    }
  }

  return {new_lengths, new_indices, new_weights, new_pos, unbucketize_permute};
}

Tensor _float_to_fused8bitrowwise_gpu(const Tensor& input) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  auto output = at::empty(
      output_dims, // 4 = sizeof(float)
      input.options().dtype(at::kByte));

  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(nrows, threads_per_block);
  // think unsigned as we use 0, 255

  if (nrows <= 20) {
    _float_to_fused8bitrowwise_cuda_kernel<<<
        num_blocks,
        threads_per_block,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), nrows, ncols, output.data_ptr<std::uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // range_tensor is used to store the range for each embedding row.
    // We save range/255.0f as row scale, and use 255.0f / (range + kEpsilon) to
    // quantize. This will guarantee the numerical match but bring some perf
    // regression.
    auto range_tensor = at::empty({nrows}, input.options().dtype(at::kFloat));

    {
      // we need a blockDim.x that is a power of 2 no larger than the warp size
      // of 32

      int blockDim_x = 1;
      if (ncols > 16) {
        // max warp size
        blockDim_x = 32;
      } else {
        while (blockDim_x < ncols) {
          blockDim_x <<= 1;
        }
      }

      const int rows_per_block = threads_per_block / blockDim_x;
      const auto num_blocks_warp =
          cuda_calc_xblock_count(nrows, rows_per_block);

      _get_8bit_qparam_cuda_kernel<<<
          num_blocks_warp,
          dim3(blockDim_x, rows_per_block),
          0,
          at::cuda::getCurrentCUDAStream()>>>(
          input.data_ptr<float>(),
          nrows,
          ncols,
          output.data_ptr<std::uint8_t>(),
          range_tensor.data_ptr<float>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    {
      const int blockDim_x = std::min(ncols, threads_per_block);
      dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
      const auto gridDim_x = cuda_calc_xblock_count(ncols, blockDim.x);
      const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
      dim3 gridDim(gridDim_x, gridDim_y);

      _compute_8bit_quantize_cuda_kernel<<<
          gridDim,
          blockDim,
          0,
          at::cuda::getCurrentCUDAStream()>>>(
          input.data_ptr<float>(),
          range_tensor.data_ptr<float>(),
          nrows,
          ncols,
          output.data_ptr<std::uint8_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  return output;
}

Tensor _fused8bitrowwise_to_float_gpu(const Tensor& input) {
  TENSOR_ON_CUDA_GPU(input);
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned - 2 * sizeof(float);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  auto output = at::empty(
      output_dims, // 4 = sizeof(float)
      input.options().dtype(at::kFloat));

  if (nrows == 0 || output_columns == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;

  const int blockDim_x = std::min(threads_per_block, output_columns);
  dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);

  const auto gridDim_x = cuda_calc_xblock_count(output_columns, blockDim.x);
  const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
  dim3 gridDim(gridDim_x, gridDim_y);

  _fused8bitrowwise_to_float_cuda_kernel<<<
      gridDim,
      blockDim,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<std::uint8_t>(), nrows, ncols, output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

Tensor _float_to_fusednbitrowwise_gpu(
    const Tensor& input,
    const int64_t bit_rate) {
  TENSOR_ON_CUDA_GPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int nrows = input.size(0);
  const int ncols = input.size(1);
  const int num_elem_per_byte = 8 / bit_rate;
  TORCH_CHECK(
      ncols % (2 * num_elem_per_byte) == 0,
      "ncols needs to be multiple of 2 Bytes (half type size) to make the address aligned");
  const int output_columns =
      (ncols + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(at::Half);

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output = at::empty(
      {nrows, output_columns},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t

  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr auto threads_per_block = 256;
  const auto num_blocks = cuda_calc_xblock_count(nrows, threads_per_block);
  // think unsigned as we use 0, 255

  _float_to_fusednbitrowwise_cuda_kernel<<<
      num_blocks,
      threads_per_block,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      bit_rate,
      input.data_ptr<float>(),
      nrows,
      ncols,
      output.data_ptr<std::uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

Tensor _fusednbitrowwise_to_float_gpu(
    const Tensor& input,
    const int64_t bit_rate) {
  TENSOR_ON_CUDA_GPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int nrows = input.size(0);
  const int ncols = input.size(1);
  const int num_elem_per_byte = 8 / bit_rate;
  const int output_columns = (ncols - 2 * sizeof(at::Half)) * num_elem_per_byte;

  // Global memory instructions support reading or writing words of size equal
  // to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to
  // data residing in global memory compiles to a single global memory
  // instruction if and only if the size of the data type is 1, 2, 4, 8, or 16
  // bytes and the data is naturally aligned (i.e., its address is a multiple of
  // that size).
  auto output = at::empty(
      {nrows, output_columns}, // 4 = sizeof(float)
      input.options().dtype(at::kFloat)); // at::kBytes for uint8_t

  if (nrows == 0 || output_columns == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;

  const int blockDim_x = std::min(output_columns, threads_per_block);
  dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
  const auto gridDim_x = cuda_calc_xblock_count(output_columns, blockDim.x);
  const auto gridDim_y = cuda_calc_block_count(nrows, blockDim.y);
  dim3 gridDim(gridDim_x, gridDim_y);

  _fusednbitrowwise_to_float_cuda_kernel<<<
      gridDim,
      blockDim,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      bit_rate,
      input.data_ptr<uint8_t>(),
      nrows,
      ncols,
      output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

template <typename Dtype>
__global__ void reorder_batched_ad_lengths_kernel(
    // reorder lengths from (ragged) [B  x T x #num_ads_b)] to
    // [T][B][#num_ads_b], i.e. [T][sum(#num_ads_b)].
    const at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        cat_ad_lengths,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        batch_offsets,
    at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        reordered_cat_ad_lengths,
    int32_t T) {
  const int32_t B = batch_offsets.size(0) - 1;

  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const int32_t num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const int32_t input_segment_start = T * batch_offsets[b] + t * num_ads_b;
  const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets[b];

  for (int32_t i = threadIdx.x; i < num_ads_b; i += blockDim.x) {
    reordered_cat_ad_lengths[output_segment_start + i] =
        cat_ad_lengths[input_segment_start + i];
  }
}

Tensor reorder_batched_ad_lengths_gpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch) {
  TENSOR_ON_CUDA_GPU(cat_ad_lengths);
  TENSOR_ON_CUDA_GPU(batch_offsets);
  TENSORS_ON_SAME_DEVICE(cat_ad_lengths, batch_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cat_ad_lengths.get_device());

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = cat_ad_lengths.numel() / num_ads_in_batch;

  Tensor reordered_cat_ad_lengths = at::empty_like(cat_ad_lengths);

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES(
      cat_ad_lengths.type(), "reorder_batched_ad_lengths_gpu_kernel", ([&] {
        reorder_batched_ad_lengths_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                batch_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                reordered_cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                T);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
  return reordered_cat_ad_lengths;
}

template <typename Dtype>
__global__ void reorder_batched_ad_indices_kernel(
    // reorder indices from (ragged) [B  x T x #num_ads_b x length_{b, t, a})]
    // to [T][B][#num_ads_b][length_{b, t, a}], i.e. [sum(length_{b, t, a})],
    // laid out as [T][B][A][L] (if all lengths were equal).
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        cat_ad_offsets,
    const at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        cat_ad_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        reordered_cat_ad_offsets,
    at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        reordered_cat_ad_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        batch_offsets,
    int32_t T) {
  const int32_t B = batch_offsets.size(0) - 1;
  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }
  // for each ad,
  const int32_t num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const int32_t b_t_start = T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_start =
      T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_end =
      T * batch_offsets[b] + t * num_ads_b + num_ads_b;

  // Idea: we want to copy the entire segment of size sum_a(length_{b, t, a})
  // from starting point (given by cat_ad_offsets[b, t])
  // to end point (given by reordered_cat_ad_indices[t][b])
  const int32_t input_segment_start =
      cat_ad_offsets[input_segment_offset_start];
  const int32_t input_segment_end = cat_ad_offsets[input_segment_offset_end];

  const int32_t output_segment_offset_start =
      t * num_ads_in_batch + batch_offsets[b];
  const int32_t output_segment_start =
      reordered_cat_ad_offsets[output_segment_offset_start];

  for (auto i = threadIdx.x; i < input_segment_end - input_segment_start;
       i += blockDim.x) {
    reordered_cat_ad_indices[output_segment_start + i] =
        cat_ad_indices[input_segment_start + i];
  }
}

Tensor reorder_batched_ad_indices_gpu(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch) {
  TENSOR_ON_CUDA_GPU(cat_ad_offsets);
  TENSOR_ON_CUDA_GPU(cat_ad_indices);
  TENSOR_ON_CUDA_GPU(reordered_cat_ad_offsets);
  TENSOR_ON_CUDA_GPU(batch_offsets);
  TENSORS_ON_SAME_DEVICE(cat_ad_offsets, cat_ad_indices);
  TENSORS_ON_SAME_DEVICE(cat_ad_offsets, reordered_cat_ad_offsets);
  TENSORS_ON_SAME_DEVICE(cat_ad_offsets, batch_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cat_ad_offsets.get_device());

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = (cat_ad_offsets.numel() - 1) / num_ads_in_batch;
  Tensor reordered_cat_ad_indices = at::empty_like(cat_ad_indices);

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES(
      cat_ad_indices.type(), "reorder_batched_ad_indices_gpu_kernel", ([&] {
        reorder_batched_ad_indices_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                cat_ad_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                cat_ad_indices
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                reordered_cat_ad_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                reordered_cat_ad_indices
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                batch_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                T);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
  return reordered_cat_ad_indices;
}

Tensor batched_unary_embeddings_forward_cuda(
    const Tensor& weight,
    const Tensor& table_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(table_offsets);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(weight);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(offsets);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(indices);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weight.get_device());
  // N: number of tasks, T: number of tables, B: batch size
  const int32_t N = weight.size(0);
  const int32_t T = table_offsets.numel() - 1;
  const int32_t B = (offsets.numel() - 1) / T;
  TORCH_CHECK(N > 0);
  TORCH_CHECK(B > 0);
  TORCH_CHECK(T > 0);
  TORCH_CHECK(T <= 65535);
  TORCH_CHECK(N <= 65535);
  int32_t threads = std::min<int32_t>(B, 512);
  dim3 blocks(cuda_calc_xblock_count(B, threads), T, N);
  auto output = at::empty({N, B, T}, weight.options());
  AT_DISPATCH_INDEX_TYPES(
      indices.type(), "batched_unary_embeddings_forward_kernel", ([&] {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            weight.type(), "batched_unary_embeddings_forward_kernel", ([&] {
              batched_unary_embeddings_forward_kernel<scalar_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      N,
                      B,
                      T,
                      weight.data_ptr<scalar_t>(),
                      table_offsets.data_ptr<index_t>(),
                      offsets.data_ptr<index_t>(),
                      indices.data_ptr<index_t>(),
                      output.data_ptr<scalar_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
      }));
  return output;
}

Tensor batched_unary_embeddings_backward_cuda(
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& table_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  TENSOR_ON_CUDA_GPU(grad_output);
  TENSOR_ON_CUDA_GPU(weight);
  TENSOR_ON_CUDA_GPU(table_offsets);
  TENSOR_ON_CUDA_GPU(offsets);
  TENSOR_ON_CUDA_GPU(indices);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  // N: number of tasks, T: number of tables, B: batch size
  const int32_t N = grad_output.size(0);
  const int32_t B = grad_output.size(1);
  const int32_t T = grad_output.size(2);
  TORCH_CHECK(N > 0);
  TORCH_CHECK(B > 0);
  TORCH_CHECK(T > 0);
  int threads = std::min<int32_t>(N * T, 512);
  dim3 blocks(cuda_calc_xblock_count(N * T, threads));
  auto grad_weight = at::zeros_like(weight);
  AT_DISPATCH_INDEX_TYPES(
      indices.type(), "batched_unary_embeddings_backward_kernel", ([&] {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_output.type(),
            "batched_unary_embeddings_backward_kernel",
            ([&] {
              batched_unary_embeddings_backward_kernel<scalar_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      N,
                      B,
                      T,
                      grad_output.data_ptr<scalar_t>(),
                      table_offsets.data_ptr<index_t>(),
                      offsets.data_ptr<index_t>(),
                      indices.data_ptr<index_t>(),
                      grad_weight.data_ptr<scalar_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
      }));
  return grad_weight;
}

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

template <typename T>
__global__ void histogram_binning_calibration_kernel(
    const int64_t num_logits,
    const int64_t num_bins,
    const double recalibrate_value,
    const double step,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const T* const logit_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    T* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_logits) {
    return;
  }

  const T pre_sigmoid = logit_data[index] + recalibrate_value;
  const double uncalibrated = 1.0 / (1.0 + exp(-pre_sigmoid));

  bin_ids_data[index] = ceil(uncalibrated / step) - 1;

  const auto curr_bin_num_examples = bin_num_examples_data[bin_ids_data[index]];
  if (curr_bin_num_examples > bin_ctr_in_use_after) {
    const auto curr_bin_ctr =
        bin_num_positives_data[bin_ids_data[index]] / curr_bin_num_examples;
    calibrated_prediction_data[index] = curr_bin_ctr * bin_ctr_weight_value +
        uncalibrated * (1.0 - bin_ctr_weight_value);
  } else {
    calibrated_prediction_data[index] = uncalibrated;
  }
}

std::tuple<Tensor, Tensor> histogram_binning_calibration_cuda(
    const Tensor& logit,
    const Tensor& bin_num_examples,
    const Tensor& bin_num_positives,
    double positive_weight,
    double lower_bound,
    double upper_bound,
    int64_t bin_ctr_in_use_after,
    double bin_ctr_weight_value) {
  TENSOR_ON_CUDA_GPU(logit);
  TENSOR_ON_CUDA_GPU(bin_num_examples);
  TENSOR_ON_CUDA_GPU(bin_num_positives);
  TORCH_CHECK(bin_num_examples.numel() == bin_num_positives.numel());

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(logit.get_device());

  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);
  const double step = (upper_bound - lower_bound) /
      static_cast<double>(bin_num_examples.numel());

  const int32_t num_threads = 512;
  const auto logit_packed = logit.contiguous();
  const auto bin_num_examples_packed = bin_num_examples.contiguous();
  const auto bin_num_positives_packed = bin_num_positives.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logit.type(), "histogram_binning_calibration_cuda", [&]() {
        histogram_binning_calibration_kernel<scalar_t>
            <<<fbgemm_gpu::div_round_up(logit.numel(), num_threads),
               num_threads,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                logit.numel(),
                bin_num_examples.numel(),
                recalibrate_value,
                step,
                bin_ctr_in_use_after,
                bin_ctr_weight_value,
                logit_packed.data_ptr<scalar_t>(),
                bin_num_examples_packed.data_ptr<double>(),
                bin_num_positives_packed.data_ptr<double>(),
                calibrated_prediction.data_ptr<scalar_t>(),
                bin_ids.data_ptr<int64_t>());
      });

  return std::make_tuple(calibrated_prediction, bin_ids);
}

template <typename T>
__global__ void to_dense_segment_value_kernel(
    const int64_t num_lengths,
    const int64_t* const segment_value_data,
    const T* const segment_offsets_data,
    int64_t* const dense_segment_value_data) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_lengths - 1) {
    return;
  }

  const auto curr_offset = segment_offsets_data[index];
  const auto next_offset = segment_offsets_data[index + 1];
  if (next_offset > curr_offset) {
    // Add 1 to distinguish between 0 inserted by densification vs. original
    // value.
    dense_segment_value_data[index] = segment_value_data[curr_offset] + 1;
  }
}

template <typename T>
__global__ void histogram_binning_calibration_by_feature_kernel(
    const int64_t num_logits,
    const int64_t num_bins,
    const int64_t num_segments,
    const double recalibrate_value,
    const double step,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const T* const logit_data,
    const int64_t* const dense_segment_value_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    T* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_logits) {
    return;
  }

  const T pre_sigmoid = logit_data[index] + recalibrate_value;
  const double uncalibrated = 1.0 / (1.0 + exp(-pre_sigmoid));

  const int64_t curr_segment_value =
      dense_segment_value_data[index] > num_segments
      ? 0
      : std::max(0L, dense_segment_value_data[index] * num_bins);

  bin_ids_data[index] = ceil(uncalibrated / step) - 1 + curr_segment_value;

  const auto curr_bin_num_examples = bin_num_examples_data[bin_ids_data[index]];
  if (curr_bin_num_examples > bin_ctr_in_use_after) {
    const auto curr_bin_ctr =
        bin_num_positives_data[bin_ids_data[index]] / curr_bin_num_examples;
    calibrated_prediction_data[index] = curr_bin_ctr * bin_ctr_weight_value +
        uncalibrated * (1.0 - bin_ctr_weight_value);
  } else {
    calibrated_prediction_data[index] = uncalibrated;
  }
}

std::tuple<Tensor, Tensor> histogram_binning_calibration_by_feature_cuda(
    const Tensor& logit,
    const Tensor& segment_value,
    const Tensor& segment_lengths,
    int64_t num_segments,
    const Tensor& bin_num_examples,
    const Tensor& bin_num_positives,
    int64_t num_bins,
    double positive_weight,
    double lower_bound,
    double upper_bound,
    int64_t bin_ctr_in_use_after,
    double bin_ctr_weight_value) {
  TENSOR_ON_CUDA_GPU(logit);
  TENSOR_ON_CUDA_GPU(segment_value);
  TENSOR_ON_CUDA_GPU(segment_lengths);
  TENSOR_ON_CUDA_GPU(bin_num_examples);
  TENSOR_ON_CUDA_GPU(bin_num_positives);
  TORCH_CHECK(bin_num_examples.numel() == bin_num_positives.numel());

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(logit.get_device());

  // Convert lengths to offsets for better handling on GPUs.
  const auto segment_lengths_packed = segment_lengths.contiguous();
  auto segment_offsets =
      asynchronous_complete_cumsum_gpu(segment_lengths_packed.view(-1));

  // dense_segment_value is used as a temporary storage.
  Tensor dense_segment_value =
      at::zeros({logit.numel()}, segment_value.options());

  const int32_t num_threads = 512;
  const auto segment_value_packed = segment_value.contiguous();
  const auto segment_offsets_packed = segment_offsets.contiguous();
  auto dense_segment_value_packed = dense_segment_value.contiguous();
  AT_DISPATCH_INDEX_TYPES(
      segment_offsets.scalar_type(), "to_dense_segment_value_cuda", [&]() {
        to_dense_segment_value_kernel<index_t>
            <<<fbgemm_gpu::div_round_up(segment_offsets.numel(), num_threads),
               num_threads,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                segment_offsets.numel(),
                segment_value_packed.data_ptr<int64_t>(),
                segment_offsets_packed.data_ptr<index_t>(),
                dense_segment_value_packed.data_ptr<int64_t>());
      });

  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);
  const double step =
      (upper_bound - lower_bound) / static_cast<double>(num_bins);

  const auto logit_packed = logit.contiguous();
  const auto bin_num_examples_packed = bin_num_examples.contiguous();
  const auto bin_num_positives_packed = bin_num_positives.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logit.type(), "histogram_binning_calibration_by_feature_cuda", [&]() {
        histogram_binning_calibration_by_feature_kernel<scalar_t>
            <<<fbgemm_gpu::div_round_up(logit.numel(), num_threads),
               num_threads,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                logit.numel(),
                num_bins,
                num_segments,
                recalibrate_value,
                step,
                bin_ctr_in_use_after,
                bin_ctr_weight_value,
                logit_packed.data_ptr<scalar_t>(),
                dense_segment_value_packed.data_ptr<int64_t>(),
                bin_num_examples_packed.data_ptr<double>(),
                bin_num_positives_packed.data_ptr<double>(),
                calibrated_prediction.data_ptr<scalar_t>(),
                bin_ids.data_ptr<int64_t>());
      });

  return std::make_tuple(calibrated_prediction, bin_ids);
}

} // namespace fbgemm_gpu
