/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/KernelUtils.h>

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"
#include "fbgemm_gpu/utils/cuda_utilities.cuh"

#include <limits>

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Kernel for bucketize lengths, with the Cyclic distribution (vs. block,
// block-cyclic distribution). Used for bucketize sparse feature with row-wise
// partition (sparse_feature is partitioned cyclically along the sparse
// dimension into my_size blocks)
template <typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void _bucketize_sparse_features_cuda_kernel1(
    int lengths_size,
    int my_size,
    const scalar_t* __restrict__ offsets_data,
    const scalar_t* __restrict__ indices_data,
    scalar_t* __restrict__ new_lengths_data) {
  using uscalar_t = std::make_unsigned_t<scalar_t>;
  CUDA_KERNEL_LOOP(r, lengths_size) {
    scalar_t rowstart = (r == 0 ? 0 : offsets_data[r - 1]);
    scalar_t rowend = offsets_data[r];
    for (scalar_t i = rowstart; i < rowend; ++i) {
      // Need to handle negative indices if we use raw indices instead of hashed
      // indices, convert to unsigned
      uscalar_t idx = static_cast<uscalar_t>(indices_data[i]);
      uscalar_t p = idx % my_size;
      new_lengths_data[p * lengths_size + r]++;
    }
  }
}

// Kernel for bucketize offsets, indices, and positional weights, with the
// Cyclic distribution (vs. block, block-cyclic distribution). Used for
// bucketize sparse feature with row-wise partition (sparse_feature is
// partitioned cyclically along the sparse dimension into my_size blocks)
template <
    bool has_weight,
    bool bucketize_pos,
    typename index_t,
    typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void _bucketize_sparse_features_cuda_kernel2(
    int lengths_size,
    int my_size,
    const index_t* __restrict__ offsets_data,
    const index_t* __restrict__ indices_data,
    const scalar_t* __restrict__ weights_data,
    index_t* __restrict__ new_offsets_data,
    index_t* __restrict__ new_indices_data,
    scalar_t* __restrict__ new_weights_data,
    index_t* __restrict__ new_pos_data) {
  using uindex_t = std::make_unsigned_t<index_t>;
  CUDA_KERNEL_LOOP(r, lengths_size) {
    index_t rowstart = r == 0 ? 0 : offsets_data[r - 1];
    index_t rowend = offsets_data[r];
    for (index_t i = rowstart; i < rowend; ++i) {
      // Need to handle negative indices if we use raw indices instead of hashed
      // indices, convert to unsigned
      uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      uindex_t p = idx % my_size;
      uindex_t new_idx = idx / my_size;
      uindex_t pos = new_offsets_data[p * lengths_size + r];
      new_indices_data[pos] = new_idx;
      new_offsets_data[p * lengths_size + r]++;
      if (has_weight) {
        new_weights_data[pos] = weights_data[i];
      }
      if (bucketize_pos) {
        new_pos_data[pos] = i - rowstart;
      }
    }
  }
}

// This function partitions sparse features
// cyclically along the sparse dimension into my_size blocks
DLL_PUBLIC
std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
bucketize_sparse_features_cuda(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const int64_t my_size,
    const std::optional<Tensor>& weights) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(lengths, indices);

  CUDA_DEVICE_GUARD(lengths);

  // allocate tensors and buffers
  TORCH_CHECK(
      lengths.numel() <= std::numeric_limits<int>::max(),
      "lengths.numel() exceeds the supported range");
  const auto lengths_size = static_cast<int>(lengths.numel());
  const auto new_lengths_size = lengths_size * my_size;
  auto offsets = at::empty({lengths_size}, lengths.options());
  auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
  auto new_offsets = at::empty({new_lengths_size}, lengths.options());
  auto new_indices = at::empty_like(indices);
  auto indices_contig = indices.contiguous();
  auto offsets_contig = offsets.contiguous();
  Tensor new_weights;
  Tensor new_pos;

  // Count Non-zeros
  offsets_contig = fbgemm_gpu::asynchronous_inclusive_cumsum_gpu(lengths);
  int threads_per_block = 256;
  const auto num_blocks = utils::cuda::cap_grid_dim_x_from_workload(
      lengths_size, threads_per_block, at::cuda::getCurrentCUDAStream());

  AT_DISPATCH_INDEX_TYPES(
      indices_contig.scalar_type(),
      "_bucketize_sparse_features_cuda_kernel1",
      ([&] {
        FBGEMM_LAUNCH_KERNEL(
            (_bucketize_sparse_features_cuda_kernel1<index_t>),
            num_blocks,
            threads_per_block,
            0,
            at::cuda::getCurrentCUDAStream(),
            lengths_size,
            my_size,
            offsets_contig.data_ptr<index_t>(),
            indices_contig.data_ptr<index_t>(),
            new_lengths.data_ptr<index_t>());
      }));

  // Bucketize non-zeros
  new_offsets = fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(new_lengths);
  const auto bucketize = [&]<bool kBucketizePos>() {
    dispatch_index_types(
        indices_contig.scalar_type(),
        weights.has_value()
            ? "_bucketize_sparse_features_weight_cuda_kernel2_1"
            : "_bucketize_sparse_features_cuda_kernel2",
        [&]<typename index_t>() {
          const auto launch = [&]<bool kHasWeight, typename weights_t>(
                                  const weights_t* weights_data,
                                  weights_t* new_weights_data) {
            FBGEMM_LAUNCH_KERNEL(
                (_bucketize_sparse_features_cuda_kernel2<
                    kHasWeight,
                    kBucketizePos,
                    index_t,
                    weights_t>),
                num_blocks,
                threads_per_block,
                0,
                at::cuda::getCurrentCUDAStream(),
                lengths_size,
                my_size,
                offsets_contig.data_ptr<index_t>(),
                indices_contig.data_ptr<index_t>(),
                weights_data,
                new_offsets.data_ptr<index_t>(),
                new_indices.data_ptr<index_t>(),
                new_weights_data,
                kBucketizePos ? new_pos.data_ptr<index_t>() : nullptr);
          };
          if (weights.has_value()) {
            const auto weights_value = weights.value();
            const auto weights_value_contig = weights_value.contiguous();
            new_weights = at::empty_like(weights_value);
            FBGEMM_DISPATCH_FLOAT_ONLY(
                weights_value.scalar_type(),
                "_bucketize_sparse_features_cuda_weight_kernel2_2",
                ([&] {
                  launch.template operator()<true>(
                      weights_value_contig.data_ptr<scalar_t>(),
                      new_weights.data_ptr<scalar_t>());
                }));
          } else {
            launch.template operator()<false, std::nullptr_t>(nullptr, nullptr);
          }
        });
  };
  if (bucketize_pos) {
    new_pos = at::empty_like(indices);
    bucketize.template operator()<true>();
  } else {
    bucketize.template operator()<false>();
  }

  return {new_lengths, new_indices, new_weights, new_pos};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "bucketize_sparse_features",
    fbgemm_gpu::bucketize_sparse_features_cuda);
