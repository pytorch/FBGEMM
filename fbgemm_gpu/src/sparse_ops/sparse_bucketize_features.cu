/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

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
std::tuple<Tensor, Tensor, c10::optional<Tensor>, c10::optional<Tensor>>
bucketize_sparse_features_cuda(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const int64_t my_size,
    const c10::optional<Tensor>& weights) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(lengths, indices);

  CUDA_DEVICE_GUARD(lengths);

  // allocate tensors and buffers
  const int lengths_size = lengths.numel();
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
  // count nonzeros
  offsets_contig = fbgemm_gpu::asynchronous_inclusive_cumsum_gpu(lengths);
  int threads_per_block = 256;
  const auto num_blocks =
      cuda_calc_xblock_count(lengths_size, threads_per_block);
  AT_DISPATCH_INDEX_TYPES(
      indices_contig.scalar_type(),
      "_bucketize_sparse_features_cuda_kernel1",
      ([&] {
        _bucketize_sparse_features_cuda_kernel1<<<
            num_blocks,
            threads_per_block,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            lengths_size,
            my_size,
            offsets_contig.data_ptr<index_t>(),
            indices_contig.data_ptr<index_t>(),
            new_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
  // bucketize nonzeros
  new_offsets = fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(new_lengths);
  if (weights.has_value() & bucketize_pos) {
    Tensor weights_value = weights.value();
    auto weights_value_contig = weights_value.contiguous();
    new_weights = at::empty_like(weights_value);
    new_pos = at::empty_like(indices);
    AT_DISPATCH_INDEX_TYPES(
        indices_contig.scalar_type(),
        "_bucketize_sparse_features_weight_cuda_kernel2_1",
        ([&] {
          AT_DISPATCH_FLOATING_TYPES(
              weights_value.scalar_type(),
              "_bucketize_sparse_features_cuda_weight_kernel2_2",
              ([&] {
                _bucketize_sparse_features_cuda_kernel2<
                    true,
                    true,
                    index_t,
                    scalar_t>
                    <<<num_blocks,
                       threads_per_block,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        lengths_size,
                        my_size,
                        offsets_contig.data_ptr<index_t>(),
                        indices_contig.data_ptr<index_t>(),
                        weights_value_contig.data_ptr<scalar_t>(),
                        new_offsets.data_ptr<index_t>(),
                        new_indices.data_ptr<index_t>(),
                        new_weights.data_ptr<scalar_t>(),
                        new_pos.data_ptr<index_t>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }));
        }));
  } else if (weights.has_value()) {
    Tensor weights_value = weights.value();
    auto weights_value_contig = weights_value.contiguous();
    new_weights = at::empty_like(weights_value);
    AT_DISPATCH_INDEX_TYPES(
        indices_contig.scalar_type(),
        "_bucketize_sparse_features_weight_cuda_kernel2_1",
        ([&] {
          AT_DISPATCH_FLOATING_TYPES(
              weights_value.scalar_type(),
              "_bucketize_sparse_features_cuda_weight_kernel2_2",
              ([&] {
                _bucketize_sparse_features_cuda_kernel2<
                    true,
                    false,
                    index_t,
                    scalar_t>
                    <<<num_blocks,
                       threads_per_block,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        lengths_size,
                        my_size,
                        offsets_contig.data_ptr<index_t>(),
                        indices_contig.data_ptr<index_t>(),
                        weights_value_contig.data_ptr<scalar_t>(),
                        new_offsets.data_ptr<index_t>(),
                        new_indices.data_ptr<index_t>(),
                        new_weights.data_ptr<scalar_t>(),
                        nullptr);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }));
        }));
  } else if (bucketize_pos) {
    new_pos = at::empty_like(indices);
    AT_DISPATCH_INDEX_TYPES(
        indices_contig.scalar_type(),
        "_bucketize_sparse_features_cuda_kernel2",
        ([&] {
          _bucketize_sparse_features_cuda_kernel2<
              false,
              true,
              index_t,
              std::nullptr_t>
              <<<num_blocks,
                 threads_per_block,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  lengths_size,
                  my_size,
                  offsets_contig.data_ptr<index_t>(),
                  indices_contig.data_ptr<index_t>(),
                  nullptr,
                  new_offsets.data_ptr<index_t>(),
                  new_indices.data_ptr<index_t>(),
                  nullptr,
                  new_pos.data_ptr<index_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
  } else {
    AT_DISPATCH_INDEX_TYPES(
        indices_contig.scalar_type(),
        "_bucketize_sparse_features_cuda_kernel2",
        ([&] {
          _bucketize_sparse_features_cuda_kernel2<
              false,
              false,
              index_t,
              std::nullptr_t>
              <<<num_blocks,
                 threads_per_block,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  lengths_size,
                  my_size,
                  offsets_contig.data_ptr<index_t>(),
                  indices_contig.data_ptr<index_t>(),
                  nullptr,
                  new_offsets.data_ptr<index_t>(),
                  new_indices.data_ptr<index_t>(),
                  nullptr,
                  nullptr);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
  }

  return {new_lengths, new_indices, new_weights, new_pos};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "bucketize_sparse_features",
    fbgemm_gpu::bucketize_sparse_features_cuda);
