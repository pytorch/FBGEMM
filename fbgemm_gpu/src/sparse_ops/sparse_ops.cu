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

// Kernel for generate 1D data permute from dimension permute index.
// Used for permutation of sparse features.
template <typename index_t, typename offsets_t>
__global__
__launch_bounds__(kMaxThreads) void expand_into_jagged_permute_kernel(
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    int32_t input_size,
    const index_t* __restrict__ permute,
    index_t* __restrict__ output_permute) {
  const int32_t t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int t = t_start; t < input_size; t += stride) {
    const offsets_t output_start = output_offsets[t];
    const offsets_t segment_length = output_offsets[t + 1] - output_offsets[t];
    const offsets_t input_start = input_offsets[permute[t]];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      output_permute[output_start + i] = input_start + i;
    }
  }
}

DLL_PUBLIC Tensor expand_into_jagged_permute_cuda(
    const Tensor& permute,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    int64_t output_size) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      permute, input_offsets, output_offsets);

  TORCH_CHECK(permute.numel() > 0);
  TORCH_CHECK(permute.numel() == input_offsets.numel() - 1);
  TORCH_CHECK(permute.numel() == output_offsets.numel() - 1);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(permute.get_device());

  const auto permute_contig = permute.contiguous();
  const auto permute_size = permute.numel();

  Tensor output_permute = at::empty({output_size}, permute.options());

  // number of table per block
  constexpr int32_t T_blocks = kMaxThreads / kWarpSize;
  dim3 threads(kWarpSize, T_blocks);
  const auto blocks = cuda_calc_xblock_count(permute_size, T_blocks);
  AT_DISPATCH_INDEX_TYPES(
      permute.scalar_type(), "expand_into_jagged_permute_kernel", [&] {
        using offsets_t = index_t;
        expand_into_jagged_permute_kernel<index_t, offsets_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_offsets.data_ptr<offsets_t>(),
                output_offsets.data_ptr<offsets_t>(),
                permute_size,
                permute.data_ptr<index_t>(),
                output_permute.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output_permute;
}

// Kernel for bucketize lengths, with the Block distribution (vs. cyclic,
// block-cyclic distribution). Used for bucketize sparse feature, especially for
// checkpointing with row-wise partition (sparse_feature is partitioned
// continuously along the sparse dimension into my_size blocks)
template <typename offset_t, typename index_t>
__global__
__launch_bounds__(kMaxThreads) void _block_bucketize_sparse_features_cuda_kernel1(
    int32_t lengths_size,
    int32_t B,
    const index_t* __restrict__ block_sizes_data,
    int my_size,
    const offset_t* __restrict__ offsets_data,
    const index_t* __restrict__ indices_data,
    offset_t* __restrict__ new_lengths_data) {
  using uindex_t = std::make_unsigned_t<index_t>;
  CUDA_KERNEL_LOOP(b_t, lengths_size) {
    int32_t t = b_t / B;
    index_t blk_size = block_sizes_data[t];
    offset_t rowstart = (b_t == 0 ? 0 : offsets_data[b_t - 1]);
    offset_t rowend = offsets_data[b_t];
    for (index_t i = rowstart; i < rowend; ++i) {
      // We have use cases using none-hashed raw indices that can be either
      // negative or larger than embedding table hash_size (blk_size *
      // my_size). In cases of none-hashed indices we need to ensure
      // bucketization can distribute them into different ranks and within
      // range of blk_size, we expect the later embedding module to take care
      // of hashing indices calculation.
      uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      uindex_t p = idx < blk_size * my_size ? idx / blk_size : idx % my_size;
      new_lengths_data[p * lengths_size + b_t]++;
    }
  }
}

// Kernel for bucketize offsets, indices, and positional weights, with the Block
// distribution (vs. cyclic, block-cyclic distribution). Used for bucketize
// sparse feature, especially for checkpointing with row-wise partition
// (sparse_feature is partitioned continuously along the sparse dimension into
// my_size blocks)
template <
    bool sequence,
    bool has_weight,
    bool bucketize_pos,
    typename offset_t,
    typename index_t,
    typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void _block_bucketize_sparse_features_cuda_kernel2(
    int lengths_size,
    int32_t B,
    const index_t* __restrict__ block_sizes_data,
    int my_size,
    const offset_t* __restrict__ offsets_data,
    const index_t* __restrict__ indices_data,
    const scalar_t* __restrict__ weights_data,
    offset_t* __restrict__ new_offsets_data,
    index_t* __restrict__ new_indices_data,
    scalar_t* __restrict__ new_weights_data,
    index_t* __restrict__ new_pos_data,
    index_t* __restrict__ unbucketize_permute_data) {
  using uindex_t = std::make_unsigned_t<index_t>;
  using uoffset_t = std::make_unsigned_t<offset_t>;
  CUDA_KERNEL_LOOP(b_t, lengths_size) {
    int32_t t = b_t / B;
    index_t blk_size = block_sizes_data[t];
    offset_t rowstart = (b_t == 0 ? 0 : offsets_data[b_t - 1]);
    offset_t rowend = offsets_data[b_t];
    for (index_t i = rowstart; i < rowend; ++i) {
      // We have use cases using none-hashed raw indices that can be either
      // negative or larger than embedding table hash_size (blk_size *
      // my_size). In cases of none-hashed indices we need to ensure
      // bucketization can distribute them into different ranks and within
      // range of blk_size, we expect the later embedding module to take care
      // of hashing indices calculation.
      uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      uindex_t p = idx < blk_size * my_size ? idx / blk_size : idx % my_size;
      uindex_t new_idx =
          idx < blk_size * my_size ? idx % blk_size : idx / my_size;
      uoffset_t pos = new_offsets_data[p * lengths_size + b_t];
      new_indices_data[pos] = new_idx;
      new_offsets_data[p * lengths_size + b_t]++;
      if (sequence) {
        unbucketize_permute_data[i] = pos;
      }
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
// continuously along the sparse dimension into my_size blocks
DLL_PUBLIC std::tuple<
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
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(lengths, indices);

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
      [&] {
        using offset_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            indices_contig.scalar_type(),
            "_block_bucketize_sparse_features_cuda_kernel2",
            [&] {
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
            });
      });

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
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                [&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      [&] {
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
                      });
                });
          });
    } else if (weights.has_value()) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                [&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      [&] {
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
                      });
                });
          });
    } else if (bucketize_pos) {
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                [&] {
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
                });
          });
    } else {
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                [&] {
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
                });
          });
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
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                [&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      [&] {
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
                      });
                });
          });
    } else if (weights.has_value()) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                [&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      [&] {
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
                      });
                });
          });
    } else if (bucketize_pos) {
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                [&] {
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
                });
          });
    } else {
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                [&] {
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
                });
          });
    }
  }

  return {new_lengths, new_indices, new_weights, new_pos, unbucketize_permute};
}

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

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lengths.get_device());
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

template <typename Dtype>
__global__
__launch_bounds__(kMaxThreads) void reorder_batched_ad_lengths_kernel(
    // reorder lengths from (ragged) [B  x T x #num_ads_b)] to
    // [T][B][#num_ads_b], i.e. [T][sum(#num_ads_b)].
    const at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        cat_ad_lengths,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        batch_offsets,
    at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths) {
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
  const int32_t input_segment_start =
      broadcast_lengths ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets[b];

  for (int32_t i = threadIdx.x; i < num_ads_b; i += blockDim.x) {
    reordered_cat_ad_lengths[output_segment_start + i] = broadcast_lengths
        ? cat_ad_lengths[input_segment_start]
        : cat_ad_lengths[input_segment_start + i];
  }
}

DLL_PUBLIC Tensor reorder_batched_ad_lengths_gpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(cat_ad_lengths, batch_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cat_ad_lengths.get_device());

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = broadcast_lengths
      ? cat_ad_lengths.numel() / B
      : cat_ad_lengths.numel() / num_ads_in_batch;

  Tensor reordered_cat_ad_lengths = broadcast_lengths
      ? at::empty({T * num_ads_in_batch}, cat_ad_lengths.options())
      : at::empty_like(cat_ad_lengths);

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES(
      cat_ad_lengths.scalar_type(),
      "reorder_batched_ad_lengths_gpu_kernel",
      [&] {
        reorder_batched_ad_lengths_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                batch_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                reordered_cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                T,
                broadcast_lengths);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return reordered_cat_ad_lengths;
}

template <typename Dtype, typename index_t = int32_t>
__global__
__launch_bounds__(kMaxThreads) void reorder_batched_ad_indices_kernel(
    // reorder indices from (ragged) [B  x T x #num_ads_b x length_{b, t, a})]
    // to [T][B][#num_ads_b][length_{b, t, a}], i.e. [sum(length_{b, t, a})],
    // laid out as [T][B][A][L] (if all lengths were equal).

    // if broadcast_indices is enabled, all the indices will be copies of the
    // first batch of the cat_ad_indices, this is useful for request-only
    // broadcast
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cat_ad_offsets,
    const at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        cat_ad_indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reordered_cat_ad_offsets,
    at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        reordered_cat_ad_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        batch_offsets,
    const int32_t T,
    const bool broadcast_indices) {
  const int32_t B = batch_offsets.size(0) - 1;
  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const auto num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const auto output_segment_offset_start =
      t * num_ads_in_batch + batch_offsets[b];
  const auto output_segment_start =
      reordered_cat_ad_offsets[output_segment_offset_start];
  const int32_t input_segment_offset_start =
      broadcast_indices ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_end = broadcast_indices
      ? input_segment_offset_start + 1
      : input_segment_offset_start + num_ads_b;
  const auto input_segment_start = cat_ad_offsets[input_segment_offset_start];
  const auto input_segment_end = cat_ad_offsets[input_segment_offset_end];
  const auto num_elements = input_segment_end - input_segment_start;

  if (broadcast_indices) {
    for (int32_t i = threadIdx.x; i < num_ads_b * num_elements;
         i += blockDim.x) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i % num_elements];
    }
  } else {
    // Idea: we want to copy the entire segment of size sum_a(length_{b, t, a})
    // from starting point (given by cat_ad_offsets[b, t])
    // to end point (given by reordered_cat_ad_indices[t][b])
    for (int32_t i = threadIdx.x; i < input_segment_end - input_segment_start;
         i += blockDim.x) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i];
    }
  }
}

DLL_PUBLIC Tensor reorder_batched_ad_indices_gpu(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    const int64_t num_indices_after_broadcast) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets, batch_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cat_ad_offsets.get_device());

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = (reordered_cat_ad_offsets.numel() - 1) / num_ads_in_batch;
  Tensor reordered_cat_ad_indices;
  if (broadcast_indices) {
    TORCH_CHECK_GE(num_indices_after_broadcast, 0);
    reordered_cat_ad_indices =
        at::empty({num_indices_after_broadcast}, cat_ad_indices.options());
  } else {
    reordered_cat_ad_indices = at::empty_like(cat_ad_indices);
  }

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES(
      cat_ad_indices.scalar_type(),
      "reorder_batched_ad_indices_gpu_kernel_1",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            cat_ad_offsets.scalar_type(),
            "reorder_batched_ad_indices_gpu_kernel_2",
            [&] {
              reorder_batched_ad_indices_kernel<scalar_t, index_t><<<
                  blocks,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  batch_offsets
                      .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                  T,
                  broadcast_indices);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return reordered_cat_ad_indices;
}

// A: m, batch_size, k
// B: batch_size, k, n
// C: m, batch_size, n
// bias: batch_size, n
DLL_PUBLIC Tensor permute102_baddbmm_permute102_cuda(
    const Tensor& bias,
    const Tensor& A,
    const Tensor& B) {
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(A);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(B);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(bias);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(A.get_device());

  TENSORS_ON_SAME_DEVICE(A, B);
  TENSORS_ON_SAME_DEVICE(A, bias);
  TENSOR_NDIM_EQUALS(A, 3);
  TENSOR_NDIM_EQUALS(B, 3);

  const auto m = A.size(0);
  const auto batch_size = A.size(1);
  const auto k = A.size(2);
  const auto n = B.size(2);
  TORCH_CHECK(B.size(0) == batch_size);
  TORCH_CHECK(B.size(1) == k);
  TORCH_CHECK(bias.size(0) == batch_size);
  TORCH_CHECK(bias.size(1) == n);

  // auto C = at::empty({m, batch_size, n}, A.options());
  // auto C = at::broadcast_to(bias, {m, batch_size, n}).contiguous();
  auto C = bias.unsqueeze(0).broadcast_to({m, batch_size, n}).contiguous();

  auto handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, c10::cuda::getCurrentCUDAStream());

  // C (m, b, n) = A (m, b, k) * B (b, k, n) ---> row major
  // C (m, b, n) = (B^T (b, k, n) * A^T (m, b, k))^T ---> column major

#ifdef __HIP_PLATFORM_HCC__
  float alpha = 1.0f;
  float beta = 1.0f;

  auto Btype = HIPBLAS_R_16F;
  auto ldb = n;
  auto strideB = n * k;

  auto Atype = HIPBLAS_R_16F;
  auto lda = k * batch_size;
  auto strideA = k;

  auto Ctype = HIPBLAS_R_16F;
  auto ldc = n * batch_size;
  auto strideC = n;

  auto computeType = HIPBLAS_R_32F;

  auto result = hipblasGemmStridedBatchedEx(
      handle,
      HIPBLAS_OP_N,
      HIPBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      B.data_ptr<at::Half>(),
      Btype,
      ldb,
      strideB,
      A.data_ptr<at::Half>(),
      Atype,
      lda,
      strideA,
      &beta,
      C.data_ptr<at::Half>(),
      Ctype,
      ldc,
      strideC,
      batch_size,
      computeType,
      HIPBLAS_GEMM_DEFAULT);
  TORCH_CHECK(result == CUBLAS_STATUS_SUCCESS);
  return C;
}
#else
  float alpha = 1.0f;
  float beta = 1.0f;

  auto Btype = CUDA_R_16F;
  auto ldb = n;
  auto strideB = n * k;

  auto Atype = CUDA_R_16F;
  auto lda = k * batch_size;
  auto strideA = k;

  auto Ctype = CUDA_R_16F;
  auto ldc = n * batch_size;
  auto strideC = n;

  auto computeType = CUBLAS_COMPUTE_32F;

  auto result = cublasGemmStridedBatchedEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      B.data_ptr<at::Half>(),
      Btype,
      ldb,
      strideB,
      A.data_ptr<at::Half>(),
      Atype,
      lda,
      strideA,
      &beta,
      C.data_ptr<at::Half>(),
      Ctype,
      ldc,
      strideC,
      batch_size,
      computeType,
      CUBLAS_GEMM_DEFAULT);
  TORCH_CHECK(result == CUBLAS_STATUS_SUCCESS);
  return C;
}
#endif

// Kernel for permuting the indices and weights. Used for permutation of
// table-wise partitioned sequence embeddings

template <typename index_t, typename scalar_t>
__global__ void permute_embeddings_kernel(
    int32_t len,
    int32_t T,
    int32_t B,
    const scalar_t* __restrict__ embeddings,
    // bag level permute
    const int32_t* __restrict__ permute,
    const index_t* __restrict__ input_offsets,
    const index_t* __restrict__ output_offsets,
    scalar_t* __restrict__ permuted_embeddings) {
  int32_t b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    index_t output_start = output_offsets[b_t];
    index_t segment_length;
    if (b_t == B * T - 1) {
      segment_length = len - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    index_t input_start = input_offsets[permute[t] * B + b];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_embeddings[output_start + i] = embeddings[input_start + i];
    }
  }
}

DLL_PUBLIC std::tuple<Tensor, Tensor> permute_sequence_embeddings_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& embeddings) {
  // wrapper for permute_2D_sparse_data_cuda, kept for BC
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, embeddings);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(embeddings.get_device());

  TORCH_CHECK(
      lengths.dim() == 2,
      "The dimension of lengths tensor should be equal to 2"
      "to correctly infer number of features and batch size.")

  Tensor permuted_lengths;
  Tensor permuted_embeddings;
  c10::optional<Tensor> weights_dummy;
  c10::optional<int64_t> permuted_lengths_sum_dummy;

  const auto T = permute.numel();
  const auto B = lengths.size(1);

  permuted_lengths = at::empty({T, B}, lengths.options());

  // ignore the third element in the tuple
  std::tie(permuted_lengths, permuted_embeddings, std::ignore) =
      fbgemm_gpu::permute_2D_sparse_data_cuda(
          permute,
          lengths,
          embeddings,
          weights_dummy,
          permuted_lengths_sum_dummy);

  return {permuted_lengths, permuted_embeddings};
}

template <typename Length_T, typename Data_T>
__global__ void pack_segments_cuda_kernel(
    const Data_T* const data_ptr,
    const int64_t data_size_0,
    const Length_T* const lengths_ptr,
    const Length_T* const lengths_cum_sum,
    const Length_T max_length,
    const int64_t num_seq,
    const int64_t cell_size,
    const Data_T padding,
    Data_T* const out_ptr,
    TORCH_DSA_KERNEL_ARGS) {
  // PackSegments requires that the sum of the lengths is equal to the first
  //  dimension of data
  CUDA_KERNEL_ASSERT2(
      data_size_0 == lengths_cum_sum[num_seq - 1] + lengths_ptr[num_seq - 1]);

  CUDA_KERNEL_LOOP(i, num_seq * max_length * cell_size) {
    const auto seq = (i / cell_size) / max_length;
    const auto cell = (i / cell_size) % max_length;
    const auto offset = i % cell_size;
    if (cell >= lengths_ptr[seq]) {
      out_ptr[i] = padding;
    } else {
      const auto idx = (lengths_cum_sum[seq] + cell) * cell_size + offset;
      out_ptr[i] = data_ptr[idx];
    }
  }
}

/// Map N dim tensor to N+1 dim based on lengths tensor.
/// Sequences that are shorter than the longest sequence are padded with
/// zeros.
/// @param t_in         N dim Tensor.
/// @param lengths      1D int/long tensor contains the length in each of the
/// output.
/// @param max_length   The pre-defined max_length for the packed segments.
/// @return packed_tensor
///         packed_tensor  N + 1 dim Tensor where dim(1) is the max length,
///                        dim(0) is the batch size.
DLL_PUBLIC Tensor pack_segments_forward_cuda(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(t_in, lengths);
  TENSOR_NDIM_IS_GE(t_in, 1);
  TENSOR_NDIM_EQUALS(lengths, 1);
  TORCH_CHECK(
      t_in.dtype() == at::ScalarType::Float ||
          t_in.dtype() == at::ScalarType::Double ||
          t_in.dtype() == at::ScalarType::Half ||
          t_in.dtype() == at::ScalarType::BFloat16,
      "t_in must be of type float or double or half or bfloat16");
  TORCH_CHECK_GT(max_length, 0);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());

  const auto t_in_c = t_in.contiguous();

  Tensor packed_tensor;

  AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "pack_segments_cuda", [&] {
    const auto* const lengths_data = lengths.data_ptr<index_t>();

    // Shape of output is batch_size x max_len x ...
    auto shape = t_in_c.sizes().vec(); // Get copy of current shape
    shape[0] = max_length; // Set first element to max_len
    shape.insert(
        shape.begin(), lengths.numel()); // Insert batch size at beginning
    packed_tensor = at::zeros(shape, t_in_c.options());

    if (t_in_c.size(0) == 0 || lengths.size(0) == 0) {
      return; // Return empty output (with the proper shape)
    }

    auto lengths_prefix_sum =
        fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths);
    auto lps_data = lengths_prefix_sum.data_ptr<index_t>();

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        t_in_c.scalar_type(),
        "pack_segments_cuda-packing",
        [&] {
          const auto* const data_ptr = t_in_c.data_ptr<scalar_t>();
          auto* const out_data = packed_tensor.data_ptr<scalar_t>();
          const auto num_seq = lengths.size(0);
          const auto cell_size = t_in_c.numel() / t_in_c.size(0);
          TORCH_DSA_KERNEL_LAUNCH(
              (pack_segments_cuda_kernel<index_t, scalar_t>),
              cuda_calc_xblock_count(num_seq * max_length * cell_size, 128),
              128,
              0,
              at::cuda::getCurrentCUDAStream(),
              data_ptr,
              t_in_c.size(0),
              lengths_data,
              lps_data,
              max_length,
              num_seq,
              cell_size,
              static_cast<scalar_t>(0),
              out_data);
        });
  });

  return packed_tensor;
}

template <typename Length_T, typename Data_T>
__global__ void unpack_segments_cuda_kernel(
    const Data_T* const data_ptr,
    const Length_T* const lengths_ptr,
    const Length_T* const lengths_cum_sum,
    const Length_T max_length,
    const int64_t num_seq,
    const int64_t cell_size,
    Data_T* const out_ptr) {
  CUDA_KERNEL_LOOP(i, num_seq * max_length * cell_size) {
    const auto seq = (i / cell_size) / max_length;
    const auto cell = (i / cell_size) % max_length;
    const auto offset = i % cell_size;
    if (cell < lengths_ptr[seq]) {
      const auto idx = (lengths_cum_sum[seq] + cell) * cell_size + offset;
      out_ptr[idx] = data_ptr[i];
    }
  }
}

/// Map N+1 dim tensor to N dim based on lengths tensor
/// Sequences that are shorter than the longest sequence are padded with
/// zeros.
/// @param data         N+1 dim Tensor.
/// @param lengths      1D int/long tensor contains the length in each of the
/// input.
/// @param total_length Sum of elements in the 1D tensor legnths
/// @param max_length   The pre-defined max_length for the packed segments.
/// @return unpacked_tensor N-dimensional tensor
DLL_PUBLIC Tensor pack_segments_backward_cuda(
    const Tensor& data,
    const Tensor& lengths,
    int64_t total_length,
    int64_t max_length) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(data, lengths);
  TENSOR_NDIM_IS_GE(data, 2);
  TENSOR_NDIM_EQUALS(lengths, 1);
  TORCH_CHECK_EQ(data.size(0), lengths.size(0));
  TORCH_CHECK(
      data.dtype() == at::ScalarType::Float ||
          data.dtype() == at::ScalarType::Double ||
          data.dtype() == at::ScalarType::Half ||
          data.dtype() == at::ScalarType::BFloat16,
      "data must be of type float or double or half or bfloat16");
  TORCH_CHECK(
      max_length == data.size(1),
      "max_length should be equal to the second dimension of the packed segments");

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(data.get_device());

  Tensor unpacked_tensor; // The output tensor

  AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "unpack_segments_cuda", [&] {
    const auto* const lengths_data = lengths.data_ptr<index_t>();

    // Create output tensor of appropriate dimensions
    auto shape = data.sizes().vec();
    shape.erase(shape.begin());
    shape[0] = total_length;
    unpacked_tensor = at::empty(shape, data.options());

    if (!(data.size(0) && data.size(1))) { // TODO: What does this mean?
      return;
    }

    auto lengths_prefix_sum =
        fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths);
    auto lps_data = lengths_prefix_sum.data_ptr<index_t>();

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        data.scalar_type(),
        "unpack_segments_cuda-unpacking",
        [&] {
          const auto num_seq = lengths.size(0);
          const auto cell_size = data.numel() / (data.size(0) * data.size(1));
          const auto* const data_ptr = data.data_ptr<scalar_t>();
          auto* const out_data = unpacked_tensor.data_ptr<scalar_t>();

          unpack_segments_cuda_kernel<index_t, scalar_t>
              <<<cuda_calc_xblock_count(num_seq * max_length * cell_size, 128),
                 128,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  data_ptr,
                  lengths_data,
                  lps_data,
                  max_length,
                  num_seq,
                  cell_size,
                  out_data);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  });

  return unpacked_tensor;
}

constexpr int MAX_ELEMENTS_PER_THREAD = 4;

template <
    typename index_t,
    typename scalar_t,
    int UNROLL_FACTOR,
    bool indices_sorted>
__global__ __launch_bounds__(kMaxThreads) void index_select_2d_kernel(
    const at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        orig_indices,
    at::PackedTensorAccessor64<scalar_t, 2> output,
    TORCH_DSA_KERNEL_ARGS) {
  const int N = indices.size(0);
  const int input_size = input.size(0);
  const int D = input.size(1);
  CUDA_KERNEL_ASSERT2(output.size(0) == N);

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const index_t src_idx = indices[row];
    const int64_t dst_idx = indices_sorted ? orig_indices[row] : row;
    CUDA_KERNEL_ASSERT2(src_idx < input_size);
    int col;
    for (col = threadIdx.x * UNROLL_FACTOR;
         col < D / UNROLL_FACTOR * UNROLL_FACTOR;
         col += blockDim.x * UNROLL_FACTOR) {
#pragma unroll
      for (int i = 0; i < UNROLL_FACTOR; i++) {
        output[dst_idx][col + i] = LDG(&input[src_idx][col + i]);
      }
    }
    for (; col < D; ++col) {
      output[dst_idx][col] = LDG(&input[src_idx][col]);
    }
  }
}

template <typename index_t, typename scalar_t, int UNROLL_FACTOR>
__global__
__launch_bounds__(kMaxThreads) void index_add_2d_with_unique_indices_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        out_grad,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        orig_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor64<scalar_t, 2> in_deduped_grad,
    const int stride_D,
    const int rounded_D,
    const int remaining_D,
    const bool consecutive_indices,
    const int consecutive_range_start) {
  const int start_offset = blockIdx.x == 0 ? 0 : offsets[blockIdx.x - 1];
  const int end_offset = offsets[blockIdx.x];
  index_t dst_idx = consecutive_indices ? blockIdx.x + consecutive_range_start
                                        : unique_indices[blockIdx.x];
  const bool has_remainder = blockIdx.y == blockDim.y - 1 && remaining_D > 0 &&
      threadIdx.x < remaining_D;

  // Buffer for storing temporary results
  scalar_t sum[MAX_ELEMENTS_PER_THREAD];
  for (int i = 0; i < MAX_ELEMENTS_PER_THREAD; i++) {
    sum[i] = 0;
  }

  scalar_t sum_remainder = 0;

  // Each thread block processes max of stride_D elements
  int start_D = (blockIdx.y * stride_D) + (threadIdx.x * UNROLL_FACTOR);

  // For each row
  for (int row = start_offset; row < end_offset; row++) {
    int64_t src_idx = orig_indices[row];
    int col, i;
    for (col = start_D, i = 0; col < start_D + stride_D && col < rounded_D;
         col += blockDim.x * UNROLL_FACTOR, i += UNROLL_FACTOR) {
#pragma unroll
      for (int j = 0; j < UNROLL_FACTOR; j++) {
        sum[i + j] += LDG(&out_grad[src_idx][col + j]);
      }
    }
    if (has_remainder) {
      sum_remainder += LDG(&out_grad[src_idx][rounded_D + threadIdx.x]);
    }
  } // for each row

  // Write results to global memory
  int col, i;
  for (col = start_D, i = 0; col < start_D + stride_D && col < rounded_D;
       col += blockDim.x * UNROLL_FACTOR, i += UNROLL_FACTOR) {
#pragma unroll
    for (int j = 0; j < UNROLL_FACTOR; j++) {
      in_deduped_grad[dst_idx][col + j] = sum[i + j];
    }
  }
  if (has_remainder) {
    in_deduped_grad[dst_idx][rounded_D + threadIdx.x] += sum_remainder;
  }
}

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void compute_frequency_sequence_kernel(
    index_t* input,
    int64_t* output,
    index_t start_input,
    const int input_size) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= input_size) {
    return;
  }
  // Atomic could become a bottleneck if frequencies are very skew
  atomicAdd(&output[input[i] - start_input], 1);
}

DLL_PUBLIC void compute_frequency_sequence(
    const Tensor& input,
    Tensor& output,
    const int start_input,
    const int output_size) {
  output = at::zeros({output_size}, input.options().dtype(at::kLong));

  AT_DISPATCH_INDEX_TYPES(
      input.scalar_type(), "compute_frequency_sequence_kernel_1", [&] {
        compute_frequency_sequence_kernel<index_t>
            <<<cuda_calc_xblock_count(input.numel(), kWarpSize),
               kWarpSize,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<index_t>(),
                output.data_ptr<int64_t>(),
                start_input,
                input.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

template <
    typename scalar_t,
    int ndim,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
at::PackedTensorAccessor32<scalar_t, ndim, PtrTraits>
dummy_packed_accessor32() {
  std::array<int64_t, ndim> zeros{};
  return {nullptr, zeros.data(), zeros.data()};
}

template <
    typename scalar_t,
    int ndim,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
at::PackedTensorAccessor64<scalar_t, ndim, PtrTraits>
dummy_packed_accessor64() {
  std::array<int64_t, ndim> zeros{};
  return {nullptr, zeros.data(), zeros.data()};
}

DLL_PUBLIC Tensor index_select_cuda(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& orig_indices,
    const bool indices_sorted) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int N = indices.size(0);
  auto output_shape = input.sizes().vec();
  output_shape[0] = N;

  if (input.numel() == 0 || N == 0) {
    return at::empty(output_shape, input.options());
  }

  Tensor input_reshaped = input.reshape({input.size(0), -1});
  const int D = input_reshaped.size(1);

  Tensor output = at::empty({N, D}, input_reshaped.options());

  const int UNROLL_FACTOR = 2;

#define LAUNCH_INDEX_SELECT(INDICES_SORTED)                                   \
  TORCH_DSA_KERNEL_LAUNCH(                                                    \
      (index_select_2d_kernel<                                                \
          index_t,                                                            \
          scalar_t,                                                           \
          UNROLL_FACTOR,                                                      \
          INDICES_SORTED>),                                                   \
      cuda_calc_xblock_count(N, 1),                                           \
      std::min(div_round_up(D, UNROLL_FACTOR), kMaxThreads),                  \
      0,                                                                      \
      at::cuda::getCurrentCUDAStream(),                                       \
      input_reshaped.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(), \
      indices.packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),         \
      INDICES_SORTED                                                          \
          ? orig_indices                                                      \
                .packed_accessor64<int64_t, 1, at::RestrictPtrTraits>()       \
          : dummy_packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),     \
      output.packed_accessor64<scalar_t, 2>());

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "index_add_2d_kernel_1", [&] {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_reshaped.scalar_type(), "index_add_2d_kernel_2", [&] {
          if (indices_sorted) {
            LAUNCH_INDEX_SELECT(true)
          } else {
            LAUNCH_INDEX_SELECT(false)
          }
        });
  });

#undef LAUNCH_INDEX_SELECT

  return output.reshape(output_shape);
}

DLL_PUBLIC Tensor index_add_with_unique_indices_cuda(
    const Tensor& grad_output,
    const Tensor& sorted_indices,
    const Tensor& orig_indices,
    std::vector<int64_t>& input_shape,
    const int consecutive_range_start,
    const int consecutive_range_length) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const int N = grad_output.size(0);

  if (grad_output.numel() == 0) {
    return at::zeros(input_shape, grad_output.options());
  }

  const Tensor grad_output_reshaped = grad_output.reshape({N, -1});
  const int D = grad_output_reshaped.size(1);

  TORCH_CHECK(sorted_indices.size(0) == N);

  Tensor input_grad = at::zeros({input_shape[0], D}, grad_output.options());
  bool consecutive_indices =
      consecutive_range_start >= 0 && consecutive_range_length > 0;

  AT_DISPATCH_INDEX_TYPES(
      sorted_indices.scalar_type(), "index_add_2d_kernel_1", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_output.scalar_type(), "index_add_2d_kernel_2", [&] {
              // UNROLL_FACTOR is determined based on the empirical study
              const int UNROLL_FACTOR = std::is_same<scalar_t, float>() ? 4 : 2;
              const int rounded_D = D / UNROLL_FACTOR * UNROLL_FACTOR;
              const int remaining_D = D - rounded_D;
              int block_size =
                  std::min(div_round_up(D, UNROLL_FACTOR), kMaxThreads);
              block_size = std::max(remaining_D, block_size);
              // Number of elements per block
              const int stride_D = MAX_ELEMENTS_PER_THREAD * block_size;

              int num_unique_indices;
              Tensor unique_indices, offsets;
              if (consecutive_indices) {
                TORCH_CHECK(
                    consecutive_range_start < input_shape[0] &&
                    consecutive_range_start + consecutive_range_length - 1 <
                        input_shape[0]);

                // Since indices are selected from consecutive range, we can
                // infer the number of unique indices from
                // consecutive_range_length
                num_unique_indices = consecutive_range_length;
                compute_frequency_sequence(
                    sorted_indices,
                    offsets,
                    consecutive_range_start,
                    num_unique_indices);
                offsets = offsets.cumsum(0);
              } else {
                Tensor unique_count;
                // Unique consecutive does D->H transfer internally
                // (enforcing synchronization between host and device)
                std::tie(unique_indices, std::ignore, unique_count) =
                    at::unique_consecutive(sorted_indices, false, true, 0);

                // This does D->H transfer
                num_unique_indices = unique_indices.numel();
                offsets = unique_count.cumsum(0);
              }

              const dim3 grid_size(
                  cuda_calc_xblock_count(num_unique_indices, 1),
                  (D + stride_D - 1) / stride_D,
                  1);

              index_add_2d_with_unique_indices_kernel<
                  index_t,
                  scalar_t,
                  UNROLL_FACTOR><<<
                  grid_size,
                  block_size,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  grad_output_reshaped
                      .packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                  consecutive_indices ? dummy_packed_accessor32<
                                            index_t,
                                            1,
                                            at::RestrictPtrTraits>()
                                      : unique_indices.packed_accessor32<
                                            index_t,
                                            1,
                                            at::RestrictPtrTraits>(),
                  orig_indices
                      .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                  offsets
                      .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                  input_grad.packed_accessor64<scalar_t, 2>(),
                  stride_D, // Pass constants as kernel args
                  rounded_D,
                  remaining_D,
                  consecutive_indices,
                  consecutive_range_start);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return input_grad.reshape(input_shape);
}

// TODO: Update UNROLL_FACTOR
constexpr int GROUP_INDEX_SELECT_UNROLL_FACTOR = 1;
constexpr int GROUP_INDEX_SELECT_COLS_PER_WARP =
    GROUP_INDEX_SELECT_UNROLL_FACTOR * kWarpSize;
// GROUP_INDEX_SELECT_COLS_PER_WARP must be power of two
constexpr int GROUP_INDEX_SELECT_LOG_COLS_PER_WARP =
    log2_calc<GROUP_INDEX_SELECT_COLS_PER_WARP>::value;

int get_group_index_select_cols_per_warp() {
  return GROUP_INDEX_SELECT_COLS_PER_WARP;
}

template <
    typename index_t,
    typename scalar_t,
    bool USE_INDEX_SELECT,
    bool USE_VAR_COLS,
    int UNROLL_FACTOR,
    int COLS_PER_WARP,
    int LOG_COLS_PER_WARP>
__global__
__launch_bounds__(kMaxThreads) void group_index_select_or_add_2d_kernel(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const int64_t max_indices,
    const int64_t num_work_rows, // number of rows to work on per member
    const int64_t group_size) {
  const auto total_num_warps = warp_offsets_group[group_size];
  for (int64_t warp_id = threadIdx.y * gridDim.x + blockIdx.x;
       warp_id < total_num_warps;
       warp_id += gridDim.x * blockDim.y) {
    int32_t member_id, member_warp_id, num_cols, warps_per_row;
    if (USE_VAR_COLS) {
      __shared__ int member_ids[kMaxThreads / kWarpSize];
      if (threadIdx.x == 0) {
        binary_search_range(
            &member_ids[threadIdx.y],
            warp_offsets_group + 1,
            warp_id,
            group_size);
      }
      syncwarp();
      member_id = member_ids[threadIdx.y];
      num_cols = num_cols_group[member_id];
      warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
      member_warp_id = warp_id - warp_offsets_group[member_id];
    } else {
      // All columns are the same
      num_cols = num_cols_group[0];
      warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
      member_id = warp_id / (warps_per_row * num_work_rows);
      member_warp_id = warp_id - (member_id * warps_per_row * num_work_rows);
    }
    const auto row = member_warp_id / warps_per_row;
    const auto col_offset =
        ((member_warp_id % warps_per_row) << LOG_COLS_PER_WARP) +
        (threadIdx.x * UNROLL_FACTOR);
    scalar_t* input =
        reinterpret_cast<scalar_t*>(input_ptrs[member_id]) + col_offset;
    scalar_t* output =
        reinterpret_cast<scalar_t*>(output_ptrs[member_id]) + col_offset;
    index_t* indices = reinterpret_cast<index_t*>(indices_ptrs[member_id]);
    const index_t idx = indices[row];
    CUDA_KERNEL_ASSERT(idx < max_indices)
#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR && col_offset + i < num_cols; i++) {
      // Compile time conditional
      if (USE_INDEX_SELECT) {
        output[row * num_cols + i] = LDG(&input[idx * num_cols + i]);
      } else {
        gpuAtomicAddNoReturn(
            &output[idx * num_cols + i], input[row * num_cols + i]);
      }
    }
  }
}

DLL_PUBLIC void group_index_select_or_add_cuda(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const c10::ScalarType& input_scalar_type,
    const c10::ScalarType& indices_scalar_type,
    const c10::DeviceIndex& device,
    const int max_indices,
    const int num_work_rows,
    const int64_t total_num_warps,
    const int group_size,
    const bool use_index_select,
    const bool use_var_cols) {
  if (group_size == 0) {
    return;
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(device);

  // Partition work based on num_work_rows
  uint32_t num_warps_per_threadblock = kMaxThreads / kWarpSize;
  uint32_t max_grid_size =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8;
  uint32_t grid_size = std::min(
      cuda_calc_xblock_count(total_num_warps, num_warps_per_threadblock),
      max_grid_size);
  dim3 block_size(kWarpSize, num_warps_per_threadblock, 1);

#define INVOKE_GROUP_INDEX_SELECT_OR_ADD(USE_INDEX_SELECT, USE_VAR_COLS) \
  group_index_select_or_add_2d_kernel<                                   \
      index_t,                                                           \
      scalar_t,                                                          \
      USE_INDEX_SELECT,                                                  \
      USE_VAR_COLS,                                                      \
      GROUP_INDEX_SELECT_UNROLL_FACTOR,                                  \
      GROUP_INDEX_SELECT_COLS_PER_WARP,                                  \
      GROUP_INDEX_SELECT_LOG_COLS_PER_WARP>                              \
      <<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(  \
          input_ptrs,                                                    \
          output_ptrs,                                                   \
          indices_ptrs,                                                  \
          warp_offsets_group,                                            \
          num_cols_group,                                                \
          max_indices,                                                   \
          num_work_rows,                                                 \
          group_size)

  AT_DISPATCH_INDEX_TYPES(
      indices_scalar_type, "group_index_select_2d_wrapper_1", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input_scalar_type, "group_index_select_2d_wrapper_2", [&] {
              if (use_index_select) {
                if (use_var_cols) {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(true, true);
                } else {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(true, false);
                }
              } else {
                if (use_var_cols) {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(false, true);
                } else {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(false, false);
                }
              }
            });
      });

#undef INVOKE_GROUP_INDEX_SELECT_OR_ADD
}

} // namespace fbgemm_gpu
