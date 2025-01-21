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

// Kernel for calulating lengthh idx to feature id mapping. Used for block
// bucketize sparse features with variable batch size for row-wise partition
template <typename offset_t>
__global__
__launch_bounds__(kMaxThreads) void _populate_length_to_feature_id_inplace_kernel(
    const uint64_t max_B,
    const int T,
    const offset_t* const __restrict__ batch_size_per_feature,
    const offset_t* const __restrict__ batch_size_offsets,
    offset_t* const __restrict__ length_to_feature_idx) {
  const auto b_t = blockIdx.x * blockDim.x + threadIdx.x;

  const auto t = b_t / max_B;
  const auto b = b_t % max_B;

  if (t >= T || b >= batch_size_per_feature[t]) {
    return;
  }

  length_to_feature_idx[batch_size_offsets[t] + b] = t;
}

void adjust_block_bucketize_sparse_features_kernel_launch_configs_based_on_smem(
    int* smem_size,
    dim3* block_dims,
    dim3* grid_dims,
    int* max_smem,
    const int lengths_size,
    const int my_size,
    const int device) {
  // V100: 96 KB; A100: 160 KB; H100: 228 KB.
  int max_shared_bytes = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared_bytes,
#ifndef __HIP_PLATFORM_AMD__
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
#else
      hipDeviceAttributeMaxSharedMemoryPerBlock,
#endif
      device));

  int shared_kb = max_shared_bytes >> 10;
  // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
  // V100: 64 KB; A100: 96 KB; H100: 144 KB
  int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
  TORCH_CHECK(used_shared_kb > 0);

  *max_smem = used_shared_kb << 10;
  while (*smem_size > *max_smem && block_dims->y > 0) {
    block_dims->y--;
    *smem_size = my_size * block_dims->y * sizeof(uint64_t);
  }
  TORCH_CHECK(
      block_dims->y > 0,
      "block_bucketize_sparse_features does not have sufficient shared memory."
      "Please contact the FBGEMM team.")
  grid_dims->x = cuda_calc_xblock_count(lengths_size, block_dims->y);
}

template <typename func_t>
void increase_gpu_max_dynamic_shared_memory(func_t kernel, const int max_smem) {
  TORCH_CHECK(max_smem > 0);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      (void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Kernel for bucketize lengths, with the Block distribution (vs. cyclic,
// block-cyclic distribution). Used for bucketize sparse feature, especially for
// checkpointing with row-wise partition (sparse_feature is partitioned
// continuously along the sparse dimension into my_size blocks)
template <typename offset_t, typename index_t>
__global__
__launch_bounds__(kMaxThreads) void _block_bucketize_sparse_features_cuda_kernel1(
    const int32_t lengths_size,
    const int32_t B,
    const index_t* const __restrict__ block_sizes_data,
    const index_t* const __restrict__ total_num_blocks,
    const int my_size,
    const offset_t* const __restrict__ offsets_data,
    const index_t* const __restrict__ indices_data,
    offset_t* const __restrict__ new_lengths_data,
    offset_t* __restrict__ length_to_feature_idx,
    const index_t* const __restrict__ block_bucketize_pos_concat,
    const index_t* const __restrict__ block_bucketize_pos_offsets,
    index_t* __restrict__ indices_to_lb) {
  using uindex_t = std::make_unsigned_t<index_t>;
  const auto bt_start = blockIdx.x * blockDim.y + threadIdx.y;
  const auto stride = gridDim.x * blockDim.y;
  for (auto b_t = bt_start; b_t < lengths_size; b_t += stride) {
    const auto t = length_to_feature_idx ? length_to_feature_idx[b_t] : b_t / B;
    index_t blk_size = block_sizes_data[t];
    const index_t local_num_blks =
        total_num_blocks == nullptr ? 1 : (total_num_blocks[t] / my_size);
    const index_t global_num_blks =
        total_num_blocks == nullptr ? my_size : total_num_blocks[t];
    const index_t global_idx_size = blk_size * global_num_blks;
    const index_t local_idx_size = blk_size * local_num_blks;
    offset_t rowstart = (b_t == 0 ? 0 : offsets_data[b_t - 1]);
    offset_t rowend = offsets_data[b_t];
    const auto use_block_bucketize_pos =
        (block_bucketize_pos_concat != nullptr);
    // We have use cases using none-hashed raw indices that can be either
    // negative or larger than embedding table hash_size (blk_size *
    // my_size). In cases of none-hashed indices we need to ensure
    // bucketization can distribute them into different ranks and within
    // range of blk_size, we expect the later embedding module to take care
    // of hashing indices calculation.
    if (!use_block_bucketize_pos) {
      for (auto i = rowstart + threadIdx.x; i < rowend; i += blockDim.x) {
        uindex_t idx = static_cast<uindex_t>(indices_data[i]);
        uindex_t p = idx < global_idx_size
            ? idx / local_idx_size
            : (idx % global_num_blks) / local_num_blks;
        atomicAdd(&new_lengths_data[p * lengths_size + b_t], 1);
      }
      return;
    }

    const index_t bucketize_max_idx = (t + 1) * (my_size + 1) - 1;
    const uindex_t blk_scalar =
        block_bucketize_pos_concat[bucketize_max_idx] / global_num_blks;
    for (auto i = rowstart + threadIdx.x; i < rowend; i += blockDim.x) {
      uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      uindex_t p = 0;
      index_t first = block_bucketize_pos_offsets[t];
      index_t last = block_bucketize_pos_offsets[t + 1];
      if (blk_size == 0) {
        idx = (idx % global_num_blks) * blk_scalar;
      }

      while (first < last) {
        index_t middle = first + ((last - first) / 2);
        if (static_cast<uindex_t>(block_bucketize_pos_concat[middle]) <= idx) {
          first = ++middle;
        } else {
          last = middle;
        }
      }
      uindex_t lb =
          static_cast<uindex_t>(first - block_bucketize_pos_offsets[t] - 1);
      indices_to_lb[i] = lb;
      p = lb < my_size ? lb : idx % my_size;
      atomicAdd(&new_lengths_data[p * lengths_size + b_t], 1);
    }
  }
}

// Kernel for bucketize offsets, indices, and positional weights, with the Block
// distribution (vs. cyclic, block-cyclic distribution). Used for bucketize
// sparse feature, especially for checkpointing with row-wise partition
// (sparse_feature is partitioned continuously along the sparse dimension into
// my_size blocks)
// This kernel handles pooled sparse features
// WHERE THE ORDER OF INDICES DOES NOT MATTER
template <
    bool has_weight,
    bool bucketize_pos,
    typename offset_t,
    typename index_t,
    typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void _block_bucketize_pooled_sparse_features_cuda_kernel2(
    int lengths_size,
    int32_t B,
    const index_t* __restrict__ block_sizes_data,
    const index_t* __restrict__ total_num_blocks,
    int my_size,
    const offset_t* __restrict__ offsets_data,
    const index_t* __restrict__ indices_data,
    const scalar_t* __restrict__ weights_data,
    offset_t* __restrict__ new_offsets_data,
    index_t* __restrict__ new_indices_data,
    scalar_t* __restrict__ new_weights_data,
    index_t* __restrict__ new_pos_data,
    const offset_t* const __restrict__ length_to_feature_idx,
    const index_t* const __restrict__ block_bucketize_pos_concat,
    const index_t* const __restrict__ block_bucketize_pos_offsets,
    const index_t* const __restrict__ indices_to_lb,
    const bool keep_orig_idx) {
  using uindex_t = std::make_unsigned_t<index_t>;
  const auto bt_start = blockIdx.x * blockDim.y + threadIdx.y;
  const auto stride = gridDim.x * blockDim.y;

  extern __shared__ uint64_t smem[];
  uint64_t* offset_in_different_ranks = &smem[my_size * threadIdx.y];

  for (auto b_t = bt_start; b_t < lengths_size; b_t += stride) {
    const auto t = length_to_feature_idx ? length_to_feature_idx[b_t] : b_t / B;
    const index_t blk_size = block_sizes_data[t];
    const offset_t rowstart = (b_t == 0 ? 0 : offsets_data[b_t - 1]);
    const offset_t rowend = offsets_data[b_t];
    const auto use_block_bucketize_pos =
        (block_bucketize_pos_concat != nullptr);

    /* Re-init the offset array to be 0 for the current iteration */
    syncwarp();
    for (auto i = threadIdx.x; i < my_size; i += blockDim.x) {
      offset_in_different_ranks[i] = 0;
    }
    syncwarp();

    const index_t local_num_blks =
        total_num_blocks == nullptr ? 1 : (total_num_blocks[t] / my_size);
    const index_t global_num_blks =
        total_num_blocks == nullptr ? my_size : total_num_blocks[t];
    const index_t global_idx_size = blk_size * global_num_blks;
    const index_t local_idx_size = blk_size * local_num_blks;
    for (auto i = rowstart + threadIdx.x; i < rowend; i += blockDim.x) {
      // We have use cases using none-hashed raw indices that can be either
      // negative or larger than embedding table hash_size (blk_size *
      // my_size). In cases of none-hashed indices we need to ensure
      // bucketization can distribute them into different ranks and within
      // range of blk_size, we expect the later embedding module to take care
      // of hashing indices calculation.
      const uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      uindex_t p = 0;
      uindex_t new_idx = 0;
      if (!use_block_bucketize_pos) { // uniform bucket sizes
        p = idx < global_idx_size ? idx / local_idx_size
                                  : (idx % global_num_blks) / local_num_blks;
        if (keep_orig_idx) {
          new_idx = idx;
        } else if (idx < global_idx_size) {
          new_idx = idx % local_idx_size;
        } else {
          new_idx = idx / global_num_blks;
        }
      } else { // variable bucket sizes
        uindex_t lb = indices_to_lb[i];
        p = lb < my_size ? lb : idx % my_size;
        if (keep_orig_idx) {
          new_idx = idx;
        } else if (blk_size == 0) {
          new_idx = idx / global_num_blks;
        } else if (lb < my_size) {
          new_idx = idx -
              block_bucketize_pos_concat[lb + block_bucketize_pos_offsets[t]];
        } else {
          new_idx = idx / my_size;
        }
      }
      static_assert(
          sizeof(unsigned long long int) == sizeof(uint64_t),
          "bitwidth change is not allowed");
      const uint64_t pos = atomicAdd(
                               reinterpret_cast<unsigned long long int*>(
                                   &offset_in_different_ranks[p]),
                               1) +
          new_offsets_data[p * lengths_size + b_t];
      new_indices_data[pos] = new_idx;
      if (has_weight) {
        new_weights_data[pos] = weights_data[i];
      }
      if (bucketize_pos) {
        new_pos_data[pos] = i - rowstart;
      }
    }
  }
}

// Kernel for bucketize offsets, indices, and positional weights, with the Block
// distribution (vs. cyclic, block-cyclic distribution). Used for bucketize
// sparse feature, especially for checkpointing with row-wise partition
// (sparse_feature is partitioned continuously along the sparse dimension into
// my_size blocks)
// This kernel handles SEQUENCE sparse features WHERE THE ORDER OF INDICES
// MATTERS
template <
    bool has_weight,
    bool bucketize_pos,
    bool return_bucket_mapping,
    typename offset_t,
    typename index_t,
    typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void _block_bucketize_sequence_sparse_features_cuda_kernel2(
    int lengths_size,
    int32_t B,
    const index_t* __restrict__ block_sizes_data,
    const index_t* __restrict__ total_num_blocks,
    int my_size,
    const offset_t* __restrict__ offsets_data,
    const index_t* __restrict__ indices_data,
    const scalar_t* __restrict__ weights_data,
    offset_t* __restrict__ new_offsets_data,
    index_t* __restrict__ new_indices_data,
    scalar_t* __restrict__ new_weights_data,
    index_t* __restrict__ new_pos_data,
    index_t* const __restrict__ unbucketize_permute_data,
    index_t* const __restrict__ bag_mapping_data,
    const offset_t* const __restrict__ length_to_feature_idx,
    const index_t* const __restrict__ block_bucketize_pos_concat,
    const index_t* const __restrict__ block_bucketize_pos_offsets,
    const index_t* const __restrict__ indices_to_lb,
    const bool keep_orig_idx) {
  using uindex_t = std::make_unsigned_t<index_t>;
  using uoffset_t = std::make_unsigned_t<offset_t>;
  CUDA_KERNEL_LOOP(b_t, lengths_size) {
    const auto t = length_to_feature_idx ? length_to_feature_idx[b_t] : b_t / B;
    index_t blk_size = block_sizes_data[t];
    const index_t local_num_blks =
        total_num_blocks == nullptr ? 1 : (total_num_blocks[t] / my_size);
    const index_t global_num_blks =
        total_num_blocks == nullptr ? my_size : total_num_blocks[t];
    const index_t global_idx_size = blk_size * global_num_blks;
    const index_t local_idx_size = blk_size * local_num_blks;

    offset_t rowstart = (b_t == 0 ? 0 : offsets_data[b_t - 1]);
    offset_t rowend = offsets_data[b_t];
    const auto use_block_bucketize_pos =
        (block_bucketize_pos_concat != nullptr);
    for (index_t i = rowstart; i < rowend; ++i) {
      // We have use cases using none-hashed raw indices that can be either
      // negative or larger than embedding table hash_size (blk_size *
      // my_size). In cases of none-hashed indices we need to ensure
      // bucketization can distribute them into different ranks and within
      // range of blk_size, we expect the later embedding module to take care
      // of hashing indices calculation.
      uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      uindex_t p = 0;
      uindex_t new_idx = 0;
      if (!use_block_bucketize_pos) {
        p = idx < global_idx_size ? idx / local_idx_size
                                  : (idx % global_num_blks) / local_num_blks;
        if (keep_orig_idx) {
          new_idx = idx;
        } else if (idx < global_idx_size) {
          new_idx = idx % local_idx_size;
        } else {
          new_idx = idx / global_num_blks;
        }
      } else {
        uindex_t lb = indices_to_lb[i];
        p = lb < my_size ? lb : idx % my_size;
        if (keep_orig_idx) {
          new_idx = idx;
        } else if (blk_size == 0) {
          new_idx = idx / global_num_blks;
        } else if (lb < my_size) {
          new_idx = idx -
              block_bucketize_pos_concat[lb + block_bucketize_pos_offsets[t]];
        } else {
          new_idx = idx / my_size;
        }
      }
      uoffset_t pos = new_offsets_data[p * lengths_size + b_t];
      new_indices_data[pos] = new_idx;
      new_offsets_data[p * lengths_size + b_t]++;
      unbucketize_permute_data[i] = pos;
      if constexpr (return_bucket_mapping) {
        bag_mapping_data[i] = p;
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

template <typename offset_t, typename index_t>
__global__
__launch_bounds__(kMaxThreads) void _populate_bucketized_permute_cuda_kernel(
    const offset_t* const length_data,
    const offset_t* const offset_data,
    offset_t* const bucketized_offsets_data,
    const index_t* const bucket_mapping_data,
    index_t* const bucketized_permute_data_out,
    int32_t lengths_size) {
  CUDA_KERNEL_LOOP(b_t, lengths_size) {
    const auto length = length_data[b_t];
    const auto offset = offset_data[b_t];
    for (size_t i = 0; i < length; i++) {
      const auto index = offset + i;
      const auto bucket = bucket_mapping_data[index];
      bucketized_permute_data_out[index] =
          bucketized_offsets_data[bucket * lengths_size + b_t]++;
    }
  }
}

#define LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITH_WEIGHT( \
    bucketize_pos, return_bucket_mapping)                                        \
  AT_DISPATCH_INDEX_TYPES(                                                       \
      offsets_contig.scalar_type(),                                              \
      "_block_bucketize_sequence_sparse_features_cuda_kernel2_1",                \
      [&] {                                                                      \
        using offset_t = index_t;                                                \
        AT_DISPATCH_INDEX_TYPES(                                                 \
            indices_contig.scalar_type(),                                        \
            "_block_bucketize_sequence_sparse_features_cuda_kernel2_2",          \
            [&] {                                                                \
              FBGEMM_DISPATCH_FLOAT_ONLY(                                        \
                  weights_value.scalar_type(),                                   \
                  "_block_bucketize_sequence_sparse_features_cuda_kernel2_3",    \
                  [&] {                                                          \
                    _block_bucketize_sequence_sparse_features_cuda_kernel2<      \
                        true,                                                    \
                        bucketize_pos,                                           \
                        return_bucket_mapping,                                   \
                        offset_t,                                                \
                        index_t,                                                 \
                        scalar_t>                                                \
                        <<<num_blocks,                                           \
                           threads_per_block,                                    \
                           0,                                                    \
                           at::cuda::getCurrentCUDAStream()>>>(                  \
                            lengths_size,                                        \
                            B,                                                   \
                            block_sizes.data_ptr<index_t>(),                     \
                            total_num_blocks.has_value()                         \
                                ? total_num_blocks.value().data_ptr<index_t>()   \
                                : static_cast<index_t*>(nullptr),                \
                            my_size,                                             \
                            offsets_contig.data_ptr<offset_t>(),                 \
                            indices_contig.data_ptr<index_t>(),                  \
                            weights_value_contig.data_ptr<scalar_t>(),           \
                            new_offsets.data_ptr<offset_t>(),                    \
                            new_indices.data_ptr<index_t>(),                     \
                            new_weights.data_ptr<scalar_t>(),                    \
                            bucketize_pos ? new_pos.data_ptr<index_t>()          \
                                          : static_cast<index_t*>(nullptr),      \
                            unbucketize_permute.data_ptr<index_t>(),             \
                            (return_bucket_mapping)                              \
                                ? bucket_mapping.data_ptr<index_t>()             \
                                : static_cast<index_t*>(nullptr),                \
                            batch_size_per_feature.has_value()                   \
                                ? length_to_feature_idx.data_ptr<offset_t>()     \
                                : static_cast<offset_t*>(nullptr),               \
                            block_bucketize_pos.has_value()                      \
                                ? block_bucketize_pos_concat                     \
                                      .data_ptr<index_t>()                       \
                                : static_cast<index_t*>(nullptr),                \
                            block_bucketize_pos.has_value()                      \
                                ? block_bucketize_pos_offsets                    \
                                      .data_ptr<index_t>()                       \
                                : static_cast<index_t*>(nullptr),                \
                            block_bucketize_pos.has_value()                      \
                                ? indices_to_lb.data_ptr<index_t>()              \
                                : static_cast<index_t*>(nullptr),                \
                            keep_orig_idx);                                      \
                    C10_CUDA_KERNEL_LAUNCH_CHECK();                              \
                  });                                                            \
            });                                                                  \
      });

#define LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITHOUT_WEIGHT( \
    bucketize_pos, return_bucket_mapping)                                           \
  AT_DISPATCH_INDEX_TYPES(                                                          \
      offsets_contig.scalar_type(),                                                 \
      "_block_bucketize_sequence_sparse_features_cuda_kernel2_1",                   \
      [&] {                                                                         \
        using offset_t = index_t;                                                   \
        AT_DISPATCH_INDEX_TYPES(                                                    \
            indices_contig.scalar_type(),                                           \
            "_block_bucketize_sequence_sparse_features_cuda_kernel_2",              \
            [&] {                                                                   \
              _block_bucketize_sequence_sparse_features_cuda_kernel2<               \
                  false,                                                            \
                  bucketize_pos,                                                    \
                  return_bucket_mapping,                                            \
                  offset_t,                                                         \
                  index_t,                                                          \
                  std::nullptr_t>                                                   \
                  <<<num_blocks,                                                    \
                     threads_per_block,                                             \
                     0,                                                             \
                     at::cuda::getCurrentCUDAStream()>>>(                           \
                      lengths_size,                                                 \
                      B,                                                            \
                      block_sizes.data_ptr<index_t>(),                              \
                      total_num_blocks.has_value()                                  \
                          ? total_num_blocks.value().data_ptr<index_t>()            \
                          : static_cast<index_t*>(nullptr),                         \
                      my_size,                                                      \
                      offsets_contig.data_ptr<offset_t>(),                          \
                      indices_contig.data_ptr<index_t>(),                           \
                      nullptr,                                                      \
                      new_offsets.data_ptr<offset_t>(),                             \
                      new_indices.data_ptr<index_t>(),                              \
                      nullptr,                                                      \
                      bucketize_pos ? new_pos.data_ptr<index_t>()                   \
                                    : static_cast<index_t*>(nullptr),               \
                      unbucketize_permute.data_ptr<index_t>(),                      \
                      return_bucket_mapping                                         \
                          ? bucket_mapping.data_ptr<index_t>()                      \
                          : static_cast<index_t*>(nullptr),                         \
                      batch_size_per_feature.has_value()                            \
                          ? length_to_feature_idx.data_ptr<offset_t>()              \
                          : static_cast<offset_t*>(nullptr),                        \
                      block_bucketize_pos.has_value()                               \
                          ? block_bucketize_pos_concat.data_ptr<index_t>()          \
                          : static_cast<index_t*>(nullptr),                         \
                      block_bucketize_pos.has_value()                               \
                          ? block_bucketize_pos_offsets.data_ptr<index_t>()         \
                          : static_cast<index_t*>(nullptr),                         \
                      block_bucketize_pos.has_value()                               \
                          ? indices_to_lb.data_ptr<index_t>()                       \
                          : static_cast<index_t*>(nullptr),                         \
                      keep_orig_idx);                                               \
              C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            });                                                                     \
      });

#define LAUNCH_BLOCK_BUCKETIZE_POOLED_SPARSE_FEATURES_CUDA_KERNEL_2_WITH_WEIGHT( \
    bucketize_pos, return_new_pos)                                               \
  AT_DISPATCH_INDEX_TYPES(                                                       \
      offsets_contig.scalar_type(),                                              \
      "_block_bucketize_pooled_sparse_features_cuda_kernel2_1",                  \
      [&] {                                                                      \
        using offset_t = index_t;                                                \
        AT_DISPATCH_INDEX_TYPES(                                                 \
            indices_contig.scalar_type(),                                        \
            "_block_bucketize_pooled_sparse_features_cuda_kernel2_2",            \
            [&] {                                                                \
              FBGEMM_DISPATCH_FLOAT_ONLY(                                        \
                  weights_value.scalar_type(),                                   \
                  "_block_bucketize_pooled_sparse_features_cuda_kernel2_3",      \
                  [&] {                                                          \
                    const auto block_bucketize_kernel =                          \
                        _block_bucketize_pooled_sparse_features_cuda_kernel2<    \
                            true,                                                \
                            bucketize_pos,                                       \
                            offset_t,                                            \
                            index_t,                                             \
                            scalar_t>;                                           \
                    if (smem_size > smem_adjust_threshold) {                     \
                      increase_gpu_max_dynamic_shared_memory(                    \
                          block_bucketize_kernel, max_smem);                     \
                    }                                                            \
                    block_bucketize_kernel<<<                                    \
                        grid_dims,                                               \
                        block_dims,                                              \
                        smem_size,                                               \
                        at::cuda::getCurrentCUDAStream()>>>(                     \
                        lengths_size,                                            \
                        B,                                                       \
                        block_sizes.data_ptr<index_t>(),                         \
                        total_num_blocks.has_value()                             \
                            ? total_num_blocks.value().data_ptr<index_t>()       \
                            : static_cast<index_t*>(nullptr),                    \
                        my_size,                                                 \
                        offsets_contig.data_ptr<offset_t>(),                     \
                        indices_contig.data_ptr<index_t>(),                      \
                        weights_value_contig.data_ptr<scalar_t>(),               \
                        new_offsets.data_ptr<offset_t>(),                        \
                        new_indices.data_ptr<index_t>(),                         \
                        new_weights.data_ptr<scalar_t>(),                        \
                        (return_new_pos) ? new_pos.data_ptr<index_t>()           \
                                         : static_cast<index_t*>(nullptr),       \
                        batch_size_per_feature.has_value()                       \
                            ? length_to_feature_idx.data_ptr<offset_t>()         \
                            : static_cast<offset_t*>(nullptr),                   \
                        block_bucketize_pos.has_value()                          \
                            ? block_bucketize_pos_concat.data_ptr<index_t>()     \
                            : static_cast<index_t*>(nullptr),                    \
                        block_bucketize_pos.has_value()                          \
                            ? block_bucketize_pos_offsets.data_ptr<index_t>()    \
                            : static_cast<index_t*>(nullptr),                    \
                        block_bucketize_pos.has_value()                          \
                            ? indices_to_lb.data_ptr<index_t>()                  \
                            : static_cast<index_t*>(nullptr),                    \
                        keep_orig_idx);                                          \
                    C10_CUDA_KERNEL_LAUNCH_CHECK();                              \
                  });                                                            \
            });                                                                  \
      });

#define LAUNCH_BLOCK_BUCKETIZE_POOLED_SPARSE_FEATURES_CUDA_KERNEL_2_WITHOUT_WEIGHT( \
    bucketize_pos, return_new_pos)                                                  \
  AT_DISPATCH_INDEX_TYPES(                                                          \
      offsets_contig.scalar_type(),                                                 \
      "_block_bucketize_pooled_sparse_features_cuda_kernel2_1",                     \
      [&] {                                                                         \
        using offset_t = index_t;                                                   \
        AT_DISPATCH_INDEX_TYPES(                                                    \
            indices_contig.scalar_type(),                                           \
            "_block_bucketize_pooled_sparse_features_cuda_kernel2_2",               \
            [&] {                                                                   \
              const auto block_bucketize_kernel =                                   \
                  _block_bucketize_pooled_sparse_features_cuda_kernel2<             \
                      false,                                                        \
                      bucketize_pos,                                                \
                      offset_t,                                                     \
                      index_t,                                                      \
                      std::nullptr_t>;                                              \
              if (smem_size > smem_adjust_threshold) {                              \
                increase_gpu_max_dynamic_shared_memory(                             \
                    block_bucketize_kernel, max_smem);                              \
              }                                                                     \
              block_bucketize_kernel<<<                                             \
                  grid_dims,                                                        \
                  block_dims,                                                       \
                  smem_size,                                                        \
                  at::cuda::getCurrentCUDAStream()>>>(                              \
                  lengths_size,                                                     \
                  B,                                                                \
                  block_sizes.data_ptr<index_t>(),                                  \
                  total_num_blocks.has_value()                                      \
                      ? total_num_blocks.value().data_ptr<index_t>()                \
                      : static_cast<index_t*>(nullptr),                             \
                  my_size,                                                          \
                  offsets_contig.data_ptr<offset_t>(),                              \
                  indices_contig.data_ptr<index_t>(),                               \
                  nullptr,                                                          \
                  new_offsets.data_ptr<offset_t>(),                                 \
                  new_indices.data_ptr<index_t>(),                                  \
                  nullptr,                                                          \
                  (return_new_pos) ? new_pos.data_ptr<index_t>()                    \
                                   : static_cast<index_t*>(nullptr),                \
                  batch_size_per_feature.has_value()                                \
                      ? length_to_feature_idx.data_ptr<offset_t>()                  \
                      : static_cast<offset_t*>(nullptr),                            \
                  block_bucketize_pos.has_value()                                   \
                      ? block_bucketize_pos_concat.data_ptr<index_t>()              \
                      : static_cast<index_t*>(nullptr),                             \
                  block_bucketize_pos.has_value()                                   \
                      ? block_bucketize_pos_offsets.data_ptr<index_t>()             \
                      : static_cast<index_t*>(nullptr),                             \
                  block_bucketize_pos.has_value()                                   \
                      ? indices_to_lb.data_ptr<index_t>()                           \
                      : static_cast<index_t*>(nullptr),                             \
                  keep_orig_idx);                                                   \
              C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            });                                                                     \
      });

// This function partitions sparse features
// continuously along the sparse dimension into my_size blocks
std::tuple<
    Tensor,
    Tensor,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>>
_block_bucketize_sparse_features_cuda(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const bool sequence,
    const Tensor& block_sizes,
    const std::optional<Tensor>& total_num_blocks,
    const int64_t my_size,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& batch_size_per_feature,
    const int64_t max_B,
    const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    const bool return_bucket_mapping,
    const bool keep_orig_idx) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(lengths, indices);

  CUDA_DEVICE_GUARD(lengths);

  // allocate tensors and buffers
  const auto lengths_size = lengths.numel();
  const auto T = block_sizes.numel();
  const auto B = lengths_size / T;
  const auto new_lengths_size = lengths_size * my_size;
  auto offsets = at::empty({lengths_size}, lengths.options());
  auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
  auto new_offsets = at::empty({new_lengths_size}, lengths.options());
  auto new_indices = at::empty_like(indices);
  auto lengths_contig = lengths.contiguous();
  auto indices_contig = indices.contiguous();
  auto offsets_contig = offsets.contiguous();
  auto batch_sizes_contig =
      batch_size_per_feature.value_or(at::empty({T}, lengths.options()))
          .contiguous();
  auto batch_sizes_offsets_contig =
      at::empty({T}, batch_sizes_contig.options());
  Tensor new_weights;
  Tensor new_pos;
  Tensor unbucketize_permute;
  Tensor bucket_mapping;
  // count nonzeros
  offsets_contig = asynchronous_inclusive_cumsum_gpu(lengths);
  if (batch_size_per_feature.has_value()) {
    TORCH_CHECK(max_B > 0);
    batch_sizes_offsets_contig =
        asynchronous_exclusive_cumsum_gpu(batch_size_per_feature.value());
  }
  auto length_to_feature_idx =
      at::empty({lengths_size}, lengths_contig.options());
  auto indices_to_lb = at::empty_like(indices);
  if (batch_size_per_feature.has_value()) {
    constexpr auto threads_per_block = 256;
    const auto num_blocks =
        cuda_calc_xblock_count(max_B * T, threads_per_block);

    AT_DISPATCH_INDEX_TYPES(
        offsets_contig.scalar_type(),
        "_populate_length_to_feature_id_inplace_kernel",
        [&] {
          using offset_t = index_t;
          _populate_length_to_feature_id_inplace_kernel<<<
              num_blocks,
              threads_per_block,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              max_B,
              T,
              batch_sizes_contig.data_ptr<offset_t>(),
              batch_sizes_offsets_contig.data_ptr<offset_t>(),
              length_to_feature_idx.data_ptr<offset_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  at::Tensor block_bucketize_pos_concat =
      at::empty({1}, indices_contig.options());
  at::Tensor block_bucketize_pos_offsets =
      at::empty({1}, indices_contig.options());

  if (block_bucketize_pos.has_value()) {
    block_bucketize_pos_concat = at::cat(block_bucketize_pos.value(), 0);
    std::vector<int64_t> sizes_;
    sizes_.reserve(block_bucketize_pos.value().size() + 1);
    for (auto const& t : block_bucketize_pos.value()) {
      sizes_.push_back(t.numel());
    }
    sizes_.push_back(0);
    at::Tensor sizes_vec =
        at::tensor(sizes_, at::TensorOptions().dtype(indices_contig.dtype()));
    block_bucketize_pos_offsets = asynchronous_exclusive_cumsum_cpu(
        sizes_vec); // expect sizes_vec to be a small tensor, using cpu instead
                    // of gpu for cumsum
    block_bucketize_pos_offsets = block_bucketize_pos_offsets.to(
        block_bucketize_pos_concat.device(), true);
  }
  static_assert(kMaxThreads % kWarpSize == 0);
  dim3 block_dims(kWarpSize, kMaxThreads / kWarpSize);
  dim3 grid_dims(cuda_calc_xblock_count(lengths_size, block_dims.y));
  const auto smem_adjust_threshold =
      at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
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
                  grid_dims,
                  block_dims,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  lengths_size,
                  B,
                  block_sizes.data_ptr<index_t>(),
                  total_num_blocks.has_value()
                      ? total_num_blocks.value().data_ptr<index_t>()
                      : static_cast<index_t*>(nullptr),
                  my_size,
                  offsets_contig.data_ptr<offset_t>(),
                  indices_contig.data_ptr<index_t>(),
                  new_lengths.data_ptr<offset_t>(),
                  batch_size_per_feature.has_value()
                      ? length_to_feature_idx.data_ptr<offset_t>()
                      : static_cast<offset_t*>(nullptr),
                  block_bucketize_pos.has_value()
                      ? block_bucketize_pos_concat.data_ptr<index_t>()
                      : static_cast<index_t*>(nullptr),
                  block_bucketize_pos.has_value()
                      ? block_bucketize_pos_offsets.data_ptr<index_t>()
                      : static_cast<index_t*>(nullptr),
                  block_bucketize_pos.has_value()
                      ? indices_to_lb.data_ptr<index_t>()
                      : static_cast<index_t*>(nullptr));
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  constexpr auto threads_per_block = 256;
  const auto num_blocks =
      cuda_calc_xblock_count(lengths_size, threads_per_block);
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
      if (return_bucket_mapping) {
        bucket_mapping = at::empty({lengths_sum}, indices.options());
        LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITH_WEIGHT(
            true, true);
      } else {
        LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITH_WEIGHT(
            true, false);
      }
    } else if (weights.has_value()) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      if (return_bucket_mapping) {
        bucket_mapping = at::empty({lengths_sum}, indices.options());
        LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITH_WEIGHT(
            false, true);
      } else {
        LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITH_WEIGHT(
            false, false);
      }
    } else if (bucketize_pos) {
      new_pos = at::empty_like(indices);
      if (return_bucket_mapping) {
        bucket_mapping = at::empty({lengths_sum}, indices.options());
        LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITHOUT_WEIGHT(
            true, true);
      } else {
        LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITHOUT_WEIGHT(
            true, false);
      }
    } else {
      if (return_bucket_mapping) {
        bucket_mapping = at::empty({lengths_sum}, indices.options());
        LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITHOUT_WEIGHT(
            false, true);
      } else {
        LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITHOUT_WEIGHT(
            false, false);
      }
    }

  } else {
    int smem_size = my_size * block_dims.y * sizeof(uint64_t);
    int max_smem = 0;
    adjust_block_bucketize_sparse_features_kernel_launch_configs_based_on_smem(
        &smem_size,
        &block_dims,
        &grid_dims,
        &max_smem,
        lengths_size,
        my_size,
        lengths.get_device());
    if (weights.has_value() & bucketize_pos) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      new_pos = at::empty_like(indices);
      LAUNCH_BLOCK_BUCKETIZE_POOLED_SPARSE_FEATURES_CUDA_KERNEL_2_WITH_WEIGHT(
          true, true);

    } else if (weights.has_value()) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      LAUNCH_BLOCK_BUCKETIZE_POOLED_SPARSE_FEATURES_CUDA_KERNEL_2_WITH_WEIGHT(
          false, false);

    } else if (bucketize_pos) {
      new_pos = at::empty_like(indices);
      LAUNCH_BLOCK_BUCKETIZE_POOLED_SPARSE_FEATURES_CUDA_KERNEL_2_WITHOUT_WEIGHT(
          true, true);

    } else {
      LAUNCH_BLOCK_BUCKETIZE_POOLED_SPARSE_FEATURES_CUDA_KERNEL_2_WITHOUT_WEIGHT(
          false, false);
    }
  }

  return {
      new_lengths,
      new_indices,
      new_weights,
      new_pos,
      unbucketize_permute,
      bucket_mapping};
}

#undef LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITHOUT_WEIGHT
#undef LAUNCH_BLOCK_BUCKETIZE_SEQUENCE_SPARSE_FEATURES_CUDA_KERNEL_WITH_WEIGHT
#undef LAUNCH_BLOCK_BUCKETIZE_POOLED_SPARSE_FEATURES_CUDA_KERNEL_2_WITH_WEIGHT
#undef LAUNCH_BLOCK_BUCKETIZE_POOLED_SPARSE_FEATURES_CUDA_KERNEL_2_WITHOUT_WEIGHT

// This function partitions sparse features
// continuously along the sparse dimension into my_size
// blocks
DLL_PUBLIC std::tuple<
    Tensor,
    Tensor,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>>
block_bucketize_sparse_features_cuda(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const bool sequence,
    const Tensor& block_sizes,
    const int64_t my_size,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& batch_size_per_feature,
    const int64_t max_B,
    const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    const bool keep_orig_idx,
    const std::optional<Tensor>& total_num_blocks) {
  Tensor new_lengths;
  Tensor new_indices;
  std::optional<Tensor> new_weights;
  std::optional<Tensor> new_pos;
  std::optional<Tensor> unbucketize_permute;
  std::tie(
      new_lengths,
      new_indices,
      new_weights,
      new_pos,
      unbucketize_permute,
      std::ignore) =
      _block_bucketize_sparse_features_cuda(
          lengths,
          indices,
          bucketize_pos,
          sequence,
          block_sizes,
          total_num_blocks,
          my_size,
          weights,
          batch_size_per_feature,
          max_B,
          block_bucketize_pos,
          false,
          keep_orig_idx);
  return {new_lengths, new_indices, new_weights, new_pos, unbucketize_permute};
}

// This function partitions sparse features
// continuously along the sparse dimension into my_size blocks
DLL_PUBLIC std::tuple<
    Tensor,
    Tensor,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>>
block_bucketize_sparse_features_inference_cuda(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const bool sequence,
    const Tensor& block_sizes,
    const int64_t my_size,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& batch_size_per_feature,
    const int64_t max_B,
    const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    const bool return_bucket_mapping,
    const bool keep_orig_idx,
    const std::optional<Tensor>& total_num_blocks) {
  return _block_bucketize_sparse_features_cuda(
      lengths,
      indices,
      bucketize_pos,
      sequence,
      block_sizes,
      total_num_blocks,
      my_size,
      weights,
      batch_size_per_feature,
      max_B,
      block_bucketize_pos,
      return_bucket_mapping,
      keep_orig_idx);
}

DLL_PUBLIC Tensor populate_bucketized_permute_cuda(
    const Tensor& lengths,
    const Tensor& bucketized_lengths,
    const Tensor& bucket_mapping) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      lengths, bucketized_lengths, bucket_mapping);
  CUDA_DEVICE_GUARD(lengths);
  const auto lengths_contig = lengths.expect_contiguous();
  const auto bucketized_lengths_contig = bucketized_lengths.expect_contiguous();
  const auto bucket_mapping_contig = bucket_mapping.expect_contiguous();
  Tensor bucketized_permute = at::empty_like(*bucket_mapping_contig);
  const auto offsets = asynchronous_complete_cumsum_gpu(*lengths_contig);
  const auto bucketized_offsets =
      asynchronous_complete_cumsum_gpu(*bucketized_lengths_contig);
  constexpr auto threads_per_block = 256;
  const auto lengths_size = lengths.numel();
  const auto num_blocks =
      cuda_calc_xblock_count(lengths_size, threads_per_block);
  AT_DISPATCH_INDEX_TYPES(
      lengths_contig->scalar_type(),
      "_populate_bucketized_permute_cuda_kernel1",
      [&] {
        using offset_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            bucket_mapping_contig->scalar_type(),
            "_populate_bucketized_permute_cuda_kernel2",
            [&] {
              _populate_bucketized_permute_cuda_kernel<<<
                  num_blocks,
                  threads_per_block,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  lengths_contig->data_ptr<offset_t>(),
                  offsets.data_ptr<offset_t>(),
                  bucketized_offsets.data_ptr<offset_t>(),
                  bucket_mapping_contig->data_ptr<index_t>(),
                  bucketized_permute.data_ptr<index_t>(),
                  lengths.numel());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return bucketized_permute;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "block_bucketize_sparse_features",
    fbgemm_gpu::block_bucketize_sparse_features_cuda);
FBGEMM_OP_DISPATCH(
    CUDA,
    "block_bucketize_sparse_features_inference",
    fbgemm_gpu::block_bucketize_sparse_features_inference_cuda);
FBGEMM_OP_DISPATCH(
    CUDA,
    "populate_bucketized_permute",
    fbgemm_gpu::populate_bucketized_permute_cuda);
