/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

// clang-format off
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Linearzie the index with the cumsum of hash size so that linearized indices
// can be sorted together.
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void linearize_index_wo_infos_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices,
    FixedDivisor fd) {
  const auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b;
  int32_t t;
  const auto total_B = offsets.size(0) - 1;
  const auto valid = b_t < total_B;

  fd.DivMod(b_t, &t, &b);

  const auto hash_offset = valid ? hash_size_cumsum[t] : -1;
  const auto indices_start = valid ? offsets[b_t] : -1;
  const int32_t L = valid ? offsets[b_t + 1] - indices_start : 0;
  const auto lane_id = threadIdx.x % fbgemm_gpu::kWarpSize;

  for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
    const auto indices_start_warp = fbgemm_gpu::shfl_sync(indices_start, j);
    const auto L_warp = fbgemm_gpu::shfl_sync(L, j);
    const auto hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
    for (int32_t i = lane_id; i < L_warp; i += fbgemm_gpu::kWarpSize) {
      const auto idx = __ldg(&indices[indices_start_warp + i]);
      linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
    }
  }
}

// Delinearize the unique indices from the reverse index info and the original
// indices. For each element in the input indices, the value should equal to
// the element from the unique indices according to the reverse index info.
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void delinearize_unique_index_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reverse_index,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices) {
  const auto total_indices = indices.size(0);
  const auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_t < total_indices) {
    const auto original_index = indices[b_t];
    const auto pos = reverse_index[b_t];
    unique_indices[pos] = original_index;
  }
}

// Compute the lengths for each feature in the unique indices. The range of
// indices for each feature equals to the difference between the max and min
// values in the reverse index array.
template <typename index_t, auto max_value, auto min_value>
__global__ __launch_bounds__(kMaxThreads) void unique_indices_length_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_offsets,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reverse_index,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> lengths) {
  typedef cub::BlockReduce<index_t, kMaxThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_max;
  __shared__ typename BlockReduce::TempStorage temp_storage_min;
  __shared__ index_t block_results[2];

  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;
  const auto num_blocks = gridDim.x;
  const int32_t batch_size = (offsets.size(0) - 1) / num_blocks;

  const auto offset_begin = hash_size_offsets[bid] * batch_size;
  const auto offset_end = hash_size_offsets[bid + 1] * batch_size;
  const auto num_lengths = (offset_end - offset_begin);

  const auto reverse_index_begin = offsets[offset_begin];
  const auto reverse_index_end = offsets[offset_end];

  if (reverse_index_begin == reverse_index_end) {
    return;
  }

  index_t t_max = min_value;
  index_t t_min = max_value;
  for (index_t i = (reverse_index_begin + tid); i < reverse_index_end;
       i += kMaxThreads) {
    const index_t value = reverse_index[i];
    t_max = (value > t_max) ? value : t_max;
    t_min = (value < t_min) ? value : t_min;
  }

  index_t block_max =
      BlockReduce(temp_storage_max).Reduce(t_max, Max<index_t>());
  index_t block_min =
      BlockReduce(temp_storage_min).Reduce(t_min, Min<index_t>());
  if (tid == 0) {
    block_results[0] = block_max;
    block_results[1] = block_min;
  }
  __syncthreads();

  t_max = block_results[0];
  t_min = block_results[1];
  const index_t total_length = (t_max - t_min) + 1;
  const index_t div_length = total_length / num_lengths;
  const index_t r_length = total_length % num_lengths;
  for (int32_t i = tid; i < num_lengths; i += kMaxThreads) {
    index_t seg_length = (i < r_length) ? (div_length + 1) : div_length;
    lengths[offset_begin + i] = seg_length;
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor> jagged_unique_indices_cuda(
    const Tensor& hash_size_cumsum,
    const Tensor& hash_size_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  const auto total_B = offsets.size(0) - 1;
  const auto T = hash_size_cumsum.size(0) - 1;

  Tensor linear_indices = at::empty_like(indices);

  using at::RestrictPtrTraits;

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "linearize_index", ([&] {
                            FBGEMM_LAUNCH_KERNEL(
                                (linearize_index_wo_infos_kernel<index_t>),
                                div_round_up(total_B, kMaxThreads),
                                kMaxThreads,
                                0,
                                at::cuda::getCurrentCUDAStream(),
                                PTA_B(hash_size_cumsum, index_t, 1, 32),
                                PTA_B(indices, index_t, 1, 32),
                                PTA_B(offsets, index_t, 1, 32),
                                PTA_B(linear_indices, index_t, 1, 32),
                                FixedDivisor(total_B / T));
                          }));

  Tensor linear_unique_indices;
  Tensor reverse_index;

  std::tie(linear_unique_indices, reverse_index) =
      at::_unique(linear_indices, true, true);

  const auto total_indices = indices.size(0);
  Tensor unique_indices = at::empty_like(linear_unique_indices);

  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "delinearize_unique_index", ([&] {
        FBGEMM_LAUNCH_KERNEL(
            (delinearize_unique_index_kernel<index_t>),
            div_round_up(total_indices + 1, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(indices, index_t, 1, 32),
            PTA_B(reverse_index, index_t, 1, 32),
            PTA_B(unique_indices, index_t, 1, 32));
      }));

  Tensor output_lengths = at::zeros({total_B}, offsets.options());
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "unique_indices_length", ([&] {
                            FBGEMM_LAUNCH_KERNEL(
                                (unique_indices_length_kernel<
                                    index_t,
                                    std::numeric_limits<index_t>::max(),
                                    std::numeric_limits<index_t>::min()>),
                                T,
                                kMaxThreads,
                                0,
                                at::cuda::getCurrentCUDAStream(),
                                PTA_B(hash_size_offsets, index_t, 1, 32),
                                PTA_B(reverse_index, index_t, 1, 32),
                                PTA_B(offsets, index_t, 1, 32),
                                PTA_B(output_lengths, index_t, 1, 32));
                          }));

  Tensor output_offsets;
  output_offsets = asynchronous_complete_cumsum_gpu(output_lengths);
  return {output_lengths, output_offsets, unique_indices, reverse_index};
}

// Compute hash size for each key using the max value of indices per key.
template <typename index_t, auto min_value>
__global__ __launch_bounds__(kMaxThreads) void compute_hash_size_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const int64_t batch_size,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> hash_size) {
  typedef cub::BlockReduce<index_t, kMaxThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_max;

  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;

  const auto offset_begin = bid * batch_size;
  const auto offset_end = (bid + 1) * batch_size;
  const auto index_begin = offsets[offset_begin];
  const auto index_end = offsets[offset_end];

  if (index_begin == index_end) {
    return;
  }

  index_t t_max = min_value;
  for (index_t i = (index_begin + tid); i < index_end; i += kMaxThreads) {
    const index_t value = indices[i];
    t_max = (value > t_max) ? value : t_max;
  }

  index_t block_max =
      BlockReduce(temp_storage_max).Reduce(t_max, Max<index_t>());
  if (tid == 0) {
    hash_size[bid] = block_max + 1;
  }
}

std::tuple<Tensor, Tensor> jagged_hash_size_cumsum_cuda(
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t batch_size) {
  const auto T = (offsets.size(0) - 1) / batch_size;
  Tensor hash_size = at::zeros({T}, offsets.options());

  using at::RestrictPtrTraits;

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "compute_hash_size", ([&] {
                            FBGEMM_LAUNCH_KERNEL(
                                (compute_hash_size_kernel<
                                    index_t,
                                    std::numeric_limits<index_t>::min()>),
                                T,
                                kMaxThreads,
                                0,
                                at::cuda::getCurrentCUDAStream(),
                                PTA_B(offsets, index_t, 1, 32),
                                PTA_B(indices, index_t, 1, 32),
                                batch_size,
                                PTA_B(hash_size, index_t, 1, 32));
                          }));

  Tensor hash_size_cumsum;
  hash_size_cumsum = asynchronous_complete_cumsum_gpu(hash_size);

  Tensor hash_size_lengths = at::ones_like(hash_size);
  Tensor hash_size_offsets;
  hash_size_offsets = asynchronous_complete_cumsum_gpu(hash_size_lengths);
  return {hash_size_cumsum, hash_size_offsets};
}

// Optimized atomic kernel with better memory access patterns
template <typename index_t, typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void accumulate_weights_and_counts_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reverse_indices,
    const pta::PackedTensorAccessor32<scalar_t, 1, at::RestrictPtrTraits>
        weights,
    pta::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits>
        accumulated_data) {
  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;
  const auto total_elements = weights.size(0);

  // Process elements with stride of kMaxThreads for better memory
  // bandwidth utilization
  for (int i = bid * kMaxThreads + tid; i < total_elements;
       i += kMaxThreads * gridDim.x) {
    const index_t unique_idx = reverse_indices[i];
    const scalar_t weight_val = weights[i];

    // Use fast atomic operations
    atomicAdd(&accumulated_data[unique_idx][0], static_cast<float>(weight_val));
    atomicAdd(&accumulated_data[unique_idx][1], 1.0f);
  }
}

// Optimized function to accumulate weights and counts using atomic operations
// Simplified approach that focuses on memory bandwidth and atomic efficiency
Tensor jagged_acc_weights_and_counts_cu(
    const Tensor& weights,
    const Tensor& reverse_indices,
    int64_t num_unique_indices) {
  // Create 2D tensor: [num_unique_indices, 2] where dim 0 = accumulated
  // weights, dim 1 = counts
  Tensor accumulated_data = at::zeros(
      {num_unique_indices, 2},
      at::TensorOptions().dtype(at::kFloat).device(weights.device()));

  const auto total_elements = weights.size(0);

  // Use optimized atomic approach - simpler and often faster than segmented
  // reduction for this use case due to reduced overhead
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.scalar_type(), "accumulate_weights_and_counts", ([&] {
        AT_DISPATCH_INDEX_TYPES(
            reverse_indices.scalar_type(),
            "accumulate_weights_and_counts_idx",
            ([&] {
              // Calculate number of blocks based on total elements
              const int num_blocks = div_round_up(total_elements, kMaxThreads);

              FBGEMM_LAUNCH_KERNEL(
                  (accumulate_weights_and_counts_kernel<index_t, scalar_t>),
                  num_blocks,
                  kMaxThreads,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(reverse_indices, index_t, 1, 32),
                  PTA_B(weights, scalar_t, 1, 32),
                  PTA_B(accumulated_data, float, 2, 32));
            }));
      }));

  return accumulated_data;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_unique_indices",
    fbgemm_gpu::jagged_unique_indices_cuda);

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_hash_size_cumsum",
    fbgemm_gpu::jagged_hash_size_cumsum_cuda);

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_acc_weights_and_counts",
    fbgemm_gpu::jagged_acc_weights_and_counts_cu);
