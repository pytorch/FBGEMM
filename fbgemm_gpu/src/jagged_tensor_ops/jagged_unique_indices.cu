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

// Linearzie the index with the cumsum of hash size so that linearized indices
// can be sorted together.
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void linearize_index_wo_infos_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices,
    FixedDivisor fd) {
  const int32_t T = hash_size_cumsum.size(0) - 1;
  const auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b;
  int32_t t;
  const auto total_B = offsets.size(0) - 1;
  const auto valid = b_t < total_B;

  fd.DivMod(b_t, &t, &b);

  const auto hash_offset = valid ? hash_size_cumsum[t] : -1;
  const auto indices_start = valid ? offsets[b_t] : -1;
  const int32_t L = valid ? offsets[b_t + 1] - indices_start : 0;
  const int32_t lane_id = threadIdx.x % fbgemm_gpu::kWarpSize;

  for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
    const auto indices_start_warp = fbgemm_gpu::shfl_sync(indices_start, j);
    const auto t_warp = fbgemm_gpu::shfl_sync(t, j);
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
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reverse_index,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
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
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reverse_index,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> lengths) {
  typedef cub::BlockReduce<index_t, kMaxThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_max;
  __shared__ typename BlockReduce::TempStorage temp_storage_min;
  __shared__ index_t block_results[2];

  const int32_t tid = threadIdx.x;
  const int32_t bid = blockIdx.x;
  const int32_t num_blocks = gridDim.x;
  const int32_t batch_size = (offsets.size(0) - 1) / num_blocks;

  const auto offset_begin = bid * batch_size;
  const auto offset_end = (bid + 1) * batch_size;

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

  index_t block_max = BlockReduce(temp_storage_max).Reduce(t_max, cub::Max());
  index_t block_min = BlockReduce(temp_storage_min).Reduce(t_min, cub::Min());
  if (tid == 0) {
    block_results[0] = block_max;
    block_results[1] = block_min;
  }
  __syncthreads();

  t_max = block_results[0];
  t_min = block_results[1];
  const index_t total_length = (t_max - t_min) + 1;
  const index_t div_length = total_length / batch_size;
  const index_t r_length = total_length % batch_size;
  for (int32_t i = tid; i < batch_size; i += kMaxThreads) {
    index_t seg_length = (i < r_length) ? (div_length + 1) : div_length;
    lengths[offset_begin + i] = seg_length;
  }
}

std::tuple<Tensor, Tensor, Tensor> jagged_unique_indices_cuda(
    const Tensor& hash_size_cumsum,
    const Tensor& offsets,
    const Tensor& indices) {
  const auto total_B = offsets.size(0) - 1;
  const auto T = hash_size_cumsum.size(0) - 1;

  Tensor linear_indices = at::empty_like(indices);

  using at::RestrictPtrTraits;

  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "linearize_index", ([&] {
        const auto linearize_index_kernel_ =
            linearize_index_wo_infos_kernel<index_t>;
        linearize_index_kernel_<<<
            div_round_up(total_B, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            hash_size_cumsum.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            linear_indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            FixedDivisor(total_B / T));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));

  Tensor linear_unique_indices;
  Tensor reverse_index;

  std::tie(linear_unique_indices, reverse_index) =
      at::_unique(linear_indices, true, true);

  const auto total_indices = indices.size(0);
  Tensor unique_indices = at::empty_like(linear_unique_indices);

  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "delinearize_unique_index", ([&] {
        const auto delinearize_unique_index_kernel_ =
            delinearize_unique_index_kernel<index_t>;
        delinearize_unique_index_kernel_<<<
            div_round_up(total_indices, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            reverse_index.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            unique_indices.packed_accessor32<index_t, 1, RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));

  Tensor lengths = at::zeros({total_B}, offsets.options());
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "unique_indices_length", ([&] {
        const auto unique_indices_length_kernel_ = unique_indices_length_kernel<
            index_t,
            std::numeric_limits<index_t>::max(),
            std::numeric_limits<index_t>::min()>;
        unique_indices_length_kernel_<<<
            T,
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            reverse_index.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            lengths.packed_accessor32<index_t, 1, RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));

  Tensor output_offsets;
  output_offsets = asynchronous_complete_cumsum_gpu(lengths);
  return {output_offsets, unique_indices, reverse_index};
}
} // namespace fbgemm_gpu

JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_unique_indices",
    fbgemm_gpu::jagged_unique_indices_cuda);
