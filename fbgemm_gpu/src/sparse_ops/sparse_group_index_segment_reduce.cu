/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef USE_ROCM

#include <algorithm>
#include <limits>

#include <ATen/AccumulateType.h>

#include "common.cuh"

namespace fbgemm_gpu {
namespace {

constexpr int64_t kSegmentReduceChunkSize = 512;

template <typename value_t>
__device__ __forceinline__ int64_t
lower_bound(const value_t* values, const int64_t size, const int64_t target) {
  int64_t first = 0;
  int64_t count = size;
  while (count > 0) {
    const int64_t step = count / 2;
    const int64_t current = first + step;
    if (static_cast<int64_t>(values[current]) < target) {
      first = current + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

__device__ __forceinline__ int64_t find_segment(
    const int64_t* offsets,
    const int64_t num_segments,
    const int64_t position) {
  int64_t first = 0;
  int64_t count = num_segments + 1;
  while (count > 0) {
    const int64_t step = count / 2;
    const int64_t current = first + step;
    if (offsets[current] <= position) {
      first = current + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first - 1;
}

template <typename index_t>
__global__ void compute_segment_metadata_kernel(
    const int64_t* sorted_indices_ptrs,
    const int64_t* row_offsets_group,
    int64_t* row_starts,
    int64_t* row_lengths,
    int64_t* chunks_per_row,
    const int64_t num_work_rows,
    const int64_t total_num_rows,
    const int64_t group_size) {
  for (int64_t global_row =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       global_row < total_num_rows;
       global_row += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t group =
        find_segment(row_offsets_group, group_size, global_row);
    const int64_t local_row = global_row - row_offsets_group[group];
    const auto* sorted_indices =
        reinterpret_cast<const index_t*>(sorted_indices_ptrs[group]);
    const int64_t begin = lower_bound(sorted_indices, num_work_rows, local_row);
    const int64_t end =
        lower_bound(sorted_indices, num_work_rows, local_row + 1);
    const int64_t length = end - begin;

    row_starts[global_row] = begin;
    row_lengths[global_row] = length;
    chunks_per_row[global_row] =
        (length + kSegmentReduceChunkSize - 1) / kSegmentReduceChunkSize;
  }
}

template <typename scalar_t, typename acc_t>
__global__ void group_index_add_segment_reduce_kernel(
    const int64_t* grad_output_ptrs,
    const int64_t* grad_input_ptrs,
    const int64_t* reverse_indices_ptrs,
    const int64_t* row_offsets_group,
    const int32_t* num_cols_group,
    const int64_t* row_starts,
    const int64_t* row_lengths,
    const int64_t* chunk_offsets,
    const int64_t total_num_rows,
    const int64_t group_size,
    const int64_t max_chunks) {
  const int64_t actual_chunks = chunk_offsets[total_num_rows];
  for (int64_t chunk = blockIdx.x; chunk < max_chunks; chunk += gridDim.x) {
    if (chunk >= actual_chunks) {
      break;
    }

    const int64_t global_row =
        find_segment(chunk_offsets, total_num_rows, chunk);
    const int64_t group =
        find_segment(row_offsets_group, group_size, global_row);
    const int64_t local_row = global_row - row_offsets_group[group];
    const int64_t row_chunk = chunk - chunk_offsets[global_row];
    const int64_t begin =
        row_starts[global_row] + row_chunk * kSegmentReduceChunkSize;
    const int64_t row_end = row_starts[global_row] + row_lengths[global_row];
    const int64_t chunk_end = begin + kSegmentReduceChunkSize;
    const int64_t end = chunk_end < row_end ? chunk_end : row_end;
    const bool single_chunk =
        chunk_offsets[global_row + 1] - chunk_offsets[global_row] == 1;
    const int64_t num_cols = num_cols_group[group];
    const auto* grad_output =
        reinterpret_cast<const scalar_t*>(grad_output_ptrs[group]);
    auto* grad_input = reinterpret_cast<acc_t*>(grad_input_ptrs[group]);
    const auto* reverse_indices =
        reinterpret_cast<const int64_t*>(reverse_indices_ptrs[group]);

    for (int64_t col = threadIdx.x; col < num_cols; col += blockDim.x) {
      acc_t sum = 0;
      for (int64_t sorted_pos = begin; sorted_pos < end; ++sorted_pos) {
        const int64_t source_row = reverse_indices[sorted_pos];
        sum += static_cast<acc_t>(grad_output[source_row * num_cols + col]);
      }

      auto* output = grad_input + local_row * num_cols + col;
      if (single_chunk) {
        *output = sum;
      } else {
        gpuAtomicAddNoReturn(output, sum);
      }
    }
  }
}

} // namespace

DLL_PUBLIC void group_index_add_2d_segment_cuda(
    const int64_t* grad_output_ptrs,
    const int64_t* grad_input_ptrs,
    const int64_t* sorted_indices_ptrs,
    const int64_t* reverse_indices_ptrs,
    const int64_t* row_offsets_group,
    const int32_t* num_cols_group,
    const c10::ScalarType& input_scalar_type,
    const c10::ScalarType& indices_scalar_type,
    const c10::DeviceIndex& device,
    const int64_t num_work_rows,
    const int64_t total_num_rows,
    const int64_t group_size) {
  if (num_work_rows == 0 || total_num_rows == 0 || group_size == 0) {
    return;
  }

  TORCH_CHECK(
      total_num_rows < std::numeric_limits<int32_t>::max(),
      "group_index_add_2d_segment_cuda supports fewer than INT_MAX output rows");
  TORCH_CHECK(
      num_work_rows <= std::numeric_limits<int32_t>::max(),
      "group_index_add_2d_segment_cuda supports at most INT_MAX input rows");
  TORCH_CHECK(
      group_size <= std::numeric_limits<int64_t>::max() / num_work_rows,
      "group_index_add_2d_segment_cuda work size overflow");

  at::cuda::OptionalCUDAGuard device_guard(device);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto metadata_options =
      at::TensorOptions().device(at::kCUDA, device).dtype(at::kLong);
  auto row_starts = at::empty({total_num_rows}, metadata_options);
  auto row_lengths = at::empty({total_num_rows}, metadata_options);
  auto chunks_per_row = at::empty({total_num_rows}, metadata_options);

  constexpr uint32_t metadata_threads = 256;
  const uint32_t metadata_blocks =
      utils::cuda::cap_grid_dim_x(total_num_rows, metadata_threads, stream);
  AT_DISPATCH_INDEX_TYPES(
      indices_scalar_type, "group_index_add_2d_segment_metadata", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (compute_segment_metadata_kernel<index_t>),
            metadata_blocks,
            metadata_threads,
            0,
            stream,
            sorted_indices_ptrs,
            row_offsets_group,
            row_starts.data_ptr<int64_t>(),
            row_lengths.data_ptr<int64_t>(),
            chunks_per_row.data_ptr<int64_t>(),
            num_work_rows,
            total_num_rows,
            group_size);
      });

  const auto chunk_offsets = asynchronous_complete_cumsum_gpu(chunks_per_row);
  const int64_t total_num_indices = group_size * num_work_rows;
  const int64_t max_chunks = total_num_rows +
      (total_num_indices + kSegmentReduceChunkSize - 1) /
          kSegmentReduceChunkSize;
  const int64_t max_blocks =
      static_cast<int64_t>(
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount) *
      8;
  const uint32_t reduce_blocks =
      static_cast<uint32_t>(std::min(max_chunks, max_blocks));
  constexpr uint32_t reduce_threads = 256;

  FBGEMM_DISPATCH_FLOATING_TYPES(
      input_scalar_type, "group_index_add_2d_segment_reduce", [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        FBGEMM_LAUNCH_KERNEL(
            (group_index_add_segment_reduce_kernel<scalar_t, acc_t>),
            reduce_blocks,
            reduce_threads,
            0,
            stream,
            grad_output_ptrs,
            grad_input_ptrs,
            reverse_indices_ptrs,
            row_offsets_group,
            num_cols_group,
            row_starts.const_data_ptr<int64_t>(),
            row_lengths.const_data_ptr<int64_t>(),
            chunk_offsets.const_data_ptr<int64_t>(),
            total_num_rows,
            group_size,
            max_chunks);
      });
}

} // namespace fbgemm_gpu

#endif // USE_ROCM
