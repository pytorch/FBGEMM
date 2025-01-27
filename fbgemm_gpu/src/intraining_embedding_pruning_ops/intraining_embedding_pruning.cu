/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <time.h>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>

#include "fbgemm_gpu/intraining_embedding_pruning.h"
#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;
using namespace torch::indexing;

namespace fbgemm_gpu {

__global__ void init_address_lookup_kernel(
    const int32_t blocks_per_table,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        address_lookups,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        emb_sizes) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t t_i = blockIdx.x / blocks_per_table;
  int32_t threads_per_table = blocks_per_table * blockDim.x;
  int32_t idx_table = idx % threads_per_table;

  int64_t rows = buffer_offsets[t_i + 1] - buffer_offsets[t_i];
  int64_t rows_per_thread = div_round_up(rows, threads_per_table);
  int64_t start = idx_table * rows_per_thread;
  int64_t end = min(start + rows_per_thread, rows);

  if (start >= rows) {
    return;
  }

  int64_t buffer_offset = buffer_offsets[t_i];
  int64_t* address_lookup = &address_lookups[buffer_offset];
  int64_t emb_size = emb_sizes[t_i];

  for (int64_t idx_row = start; idx_row < end; idx_row++) {
    if (idx_row < emb_size) {
      address_lookup[idx_row] = idx_row;
    } else {
      address_lookup[idx_row] = 0;
    }
  }
}

// Update utility of not accessed rows
__global__ void decay_row_utils(
    float decay_factor,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> row_utils) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t row_utils_size = row_utils.size(0);
  if (idx >= row_utils_size) {
    return;
  }

  // decay row_utils
  row_utils[idx] *= decay_factor;
  CUDA_KERNEL_ASSERT(row_utils[idx] >= 0);
}

__global__ void get_util_samples(
    int64_t iter,
    const int32_t blocks_per_table,
    const int32_t rows_per_sample,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits>
        sampled_utilities,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        sampling_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets,
    const at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits>
        row_utils) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t t_i = blockIdx.x / blocks_per_table;
  int32_t num_tables = buffer_offsets.size(0) - 1;

  if (t_i >= num_tables) {
    return;
  }

  int32_t threads_per_table = blocks_per_table * blockDim.x;
  int32_t idx_table = idx % threads_per_table;

  int64_t rows = buffer_offsets[t_i + 1] - buffer_offsets[t_i];
  int64_t num_samples = sampling_offsets[t_i + 1] - sampling_offsets[t_i];
  int64_t num_samples_per_thread = div_round_up(num_samples, threads_per_table);
  int64_t start = idx_table * num_samples_per_thread;
  int64_t end = min(start + num_samples_per_thread, num_samples);

  if (start >= num_samples) {
    return;
  }

  // Add randomness in the first sampling interval as the bias term
  int64_t offset = iter % rows_per_sample;

  for (int32_t i = start; i < end; i++) {
    int64_t sampled_utilities_addr = sampling_offsets[t_i] + i;
    int64_t row_util_addr =
        buffer_offsets[t_i] + min(rows - 1, offset + i * rows_per_sample);
    sampled_utilities[sampled_utilities_addr] = row_utils[row_util_addr];
    CUDA_KERNEL_ASSERT(sampled_utilities[sampled_utilities_addr] >= 0);
  }
}

__global__ void get_util_thresholds(
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> util_thresholds,
    const at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits>
        sampled_utilities_sorted,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        sampling_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        emb_sizes) {
  int64_t t_i = blockIdx.x * blockDim.x + threadIdx.x;

  int64_t segment_start = sampling_offsets[t_i];
  int64_t num_samples = sampling_offsets[t_i + 1] - sampling_offsets[t_i];

  int64_t buffer_rows = buffer_offsets[t_i + 1] - buffer_offsets[t_i];
  float pruning_ratio = 1.0 -
      static_cast<float>(emb_sizes[t_i]) / static_cast<float>(buffer_rows);
  int64_t threshold_index =
      min(static_cast<int64_t>(floor(pruning_ratio * num_samples)),
          num_samples - 1);
  util_thresholds[t_i] =
      sampled_utilities_sorted[segment_start + threshold_index];
}

__global__ void prune_indices_per_table(
    const int32_t blocks_per_table,
    const at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits>
        util_thresholds,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets,
    const at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> row_utils,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        address_lookups) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t t_i = blockIdx.x / blocks_per_table;
  int32_t threads_per_table = blocks_per_table * blockDim.x;
  int32_t idx_table = idx % threads_per_table;

  int64_t rows = buffer_offsets[t_i + 1] - buffer_offsets[t_i];
  int64_t rows_per_thread = div_round_up(rows, threads_per_table);
  int64_t start = idx_table * rows_per_thread;
  int64_t end = min(start + rows_per_thread, rows);

  if (start >= rows) {
    return;
  }

  // Load row utility and address lookup
  int64_t buffer_offset = buffer_offsets[t_i];
  const float* row_util = &row_utils[buffer_offset];
  int64_t* address_lookup = &address_lookups[buffer_offset];

  // All pruned rows will be directed to the first row of embedding storage.
  // Weights of the first row are all zeros throughout the training.
  int64_t PAST_PRUNED_ROW_PLACEHOLDER = 0;
  // This placeholder will never be accessed.
  int64_t INSERTED_ROW_PLACEHOLDER = rows;

  for (int64_t idx_row = start; idx_row < end; idx_row++) {
    if (row_util[idx_row] > util_thresholds[t_i]) {
      // If the row was not pruned, then skip;
      // Otherwise, reset its physical address to placeholder
      if (address_lookup[idx_row] == PAST_PRUNED_ROW_PLACEHOLDER) {
        address_lookup[idx_row] = INSERTED_ROW_PLACEHOLDER;
      }
    } else {
      // If the row was pruned in last interval,
      // then skip as no address can be reclaimed;
      // Otherwise, flip its embedding address to negative value
      // So that we can memorize the new reclaimed physical address
      if (address_lookup[idx_row] != PAST_PRUNED_ROW_PLACEHOLDER) {
        address_lookup[idx_row] = -address_lookup[idx_row];
      }
    }
  }
}

__global__ void get_pruning_lengths(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        address_lookups,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_row_lengths,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        inserted_row_lengths,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruning_lengths) {
  int32_t t_i = blockIdx.x;
  int64_t rows = buffer_offsets[t_i + 1] - buffer_offsets[t_i];
  int64_t segment_length = div_round_up(rows, blockDim.x);
  int64_t segment_start = threadIdx.x * segment_length;
  int64_t segment_end = min(segment_start + segment_length, rows);

  if (segment_start >= rows) {
    return;
  }

  int64_t buffer_offset = buffer_offsets[t_i];
  const int64_t* address_lookup = &address_lookups[buffer_offset];
  int64_t* inserted_row_length = &inserted_row_lengths[t_i * kMaxThreads];
  int64_t* pruned_row_length = &pruned_row_lengths[t_i * kMaxThreads];

  // This placeholder will never be accessed.
  int64_t INSERTED_ROW_PLACEHOLDER = rows;

  for (int64_t idx = segment_start; idx < segment_end; idx++) {
    if (address_lookup[idx] < 0) {
      pruned_row_length[threadIdx.x] += 1;
    } else if (address_lookup[idx] == INSERTED_ROW_PLACEHOLDER) {
      inserted_row_length[threadIdx.x] += 1;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    int64_t inserted_rows = 0;
    int64_t pruned_rows = 0;
    for (int32_t i = 0; i < kMaxThreads; i++) {
      inserted_rows += inserted_row_length[i];
      pruned_rows += pruned_row_length[i];
    }
    CUDA_KERNEL_ASSERT(inserted_rows >= 0);
    CUDA_KERNEL_ASSERT(pruned_rows >= 0);
    pruning_lengths[t_i] = min(inserted_rows, pruned_rows);
    CUDA_KERNEL_ASSERT(pruning_lengths[t_i] >= 0);
  }
}

__global__ void retrieve_pruned_and_inserted_rows(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_row_offsets,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> pruned_rows,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        inserted_row_offsets,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> inserted_rows,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        address_lookups) {
  int32_t t_i = blockIdx.x;
  int64_t rows = buffer_offsets[t_i + 1] - buffer_offsets[t_i];
  int64_t segment_length = div_round_up(rows, blockDim.x);
  int64_t segment_start = threadIdx.x * segment_length;
  int64_t segment_end = min(segment_start + segment_length, rows);

  if (segment_start >= rows) {
    return;
  }

  int64_t buffer_offset = buffer_offsets[t_i];
  const int64_t* address_lookup = &address_lookups[buffer_offset];

  int64_t idx_offset = t_i * kMaxThreads + threadIdx.x;
  int64_t pruned_row_offset = pruned_row_offsets[idx_offset];
  int64_t* pruned_row = &pruned_rows[pruned_row_offset];
  int64_t inserted_row_offset = inserted_row_offsets[idx_offset];
  int64_t* inserted_row = &inserted_rows[inserted_row_offset];

  // This placeholder will never be accessed
  int64_t INSERTED_ROW_PLACEHOLDER = rows;

  int64_t pruned_idx = 0, inserted_idx = 0;
  for (int64_t idx = segment_start; idx < segment_end; idx++) {
    if (address_lookup[idx] < 0) {
      pruned_row[pruned_idx++] = idx;
    } else if (address_lookup[idx] == INSERTED_ROW_PLACEHOLDER) {
      inserted_row[inserted_idx++] = idx;
    }
  }
}

__global__ void retrieve_pruned_indices(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_row_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_rows,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        inserted_row_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        inserted_rows,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruning_offsets,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruning_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        address_lookups) {
  int32_t t_i = blockIdx.x;
  int64_t rows = pruning_offsets[t_i + 1] - pruning_offsets[t_i];
  int64_t segment_length = div_round_up(rows, blockDim.x);
  int64_t segment_start = threadIdx.x * segment_length;
  int64_t segment_end = min(segment_start + segment_length, rows);

  if (segment_start >= rows) {
    return;
  }

  int64_t buffer_offset = buffer_offsets[t_i];
  int64_t* address_lookup = &address_lookups[buffer_offset];

  int64_t pruning_offset = pruning_offsets[t_i];
  int64_t* table_pruning_indices = &pruning_indices[pruning_offset];

  int64_t pruned_row_offset = pruned_row_offsets[t_i * kMaxThreads];
  const int64_t* pruned_row = &pruned_rows[pruned_row_offset];
  int64_t inserted_row_offset = inserted_row_offsets[t_i * kMaxThreads];
  const int64_t* inserted_row = &inserted_rows[inserted_row_offset];

  // All pruned rows will be directed to the first row of embedding storage
  int64_t PAST_PRUNED_ROW_PLACEHOLDER = 0;

  for (int64_t idx = segment_start; idx < segment_end; idx++) {
    address_lookup[inserted_row[idx]] = -address_lookup[pruned_row[idx]];
    address_lookup[pruned_row[idx]] = PAST_PRUNED_ROW_PLACEHOLDER;
    table_pruning_indices[idx] = address_lookup[inserted_row[idx]];
  }
}

__global__ void cleanup_address_lookups(
    const int32_t blocks_per_table,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        address_lookups) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t t_i = blockIdx.x / blocks_per_table;
  int32_t threads_per_table = blocks_per_table * blockDim.x;
  int32_t idx_table = idx % threads_per_table;

  int64_t rows = buffer_offsets[t_i + 1] - buffer_offsets[t_i];
  int64_t rows_per_thread = div_round_up(rows, threads_per_table);
  int64_t start = idx_table * rows_per_thread;
  int64_t end = min(start + rows_per_thread, rows);

  if (start >= rows) {
    return;
  }

  int64_t buffer_offset = buffer_offsets[t_i];
  int64_t* address_lookup = &address_lookups[buffer_offset];

  // All pruned rows will be directed to the first row of embedding storage
  int64_t PAST_PRUNED_ROW_PLACEHOLDER = 0;
  // This placeholder will be never accessed
  int64_t INSERTED_ROW_PLACEHOLDER = rows;

  for (int64_t idx = start; idx < end; idx++) {
    if (address_lookup[idx] < 0) {
      // If exists more space due to sampling difference, keep as it is
      address_lookup[idx] = -address_lookup[idx];
    } else if (address_lookup[idx] == INSERTED_ROW_PLACEHOLDER) {
      // If lacks space due to sampling difference, keep them pruned.
      address_lookup[idx] = PAST_PRUNED_ROW_PLACEHOLDER;
    }
  }
}

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void remap_indices_update_utils_per_table_sorted_kernel(
    const int32_t buf_idx,
    const int64_t values_offset,
    const int64_t num_indices,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> values,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        values_sorted_unique_run,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        values_sorted_counts_run,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        values_sorted_num_runs,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        address_lookup,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> row_util,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_indices) {
    return;
  }

  const auto buffer_offset = buffer_offsets[buf_idx];
  auto* val = &values[values_offset + idx];

  // remap index
  const int64_t address_lookup_idx = buffer_offset + *val;
  *val = address_lookup[address_lookup_idx];

  if (idx >= values_sorted_num_runs[0]) {
    return;
  }

  // update row util
  const int64_t row_util_idx = buffer_offset + values_sorted_unique_run[idx];
  const int32_t util_count = values_sorted_counts_run[idx];
  row_util[row_util_idx] += util_count;
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void remap_indices_per_table_kernel(
    const int32_t buf_idx,
    const int64_t values_offset,
    const int64_t num_indices,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> values,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        address_lookup,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        buffer_offsets) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_indices) {
    return;
  }

  auto* val = &values[values_offset + idx];

  // remap index
  const int64_t buffer_idx = buffer_offsets[buf_idx] + *val;
  *val = address_lookup[buffer_idx];
}

int get_sm_count_() {
  cudaDeviceProp* deviceProp =
      at::cuda::getDeviceProperties(c10::cuda::current_device());
  return deviceProp->multiProcessorCount;
}

void init_address_lookup_cuda(
    Tensor address_lookups,
    Tensor buffer_offsets,
    Tensor emb_sizes) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      address_lookups, buffer_offsets, emb_sizes);

  CUDA_DEVICE_GUARD(address_lookups);

  const int32_t num_tables = buffer_offsets.size(0) - 1;
  if (num_tables <= 0) {
    return;
  }

  // Get number of SMs in the GPU
  const int32_t blocks_per_table = get_sm_count_();

  // Table entries can be as small as 0.1 million and as large as 4 millions.
  // Allocate all SMs to each table
  init_address_lookup_kernel<<<
      num_tables * blocks_per_table,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      blocks_per_table,
      address_lookups.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      buffer_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      emb_sizes.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<Tensor, Tensor, int64_t> prune_embedding_tables_cuda(
    int64_t iter,
    int64_t pruning_interval,
    Tensor address_lookups,
    Tensor row_utils,
    Tensor buffer_offsets,
    Tensor emb_sizes) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      address_lookups, row_utils, buffer_offsets, emb_sizes);

  CUDA_DEVICE_GUARD(address_lookups);

  const int32_t num_tables = buffer_offsets.size(0) - 1;
  if (num_tables <= 0) {
    return std::tuple(
        at::zeros({1}, buffer_offsets.options().dtype(at::kLong)),
        at::zeros({1}, buffer_offsets.options().dtype(at::kLong)),
        0);
  }

  const int32_t rows_per_sample = 23;
  // Get number of SMs in the GPU
  const int32_t blocks_per_table = get_sm_count_();

  // 1. Decay the row utility of all tables to account for stale values;
  int64_t total_buffer_size = row_utils.size(0);
  float decay_factor =
      iter > pruning_interval ? 0.98 : 1.0; // i.e., expf(-logf(1.1));
  decay_row_utils<<<
      div_round_up(total_buffer_size, kMaxThreads),
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      decay_factor,
      row_utils.packed_accessor32<float, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 2. Get number of samples per table to create sampling buffers
  // For each table, we sampled some utility rows instead of sorting all
  // utility tables. Otherwise, it results in great memory/computation overhead
  auto sampling_offsets =
      at::zeros({num_tables + 1}, buffer_offsets.options().dtype(at::kLong));

  auto rows = at::diff(buffer_offsets);
  auto sampled_rows = at::floor(rows / rows_per_sample);
  sampling_offsets.index({Slice(1, num_tables + 1)}) =
      at::cumsum(sampled_rows, 0);

  auto sampling_offsets_h = sampling_offsets.cpu();
  auto sampling_offsets_a = sampling_offsets_h.accessor<int64_t, 1>();

  int64_t total_samples_h = sampling_offsets_a[num_tables];

  // 3. Sample certain number of row utilities to determine the utility
  // threshold
  auto sampled_utilities = at::zeros({total_samples_h}, row_utils.options());
  // Allocate all SMs to each table
  get_util_samples<<<
      num_tables * blocks_per_table,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      iter,
      blocks_per_table,
      rows_per_sample,
      sampled_utilities.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
      sampling_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      buffer_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      row_utils.packed_accessor32<float, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 4. Sort sampled row utilities to determine the utility threshold
  // First, sort sampled row utilities
  auto sampled_utilities_sorted =
      at::zeros({total_samples_h}, row_utils.options());
  for (int32_t i = 0; i < num_tables; i++) {
    int64_t length = sampling_offsets_a[i + 1] - sampling_offsets_a[i];
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(
        nullptr,
        temp_storage_bytes,
        sampled_utilities.data_ptr<float>() + sampling_offsets_a[i],
        sampled_utilities_sorted.data_ptr<float>() + sampling_offsets_a[i],
        length,
        0,
        sizeof(float) * 8,
        at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        row_utils.options().dtype(at::kByte));

    cub::DeviceRadixSort::SortKeys(
        temp_storage.data_ptr(),
        temp_storage_bytes,
        sampled_utilities.data_ptr<float>() + sampling_offsets_a[i],
        sampled_utilities_sorted.data_ptr<float>() + sampling_offsets_a[i],
        length,
        0,
        sizeof(float) * 8,
        at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // Second, get util thresholds
  auto util_thresholds = at::zeros({num_tables}, row_utils.options());
  CUDA_KERNEL_ASSERT(num_tables <= kMaxThreads);
  get_util_thresholds<<<1, num_tables, 0, at::cuda::getCurrentCUDAStream()>>>(
      util_thresholds.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
      sampled_utilities_sorted
          .packed_accessor32<float, 1, at::RestrictPtrTraits>(),
      sampling_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      buffer_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      emb_sizes.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 5. Prune embedding tables based on row utilities threshold.
  // Table entries can be as small as 0.1 million and as large as 4 millions.
  // Allocate all SMs to each table
  prune_indices_per_table<<<
      blocks_per_table * num_tables,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      blocks_per_table,
      util_thresholds.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
      buffer_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      row_utils.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
      address_lookups.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 6. Get pruning length for each table
  int64_t num_table_segments = num_tables * kMaxThreads;
  auto pruned_row_lengths = at::zeros(
      {num_table_segments}, buffer_offsets.options().dtype(at::kLong));
  auto inserted_row_lengths = at::zeros(
      {num_table_segments}, buffer_offsets.options().dtype(at::kLong));
  auto pruning_lengths =
      at::zeros({num_tables}, buffer_offsets.options().dtype(at::kLong));

  get_pruning_lengths<<<
      num_tables,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      buffer_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      address_lookups.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      pruned_row_lengths.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      inserted_row_lengths
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      pruning_lengths.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 7. Retrieve pruned and inserted rows
  auto pruned_row_offsets = at::zeros(
      {num_table_segments + 1}, buffer_offsets.options().dtype(at::kLong));
  pruned_row_offsets.index({Slice(1, num_table_segments + 1)}) =
      at::cumsum(pruned_row_lengths, 0);
  auto pruned_row_offsets_h = pruned_row_offsets.cpu();
  auto pruned_row_offsets_a = pruned_row_offsets_h.accessor<int64_t, 1>();
  int64_t pruned_row_len = pruned_row_offsets_a[num_table_segments];
  if (pruned_row_len <= 0) {
    pruned_row_len = 1; // Avoid allocate empty tensor
  }
  auto pruned_rows =
      at::zeros({pruned_row_len}, buffer_offsets.options().dtype(at::kLong));

  auto inserted_row_offsets = at::zeros(
      {num_table_segments + 1}, buffer_offsets.options().dtype(at::kLong));
  inserted_row_offsets.index({Slice(1, num_table_segments + 1)}) =
      at::cumsum(inserted_row_lengths, 0);
  auto inserted_row_offsets_h = inserted_row_offsets.cpu();
  auto inserted_row_offsets_a = inserted_row_offsets_h.accessor<int64_t, 1>();
  int64_t inserted_row_len = inserted_row_offsets_a[num_table_segments];
  if (inserted_row_len <= 0) {
    inserted_row_len = 1; // Avoid allocate empty tensor
  }
  auto inserted_rows =
      at::zeros({inserted_row_len}, buffer_offsets.options().dtype(at::kLong));

  retrieve_pruned_and_inserted_rows<<<
      num_tables,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      pruned_row_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      pruned_rows.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      inserted_row_offsets
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      inserted_rows.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      buffer_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      address_lookups.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 8. Retrieve the pruned indices
  auto pruning_offsets =
      at::zeros({num_tables + 1}, buffer_offsets.options().dtype(at::kLong));
  pruning_offsets.index({Slice(1, num_tables + 1)}) =
      at::cumsum(pruning_lengths, 0);
  auto pruning_offsets_h = pruning_offsets.cpu();
  auto pruning_offsets_a = pruning_offsets_h.accessor<int64_t, 1>();
  int64_t pruning_indices_len = pruning_offsets_a[num_tables];
  int64_t pruning_total_length = pruning_indices_len;
  if (pruning_indices_len <= 0) {
    pruning_indices_len = 1; // Avoid allocate empty tensor
  }
  auto pruning_indices = at::zeros(
      {pruning_indices_len}, buffer_offsets.options().dtype(at::kLong));

  retrieve_pruned_indices<<<
      num_tables,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      pruned_row_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      pruned_rows.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      inserted_row_offsets
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      inserted_rows.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      pruning_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      pruning_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      buffer_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      address_lookups.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 9. Cleanup remaining marked rows in address lookups
  cleanup_address_lookups<<<
      num_tables * blocks_per_table,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      blocks_per_table,
      buffer_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      address_lookups.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::tuple(pruning_indices, pruning_offsets, pruning_total_length);
}

// per table sorted, periodical row util update
Tensor remap_indices_update_utils_cuda(
    const int64_t iter,
    const Tensor& buffer_idx,
    const Tensor& feature_lengths,
    const Tensor& feature_offsets,
    const Tensor& values,
    const Tensor& address_lookup,
    Tensor& row_util,
    const Tensor& buffer_offsets,
    // full_values_list is required for grouped data training (GDT) in order to
    // correctly update row_util. GDT deduplicates values which distorts value
    // frequencies which are used for updating row_util. Since row_util is used
    // during pruning, updating row_util with the distorted value frequencies
    // can affect training accuracy.
    const std::optional<std::vector<Tensor>>& full_values_list,
    const std::optional<bool>& update_util) {
  // buffer_idx, feature_lengths, feature_offsets are placed on CPU.
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      values, address_lookup, row_util, buffer_offsets);
  TENSOR_ON_CPU(buffer_idx);
  TENSOR_ON_CPU(feature_lengths);
  TENSOR_ON_CPU(feature_offsets);

  CUDA_DEVICE_GUARD(values);

  const int32_t num_tables = buffer_offsets.size(0) - 1;
  if (num_tables <= 0) {
    return values;
  }

  const int32_t num_indices = values.size(0);
  if (num_indices <= 0) {
    return values;
  }

  const auto buffer_idx_a = buffer_idx.accessor<int32_t, 1>();
  const auto feature_lengths_a = feature_lengths.accessor<int64_t, 1>();
  const auto feature_offsets_a = feature_offsets.accessor<int64_t, 1>();

  const auto use_gdt = full_values_list.has_value();
  const int32_t num_features = feature_lengths.numel();
  const bool update_util_value = update_util.has_value()
      ? update_util.value()
      : ((iter < 10) || (iter < 100 && (iter + 1) % 19 == 0) ||
         ((iter + 1) % 39 == 0));

  AT_DISPATCH_INDEX_TYPES(
      values.scalar_type(), "remap_indices_update_utils_cuda", [&] {
        for (int32_t i = 0; i < num_features; i++) {
          const auto start = feature_offsets_a[i];
          const auto length = feature_lengths_a[i];

          if (length == 0) {
            continue;
          }

          if (update_util_value) {
            const Tensor& full_values =
                use_gdt ? full_values_list.value()[i] : values;
            const index_t full_start = use_gdt ? 0 : start;
            const index_t full_length = use_gdt ? full_values.numel() : length;
            if (use_gdt) {
              TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(values, full_values);
            }

            // sort indices for each feature
            auto values_sorted =
                at::zeros({full_length}, full_values.options());
            size_t temp_storage_bytes_0 = 0;

            cub::DeviceRadixSort::SortKeys(
                nullptr,
                temp_storage_bytes_0,
                full_values.data_ptr<index_t>() + full_start,
                values_sorted.data_ptr<index_t>(),
                full_length,
                0,
                sizeof(index_t) * 8,
                at::cuda::getCurrentCUDAStream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            auto temp_storage_0 = at::empty(
                {static_cast<index_t>(temp_storage_bytes_0)},
                values.options().dtype(at::kByte));

            cub::DeviceRadixSort::SortKeys(
                temp_storage_0.data_ptr(),
                temp_storage_bytes_0,
                full_values.data_ptr<index_t>() + full_start,
                values_sorted.data_ptr<index_t>(),
                full_length,
                0,
                sizeof(index_t) * 8,
                at::cuda::getCurrentCUDAStream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // run length encode to count access frequency to each row
            auto values_sorted_unique_run = at::empty_like(values_sorted);
            auto values_sorted_counts_run = at::zeros(
                values_sorted.sizes(), values_sorted.options().dtype(at::kInt));
            auto values_sorted_num_runs =
                at::zeros({1}, values_sorted.options().dtype(at::kInt));
            size_t temp_storage_bytes_1 = 0;

            cub::DeviceRunLengthEncode::Encode(
                nullptr,
                temp_storage_bytes_1,
                values_sorted.data_ptr<index_t>(),
                values_sorted_unique_run.data_ptr<index_t>(),
                values_sorted_counts_run.data_ptr<int32_t>(),
                values_sorted_num_runs.data_ptr<int32_t>(),
                full_length,
                at::cuda::getCurrentCUDAStream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            auto temp_storage_1 = at::empty(
                {static_cast<index_t>(temp_storage_bytes_1)},
                values.options().dtype(at::kByte));

            cub::DeviceRunLengthEncode::Encode(
                temp_storage_1.data_ptr(),
                temp_storage_bytes_1,
                values_sorted.data_ptr<index_t>(),
                values_sorted_unique_run.data_ptr<index_t>(),
                values_sorted_counts_run.data_ptr<int32_t>(),
                values_sorted_num_runs.data_ptr<int32_t>(),
                full_length,
                at::cuda::getCurrentCUDAStream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // remap indices and update row utils
            const int32_t buf_idx = buffer_idx_a[i];
            remap_indices_update_utils_per_table_sorted_kernel<<<
                div_round_up(length, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                buf_idx,
                start,
                length,
                values.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                values_sorted_unique_run
                    .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                values_sorted_counts_run
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                values_sorted_num_runs
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                address_lookup
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                row_util.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
                buffer_offsets
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            // remap indices and update row utils
            const int32_t buf_idx = buffer_idx_a[i];
            remap_indices_per_table_kernel<<<
                div_round_up(length, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                buf_idx,
                start,
                length,
                values.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                address_lookup
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                buffer_offsets
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }
      });

  return values;
}

} // namespace fbgemm_gpu
