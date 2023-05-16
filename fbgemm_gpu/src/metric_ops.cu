/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>
#include <ATen/cuda/Atomic.cuh>
#include <algorithm>

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "metric_ops.h"

constexpr int MAX_ENTRIES_PER_BLOCK = 512;
constexpr int NUM_THREADS_PER_BLOCK = 256;

namespace fbgemm_gpu {

template <typename scalar_t>
__inline__ __device__ void trapz_kernel(
    scalar_t* output,
    const scalar_t* y,
    const scalar_t* x,
    const scalar_t* block_y,
    const scalar_t* block_x,
    const int num_entries_per_block,
    const int block_id) {
  scalar_t sum = 0;
  // Compute inter-block pair
  if (block_id > 0 && threadIdx.x == 0) {
    sum +=
        0.5 * (x[0] - block_x[block_id - 1]) * (y[0] + block_y[block_id - 1]);
  }
  for (int i = threadIdx.x + 1; i < num_entries_per_block; i += blockDim.x) {
    sum += 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1]);
  }
  sum = warpReduceAllSum(sum);
  // Only first lane threads accumulate results
  // Expect output to be initialize with zero (we initialize output outside of
  // trapz_kernel to avoid calling another __syncthreads() here)
  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicAdd(output, sum);
  }
}

template <
    typename index_t,
    typename label_t,
    typename weight_t,
    typename acc_t,
    int PADDED_SECTION_SIZE>
__global__ void auc_kernel(
    acc_t* output,
    const index_t* indices,
    const label_t* labels,
    const weight_t* weights,
    int* block_flags,
    acc_t* block_sums,
    const int num_entries,
    const int last_block_num_entries,
    const int padded_num_entries_per_block,
    const int num_blocks) {
  typedef cub::BlockScan<acc_t, NUM_THREADS_PER_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage bs_temp_storage;
  __shared__ acc_t smem[MAX_ENTRIES_PER_BLOCK * 2 + 3];
  acc_t* smem_fp = smem;
  acc_t* smem_tp = smem + padded_num_entries_per_block;
  acc_t* smem_tmp = smem_tp + padded_num_entries_per_block;
  acc_t* smem_auc = smem_tmp + 2;

  const int block_id = blockIdx.x % num_blocks;
  const int task_id = blockIdx.x / num_blocks;

  const int num_entries_per_block = block_id == num_blocks - 1
      ? last_block_num_entries
      : MAX_ENTRIES_PER_BLOCK;
  const int input_offset = task_id * num_entries;
  const int block_sums_offset = task_id * num_blocks;
  const bool is_multi_block = num_blocks > 1;
  const int section_offset = PADDED_SECTION_SIZE * threadIdx.x;

  indices += input_offset + (block_id * MAX_ENTRIES_PER_BLOCK);
  labels += input_offset;
  weights += input_offset;
  output += task_id;

  acc_t* block_sums_fp =
      is_multi_block ? (block_sums + (block_sums_offset << 1)) : nullptr;
  acc_t* block_sums_tp =
      is_multi_block ? (block_sums_fp + num_blocks) : nullptr;
  block_flags = is_multi_block ? block_flags + block_sums_offset : nullptr;

  acc_t local_fp[PADDED_SECTION_SIZE];
  acc_t local_tp[PADDED_SECTION_SIZE];

  // Load data into shared memory
  for (int i = 0;
       i < PADDED_SECTION_SIZE && section_offset + i < num_entries_per_block;
       i++) {
    const index_t idx = indices[section_offset + i];
    const acc_t weight = weights[idx];
    const label_t label = labels[idx];
    local_fp[i] = weight * (1.0 - label);
    local_tp[i] = weight * label;
  }

  if (threadIdx.x == 0) {
    *smem_auc = 0.0;
  }

  __syncthreads();

  inclusive_sum_scan_kernel<acc_t, PADDED_SECTION_SIZE, NUM_THREADS_PER_BLOCK>(
      local_fp,
      bs_temp_storage,
      block_flags,
      block_sums_fp,
      &smem_tmp[0],
      num_entries_per_block,
      block_id,
      is_multi_block,
      /*signal=*/1);

  inclusive_sum_scan_kernel<acc_t, PADDED_SECTION_SIZE, NUM_THREADS_PER_BLOCK>(
      local_tp,
      bs_temp_storage,
      block_flags,
      block_sums_tp,
      &smem_tmp[0],
      num_entries_per_block,
      block_id,
      is_multi_block,
      /*signal=*/2);

  for (int i = 0; i < PADDED_SECTION_SIZE; i++) {
    smem_fp[section_offset + i] = local_fp[i];
    smem_tp[section_offset + i] = local_tp[i];
  }

  __syncthreads();

  // Get last fp and tp
  acc_t last_fp, last_tp;
  if (is_multi_block) {
    if (block_id == num_blocks - 1) {
      last_fp = smem_fp[num_entries_per_block - 1];
      last_tp = smem_tp[num_entries_per_block - 1];
    } else {
      if (threadIdx.x == 0) {
        // This ensures that the all blocks are done writing block prefix sums
        // to global memory
        while (atomicAdd(&block_flags[num_blocks - 1], 0) < 2)
          ;
        last_fp = block_sums_fp[num_blocks - 1];
        last_tp = block_sums_tp[num_blocks - 1];
        smem_tmp[0] = last_fp;
        smem_tmp[1] = last_tp;
      }
      __syncthreads();
      if (threadIdx.x != 0) {
        last_fp = smem_tmp[0];
        last_tp = smem_tmp[1];
      }
    }
  } else {
    last_fp = smem_fp[num_entries - 1];
    last_tp = smem_tp[num_entries - 1];
  }

  if (last_fp * last_tp == 0.0) {
    if (threadIdx.x == 0) {
      *output = 0.5;
    }
  } else {
    trapz_kernel(
        smem_auc,
        smem_tp,
        smem_fp,
        block_sums_tp,
        block_sums_fp,
        num_entries_per_block,
        block_id);

    // Ensure that atomic add in trapz_kernel is done
    __syncthreads();

    if (threadIdx.x == 0) {
      gpuAtomicAdd(output, *smem_auc / last_fp / last_tp);
    }
  }
}

at::Tensor batch_auc(
    const int64_t num_tasks,
    const at::Tensor& indices,
    const at::Tensor& labels,
    const at::Tensor& weights) {
  auto dim = indices.dim();
  auto num_entries = indices.size(dim - 1);
  auto num_entries_all_tasks = indices.numel();

  TORCH_CHECK(labels.dim() == dim && weights.dim() == dim)
  TORCH_CHECK(num_entries_all_tasks == num_entries * num_tasks)
  TORCH_CHECK(
      labels.size(dim - 1) == num_entries &&
      weights.size(dim - 1) == num_entries &&
      labels.numel() == num_entries_all_tasks &&
      weights.numel() == num_entries_all_tasks)

  const int log_num_threads = std::log2(NUM_THREADS_PER_BLOCK);
  const int num_blocks =
      (num_entries + MAX_ENTRIES_PER_BLOCK - 1) / MAX_ENTRIES_PER_BLOCK;
  const int num_entries_per_block =
      num_blocks > 1 ? MAX_ENTRIES_PER_BLOCK : num_entries;
  const int rounded_section_size = num_entries_per_block >> log_num_threads;
  const int rounded_num_entries_per_block = rounded_section_size
      << log_num_threads;
  const int padded_num_entries_per_block = rounded_num_entries_per_block +
      (rounded_num_entries_per_block != num_entries_per_block
           ? NUM_THREADS_PER_BLOCK
           : 0);
  const int padded_section_size =
      padded_num_entries_per_block / NUM_THREADS_PER_BLOCK;
  const int last_block_num_entries =
      num_entries - ((num_blocks - 1) * MAX_ENTRIES_PER_BLOCK);

  const auto output_options = weights.scalar_type() == at::ScalarType::Half
      ? weights.options().dtype(at::kFloat)
      : weights.options();
  at::Tensor output = at::zeros({num_tasks}, output_options);

  const int grid_size = num_blocks * num_tasks;

  at::Tensor block_flags;
  at::Tensor block_sums;
  if (num_blocks > 1) {
    block_flags = at::zeros({grid_size}, weights.options().dtype(at::kInt));
    block_sums = at::empty({grid_size * 2}, output_options);
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  auto max_smem_size =
      at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;

#define LAUNCH_AUC_KERNEL(pad)                                     \
  typedef cub::BlockScan<acc_t, NUM_THREADS_PER_BLOCK> BlockScan;  \
  TORCH_CHECK(                                                     \
      sizeof(BlockScan::TempStorage) +                             \
          ((MAX_ENTRIES_PER_BLOCK * 2 + 3) * sizeof(acc_t)) <=     \
      max_smem_size)                                               \
  auc_kernel<index_t, label_t, scalar_t, acc_t, pad>               \
      <<<dim3(grid_size),                                          \
         dim3(NUM_THREADS_PER_BLOCK),                              \
         0,                                                        \
         at::cuda::getCurrentCUDAStream()>>>(                      \
          output.data_ptr<acc_t>(),                                \
          indices.data_ptr<index_t>(),                             \
          labels.data_ptr<label_t>(),                              \
          weights.data_ptr<scalar_t>(),                            \
          num_blocks > 1 ? block_flags.data_ptr<int>() : nullptr,  \
          num_blocks > 1 ? block_sums.data_ptr<acc_t>() : nullptr, \
          num_entries,                                             \
          last_block_num_entries,                                  \
          padded_num_entries_per_block,                            \
          num_blocks);

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "auc_wrapper_1", [&] {
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, labels.scalar_type(), "auc_wrapper_2", [&] {
          using label_t = scalar_t;
          AT_DISPATCH_FLOATING_TYPES_AND_HALF(
              weights.scalar_type(), "auc_wrapper_3", [&] {
                using acc_t = at::acc_type<scalar_t, true>;
                if (padded_section_size == 1) {
                  LAUNCH_AUC_KERNEL(1)
                } else {
                  LAUNCH_AUC_KERNEL(2)
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  });

#undef LAUNCH_AUC_KERNEL

  return output;
}

} // namespace fbgemm_gpu
