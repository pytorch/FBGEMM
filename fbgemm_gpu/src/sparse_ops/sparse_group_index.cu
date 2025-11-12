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

#ifdef USE_ROCM
// The wave size is forced to be 32 on ROCm devices in favor 
// of granularity losses reduction.
constexpr int EMULATED_WARP_SIZE = 32;
#else
constexpr int EMULATED_WARP_SIZE = kWarpSize;
#endif

// TODO: Update UNROLL_FACTOR
constexpr int GROUP_INDEX_SELECT_UNROLL_FACTOR = 1;

// SELECT (fwd): use EMULATED_WARP_SIZE
constexpr int GROUP_INDEX_SELECT_COLS_PER_WARP_FWD =
    GROUP_INDEX_SELECT_UNROLL_FACTOR * EMULATED_WARP_SIZE;
constexpr int GROUP_INDEX_SELECT_LOG_COLS_PER_WARP_FWD =
    log2_calc<GROUP_INDEX_SELECT_COLS_PER_WARP_FWD>::value;

// ADD (bwd): use kWarpSize
constexpr int GROUP_INDEX_SELECT_COLS_PER_WARP_BWD =
    GROUP_INDEX_SELECT_UNROLL_FACTOR * kWarpSize;
constexpr int GROUP_INDEX_SELECT_LOG_COLS_PER_WARP_BWD =
    log2_calc<GROUP_INDEX_SELECT_COLS_PER_WARP_BWD>::value;

int get_group_index_select_cols_per_warp() {
  return GROUP_INDEX_SELECT_COLS_PER_WARP_BWD;
}

int get_group_index_select_cols_per_warp(bool use_index_select) {
  return use_index_select ? GROUP_INDEX_SELECT_COLS_PER_WARP_FWD
                          : GROUP_INDEX_SELECT_COLS_PER_WARP_BWD;
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
    const int64_t num_work_rows, 
    const int64_t group_size) {
  const auto total_num_warps = warp_offsets_group[group_size];
  int32_t num_cols = 0; 
  int32_t warps_per_row = 0;
#ifdef USE_ROCM
  // USE_INDEX_SELECT is a template argument; the compiler prunes the unused branch.
  if constexpr (USE_INDEX_SELECT) {
    for (int64_t warp_id = threadIdx.y * gridDim.x + blockIdx.x;
         warp_id < total_num_warps;
         warp_id += gridDim.x * blockDim.y) {
      int32_t member_id, member_warp_id, num_cols, warps_per_row;
      if (USE_VAR_COLS) {
        __shared__ int member_ids[kMaxThreads / EMULATED_WARP_SIZE];
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
#pragma unroll
      for (int i = 0; i < UNROLL_FACTOR && col_offset + i < num_cols; i++) {
        output[row * num_cols + i] = LDG(&input[idx * num_cols + i]);
      }
    }
  } else {
    constexpr int kCacheSlots = 2;
    index_t cached_idx[kCacheSlots];
    scalar_t cached_vals[kCacheSlots][UNROLL_FACTOR];
    bool cached_valid[kCacheSlots];
#pragma unroll
    for (int slot = 0; slot < kCacheSlots; ++slot) {
      cached_valid[slot] = false;
    }
    int32_t active_member_id = -1;
    int32_t active_num_cols = 0;
    int32_t active_col_offset = -1;
    scalar_t* active_input_base = nullptr;
    scalar_t* active_output_base = nullptr;
    index_t* active_indices = nullptr;

    auto flush_cache = [&](scalar_t* out_base,
                           int32_t num_cols,
                           int32_t col_offset) {
      if (!out_base) {
        return;
      }
#pragma unroll
      for (int slot = 0; slot < kCacheSlots; ++slot) {
        if (!cached_valid[slot]) {
          continue;
        }
        const int64_t row_offset =
            static_cast<int64_t>(cached_idx[slot]) * num_cols;
#pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; ++j) {
          const int32_t col = col_offset + j;
          if (col >= num_cols) {
            break;
          }
          gpuAtomicAddNoReturn(
              out_base + row_offset + col, cached_vals[slot][j]);
        }
        cached_valid[slot] = false;
      }
    };

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
      const int64_t row = member_warp_id / warps_per_row;
      const int32_t col_offset =
          static_cast<int32_t>(((member_warp_id % warps_per_row)
                                << LOG_COLS_PER_WARP) +
                               (threadIdx.x * UNROLL_FACTOR));

      const bool member_changed = member_id != active_member_id;
      const bool num_cols_changed =
          member_changed ? false : (num_cols != active_num_cols);
      const bool col_changed =
          member_changed ? false : (col_offset != active_col_offset);
      if (member_changed || num_cols_changed || col_changed) {
        flush_cache(active_output_base, active_num_cols, active_col_offset);
        active_member_id = member_id;
        active_num_cols = num_cols;
        active_col_offset = col_offset;
        active_input_base =
            reinterpret_cast<scalar_t*>(input_ptrs[member_id]);
        active_output_base =
            reinterpret_cast<scalar_t*>(output_ptrs[member_id]);
        active_indices =
            reinterpret_cast<index_t*>(indices_ptrs[member_id]);
      }

      if (col_offset >= active_num_cols) {
        continue;
      }

      const index_t idx = active_indices[row];

      scalar_t local_vals[UNROLL_FACTOR];
#pragma unroll
      for (int j = 0; j < UNROLL_FACTOR; ++j) {
        local_vals[j] = static_cast<scalar_t>(0);
      }
      const int64_t input_offset =
          static_cast<int64_t>(row) * active_num_cols + active_col_offset;
#pragma unroll
      for (int j = 0; j < UNROLL_FACTOR; ++j) {
        const int32_t col = active_col_offset + j;
        if (col >= active_num_cols) {
          break;
        }
        local_vals[j] = active_input_base[input_offset + j];
      }

      bool appended = false;
#pragma unroll
      for (int slot = 0; slot < kCacheSlots; ++slot) {
        if (cached_valid[slot] && cached_idx[slot] == idx) {
#pragma unroll
          for (int j = 0; j < UNROLL_FACTOR; ++j) {
            const int32_t col = active_col_offset + j;
            if (col >= active_num_cols) {
              break;
            }
            cached_vals[slot][j] += local_vals[j];
          }
          appended = true;
          break;
        }
      }

      if (!appended) {
        int slot_to_use = -1;
#pragma unroll
        for (int slot = 0; slot < kCacheSlots; ++slot) {
          if (!cached_valid[slot]) {
            slot_to_use = slot;
            break;
          }
        }
        if (slot_to_use == -1) {
          slot_to_use = 0;
          const int64_t row_offset =
              static_cast<int64_t>(cached_idx[slot_to_use]) *
              active_num_cols;
#pragma unroll
          for (int j = 0; j < UNROLL_FACTOR; ++j) {
            const int32_t col = active_col_offset + j;
            if (col >= active_num_cols) {
              break;
            }
            gpuAtomicAddNoReturn(
                active_output_base + row_offset + col,
                cached_vals[slot_to_use][j]);
          }
          cached_valid[slot_to_use] = false;
        }

        cached_idx[slot_to_use] = idx;
#pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; ++j) {
          cached_vals[slot_to_use][j] = local_vals[j];
        }
        cached_valid[slot_to_use] = true;
      }
    }

    flush_cache(active_output_base, active_num_cols, active_col_offset);
  }
#else  // Original CUDA implementation
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
#endif 
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
    const int num_work_rows,
    const int64_t total_num_warps,
    const int group_size,
    const bool use_index_select,
    const bool use_var_cols) {
  if (group_size == 0) {
    return;
  }

  at::cuda::OptionalCUDAGuard device_guard(device);
  uint32_t num_warps_per_threadblock;
  dim3 block_size;
  
  if (use_index_select) {
    // Forward pass uses EMULATED_WARP_SIZE
    num_warps_per_threadblock = kMaxThreads / EMULATED_WARP_SIZE;
    block_size = dim3(EMULATED_WARP_SIZE, num_warps_per_threadblock, 1);
  } else {
    // Backward pass uses kWarpSize
    num_warps_per_threadblock = kMaxThreads / kWarpSize;
    block_size = dim3(kWarpSize, num_warps_per_threadblock, 1);
  }

  uint32_t max_grid_size =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8;
  uint32_t grid_size = std::min(
      cuda_calc_xblock_count(total_num_warps, num_warps_per_threadblock),
      max_grid_size);
  

#define INVOKE_GROUP_INDEX_SELECT_OR_ADD(USE_INDEX_SELECT, USE_VAR_COLS) \
  FBGEMM_LAUNCH_KERNEL(                                                  \
      (group_index_select_or_add_2d_kernel<                              \
          index_t,                                                       \
          scalar_t,                                                      \
          USE_INDEX_SELECT,                                              \
          USE_VAR_COLS,                                                  \
          GROUP_INDEX_SELECT_UNROLL_FACTOR,                              \
          (USE_INDEX_SELECT                                              \
               ? GROUP_INDEX_SELECT_COLS_PER_WARP_FWD                    \
               : GROUP_INDEX_SELECT_COLS_PER_WARP_BWD),                  \
          /* LOG_COLS_PER_WARP */                                        \
          (USE_INDEX_SELECT                                              \
               ? GROUP_INDEX_SELECT_LOG_COLS_PER_WARP_FWD                \
               : GROUP_INDEX_SELECT_LOG_COLS_PER_WARP_BWD)>),            \
      grid_size,                                                         \
      block_size,                                                        \
      0,                                                                 \
      at::cuda::getCurrentCUDAStream(),                                  \
      input_ptrs,                                                        \
      output_ptrs,                                                       \
      indices_ptrs,                                                      \
      warp_offsets_group,                                                \
      num_cols_group,                                                    \
      num_work_rows,                                                     \
      group_size)

  AT_DISPATCH_INDEX_TYPES(
      indices_scalar_type, "group_index_select_2d_wrapper_1", [&] {
        FBGEMM_DISPATCH_FLOATING_TYPES(
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
