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
