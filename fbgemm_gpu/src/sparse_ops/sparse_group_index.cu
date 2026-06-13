/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cassert>
#include <limits>
#include <type_traits>
#include <variant>

#include "common.cuh"
#ifdef USE_ROCM
#include "fbgemm_gpu/utils/rocm/sparse_group_utils.h"
#endif

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
constexpr int GROUP_INDEX_SELECT_COLS_PER_WARP =
    GROUP_INDEX_SELECT_UNROLL_FACTOR * EMULATED_WARP_SIZE;

// GROUP_INDEX_SELECT_COLS_PER_WARP must be power of two
constexpr int GROUP_INDEX_SELECT_LOG_COLS_PER_WARP =
    log2_calc<GROUP_INDEX_SELECT_COLS_PER_WARP>::value;

int get_group_index_select_cols_per_warp() {
  return GROUP_INDEX_SELECT_COLS_PER_WARP;
}

int get_group_index_select_unroll_factor() {
  return GROUP_INDEX_SELECT_UNROLL_FACTOR;
}

template <
    typename index_t,
    typename scalar_t,
    bool USE_INDEX_SELECT,
    bool USE_VAR_COLS,
    bool USE_CONTIGUOUS_WARPS, 
    bool USE_SORTED_INDICES,
    bool USE_CACHE,
    bool USE_PACKED_ROWS,
    int UNROLL_FACTOR,
    int COLS_PER_WARP,
    int LOG_COLS_PER_WARP>
__global__
__launch_bounds__(kMaxThreads) void group_index_select_or_add_2d_kernel(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* reverse_indices_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const int64_t num_work_rows, // number of rows to work on per member
    const int64_t group_size) {
  static_assert(
      !USE_CACHE || UNROLL_FACTOR == 1,
      "Cache path only supports UNROLL_FACTOR == 1");

  constexpr index_t kInvalidIdx = std::numeric_limits<index_t>::max();

  const auto total_num_warps = warp_offsets_group[group_size];
  int32_t num_cols = 0;
  int32_t warps_per_row = 0;

  if constexpr (!USE_VAR_COLS) {
    num_cols = num_cols_group[0];
    warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
  }

  [[maybe_unused]] int cached_member_id = -1;
  [[maybe_unused]] int64_t cached_upper_bound = -1;
  [[maybe_unused]] int32_t last_member_id_for_accum = -1;
  [[maybe_unused]] int32_t last_member_num_cols = 0;
  [[maybe_unused]] scalar_t* last_member_output_tile = nullptr;

  int64_t start_warp_id = 0;
  int64_t warp_end = 0;
  int64_t warp_stride = 0;

  if constexpr (USE_CONTIGUOUS_WARPS) {
    const int64_t linear_warp_id = threadIdx.y * gridDim.x + blockIdx.x;
    const int64_t warps_per_launch = gridDim.x * blockDim.y;
    const int64_t chunk_size =
        (total_num_warps + warps_per_launch - 1) / warps_per_launch;
    start_warp_id = linear_warp_id * chunk_size;
    warp_end = start_warp_id + chunk_size < total_num_warps
        ? start_warp_id + chunk_size
        : total_num_warps;
    warp_stride = 1;
  } else {
    start_warp_id = threadIdx.y * gridDim.x + blockIdx.x;
    warp_end = total_num_warps;
    warp_stride = gridDim.x * blockDim.y;
  }

  auto storage = scalar_t(0);
  auto cached_idx = kInvalidIdx;
  // TODO: Account for UNROLL_FACTOR
  auto flush_cache_accumulator = [&](scalar_t* target_output,
                                     int32_t target_num_cols) {
    if constexpr (!USE_INDEX_SELECT && USE_CACHE) {
      if (target_output && cached_idx != kInvalidIdx) {
        gpuAtomicAddNoReturn(
            &target_output[cached_idx * target_num_cols],
            storage);
        cached_idx = kInvalidIdx;
      }
    }
  };

  for (int64_t warp_id = start_warp_id; warp_id < warp_end; warp_id += warp_stride) {
    bool use_small_dim_path = false;
    int rows_per_warp_small = 0;
    int32_t member_id = 0;
    int32_t member_warp_id = 0;
    if constexpr (USE_VAR_COLS) {
#ifdef USE_ROCM
      if (warp_id >= cached_upper_bound) {
        rocm::warp_upper_bound<int64_t, EMULATED_WARP_SIZE>(
            &member_id,
            &cached_upper_bound,
            warp_offsets_group + 1,
            warp_id,
            group_size);
        cached_member_id = member_id;
      } else {
        member_id = cached_member_id;
      }
#else
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
#endif
      num_cols = num_cols_group[member_id];
      warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
      member_warp_id = warp_id - warp_offsets_group[member_id];
    } else {
      // All columns are the same
      member_id = warp_id / (warps_per_row * num_work_rows);
      member_warp_id = warp_id - (member_id * warps_per_row * num_work_rows);
    }

    if constexpr (USE_PACKED_ROWS) {
      if (num_cols < COLS_PER_WARP && num_cols >= UNROLL_FACTOR) {
        // Need to ensure that [member_id] and [member_warp_id] are calculated
        // correctly for the small embedding dimension path below
        rows_per_warp_small = COLS_PER_WARP / num_cols;
        if constexpr (!USE_VAR_COLS) {
          const auto warps_per_member =
              (num_work_rows + rows_per_warp_small - 1) / rows_per_warp_small;
          member_id = warp_id / warps_per_member;
          member_warp_id = warp_id % warps_per_member;
        }
        use_small_dim_path = true;
      }
    }

    index_t* indices = reinterpret_cast<index_t*>(indices_ptrs[member_id]);
    const int64_t* reverse_indices = USE_SORTED_INDICES
        ? reinterpret_cast<int64_t*>(reverse_indices_ptrs[member_id])
        : nullptr;

    int64_t logical_row = 0;
    int64_t row = 0;
    int64_t col_offset = 0;
    bool handled_small_dim_path = false;

    if constexpr (USE_PACKED_ROWS) {
      if (use_small_dim_path) {
        // Optimized path for small embedding dimensions
        // Each warp processes 'rows_per_warp' rows
        const int rows_per_warp = rows_per_warp_small;
        const int64_t start_row = member_warp_id * rows_per_warp;
        // Since we are processing multiple rows within the warp, we need to
        // map each lane to a specific row, in addition to the column
        const int local_row = (threadIdx.x * UNROLL_FACTOR) / 
            num_cols; // the row ID within the set of rows handled by this warp
        const int64_t current_row = start_row + 
            local_row; // the actual row within the table processed by this lane
        const int col_offset_small = (threadIdx.x * UNROLL_FACTOR) % num_cols;
        // local_row may be out of bounds for the last few lanes in the warp if
        // [COLS_PER_WARP % num_cols != 0] and we also need to confirm that we are
        // within num_work_rows
        if (local_row < rows_per_warp && current_row < num_work_rows) {
          logical_row = current_row;
          row = USE_SORTED_INDICES ? reverse_indices[current_row] : current_row;
          col_offset = col_offset_small;
          handled_small_dim_path = true;
        } else {
          flush_cache_accumulator(last_member_output_tile, last_member_num_cols);
          continue;
        }
      }
    }

    if (!handled_small_dim_path) {
      int64_t row_in_member = 0;
      int64_t col_tile = 0;
      if constexpr (USE_CONTIGUOUS_WARPS) {
        // Contiguous warp traversal: iterate rows sequentially while column tiles
        // remain strided so each warp processes a different tile for successive rows.
        row_in_member = member_warp_id % num_work_rows;
        col_tile = member_warp_id / num_work_rows;
      } else {
        // Original strided mapping: each warp walks tiles first, distributing rows round-robin.
        row_in_member = member_warp_id / warps_per_row;
        col_tile = member_warp_id % warps_per_row;
      }

      logical_row = row_in_member;
      row = USE_SORTED_INDICES ? reverse_indices[row_in_member] : row_in_member;
      col_offset =
          (static_cast<int64_t>(col_tile) << LOG_COLS_PER_WARP) +
          (threadIdx.x * UNROLL_FACTOR);
    }
    scalar_t* input =
        reinterpret_cast<scalar_t*>(input_ptrs[member_id]) + col_offset;
    scalar_t* output =
        reinterpret_cast<scalar_t*>(output_ptrs[member_id]) + col_offset;

    const index_t idx = indices[logical_row];
    // TODO: Account for UNROLL_FACTOR
#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR && col_offset + i < num_cols; i++) {
      // Compile time conditional
      if constexpr (USE_INDEX_SELECT) {
        if constexpr (USE_CACHE) {
          if (cached_idx != idx) {
            storage = LDG(&input[idx * num_cols + i]);
            cached_idx = idx;
          }

          output[row * num_cols + i] = storage;
        } else {
          output[row * num_cols + i] = LDG(&input[idx * num_cols + i]);
        }
      } else {
        if constexpr (USE_CACHE) {
          const bool member_changed = (last_member_id_for_accum != -1 &&
                                      member_id != last_member_id_for_accum);
          // Probably might be merged into following if-else cascade
          if (member_changed) {
            flush_cache_accumulator(last_member_output_tile, last_member_num_cols);
          }

          const bool is_first_warp = member_changed || (warp_id == start_warp_id);
          const bool is_last_warp = (warp_id + warp_stride >= warp_end);
          if (is_first_warp) {
            storage = input[row * num_cols + i];
            cached_idx = idx;
          } else if (cached_idx != idx) {
            // Flush using the output tile that owns the cached accumulator.
            flush_cache_accumulator(last_member_output_tile, last_member_num_cols);
            storage = input[row * num_cols + i];
            cached_idx = idx;
          } else {
            storage += input[row * num_cols + i];
          }

          if (is_last_warp) {
            flush_cache_accumulator(output, num_cols);
          }

          last_member_output_tile = output;
          last_member_num_cols = num_cols;
          last_member_id_for_accum = member_id;
        } else {
          gpuAtomicAddNoReturn(
              &output[idx * num_cols + i], input[row * num_cols + i]);
        }
      }
    }
  }

  flush_cache_accumulator(last_member_output_tile, last_member_num_cols);
}

DLL_PUBLIC void group_index_select_or_add_cuda(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* sorted_indices_ptrs,
    const int64_t* reverse_indices_ptrs, 
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const c10::ScalarType& input_scalar_type,
    const c10::ScalarType& indices_scalar_type,
    const c10::DeviceIndex& device,
    const int num_work_rows,
    const int64_t total_num_warps,
    const int group_size,
    const bool use_index_select,
    const bool use_var_cols,
    const bool use_contiguous_warps,
    const bool use_cache,
    const bool use_packed_rows) {
  if (group_size == 0) {
    return;
  }

  at::cuda::OptionalCUDAGuard device_guard(device);
  const bool use_sorted_indices = (sorted_indices_ptrs && reverse_indices_ptrs);

  // Partition work based on num_work_rows
  uint32_t num_warps_per_threadblock = kMaxThreads / EMULATED_WARP_SIZE;
  uint32_t max_grid_size =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8;
  uint32_t grid_size = std::min(
      cuda_calc_xblock_count(total_num_warps, num_warps_per_threadblock),
      max_grid_size);
  dim3 block_size(EMULATED_WARP_SIZE, num_warps_per_threadblock, 1);

  auto invoke_group_index_select_or_add = [&]<typename index_t,
                                                        typename scalar_t,
                                                        bool USE_INDEX_SELECT,
                                                        bool USE_VAR_COLS,
                                                        bool USE_CONTIGUOUS_WARPS,
                                                        bool USE_SORTED_INDICES,
                                                        bool USE_CACHE,
                                                        bool USE_PACKED_ROWS>() {
    FBGEMM_LAUNCH_KERNEL(
        (group_index_select_or_add_2d_kernel<
            index_t,
            scalar_t,
            USE_INDEX_SELECT,
            USE_VAR_COLS,
            USE_CONTIGUOUS_WARPS,
            USE_SORTED_INDICES,
            USE_CACHE,
            USE_PACKED_ROWS,
            GROUP_INDEX_SELECT_UNROLL_FACTOR,
            GROUP_INDEX_SELECT_COLS_PER_WARP,
            GROUP_INDEX_SELECT_LOG_COLS_PER_WARP>),
        grid_size,
        block_size,
        0,
        at::cuda::getCurrentCUDAStream(),
        input_ptrs,
        output_ptrs,
        use_sorted_indices ? sorted_indices_ptrs : indices_ptrs,
        reverse_indices_ptrs,
        warp_offsets_group,
        num_cols_group,
        num_work_rows,
        group_size);
  };
  
  using bool_variant_t = std::variant<std::true_type, std::false_type>;

// Split is needed to avoid additional code generation for unsupported
// algorithms on CUDA
#ifdef USE_ROCM
  using platform_bool_variant_t = std::variant<std::true_type, std::false_type>;
#else
  using platform_bool_variant_t = std::variant<std::false_type>;
#endif

  auto get_bool_type = [](const bool var) -> bool_variant_t {
      if (var) {
          return std::true_type{};
      } else {
          return std::false_type{};
      }
  };

  const bool_variant_t use_index_select_variant =
      get_bool_type(use_index_select);
  const bool_variant_t use_var_cols_variant = get_bool_type(use_var_cols);
#ifdef USE_ROCM
  const platform_bool_variant_t use_contiguous_warps_variant =
      get_bool_type(use_contiguous_warps);
  const platform_bool_variant_t use_sorted_indices_variant =
      get_bool_type(use_sorted_indices);
  const platform_bool_variant_t use_cache_variant = get_bool_type(use_cache);
  const platform_bool_variant_t use_packed_rows_variant =
      get_bool_type(use_packed_rows);
#else
  const platform_bool_variant_t use_contiguous_warps_variant{std::false_type{}};
  const platform_bool_variant_t use_sorted_indices_variant{std::false_type{}};
  const platform_bool_variant_t use_cache_variant{std::false_type{}};
  const platform_bool_variant_t use_packed_rows_variant{std::false_type{}};
#endif

  AT_DISPATCH_INDEX_TYPES(
      indices_scalar_type, "group_index_select_2d_wrapper_1", [&] {
        FBGEMM_DISPATCH_FLOATING_TYPES(
            input_scalar_type, "group_index_select_2d_wrapper_2", [&] {
              std::visit(
                  [&](auto use_index_select_arg, 
                      auto use_var_cols_arg,
                      auto use_contiguous_warps_arg,
                      auto use_sorted_indices_arg,
                      auto use_cache_arg,
                      auto use_packed_rows_arg) {
                    invoke_group_index_select_or_add.template operator()<
                        index_t, 
                        scalar_t, 
                        use_index_select_arg.value,
                        use_var_cols_arg.value, 
                        use_contiguous_warps_arg.value,
                        use_sorted_indices_arg.value,
                        use_cache_arg.value,
                        use_packed_rows_arg.value>();
                  },
                  use_index_select_variant, 
                  use_var_cols_variant,
                  use_contiguous_warps_variant, 
                  use_sorted_indices_variant, 
                  use_cache_variant,
                  use_packed_rows_variant);
            });
      });
}

} // namespace fbgemm_gpu
