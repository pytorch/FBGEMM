/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <limits>

#include <ATen/Dispatch.h>
#include <ATen/ATen.h>

#include <hip/hip_runtime.h>
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/device/device_segmented_radix_sort.hpp>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/function_types.h"

namespace fbgemm_gpu::rocm {
// Selected empirically: rocprim uses merge sort when num_items < this threshold,
// which is faster for small inputs. Must match across sizing and sort calls.
constexpr unsigned int k_sort_merge_threshold = 400'000;
using sort_config = rocprim::radix_sort_config<
    rocprim::default_config,
    rocprim::default_config,
    rocprim::default_config,
    k_sort_merge_threshold>;

namespace {

template <typename scalar_t, int kLogicalWarpSize = kWarpSize>
__device__ __forceinline__ void warp_upper_bound(
    int* found,
    scalar_t* cached_boundary,
    const scalar_t* arr,
    const scalar_t target,
    const int num_entries) {
  const auto active_mask = __activemask();
  using mask_t = std::remove_const_t<decltype(active_mask)>;

  constexpr int kHardwareWarpSize = kWarpSize;
  constexpr int kMaskBits = sizeof(mask_t) * 8;

  const int hardware_lane = __lane_id();
  const int logical_lane = hardware_lane % kLogicalWarpSize;
  const int logical_warp_id = hardware_lane / kLogicalWarpSize;

  mask_t logical_mask = mask_t(0);
  if constexpr (kLogicalWarpSize >= kMaskBits) {
    logical_mask = active_mask;
  } else {
    const mask_t group_bits = (mask_t(1) << kLogicalWarpSize) - 1;
    const mask_t group_mask = group_bits
        << (logical_warp_id * kLogicalWarpSize);
    logical_mask = group_mask & active_mask;
  }
  if (!logical_mask) {
    logical_mask = active_mask;
  }

  int result = -1;
  scalar_t cached_result = *cached_boundary;
  for (int base = 0; base < num_entries; base += kLogicalWarpSize) {
    const int idx = base + logical_lane;
    const bool valid = idx < num_entries;
    const scalar_t val = valid ? arr[idx] : scalar_t(0);
    const mask_t ballot = __ballot_sync(logical_mask, valid && val > target);
    const mask_t logical_ballot = ballot & logical_mask;
    if (logical_ballot) {
      const int first_lane_hw =
          __ffsll(static_cast<long long>(logical_ballot)) - 1;
      const int first_lane = first_lane_hw - logical_warp_id * kLogicalWarpSize;
      result = base + first_lane;
      cached_result = arr[result];
      break;
    }
  }

  *found = result;
  *cached_boundary = cached_result;
}
} // namespace

// Returns temp storage size for a single-segment sort of num_items elements.
size_t get_sort_temp_storage_bytes(
    const size_t num_items,
    const c10::ScalarType scalar_type,
    const at::cuda::CUDAStream& stream);
// Returns temp storage size for segmented sort of num_groups segments each
// with num_items_per_segment elements.
size_t get_segmented_sort_temp_storage_bytes(
    const size_t num_items_per_segment,
    const int64_t num_groups,
    const c10::ScalarType scalar_type,
    const at::cuda::CUDAStream& stream);
// Sort all groups' indices with one rocprim::segmented_radix_sort_pairs call,
// eliminating all per-group CPU launch overhead.
//
// Inputs must be contiguous across groups:
//   all_keys_in    : [num_groups * num_items_per_segment] — packed input indices
//   all_values_in  : [num_groups * num_items_per_segment] — tiled 0..N-1 per segment
//   segment_offsets: [num_groups + 1] device tensor — [0, N, 2N, ..., K*N]
//   all_keys_out / all_values_out: pre-allocated output buffers (same shape)
//   temp_storage   : pre-allocated via get_segmented_sort_temp_storage_bytes()
void sort_indices_segmented_rocprim(
    const at::Tensor& all_keys_in,
    at::Tensor& all_keys_out,
    const at::Tensor& all_values_in,
    at::Tensor& all_values_out,
    const at::Tensor& segment_offsets,
    const size_t num_items_per_segment,
    const int64_t num_groups,
    at::Tensor& temp_storage,
    const at::cuda::CUDAStream& stream);
// Sort all groups in a batch with one AT_DISPATCH and one stream lookup.
// Uses radix_sort_pairs<sort_config> per group, preserving the merge sort
// fallback for small segment sizes (num_items < k_sort_merge_threshold).
void sort_indices_batch_rocprim(
    const int64_t* keys_in_ptrs,
    void* keys_out_base,
    int64_t* values_out_base,
    const int64_t* values_in,
    const size_t num_items,
    const int64_t num_groups,
    at::Tensor& temp_storage,
    const c10::ScalarType scalar_type,
    const at::cuda::CUDAStream& stream);
} // namespace fbgemm_gpu::rocm
