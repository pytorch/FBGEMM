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

#include "fbgemm_gpu/utils/cuda_prelude.cuh"

namespace fbgemm_gpu::rocm {
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
} // namespace fbgemm_gpu::rocm
