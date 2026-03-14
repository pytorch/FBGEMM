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

std::tuple<at::Tensor, at::Tensor> sort_indices_with_rocprim(const at::Tensor& indices) {
    TORCH_CHECK(
        indices.dim() == 1,
        "sort_indices_with_rocprim expects a 1D tensor, got ",
        indices.dim());
    TORCH_CHECK(
        indices.is_cuda(),
        "sort_indices_with_rocprim expects a CUDA tensor for indices");

    CUDA_DEVICE_GUARD(indices);
    auto contiguous_indices = indices.contiguous();
    auto sorted_indices = at::empty_like(contiguous_indices);
    auto reverse_indices = at::empty(
        contiguous_indices.sizes(),
        contiguous_indices.options().dtype(at::kLong));
    auto original_positions = at::arange(
        contiguous_indices.numel(),
        contiguous_indices.options().dtype(at::kLong));

    const auto numel = contiguous_indices.numel();
    if (numel == 0) {
        return {sorted_indices, reverse_indices};
    }

    const auto num_items = static_cast<size_t>(numel);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto scalar_type = contiguous_indices.scalar_type();
    AT_DISPATCH_INTEGRAL_TYPES(
        scalar_type, "sort_indices_with_rocprim", [&] {
            using index_t = scalar_t;
            auto keys_in = contiguous_indices.data_ptr<index_t>();
            auto keys_out = sorted_indices.data_ptr<index_t>();
            auto values_in = original_positions.data_ptr<int64_t>();
            auto values_out = reverse_indices.data_ptr<int64_t>();

            size_t temp_storage_bytes = 0;
            // Selected empirically
            constexpr int k_merge_sort_threshold = 400'000;

            using sort_config = rocprim::radix_sort_config<
                rocprim::default_config,
                rocprim::default_config,
                rocprim::default_config,
                k_merge_sort_threshold>;
            AT_CUDA_CHECK(rocprim::radix_sort_pairs<sort_config>(
                nullptr,
                temp_storage_bytes,
                keys_in,
                keys_out,
                values_in,
                values_out,
                num_items,
                0,
                sizeof(index_t) * 8,
                stream,
                false));
            auto temp_storage = at::empty(
                {static_cast<int64_t>(temp_storage_bytes)},
                contiguous_indices.options().dtype(at::kByte));
            AT_CUDA_CHECK(rocprim::radix_sort_pairs<sort_config>(
                temp_storage.data_ptr(),
                temp_storage_bytes,
                keys_in,
                keys_out,
                values_in,
                values_out,
                num_items,
                0,
                sizeof(index_t) * 8,
                stream,
                false));
    });

    return {sorted_indices, reverse_indices};
}
} // namespace
} // namespace fbgemm_gpu::rocm
