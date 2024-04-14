/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/cuda/CUDAGuard.h>
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/input_combine.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <typename src_t, typename dst_t, uint32_t VEC_WIDTH>
DEVICE_INLINE void vec_copy_with_implicit_type_cast(
    dst_t* const __restrict__ dst,
    const uint64_t src_addr,
    const uint64_t src_offset,
    const uint64_t dst_offset,
    const uint64_t src_bound) {
  // TODO: Use vector load/store if address aligns with the vector type
  const src_t* const src = reinterpret_cast<src_t*>(src_addr);
#pragma unroll
  for (uint64_t i = 0; i < VEC_WIDTH && src_offset + i < src_bound; i++) {
    dst[dst_offset + i] = src[src_offset + i];
  }
}

template <uint32_t VEC_WIDTH, uint32_t IS_LONG_NUM_BITS>
__global__
__launch_bounds__(kMaxThreads) void tbe_input_combine_with_length_kernel(
    int32_t* const __restrict__ combined_indices,
    int32_t* const __restrict__ combined_lengths,
    float* const __restrict__ combined_weights,
    const uint64_t* const __restrict__ indices_addrs,
    const uint64_t* const __restrict__ lengths_addrs,
    const uint64_t* const __restrict__ per_sample_weights_addrs,
    const uint32_t* const __restrict__ indices_is_long,
    const uint32_t* const __restrict__ lengths_is_long,
    const uint64_t* const __restrict__ indices_offsets,
    const uint64_t* const __restrict__ lengths_offsets,
    const uint64_t num_lists,
    const FixedDivisor fd_num_warps_per_list) {
  const auto global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  uint32_t list_id;
  uint32_t warp_id;
  fd_num_warps_per_list.DivMod(
      global_warp_id,
      reinterpret_cast<int32_t*>(&list_id),
      reinterpret_cast<int32_t*>(&warp_id));

  if (list_id >= num_lists) {
    return;
  }

  // IS_LONG_NUM_BITS is power of 2 (default = 32); div and mod should be cheap
  const uint32_t is_long_idx = list_id / IS_LONG_NUM_BITS;
  const uint32_t is_long_mask = 1u << (list_id % IS_LONG_NUM_BITS);
  const uint64_t src_idx = (warp_id * kWarpSize + threadIdx.x) * VEC_WIDTH;
  const auto indices_start = indices_offsets[list_id];
  const auto indices_end = indices_offsets[list_id + 1];
  const auto lengths_start = lengths_offsets[list_id];
  const auto lengths_end = lengths_offsets[list_id + 1];

  // Invoke a function based on the indices type
  ((indices_is_long[is_long_idx] & is_long_mask)
       ? vec_copy_with_implicit_type_cast<int64_t, int32_t, VEC_WIDTH>
       : vec_copy_with_implicit_type_cast<
             int32_t,
             int32_t,
             VEC_WIDTH>)(combined_indices,
                         indices_addrs[list_id],
                         src_idx,
                         indices_start + src_idx,
                         indices_end - indices_start);

  // Invoke a function based on the lengths type
  ((lengths_is_long[is_long_idx] & is_long_mask)
       ? vec_copy_with_implicit_type_cast<int64_t, int32_t, VEC_WIDTH>
       : vec_copy_with_implicit_type_cast<
             int32_t,
             int32_t,
             VEC_WIDTH>)(combined_lengths,
                         lengths_addrs[list_id],
                         src_idx,
                         lengths_start + src_idx,
                         lengths_end - lengths_start);

  if (per_sample_weights_addrs) {
    vec_copy_with_implicit_type_cast<float, float, VEC_WIDTH>(
        combined_weights,
        per_sample_weights_addrs[list_id],
        src_idx,
        indices_start + src_idx,
        indices_end - indices_start);
  }
}

std::tuple<Tensor, Tensor, Tensor> tbe_input_combine_with_length_cuda(
    const uint64_t* const indices_addrs,
    const uint64_t* const lengths_addrs,
    const uint64_t* const per_sample_weights_addrs,
    const uint32_t* const indices_is_long,
    const uint32_t* const lengths_is_long,
    const uint64_t* const indices_offsets,
    const uint64_t* const lengths_offsets,
    const uint64_t num_lists,
    const uint64_t total_indices,
    const uint64_t total_lengths,
    const uint64_t max_list_size,
    const c10::DeviceIndex& device) {
  constexpr uint32_t IS_LONG_NUM_BITS = 32;
  at::cuda::OptionalCUDAGuard device_guard(device);

  // combined_indices and combined_lengths are int tensors
  const auto int_options = at::TensorOptions().dtype(at::kInt).device(
      at::kCUDA, at::cuda::current_device());
  Tensor combined_indices =
      at::empty({static_cast<int64_t>(total_indices)}, int_options);
  Tensor combined_lengths =
      at::empty({static_cast<int64_t>(total_lengths)}, int_options);
  // combined_weights is a float tensor
  Tensor combined_weights = at::empty(
      {per_sample_weights_addrs ? static_cast<int64_t>(total_indices)
                                : static_cast<int64_t>(0)},
      at::TensorOptions()
          .dtype(at::kFloat)
          .device(at::kCUDA, at::cuda::current_device()));

  // Each thread loads 4 elements (rule of thumb; should work well with 32-bit
  // inputs)
  constexpr uint32_t VEC_WIDTH = 4;
  constexpr uint32_t NUM_WARPS_PER_BLOCK = kMaxThreads / kWarpSize;
  const auto num_warps_per_list =
      div_round_up(max_list_size, kWarpSize * VEC_WIDTH);
  const auto num_blocks =
      div_round_up(num_warps_per_list * num_lists, NUM_WARPS_PER_BLOCK);

  tbe_input_combine_with_length_kernel<VEC_WIDTH, IS_LONG_NUM_BITS>
      <<<num_blocks,
         dim3(kWarpSize, NUM_WARPS_PER_BLOCK),
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          combined_indices.data_ptr<int32_t>(),
          combined_lengths.data_ptr<int32_t>(),
          per_sample_weights_addrs ? combined_weights.data_ptr<float>()
                                   : nullptr,
          indices_addrs,
          lengths_addrs,
          per_sample_weights_addrs,
          indices_is_long,
          lengths_is_long,
          indices_offsets,
          lengths_offsets,
          num_lists,
          FixedDivisor(num_warps_per_list));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      std::move(combined_indices),
      std::move(combined_lengths),
      std::move(combined_weights)};
}

} // namespace fbgemm_gpu
