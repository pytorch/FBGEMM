/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include "fbgemm_gpu/input_combine.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

constexpr uint32_t IS_LONG_NUM_BITS = 32;
constexpr uint32_t NUM_ARGS = 7;
enum args_pos {
  P_indices_prts = 0,
  P_lengths_addrs = 1,
  P_indices_offsets = 2,
  P_lengths_offsets = 3,
  P_per_sample_weight = 4,
  P_indices_is_long = 5,
  P_lengths_is_long = 6
};

template <typename T>
uint64_t compute_num_uint64s(const uint64_t num_elements) {
  const uint64_t ratio = sizeof(uint64_t) / sizeof(T);
  return (num_elements + ratio - 1) / ratio;
}

void offset_tbe_input_combine_with_length_args(
    uint64_t** indices_addrs,
    uint64_t** lengths_addrs,
    uint64_t** indices_offsets,
    uint64_t** lengths_offsets,
    uint64_t** per_sample_weights_addrs,
    uint32_t** indices_is_long,
    uint32_t** lengths_is_long,
    uint64_t* base_addr,
    const uint64_t* const ptr_offsets,
    const bool need_weights) {
  *indices_addrs = base_addr + ptr_offsets[P_indices_prts];
  *lengths_addrs = base_addr + ptr_offsets[P_lengths_addrs];
  *indices_offsets = base_addr + ptr_offsets[P_indices_offsets];
  *lengths_offsets = base_addr + ptr_offsets[P_lengths_offsets];
  *per_sample_weights_addrs =
      need_weights ? (base_addr + ptr_offsets[P_per_sample_weight]) : nullptr;
  *indices_is_long =
      reinterpret_cast<uint32_t*>(base_addr + ptr_offsets[P_indices_is_long]);
  *lengths_is_long =
      reinterpret_cast<uint32_t*>(base_addr + ptr_offsets[P_lengths_is_long]);
}

std::tuple<Tensor, Tensor, Tensor> tbe_input_combine_with_length_gpu(
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& lengths_list,
    const std::vector<Tensor>& per_sample_weights) {
  const auto num_lists = indices_list.size();
  TORCH_CHECK_GT(num_lists, 0);
  TORCH_CHECK_EQ(lengths_list.size(), num_lists);
  TORCH_CHECK_EQ(per_sample_weights.size(), num_lists);
  const bool need_weights = std::any_of(
      per_sample_weights.begin(), per_sample_weights.end(), [](const auto& x) {
        return x.numel() > 0;
      });

  // Store is_longs in 32-bit variables. i-th bit (LSB) indicates if
  // list i-th is long.
  const uint64_t num_is_longs =
      (num_lists + IS_LONG_NUM_BITS - 1) / IS_LONG_NUM_BITS;
  const uint64_t num_is_longs_64 = compute_num_uint64s<uint32_t>(num_is_longs);
  // args_tensor stores kernel arguments:
  // - indices_prts (num_lists uint64_t elements)
  // - lengths_addrs (num_lists uint64_t elements)
  // - indices_offsets (num_lists + 1 uint64_t elements)
  // - lengths_offsets (num_lists + 1 uint64_t elements)
  // - per_sample_weight (num_lists uint64_t elements; optional)
  // - indices_is_long (num_is_longs uint32_t elements)
  // - lengths_is_long (num_is_longs uint32_t elements)
  uint64_t args_offsets[NUM_ARGS + 1];
  // Initialize offsets with lengths first
  args_offsets[P_indices_prts] = num_lists;
  args_offsets[P_lengths_addrs] = num_lists;
  args_offsets[P_indices_offsets] = num_lists + 1;
  args_offsets[P_lengths_offsets] = num_lists + 1;
  args_offsets[P_per_sample_weight] = need_weights ? num_lists : 0;
  args_offsets[P_indices_is_long] = num_is_longs_64;
  args_offsets[P_lengths_is_long] = num_is_longs_64;

  // Compute offsets
  uint64_t offset = 0;
  auto next = args_offsets[0];
  for (const auto i : c10::irange(NUM_ARGS)) {
    args_offsets[i] = offset;
    offset += next;
    next = args_offsets[i + 1];
  }
  args_offsets[NUM_ARGS] = offset; // total number of uint64_t elements required

  Tensor args_tensor = at::empty(
      {static_cast<int64_t>(args_offsets[NUM_ARGS] * sizeof(uint64_t))},
      at::TensorOptions().dtype(at::kByte).pinned_memory(true));

  uint64_t* indices_addrs = nullptr;
  uint64_t* lengths_addrs = nullptr;
  uint64_t* indices_offsets = nullptr;
  uint64_t* lengths_offsets = nullptr;
  uint64_t* per_sample_weights_addrs = nullptr;
  uint32_t* indices_is_long = nullptr;
  uint32_t* lengths_is_long = nullptr;

  // Offset host pointers
  offset_tbe_input_combine_with_length_args(
      &indices_addrs,
      &lengths_addrs,
      &indices_offsets,
      &lengths_offsets,
      &per_sample_weights_addrs,
      &indices_is_long,
      &lengths_is_long,
      reinterpret_cast<uint64_t*>(args_tensor.data_ptr()),
      args_offsets,
      need_weights);

  const auto& indices_0 = indices_list[0];
  uint64_t total_indices = 0;
  uint64_t total_lengths = 0;
  uint64_t max_list_size = 0;
  for (const auto i : c10::irange(num_lists)) {
    const uint64_t is_long_idx = i / IS_LONG_NUM_BITS;
    auto& indices_is_long_ = indices_is_long[is_long_idx];
    auto& lengths_is_long_ = lengths_is_long[is_long_idx];
    if (i % IS_LONG_NUM_BITS == 0) {
      indices_is_long_ = 0;
      lengths_is_long_ = 0;
    }
    const auto& indices = indices_list[i];
    const auto& lengths = lengths_list[i];
    TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(indices);
    TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(lengths);
    TENSORS_ON_SAME_DEVICE(indices, indices_0);
    TENSORS_ON_SAME_DEVICE(lengths, indices_0);
    TORCH_CHECK(indices.dtype() == c10::kInt || indices.dtype() == c10::kLong);
    TORCH_CHECK(lengths.dtype() == c10::kInt || lengths.dtype() == c10::kLong);
    TENSOR_NDIM_EQUALS(indices, 1);
    TENSOR_NDIM_EQUALS(lengths, 1);

    const auto indices_numel = indices.numel();
    const auto lengths_numel = lengths.numel();
    indices_offsets[i] = total_indices;
    lengths_offsets[i] = total_lengths;
    total_indices += indices_numel;
    total_lengths += lengths_numel;
    max_list_size =
        std::max(max_list_size, static_cast<uint64_t>(indices_numel));
    max_list_size =
        std::max(max_list_size, static_cast<uint64_t>(lengths_numel));

    // Store pointers in args_tensor
    indices_addrs[i] = reinterpret_cast<uint64_t>(indices.data_ptr());
    lengths_addrs[i] = reinterpret_cast<uint64_t>(lengths.data_ptr());
    indices_is_long_ |= static_cast<uint32_t>(indices.dtype() == c10::kLong)
        << (i % IS_LONG_NUM_BITS);
    lengths_is_long_ |= static_cast<uint32_t>(lengths.dtype() == c10::kLong)
        << (i % IS_LONG_NUM_BITS);

    const auto& weights = per_sample_weights[i];
    if (weights.numel() > 0) {
      TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(weights);
      TENSORS_ON_SAME_DEVICE(weights, indices_0);
      TENSOR_TYPE_MUST_BE(weights, c10::kFloat);
      TENSOR_NDIM_EQUALS(weights, 1);
      TENSORS_HAVE_SAME_NUMEL(weights, indices);

      per_sample_weights_addrs[i] =
          reinterpret_cast<uint64_t>(weights.data_ptr());
    }
  }
  indices_offsets[num_lists] = total_indices;
  lengths_offsets[num_lists] = total_lengths;

  const auto& device = indices_0.device();
  // Transfer args_tensor from host to device
  args_tensor = args_tensor.to(device, /*non_blocking=*/true);

  // Offset device pointers
  offset_tbe_input_combine_with_length_args(
      &indices_addrs,
      &lengths_addrs,
      &indices_offsets,
      &lengths_offsets,
      &per_sample_weights_addrs,
      &indices_is_long,
      &lengths_is_long,
      reinterpret_cast<uint64_t*>(args_tensor.data_ptr()),
      args_offsets,
      need_weights);

  return tbe_input_combine_with_length_cuda(
      indices_addrs,
      lengths_addrs,
      per_sample_weights_addrs,
      indices_is_long,
      lengths_is_long,
      indices_offsets,
      lengths_offsets,
      num_lists,
      total_indices,
      total_lengths,
      max_list_size,
      device.index());
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA(
      "tbe_input_combine_with_length",
      fbgemm_gpu::tbe_input_combine_with_length_gpu);
};

} // namespace fbgemm_gpu
