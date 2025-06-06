/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

namespace {

template <typename index_t, typename offset_t>
__global__ __launch_bounds__(kMaxThreads) void linearize_cache_indices_kernel(
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<offset_t, 1, at::RestrictPtrTraits>
        table_offsets,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    const int64_t indices_base_offset) {
  const index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= indices.size(0)) {
    return;
  }

  // Perform binary search.
  int left = 0;
  int right = table_offsets.size(0);
  const auto index_with_offset = index + indices_base_offset;
  while (left != right) {
    const int middle =
        left + (right - left) / 2; // Avoid overflow in midpoint calculation
    if (table_offsets[middle] <= index_with_offset) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }
  const int table_index = left;

  const auto max_offset =
      ::__ldg(&cache_hash_size_cumsum[cache_hash_size_cumsum.size(0) - 1]);
  const auto curr_offset = ::__ldg(&cache_hash_size_cumsum[table_index]);
  if (curr_offset >= 0 && indices[index] >= 0) {
    linear_cache_indices[index] = indices[index] + curr_offset;
  } else {
    // Either table index is wrong, or index value is negative (due to pruning):
    // set it to invalid value.
    linear_cache_indices[index] = max_offset;
  }
}

} // namespace

DLL_PUBLIC Tensor linearize_cache_indices_cuda(
    const Tensor& cache_hash_size_cumsum,
    const Tensor& indices,
    const Tensor& offsets,
    const std::optional<Tensor>& B_offsets,
    const int64_t max_B,
    const int64_t indices_base_offset) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cache_hash_size_cumsum, indices, offsets);

  const auto T = cache_hash_size_cumsum.size(0) - 1;
  TORCH_CHECK(T > 0);
  const int32_t total_B = offsets.size(0) - 1;
  const auto num_indices = indices.numel();

  CUDA_DEVICE_GUARD(cache_hash_size_cumsum);

  auto linear_cache_indices =
      at::empty(indices.sizes(), indices.options().dtype(at::kLong));

  if (total_B == 0 || num_indices == 0) {
    return linear_cache_indices;
  }

  const auto vbe = B_offsets.has_value();

  Tensor table_offsets;
  if (vbe) {
    TORCH_CHECK(max_B >= 0, "Invalid max_B ", max_B, ". max_B must be >= 0");
    table_offsets =
        at::index_select(offsets, 0, B_offsets.value().slice(0, 1, T, 1));
  } else {
    // offsets = [B x T + 1]
    const auto B = total_B / T;
    TORCH_CHECK(
        B >= 0,
        "Invalid B ",
        B,
        ". Please check the size of offsets and cache_hash_size_cumsum.");
    table_offsets = offsets.slice(0, B, B * T, B);
  }

  AT_DISPATCH_INDEX_TYPES(
      table_offsets.scalar_type(), "linearize_cache_indices_kernel_1", [&] {
        using offset_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "linearize_cache_indices_kernel_2", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (linearize_cache_indices_kernel<index_t, offset_t>),
                  div_round_up(num_indices, kMaxThreads),
                  kMaxThreads,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(cache_hash_size_cumsum, int64_t, 1, 32),
                  PTA_B(indices, index_t, 1, 32),
                  PTA_B(table_offsets, offset_t, 1, 32),
                  PTA_B(linear_cache_indices, int64_t, 1, 32),
                  indices_base_offset);
            });
      });
  return linear_cache_indices;
}

namespace {

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void linearize_cache_indices_from_row_idx_kernel(
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        update_table_indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        update_row_indices,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices) {
  const index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= update_row_indices.size(0)) {
    return;
  }
  const int table_index = update_table_indices[index];

  const auto max_offset =
      ::__ldg(&cache_hash_size_cumsum[cache_hash_size_cumsum.size(0) - 1]);
  const auto curr_offset = ::__ldg(&cache_hash_size_cumsum[table_index]);
  if (curr_offset >= 0 && update_row_indices[index] >= 0) {
    linear_cache_indices[index] = update_row_indices[index] + curr_offset;
  } else {
    // Either table index is wrong, or index value is negative (due to pruning):
    // set it to invalid value.
    linear_cache_indices[index] = max_offset;
  }
}

} // namespace

DLL_PUBLIC Tensor linearize_cache_indices_from_row_idx_cuda(
    Tensor cache_hash_size_cumsum,
    Tensor update_table_indices,
    Tensor update_row_indices) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cache_hash_size_cumsum, update_table_indices, update_row_indices);

  CUDA_DEVICE_GUARD(cache_hash_size_cumsum);

  const auto T = cache_hash_size_cumsum.size(0) - 1;
  TORCH_CHECK(T > 0);

  auto linear_cache_indices = at::empty_like(update_row_indices);
  const auto num_indices = update_row_indices.numel();
  if (num_indices == 0) {
    return linear_cache_indices;
  }

  AT_DISPATCH_INDEX_TYPES(
      update_row_indices.scalar_type(),
      "linearize_cache_indices_from_row_idx_kernel",
      [&] {
        FBGEMM_LAUNCH_KERNEL(
            (linearize_cache_indices_from_row_idx_kernel<index_t>),
            div_round_up(num_indices, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(cache_hash_size_cumsum, int64_t, 1, 32),
            PTA_B(update_table_indices, index_t, 1, 32),
            PTA_B(update_row_indices, index_t, 1, 32),
            PTA_B(linear_cache_indices, index_t, 1, 32));
      });
  return linear_cache_indices;
}

DLL_PUBLIC
std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
get_unique_indices_cuda_impl(
    const Tensor& linear_indices,
    const int64_t max_indices,
    const bool compute_count,
    const bool compute_inverse_indices) {
  TENSOR_ON_CUDA_GPU(linear_indices);

  CUDA_DEVICE_GUARD(linear_indices);

  TORCH_CHECK(linear_indices.numel() < std::numeric_limits<int32_t>::max());
  const int32_t N = linear_indices.numel();
  auto sorted_indices = at::empty_like(linear_indices);
  auto unique_indices = at::empty_like(linear_indices);
  auto unique_indices_length =
      at::empty({1}, linear_indices.options().dtype(at::kInt));
  std::optional<Tensor> unique_indices_count = std::nullopt;
  std::optional<Tensor> linear_index_positions_sorted = std::nullopt;

  Tensor linear_index_positions;
  if (compute_inverse_indices) {
    linear_index_positions = at::arange(
        {linear_indices.numel()}, linear_indices.options().dtype(at::kInt));
    linear_index_positions_sorted = at::empty_like(linear_index_positions);
  }
  if (compute_count) {
    unique_indices_count = at::empty(
        {linear_indices.numel()}, linear_indices.options().dtype(at::kInt));
  }

#define INVOKE_CUB_SORT_PAIRS(TEMP_STORAGE_PTR)                           \
  AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs( \
      TEMP_STORAGE_PTR,                                                   \
      temp_storage_bytes,                                                 \
      linear_indices.data_ptr<index_t>(),                                 \
      sorted_indices.data_ptr<index_t>(),                                 \
      linear_index_positions.data_ptr<int32_t>(),                         \
      linear_index_positions_sorted->data_ptr<int32_t>(),                 \
      N,                                                                  \
      0,                                                                  \
      int(log2(float(max_indices + 1)) + 1),                              \
      at::cuda::getCurrentCUDAStream(),                                   \
      false))

#define INVOKE_CUB_SORT_KEYS(TEMP_STORAGE_PTR)                           \
  AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortKeys( \
      TEMP_STORAGE_PTR,                                                  \
      temp_storage_bytes,                                                \
      linear_indices.data_ptr<index_t>(),                                \
      sorted_indices.data_ptr<index_t>(),                                \
      N,                                                                 \
      0,                                                                 \
      int(log2(float(max_indices + 1)) + 1),                             \
      at::cuda::getCurrentCUDAStream(),                                  \
      false))

#define INVOKE_CUB_ENCODE(TEMP_STORAGE_PTR)                                  \
  AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode( \
      TEMP_STORAGE_PTR,                                                      \
      temp_storage_bytes,                                                    \
      sorted_indices.data_ptr<index_t>(),                                    \
      unique_indices.data_ptr<index_t>(),                                    \
      unique_indices_count->data_ptr<int32_t>(),                             \
      unique_indices_length.data_ptr<int32_t>(),                             \
      N,                                                                     \
      at::cuda::getCurrentCUDAStream(),                                      \
      false))

#define INVOKE_CUB_UNIQUE(TEMP_STORAGE_PTR)                         \
  AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceSelect::Unique( \
      TEMP_STORAGE_PTR,                                             \
      temp_storage_bytes,                                           \
      sorted_indices.data_ptr<index_t>(),                           \
      unique_indices.data_ptr<index_t>(),                           \
      unique_indices_length.data_ptr<int32_t>(),                    \
      N,                                                            \
      at::cuda::getCurrentCUDAStream(),                             \
      false))

  AT_DISPATCH_INDEX_TYPES(
      linear_indices.scalar_type(), "get_unique_indices_cuda", [&] {
        // sort indices
        if (compute_inverse_indices) {
          size_t temp_storage_bytes = 0;
          INVOKE_CUB_SORT_PAIRS(nullptr);
          auto temp_storage = at::empty(
              {static_cast<int64_t>(temp_storage_bytes)},
              linear_indices.options().dtype(at::kByte));
          INVOKE_CUB_SORT_PAIRS(temp_storage.data_ptr());
        } else {
          size_t temp_storage_bytes = 0;
          INVOKE_CUB_SORT_KEYS(nullptr);
          auto temp_storage = at::empty(
              {static_cast<index_t>(temp_storage_bytes)},
              linear_indices.options().dtype(at::kByte));
          INVOKE_CUB_SORT_KEYS(temp_storage.data_ptr());
        }
        // get unique indices
        if (compute_count) {
          size_t temp_storage_bytes = 0;
          INVOKE_CUB_ENCODE(nullptr);
          auto temp_storage = at::empty(
              {static_cast<index_t>(temp_storage_bytes)},
              linear_indices.options().dtype(at::kByte));
          INVOKE_CUB_ENCODE(temp_storage.data_ptr());
        } else {
          size_t temp_storage_bytes = 0;
          INVOKE_CUB_UNIQUE(nullptr);
          auto temp_storage = at::empty(
              {static_cast<index_t>(temp_storage_bytes)},
              linear_indices.options().dtype(at::kByte));
          INVOKE_CUB_UNIQUE(temp_storage.data_ptr());
        }
      });

  return std::make_tuple(
      unique_indices,
      unique_indices_length,
      unique_indices_count,
      linear_index_positions_sorted);

#undef INVOKE_CUB_SORT_PAIRS
#undef INVOKE_CUB_SORT_KEYS
#undef INVOKE_CUB_ENCODE
#undef INVOKE_CUB_UNIQUE
}

DLL_PUBLIC
std::tuple<Tensor, Tensor, std::optional<Tensor>> get_unique_indices_cuda(
    const Tensor& linear_indices,
    const int64_t max_indices,
    const bool compute_count) {
  const auto ret = get_unique_indices_cuda_impl(
      linear_indices,
      max_indices,
      compute_count,
      /*compute_inverse_indices=*/false);

  return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
}

DLL_PUBLIC
std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
get_unique_indices_with_inverse_cuda(
    const Tensor& linear_indices,
    const int64_t max_indices,
    const bool compute_count,
    const bool compute_inverse_indices) {
  return get_unique_indices_cuda_impl(
      linear_indices, max_indices, compute_count, compute_inverse_indices);
}
