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
        linear_cache_indices) {
  const index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= indices.size(0)) {
    return;
  }

  // Perform binary search.
  int left = 0;
  int right = table_offsets.size(0);
  while (left != right) {
    const int middle =
        left + (right - left) / 2; // Avoid overflow in midpoint calculation
    if (table_offsets[middle] <= index) {
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
    Tensor cache_hash_size_cumsum,
    Tensor indices,
    Tensor offsets) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cache_hash_size_cumsum, indices, offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cache_hash_size_cumsum.get_device());

  const auto T = cache_hash_size_cumsum.size(0) - 1;
  TORCH_CHECK(T > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B >= 0);

  auto linear_cache_indices =
      at::empty(indices.sizes(), indices.options().dtype(at::kLong));
  const auto num_indices = indices.numel();
  if (B == 0 || num_indices == 0) {
    return linear_cache_indices;
  }

  const auto table_offsets = offsets.slice(0, B, B * T, B);

  AT_DISPATCH_INDEX_TYPES(
      table_offsets.scalar_type(), "linearize_cache_indices_kernel_1", [&] {
        using offset_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "linearize_cache_indices_kernel_2", [&] {
#ifdef FBGEMM_GPU_MEMCHECK
              const char* func_name = "linearize_cache_indices_kernel";
#endif
              linearize_cache_indices_kernel<<<
                  div_round_up(num_indices, kMaxThreads),
                  kMaxThreads,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  MAKE_PTA_WITH_NAME(
                      func_name, cache_hash_size_cumsum, int64_t, 1, 32),
                  MAKE_PTA_WITH_NAME(func_name, indices, index_t, 1, 32),
                  MAKE_PTA_WITH_NAME(func_name, table_offsets, offset_t, 1, 32),
                  MAKE_PTA_WITH_NAME(
                      func_name, linear_cache_indices, int64_t, 1, 32));
              C10_CUDA_KERNEL_LAUNCH_CHECK();
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

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cache_hash_size_cumsum.get_device());

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
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "linearize_cache_indices_from_row_idx_kernel";
#endif
        linearize_cache_indices_from_row_idx_kernel<<<
            div_round_up(num_indices, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            MAKE_PTA_WITH_NAME(
                func_name, cache_hash_size_cumsum, int64_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, update_table_indices, index_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, update_row_indices, index_t, 1, 32),
            MAKE_PTA_WITH_NAME(
                func_name, linear_cache_indices, index_t, 1, 32));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return linear_cache_indices;
}

DLL_PUBLIC std::tuple<Tensor, Tensor, c10::optional<Tensor>>
get_unique_indices_cuda(
    Tensor linear_indices,
    int64_t max_indices,
    bool compute_count) {
  TENSOR_ON_CUDA_GPU(linear_indices);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(linear_indices.get_device());

  TORCH_CHECK(linear_indices.numel() < std::numeric_limits<int32_t>::max());
  const int32_t N = linear_indices.numel();
  auto sorted_indices = at::empty_like(linear_indices);
  auto unique_indices = at::empty_like(linear_indices);
  auto unique_indices_length =
      at::empty({1}, linear_indices.options().dtype(at::kInt));
  c10::optional<Tensor> unique_indices_count = c10::nullopt;
  if (compute_count) {
    unique_indices_count = at::empty(
        {linear_indices.numel()}, linear_indices.options().dtype(at::kInt));
  }
  AT_DISPATCH_INDEX_TYPES(
      linear_indices.scalar_type(), "get_unique_indices_cuda", [&] {
        // sort indices
        size_t temp_storage_bytes_0 = 0;
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortKeys(
            nullptr,
            temp_storage_bytes_0,
            linear_indices.data_ptr<index_t>(),
            sorted_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(max_indices + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage_0 = at::empty(
            {static_cast<index_t>(temp_storage_bytes_0)},
            linear_indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortKeys(
            temp_storage_0.data_ptr(),
            temp_storage_bytes_0,
            linear_indices.data_ptr<index_t>(),
            sorted_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(max_indices + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
        // get unique indices
        if (compute_count) {
          size_t temp_storage_bytes_1 = 0;
          AT_CUDA_CHECK(
              FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                  nullptr,
                  temp_storage_bytes_1,
                  sorted_indices.data_ptr<index_t>(),
                  unique_indices.data_ptr<index_t>(),
                  unique_indices_count->data_ptr<int32_t>(),
                  unique_indices_length.data_ptr<int32_t>(),
                  N,
                  at::cuda::getCurrentCUDAStream(),
                  false));
          auto temp_storage_1 = at::empty(
              {static_cast<index_t>(temp_storage_bytes_1)},
              linear_indices.options().dtype(at::kByte));
          AT_CUDA_CHECK(
              FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                  temp_storage_1.data_ptr(),
                  temp_storage_bytes_1,
                  sorted_indices.data_ptr<index_t>(),
                  unique_indices.data_ptr<index_t>(),
                  unique_indices_count->data_ptr<int32_t>(),
                  unique_indices_length.data_ptr<int32_t>(),
                  N,
                  at::cuda::getCurrentCUDAStream(),
                  false));
        } else {
          size_t temp_storage_bytes_1 = 0;
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceSelect::Unique(
              nullptr,
              temp_storage_bytes_1,
              sorted_indices.data_ptr<index_t>(),
              unique_indices.data_ptr<index_t>(),
              unique_indices_length.data_ptr<int32_t>(),
              N,
              at::cuda::getCurrentCUDAStream(),
              false));
          auto temp_storage_1 = at::empty(
              {static_cast<index_t>(temp_storage_bytes_1)},
              linear_indices.options().dtype(at::kByte));
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceSelect::Unique(
              temp_storage_1.data_ptr(),
              temp_storage_bytes_1,
              sorted_indices.data_ptr<index_t>(),
              unique_indices.data_ptr<index_t>(),
              unique_indices_length.data_ptr<int32_t>(),
              N,
              at::cuda::getCurrentCUDAStream(),
              false));
        }
      });
  return std::make_tuple(
      unique_indices, unique_indices_length, unique_indices_count);
}
