/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm_gpu/split_embeddings_utils.cuh"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

inline at::Tensor asynchronous_complete_cumsum(at::Tensor t_in) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK_LT(t_in.numel(), std::numeric_limits<int32_t>::max());
  TORCH_CHECK_EQ(t_in.dim(), 1);
  auto t_out = at::empty({t_in.numel() + 1}, t_in.options());
  t_out[0].zero_();
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      t_in.options().dtype(at::kByte));
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper2", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  return t_out;
}

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void linearize_index_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> infos,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices) {
  const int32_t T = hash_size_cumsum.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;
  const int32_t b_t = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  const bool valid = t < T;

  const index_t hash_offset = valid ? hash_size_cumsum[t] : -1;
  const index_t indices_start = valid ? offsets[t * B + b] : -1;
  const int32_t L = valid ? offsets[t * B + b + 1] - indices_start : 0;
  const int32_t lane_id = threadIdx.x % fbgemm_gpu::kWarpSize;

  for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
    const index_t indices_start_warp = fbgemm_gpu::shfl_sync(indices_start, j);
    const int32_t b_t_warp = fbgemm_gpu::shfl_sync(b_t, j);
    const int32_t L_warp = fbgemm_gpu::shfl_sync(L, j);
    const index_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
    for (int32_t i = lane_id; i < L_warp; i += fbgemm_gpu::kWarpSize) {
      const index_t idx = __ldg(&indices[indices_start_warp + i]);
      infos[indices_start_warp + i] = b_t_warp;
      linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
    }
  }
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void nobag_linearize_index_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> infos,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices) {
  const int32_t T = hash_size_cumsum.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;
  const int32_t b_t = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  const bool valid = t < T;

  const index_t hash_offset = valid ? hash_size_cumsum[t] : -1;
  const index_t indices_start = valid ? offsets[t * B + b] : -1;
  const int32_t L = valid ? offsets[t * B + b + 1] - indices_start : 0;
  const int32_t lane_id = threadIdx.x % fbgemm_gpu::kWarpSize;

  for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
    const index_t indices_start_warp = fbgemm_gpu::shfl_sync(indices_start, j);
    const int32_t t_warp = fbgemm_gpu::shfl_sync(t, j);
    const int32_t L_warp = fbgemm_gpu::shfl_sync(L, j);
    const index_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
    for (int32_t i = lane_id; i < L_warp; i += fbgemm_gpu::kWarpSize) {
      const index_t idx = __ldg(&indices[indices_start_warp + i]);
      const int64_t l_t = (indices_start_warp + i) * T + t_warp;
      infos[indices_start_warp + i] = l_t;
      linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
    }
  }
}

std::tuple<
    Tensor /*linear_indices*/,
    Tensor /*linear_indices_sorted*/,
    Tensor /*infos_sorted*/,
    Tensor /*sorted_linear_indices_run*/,
    Tensor /*sorted_linear_indices_run_lengths*/,
    Tensor /*sorted_linear_indices_num_runs*/,
    Tensor /*sorted_linear_indices_cumulative_run_lengths*/>
transpose_embedding_input(
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    bool nobag,
    const c10::optional<Tensor>& vbe_b_t_map,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask) {
  const int32_t T = hash_size_cumsum.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;

  auto infos = at::empty_like(
      indices, indices.options().dtype(nobag ? at::kLong : at::kInt));
  auto infos_sorted = at::empty_like(infos);
  auto linear_indices = at::empty_like(indices);
  auto linear_indices_sorted = at::empty_like(indices);

  Tensor sorted_linear_indices_run;
  Tensor sorted_linear_indices_run_lengths;
  Tensor sorted_linear_indices_num_runs;

  using at::RestrictPtrTraits;

  AT_DISPATCH_INDEX_TYPES(
      infos.scalar_type(), "transpose_embedding_input1", [&] {
        using info_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "transpose_embedding_input2", [&] {
              if (!nobag) {
                linearize_index_kernel<<<
                    div_round_up(B * T, kMaxThreads),
                    kMaxThreads,
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    hash_size_cumsum
                        .packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    infos.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    linear_indices
                        .packed_accessor32<index_t, 1, RestrictPtrTraits>());
              } else {
                nobag_linearize_index_kernel<<<
                    div_round_up(B * T, kMaxThreads),
                    kMaxThreads,
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    hash_size_cumsum
                        .packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    infos.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                    linear_indices
                        .packed_accessor32<index_t, 1, RestrictPtrTraits>());
              }
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              {
                size_t temp_storage_bytes = 0;
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
                        nullptr,
                        temp_storage_bytes,
                        linear_indices.data_ptr<index_t>(),
                        linear_indices_sorted.data_ptr<index_t>(),
                        infos.data_ptr<info_t>(),
                        infos_sorted.data_ptr<info_t>(),
                        linear_indices.numel(),
                        0,
                        total_hash_size_bits,
                        at::cuda::getCurrentCUDAStream(),
                        false));
                auto temp_storage = at::empty(
                    {static_cast<int64_t>(temp_storage_bytes)},
                    indices.options().dtype(at::kByte));
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
                        temp_storage.data_ptr(),
                        temp_storage_bytes,
                        linear_indices.data_ptr<index_t>(),
                        linear_indices_sorted.data_ptr<index_t>(),
                        infos.data_ptr<info_t>(),
                        infos_sorted.data_ptr<info_t>(),
                        linear_indices.numel(),
                        0,
                        total_hash_size_bits,
                        at::cuda::getCurrentCUDAStream(),
                        false));
              }

              sorted_linear_indices_run = at::empty_like(indices);
              sorted_linear_indices_run_lengths =
                  at::zeros_like(indices, indices.options().dtype(at::kInt));
              sorted_linear_indices_num_runs =
                  at::zeros({1}, indices.options().dtype(at::kInt));

              {
                size_t temp_storage_bytes = 0;
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                        nullptr,
                        temp_storage_bytes,
                        linear_indices_sorted.data_ptr<index_t>(),
                        sorted_linear_indices_run.data_ptr<index_t>(),
                        sorted_linear_indices_run_lengths.data_ptr<int32_t>(),
                        sorted_linear_indices_num_runs.data_ptr<int32_t>(),
                        linear_indices_sorted.numel(),
                        at::cuda::getCurrentCUDAStream()));
                // Allocate temporary storage
                auto temp_storage = at::empty(
                    {static_cast<int64_t>(temp_storage_bytes)},
                    indices.options().dtype(at::kByte));
                // Run encoding
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                        temp_storage.data_ptr(),
                        temp_storage_bytes,
                        linear_indices_sorted.data_ptr<index_t>(),
                        sorted_linear_indices_run.data_ptr<index_t>(),
                        sorted_linear_indices_run_lengths.data_ptr<int32_t>(),
                        sorted_linear_indices_num_runs.data_ptr<int32_t>(),
                        linear_indices_sorted.numel(),
                        at::cuda::getCurrentCUDAStream()));
              }
            });
      });

  auto sorted_linear_indices_cumulative_run_lengths =
      asynchronous_complete_cumsum(sorted_linear_indices_run_lengths);

  return {
      linear_indices,
      linear_indices_sorted,
      infos_sorted,
      sorted_linear_indices_run,
      sorted_linear_indices_run_lengths,
      sorted_linear_indices_num_runs,
      sorted_linear_indices_cumulative_run_lengths};
}

#define DEF_RADIX_SORT_PAIRS_FN(KeyT, ValueT)                        \
  cudaError_t radix_sort_pairs(                                      \
      void* d_temp_storage,                                          \
      size_t& temp_storage_bytes,                                    \
      const KeyT* d_keys_in,                                         \
      KeyT* d_keys_out,                                              \
      const ValueT* d_values_in,                                     \
      ValueT* d_values_out,                                          \
      const int num_items,                                           \
      const int begin_bit,                                           \
      const int end_bit,                                             \
      cudaStream_t stream,                                           \
      const bool debug_synchronous) {                                \
    return FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs( \
        d_temp_storage,                                              \
        temp_storage_bytes,                                          \
        d_keys_in,                                                   \
        d_keys_out,                                                  \
        d_values_in,                                                 \
        d_values_out,                                                \
        num_items,                                                   \
        begin_bit,                                                   \
        end_bit,                                                     \
        stream,                                                      \
        debug_synchronous);                                          \
  }

DEF_RADIX_SORT_PAIRS_FN(int64_t, float);
DEF_RADIX_SORT_PAIRS_FN(int64_t, double);
DEF_RADIX_SORT_PAIRS_FN(int64_t, int64_t);
DEF_RADIX_SORT_PAIRS_FN(int64_t, int32_t);
