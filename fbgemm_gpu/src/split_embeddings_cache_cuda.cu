/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <limits>
#include <mutex>

#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/ops_utils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

constexpr size_t kCacheMaxThreads = 512;

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

namespace {

// // TODO: do we care about 64-bit indices? Currently we just ignore.
// __host__ DEVICE_INLINE uint32_t cache_slot(int32_t h_in, int32_t C) {
//   // MurmorHash3 32-bit mixing function.
//   uint32_t h = (uint32_t)h_in;
//   h ^= h >> 16;
//   h *= 0x85ebca6b;
//   h ^= h >> 13;
//   h *= 0xc2b2ae35;
//   h ^= h >> 16;
//   //
//   https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
//   return ((uint64_t)h * (uint64_t)C) >> 32;
// }

__host__ DEVICE_INLINE uint32_t
cache_slot(const int64_t h_in, const int32_t C) {
  // MurmurHash3 64-bit mixing function.
  uint64_t h = (uint64_t)h_in;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccd;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53;
  h ^= h >> 33;

  return h % (uint32_t)C;
}

enum uvm_cache_stats_index {
  num_calls = 0,
  num_requested_indices = 1,
  num_unique_indices = 2,
  num_unique_misses = 3,
  num_conflict_unique_misses = 4,
  num_conflict_misses = 5,
};

// Experiments showed that performance of lru/lxu_cache_find_uncached_kernel is
// not sensitive to grid size as long as the number thread blocks per SM is not
// too small nor too big.
constexpr int MAX_THREAD_BLOCKS_PER_SM_FOR_CACHE_KERNELS = 16;

int get_max_thread_blocks_for_cache_kernels_() {
  cudaDeviceProp* deviceProp =
      at::cuda::getDeviceProperties(c10::cuda::current_device());
  return deviceProp->multiProcessorCount *
      MAX_THREAD_BLOCKS_PER_SM_FOR_CACHE_KERNELS;
}

} // namespace

DLL_PUBLIC int64_t host_lxu_cache_slot(int64_t h_in, int64_t C) {
  return static_cast<int64_t>(cache_slot(h_in, static_cast<int32_t>(C)));
}

namespace {

constexpr int32_t kCacheLocationMissing = -1;
constexpr int64_t kCacheStateInvalid = -1;

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_flush_kernel(
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  const int32_t B = lxu_cache_weights.size(0);
  const int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  const int32_t slot = b % kWarpSize;
  const int32_t cache_set = b / kWarpSize;
  const int64_t current_idx = lxu_cache_state[cache_set][slot];
  if (current_idx != static_cast<int64_t>(kCacheStateInvalid)) {
    // evict from slot to backing storage
    const int32_t t_current = cache_index_table_map[current_idx];
    const int64_t idx_current = current_idx - cache_hash_size_cumsum[t_current];
    const int64_t weights_offset_current = weights_offsets[t_current];
    const int32_t D_start_current = D_offsets[t_current];
    const int32_t D_end_current = D_offsets[t_current + 1];
    const int32_t D_current = D_end_current - D_start_current;

    int32_t D_emb = D_current;
    if (std::is_same<emb_t, uint8_t>::value) {
      D_emb += kINT8QparamsBytes;
    }
    auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
        &weights[weights_offset_current + idx_current * D_emb + 0],
        &lxu_cache_weights[b][0],
        D_current,
        nullptr);
    if (!std::is_same<emb_t, float>::value && stochastic_rounding) {
      StochasticRoundingRNGState state;
      // different for every *run* and every *thread*.
      auto stochastic_rounding_seeds =
          at::cuda::philox::unpack(stochastic_rounding_philox_args);
      stochastic_rounding_init(
          std::get<0>(stochastic_rounding_seeds) ^
              std::get<1>(stochastic_rounding_seeds),
          blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
              threadIdx.x,
          &state);
      weight_row.set_stoc_state(&state);
    }

    float2 qparams;
    if (std::is_same<emb_t, uint8_t>::value) {
      qparams =
          thrust_find_qparams<cache_t>(&lxu_cache_weights[b][0], D_current);
      if (threadIdx.x == 0) {
        weight_row.store_qparams(qparams);
      }
    }
    for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
      Vec4T<at::acc_type<cache_t, true>> cache_weights_vec =
          weight_row.load(d * 4, qparams);
      weight_row.evict(cache_weights_vec, d * 4, qparams);
    }
  }
}

} // namespace

DLL_PUBLIC void lxu_cache_flush_cuda(
    Tensor uvm_weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    bool stochastic_rounding) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      uvm_weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      lxu_cache_state,
      lxu_cache_weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lxu_cache_weights.get_device());

  const int32_t T = D_offsets.numel() - 1;
  const int32_t S = lxu_cache_weights.size(0);
  const int32_t tx = std::min<int32_t>(total_D / 4 / T, kMaxThreads);
  const dim3 threads(tx, kMaxThreads / tx);
  const dim3 blocks(div_round_up(S, kMaxThreads / tx));

  DISPATCH_EMB_CACHE_TYPES(
      uvm_weights.scalar_type(),
      lxu_cache_weights.scalar_type(),
      "lxu_cache_flush_kernel_2",
      ([&] {
        at::PhiloxCudaState rng_engine_inputs;
        if (stochastic_rounding && std::is_same<emb_t, at::Half>::value) {
          auto gen = at::cuda::detail::getDefaultCUDAGenerator();
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)
                                  ->philox_cuda_state(4);
        }
        lxu_cache_flush_kernel<emb_t, cache_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                uvm_weights
                    .packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                cache_hash_size_cumsum
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                cache_index_table_map
                    .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
                weights_offsets
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                lxu_cache_state
                    .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
                lxu_cache_weights
                    .packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                stochastic_rounding,
                rng_engine_inputs);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

namespace {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void linearize_cache_indices_kernel(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        table_offsets,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
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

  auto linear_cache_indices = at::empty_like(indices);
  const auto num_indices = indices.numel();
  if (B == 0 || num_indices == 0) {
    return linear_cache_indices;
  }

  auto table_offsets = offsets.slice(0, B, B * T, B);
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "linearize_cache_indices_kernel", [&] {
        linearize_cache_indices_kernel<<<
            div_round_up(num_indices, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            cache_hash_size_cumsum
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            table_offsets
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            linear_cache_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return linear_cache_indices;
}

namespace {

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void linearize_cache_indices_from_row_idx_kernel(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        update_table_indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        update_row_indices,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
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
        linearize_cache_indices_from_row_idx_kernel<<<
            div_round_up(num_indices, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            cache_hash_size_cumsum
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            update_table_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            update_row_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            linear_cache_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>());
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

namespace {

__global__ __launch_bounds__(kMaxThreads) void emulate_cache_miss_kernel(
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const int64_t enforced_misses_per_256,
    const bool gather_cache_stats,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats) {
  const int32_t N = lxu_cache_locations.size(0);
  int64_t n_enforced_misses = 0;
  CUDA_KERNEL_LOOP(n, N) {
    if ((n & 0x00FF) < enforced_misses_per_256) {
      if (lxu_cache_locations[n] >= 0) {
        n_enforced_misses++;
      }
      lxu_cache_locations[n] = kCacheLocationMissing;
    }
  }
  if (gather_cache_stats && n_enforced_misses > 0) {
    atomicAdd(
        &uvm_cache_stats[uvm_cache_stats_index::num_conflict_misses],
        n_enforced_misses);
  }
}
} // namespace

DLL_PUBLIC Tensor emulate_cache_miss(
    Tensor lxu_cache_locations,
    const int64_t enforced_misses_per_256,
    const bool gather_cache_stats,
    Tensor uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      lxu_cache_locations, uvm_cache_stats);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lxu_cache_locations.get_device());

  const auto N = lxu_cache_locations.numel();
  if (N == 0) {
    // nothing to do
    return lxu_cache_locations;
  }

  const dim3 blocks(std::min(
      div_round_up(N, kMaxThreads),
      get_max_thread_blocks_for_cache_kernels_()));

  emulate_cache_miss_kernel<<<
      blocks,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      lxu_cache_locations
          .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      enforced_misses_per_256,
      gather_cache_stats,
      uvm_cache_stats.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return lxu_cache_locations;
}

namespace {
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lru_cache_find_uncached_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const int32_t* __restrict__ N_unique,
    int64_t max_indices,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> cache_sets,
    int64_t time_stamp,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    const bool gather_cache_stats,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats) {
  if (gather_cache_stats) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_calls], 1); // N_called.
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_requested_indices],
          unique_indices.size(0)); // N_requested_indices.
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_unique_indices],
          *N_unique); // N_unique_indices.
    }
  }

  const int32_t C = lxu_cache_state.size(0);
  int32_t n_misses = 0;

  for (int32_t n = blockIdx.x * blockDim.y + threadIdx.y; n < *N_unique;
       n += gridDim.x * blockDim.y) {
    int64_t idx = unique_indices[n];
    if (idx == max_indices) {
      // cache_sets are initialized with sentinel values in
      // lru_cache_find_uncached_cuda
      continue;
    }
    int32_t cache_set = cache_slot(idx, C);

    const auto slot = threadIdx.x;
    const bool found = ::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx;
    if (found) {
      // mark it as recently accessed so we don't evict.
      lru_state[cache_set][slot] = time_stamp;
    }

#ifdef __HIP_PLATFORM_HCC__
    if (!__any_sync(0xFFFFFFFFFFFFFFFF, found)) {
#else
    if (!__any_sync(0xFFFFFFFF, found)) {
#endif
      if (threadIdx.x == 0) {
        cache_sets[n] = cache_set;
        n_misses++;
      }
    }
  }
  if (gather_cache_stats && threadIdx.x == 0) {
    atomicAdd(
        &uvm_cache_stats[uvm_cache_stats_index::num_unique_misses],
        n_misses); // N_unique_misses.
  }
}

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void direct_mapped_lru_cache_find_uncached_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> cache_sets,
    const int64_t max_indices,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    const int64_t time_stamp,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_miss_timestamp) {
  const int32_t N = linear_cache_indices.size(0);
  const int32_t C = lxu_cache_state.size(0);

  CUDA_KERNEL_LOOP(n, N) {
    int64_t idx = linear_cache_indices[n];
    if (idx == max_indices) {
      // Invalid or pruned row: set it to sentinel value.
      // 32-way uses C as the sentinel value to reduce the maximum value during
      // radix sort to make it faster but for direct_mapped we use -1
      cache_sets[n] = -1;
      continue;
    }
    int32_t cache_set = cache_slot(idx, C);

    const bool found = ::__ldg((&lxu_cache_state[cache_set][0])) == idx;
    if (found) {
      // After all threads run, timestamp will be current timestamp
      // if any idx was hit
      // +1 because AMD doesn't have atomicMax for signed long so we should
      // initialize lxu_cache_miss_timestamp with 0 vs. -1.
      lru_state[cache_set][0] = time_stamp;
      cache_sets[n] = -1; // sentinel value
    } else {
      // There is no atomicMax for int64_t...
#ifdef __HIP_PLATFORM_HCC__
      auto addr = reinterpret_cast<unsigned long long*>(
          &lxu_cache_miss_timestamp[cache_set][0]);
      auto val = static_cast<unsigned long long>(time_stamp + 1);
      auto old = static_cast<int64_t>(atomicMax(addr, val));
#else
      auto addr = reinterpret_cast<long long int*>(
          &lxu_cache_miss_timestamp[cache_set][0]);
      auto val = static_cast<long long int>(time_stamp + 1);
      auto old = static_cast<int64_t>(atomicMax(addr, val));
#endif

      if (old < time_stamp + 1) {
        // This is the lucky thread that gets to insert its idx in the cache
        // slot. So the number of elements in cache_sets array that has the
        // value of cache_set is 1 at maximum
        cache_sets[n] = cache_set;
      } else {
        // Otherwise (too late to get this set)
        // set it to sentinel value.
        cache_sets[n] = -1;
      }
    }
  }
}
} // namespace

DLL_PUBLIC std::pair<Tensor, Tensor> lru_cache_find_uncached_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    int64_t max_indices,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    Tensor lru_state,
    bool gather_cache_stats,
    Tensor uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lru_state,
      uvm_cache_stats);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(unique_indices.get_device());

  // Fill with sentinel value
  auto cache_sets = full_like(
      unique_indices,
      lxu_cache_state.size(0),
      unique_indices.options().dtype(at::kInt));
  const int32_t N = unique_indices.numel();
  auto sorted_cache_sets = empty_like(cache_sets);
  auto cache_set_sorted_unique_indices = empty_like(unique_indices);

  AT_DISPATCH_INDEX_TYPES(
      unique_indices.scalar_type(), "lru_cache_find_uncached_cuda", [&] {
        // Find uncached indices
        lru_cache_find_uncached_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads / kWarpSize),
                get_max_thread_blocks_for_cache_kernels_()),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            max_indices,
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            cache_sets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            time_stamp,
            lru_state.packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            gather_cache_stats,
            uvm_cache_stats
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // Sort the cache sets and ids
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            cache_sets.data_ptr<int32_t>(),
            sorted_cache_sets.data_ptr<int32_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<index_t>(temp_storage_bytes)},
            unique_indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            cache_sets.data_ptr<int32_t>(),
            sorted_cache_sets.data_ptr<int32_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
      });
  return {sorted_cache_sets, cache_set_sorted_unique_indices};
}

namespace {

Tensor direct_mapped_lru_cache_find_uncached_cuda(
    Tensor linear_cache_indices,
    int64_t max_indices,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    Tensor lru_state,
    Tensor lxu_cache_miss_timestamp) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_cache_indices,
      lxu_cache_state,
      lru_state,
      lxu_cache_miss_timestamp);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(linear_cache_indices.get_device());

  const int32_t N = linear_cache_indices.numel();

  auto cache_sets = empty_like(
      linear_cache_indices, linear_cache_indices.options().dtype(at::kInt));

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(),
      "direct_mapped_lru_cache_find_uncached_cuda",
      [&] {
        // Find uncached indices
        direct_mapped_lru_cache_find_uncached_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads),
                get_max_thread_blocks_for_cache_kernels_()),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            linear_cache_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            cache_sets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            max_indices,
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            time_stamp,
            lru_state.packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            lxu_cache_miss_timestamp
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return cache_sets;
}

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kMaxThreads) void lru_cache_insert_kernel(
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_cache_sets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const int64_t time_stamp,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    const bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    const bool gather_cache_stats,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats) {
  const int32_t C = lxu_cache_state.size(0);
  int32_t n_conflict_misses = 0;
  for (int32_t n = blockIdx.x * blockDim.y + threadIdx.y; n < *N_unique;
       n += gridDim.x * blockDim.y) {
    // check if this warp is responsible for this whole segment.
    const bool segment_start =
        (n == 0 || sorted_cache_sets[n - 1] != sorted_cache_sets[n]);

    if (!segment_start) {
      // don't have *warp* divergence since we launch full warps in blockDim.x,
      // so we can just exit this warp entirely.
      continue;
    }
    const int32_t cache_set = sorted_cache_sets[n];
    if (cache_set == C) {
      // ignore the already-existing elements
      continue;
    }

    int32_t SL = 1;
    while (n + SL < *N_unique && sorted_cache_sets[n + SL] == cache_set) {
      SL += 1;
    }
    int32_t n_inserted = 0;

    // now, we need to insert the (unique!) values in indices[n:n + SL] into
    // our slots.
    const int32_t slot = threadIdx.x;
    const int64_t slot_time = lru_state[cache_set][slot];
    int64_t costs[1] = {slot_time};
    int32_t slots[1] = {slot};

    BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
    const int32_t sorted_slot = slots[0];
    const int64_t sorted_lru_cost = costs[0];

    for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
      const int32_t insert_slot = shfl_sync(sorted_slot, l);
      const int64_t insert_current_lru_cost = shfl_sync(sorted_lru_cost, l);
      if (insert_current_lru_cost == time_stamp) {
        break;
      }
      const int64_t insert_idx = cache_set_sorted_indices[n + l];
      const int32_t t_insert = cache_index_table_map[insert_idx];
      const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
      const int64_t weights_offset_insert = weights_offsets[t_insert];
      const int32_t D_start_insert = D_offsets[t_insert];
      const int32_t D_end_insert = D_offsets[t_insert + 1];
      const int32_t D_insert = D_end_insert - D_start_insert;

      // ensure that threadIdx.x is the only thread reading/writing to
      // lxu_cache_state
      int64_t current_idx =
          threadIdx.x == 0 ? lxu_cache_state[cache_set][insert_slot] : 0;
      current_idx = shfl_sync(current_idx, 0);

      // not empty
      if (current_idx != static_cast<int64_t>(kCacheStateInvalid)) {
        // evict from slot to backing storage
        const int32_t t_current = cache_index_table_map[current_idx];
        const int64_t idx_current =
            current_idx - cache_hash_size_cumsum[t_current];
        const int64_t weights_offset_current = weights_offsets[t_current];
        const int32_t D_start_current = D_offsets[t_current];
        const int32_t D_end_current = D_offsets[t_current + 1];
        const int32_t D_current = D_end_current - D_start_current;
        int32_t D_emb = D_current;
        if (std::is_same<emb_t, uint8_t>::value) {
          D_emb += kINT8QparamsBytes;
        }
        auto weight_row = WeightRow<emb_t, cache_t, cache_t>(
            &weights[weights_offset_current + idx_current * D_emb + 0],
            &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
            D_current,
            nullptr);
        if (!std::is_same<emb_t, float>::value && stochastic_rounding) {
          StochasticRoundingRNGState state;
          // different for every *run* and every *thread*.
          auto stochastic_rounding_seeds =
              at::cuda::philox::unpack(stochastic_rounding_philox_args);
          stochastic_rounding_init(
              std::get<0>(stochastic_rounding_seeds) ^
                  std::get<1>(stochastic_rounding_seeds),
              (blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
               threadIdx.x) *
                      kWarpSize +
                  l,
              &state);
          weight_row.set_stoc_state(&state);
        }
        float2 qparams;
        at::acc_type<cache_t, true> local_min =
            std::numeric_limits<at::acc_type<cache_t, true>>::max();
        at::acc_type<cache_t, true> local_max =
            std::numeric_limits<at::acc_type<cache_t, true>>::lowest();
        if (std::is_same<emb_t, uint8_t>::value) {
          for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
            Vec4T<cache_t> cache_weights_vec =
                weight_row.load(d * 4, qparams); // qparams not used
            local_max = max(local_max, vec4_max(cache_weights_vec));
            local_min = min(local_min, vec4_min(cache_weights_vec));
          }
          qparams = warp_find_qparams(local_min, local_max);
          if (threadIdx.x == 0) {
            weight_row.store_qparams(qparams);
          }
        }
        for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
          Vec4T<cache_t> cache_weights_vec = weight_row.load(d * 4, qparams);
          weight_row.evict(
              cache_weights_vec, d * 4, qparams); // FP32 -> FP16/FP32
        }
      }
      int32_t D_emb = D_insert;
      if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
      }
      // insert into cache
      auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
          &weights[weights_offset_insert + idx_insert * D_emb + 0],
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
          D_insert,
          nullptr);

      auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
          &weights[weights_offset_insert + idx_insert * D_emb + 0],
          nullptr,
          D_insert,
          nullptr);

      float2 qparams;
      if (std::is_same<emb_t, uint8_t>::value) {
        qparams = weight_row_emb.load_qparams();
      }
      for (int32_t d = threadIdx.x; d * 4 < D_insert; d += blockDim.x) {
        auto row = weight_row_emb.load(d * 4, qparams);
        weight_row_cache.store(row, d * 4, qparams);
      }
      if (threadIdx.x == 0) {
        lxu_cache_state[cache_set][insert_slot] = insert_idx;
        lru_state[cache_set][insert_slot] = time_stamp;
      }
      n_inserted++;
    }
    n_conflict_misses += (SL - n_inserted);
  }
  if (gather_cache_stats && n_conflict_misses > 0 && threadIdx.x == 0) {
    atomicAdd(
        &uvm_cache_stats[uvm_cache_stats_index::num_conflict_unique_misses],
        n_conflict_misses);
  }
}

void lru_cache_insert_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    const int64_t time_stamp,
    Tensor lru_state,
    const bool stochastic_rounding,
    bool gather_cache_stats,
    Tensor uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state,
      uvm_cache_stats);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const int32_t N = cache_set_sorted_unique_indices.numel();

  DISPATCH_EMB_CACHE_TYPES(
      weights.scalar_type(),
      lxu_cache_weights.scalar_type(),
      "lru_cache_insert_kernel_2",
      ([&] {
        at::PhiloxCudaState rng_engine_inputs;
        if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
          auto gen = at::cuda::detail::getDefaultCUDAGenerator();
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)
                                  ->philox_cuda_state(4);
        }

        lru_cache_insert_kernel<emb_t, cache_t>
            <<<div_round_up(N, kMaxThreads / kWarpSize),
               dim3(kWarpSize, kMaxThreads / kWarpSize),
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                cache_hash_size_cumsum
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                cache_index_table_map
                    .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
                weights_offsets
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                sorted_cache_sets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                cache_set_sorted_unique_indices
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                unique_indices_length.data_ptr<int32_t>(),
                lxu_cache_state
                    .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
                lxu_cache_weights
                    .packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                time_stamp,
                lru_state
                    .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
                stochastic_rounding,
                rng_engine_inputs,
                gather_cache_stats,
                uvm_cache_stats
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

} // namespace

DLL_PUBLIC void lru_cache_populate_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    const int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    const int64_t time_stamp,
    Tensor lru_state,
    const bool stochastic_rounding,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      linear_cache_indices,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state);

  Tensor uvm_cache_stats_ = at::empty({0}, weights.options().dtype(at::kInt));
  if (gather_cache_stats) {
    TORCH_CHECK(uvm_cache_stats.has_value());
    uvm_cache_stats_ = uvm_cache_stats.value();
    TENSOR_ON_CUDA_GPU(uvm_cache_stats_);
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  // Get unqiue indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(
          linear_cache_indices, total_cache_hash_size, false);

  auto cache_sets_and_unique_indices = lru_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      time_stamp,
      lru_state,
      gather_cache_stats,
      uvm_cache_stats_);
  auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  auto cache_set_sorted_unique_indices = cache_sets_and_unique_indices.second;

  // insert caching weights
  lru_cache_insert_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      time_stamp,
      lru_state,
      stochastic_rounding,
      gather_cache_stats,
      uvm_cache_stats_);
}

namespace {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lru_cache_insert_byte_kernel(
    at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
        weights_tys,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_cache_sets,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    int64_t time_stamp,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    const bool gather_cache_stats,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats,
    const int64_t row_alignment) {
  const int32_t C = lxu_cache_state.size(0);
  int64_t n_conflict_misses = 0;
  for (int32_t n = blockIdx.x * blockDim.y + threadIdx.y; n < *N_unique;
       n += gridDim.x * blockDim.y) {
    // check if this warp is responsible for this whole segment.
    const bool segment_start =
        (n == 0 || sorted_cache_sets[n - 1] != sorted_cache_sets[n]);

    if (!segment_start) {
      // don't have *warp* divergence since we launch full warps in blockDim.x,
      // so we can just exit this warp entirely.
      continue;
    }
    const int32_t cache_set = sorted_cache_sets[n];
    if (cache_set == C) {
      // ignore the already-existing elements
      continue;
    }

    int32_t SL = 1;
    while (n + SL < *N_unique && sorted_cache_sets[n + SL] == cache_set) {
      SL += 1;
    }
    int64_t n_inserted = 0;

    // now, we need to insert the (unique!) values in indices[n:n + SL] into
    // our slots.
    const int32_t slot = threadIdx.x;
    const int64_t slot_time = lru_state[cache_set][slot];
    int64_t costs[1] = {slot_time};
    int32_t slots[1] = {slot};

    BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
    const int32_t sorted_slot = slots[0];
    const int64_t sorted_lru_cost = costs[0];

    for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
      const int32_t insert_slot = shfl_sync(sorted_slot, l);
      const int64_t insert_current_lru_cost = shfl_sync(sorted_lru_cost, l);
      if (insert_current_lru_cost == time_stamp) {
        break;
      }
      index_t insert_idx = cache_set_sorted_indices[n + l];
      const int32_t t_insert = cache_index_table_map[insert_idx];
      SparseType weight_ty_insert =
          static_cast<SparseType>(weights_tys[t_insert]);
      const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
      const int64_t weights_offset_insert = weights_offsets[t_insert];
      const int32_t D_start_insert = D_offsets[t_insert];
      const int32_t D_end_insert = D_offsets[t_insert + 1];
      const int32_t D_insert = D_end_insert - D_start_insert;

      const int32_t D_insert_bytes = nbit::padded_row_size_in_bytes(
          D_insert, weight_ty_insert, row_alignment);

      // insert into cache. Note that nbit::padded_row_size_in_bytes pad each
      // row with row_alignment (16 bytes on GPUs) So each row will be multiple
      // of 16 bytes (uint4 = 32bit x 4 = 16 bytes).
      auto row = reinterpret_cast<const uint4*>(
          &weights[weights_offset_insert + idx_insert * D_insert_bytes + 0]);
      auto cache_row = reinterpret_cast<uint4*>(
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0]);
      for (int32_t d = threadIdx.x; d * sizeof(uint4) < D_insert_bytes;
           d += blockDim.x) {
        cache_row[d] = row[d];
      }

      if (threadIdx.x == 0) {
        lxu_cache_state[cache_set][insert_slot] = insert_idx;
        lru_state[cache_set][insert_slot] = time_stamp;
      }
      n_inserted++;
    }
    n_conflict_misses += (SL - n_inserted);
  }
  if (gather_cache_stats && n_conflict_misses > 0 && threadIdx.x == 0) {
    atomicAdd(
        &uvm_cache_stats[uvm_cache_stats_index::num_conflict_unique_misses],
        n_conflict_misses);
  }
}

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void direct_mapped_lru_cache_insert_byte_kernel(
    at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
        weights_tys,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    int64_t time_stamp,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_miss_timestamp,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> cache_sets,
    const int64_t row_alignment) {
  const int32_t N = cache_sets.size(0);

  // one warp for each set (multiple times)
  // (no divergence for each control branch)
  for (int32_t pos = blockIdx.x * blockDim.y + threadIdx.y; pos < N;
       pos += gridDim.x * blockDim.y) {
    auto cache_set = cache_sets[pos];

    if (cache_set == -1) {
      // Cache hit, index invalid (e.g., pruned), or too late to grab this set.
      continue;
    }

    if (lru_state[cache_set][0] == time_stamp) {
      // we have a missing index but
      // current cache row is a hit
      // so abort unnecessary insert
      continue;
    }

    // no need to check because cache_sets[pos] != -1 only when it was the
    // first one to set the buffer time_stamp
    // if (lxu_cache_miss_timestamp[cache_set][0] != time_stamp) {
    //   continue;
    // }

    // insert the index in the buffer into our only slot
    const int32_t insert_slot = 0;

    int64_t insert_idx = linear_cache_indices[pos];
    const int32_t t_insert = cache_index_table_map[insert_idx];
    SparseType weight_ty_insert =
        static_cast<SparseType>(weights_tys[t_insert]);
    const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
    const int64_t weights_offset_insert = weights_offsets[t_insert];
    const int32_t D_start_insert = D_offsets[t_insert];
    const int32_t D_end_insert = D_offsets[t_insert + 1];
    const int32_t D_insert = D_end_insert - D_start_insert;
    const int32_t D_insert_bytes = nbit::padded_row_size_in_bytes(
        D_insert, weight_ty_insert, row_alignment);

    // insert into cache. Note that nbit::padded_row_size_in_bytes pad each
    // row with row_alignment (16 bytes on GPUs) So each row will be multiple
    // of 16 bytes (uint4 = 32bit x 4 = 16 bytes).
    auto row = reinterpret_cast<const uint4*>(
        &weights[weights_offset_insert + idx_insert * D_insert_bytes + 0]);
    auto cache_row = reinterpret_cast<uint4*>(&lxu_cache_weights[cache_set][0]);
    for (int32_t d = threadIdx.x; d * sizeof(uint4) < D_insert_bytes;
         d += blockDim.x) {
      cache_row[d] = row[d];
    }

    if (threadIdx.x == 0) {
      lxu_cache_state[cache_set][insert_slot] = insert_idx;
      lru_state[cache_set][insert_slot] = time_stamp;
    }
  }
}

void lru_cache_insert_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    bool gather_cache_stats,
    Tensor uvm_cache_stats,
    int64_t row_alignment) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state,
      uvm_cache_stats);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const int32_t N = cache_set_sorted_unique_indices.numel();

  AT_DISPATCH_INDEX_TYPES(
      cache_set_sorted_unique_indices.scalar_type(),
      "lru_cache_insert_byte_cuda",
      [&] {
        lru_cache_insert_byte_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads / kWarpSize),
                get_max_thread_blocks_for_cache_kernels_()),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(),
            cache_hash_size_cumsum
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            cache_index_table_map
                .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
            weights_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            sorted_cache_sets
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            cache_set_sorted_unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(),
            time_stamp,
            lru_state.packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            gather_cache_stats,
            uvm_cache_stats
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            row_alignment);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

void direct_mapped_lru_cache_insert_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    Tensor linear_cache_indices,
    Tensor lxu_cache_miss_timestamp,
    Tensor cache_sets,
    int64_t row_alignment) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state,
      linear_cache_indices,
      lxu_cache_miss_timestamp);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const int32_t N = cache_sets.size(0);

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(),
      "direct_mapped_lru_cache_insert_byte_cuda",
      [&] {
        direct_mapped_lru_cache_insert_byte_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads / kWarpSize),
                get_max_thread_blocks_for_cache_kernels_()),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(),
            cache_hash_size_cumsum
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            cache_index_table_map
                .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
            weights_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(),
            time_stamp,
            lru_state.packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            linear_cache_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_miss_timestamp
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            cache_sets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            row_alignment);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace

DLL_PUBLIC void lru_cache_populate_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    int64_t row_alignment,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      linear_cache_indices,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state);

  Tensor uvm_cache_stats_ = at::empty({0}, weights.options().dtype(at::kInt));
  if (gather_cache_stats) {
    TORCH_CHECK(uvm_cache_stats.has_value());
    uvm_cache_stats_ = uvm_cache_stats.value();
    TENSOR_ON_CUDA_GPU(uvm_cache_stats_);
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  // Get unqiue indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(
          linear_cache_indices, total_cache_hash_size, false);

  // Find uncached indices
  auto cache_sets_and_unique_indices = lru_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      time_stamp,
      lru_state,
      gather_cache_stats,
      uvm_cache_stats_);
  auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  auto cache_set_sorted_unique_indices = cache_sets_and_unique_indices.second;

  // insert caching weights
  lru_cache_insert_byte_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      time_stamp,
      lru_state,
      gather_cache_stats,
      uvm_cache_stats_,
      row_alignment);
}

DLL_PUBLIC void direct_mapped_lru_cache_populate_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    Tensor lxu_cache_miss_timestamp,
    int64_t row_alignment) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      linear_cache_indices,
      lxu_cache_state,
      lxu_cache_weights,
      lru_state,
      lxu_cache_miss_timestamp);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  /*
  populate_byte normal flow:
  (1) get_unique (sort, dedup)
  (2) find_uncached
  (3) sort by set_idx
  (4) insert rows

  merged kernels flow:
  (1) find_uncached
        No need for sorting.
        Each hit idx will just update the timestamp in lru_state.
        Only one of miss indices will atomically set miss_timestamp,
                                      and have cache_sets[pos] = set
                                          where pos is the position of that idx
                                          in the linear_cache_indices array
        After this, for each set, we either have
          (a) lru_state timestamp is recent (hit) => no need to insert row
          (b) lru_state timestamp is not recent (no hit)
              (b-1) miss_timestamp is recent
                    => insert row for idx = linear_cache_indices[pos]
              (b-2) insert_timestamp_buffer is not recent
                    => no need to insert since there was no miss idx this time
  (2) insert rows
        Use buffer info to insert rows as the above logic.
  */

  auto cache_sets = direct_mapped_lru_cache_find_uncached_cuda(
      linear_cache_indices,
      total_cache_hash_size,
      lxu_cache_state,
      time_stamp,
      lru_state,
      lxu_cache_miss_timestamp);

  // insert caching weights
  direct_mapped_lru_cache_insert_byte_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      lxu_cache_state,
      lxu_cache_weights,
      time_stamp,
      lru_state,
      linear_cache_indices,
      lxu_cache_miss_timestamp,
      cache_sets,
      row_alignment);
}

namespace {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lfu_update_counts_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const int32_t* __restrict__ N_unique,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        unique_indices_count,
    at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lfu_state) {
  CUDA_KERNEL_LOOP(n, *N_unique) {
    const auto idx = unique_indices[n];
    lfu_state[idx] += unique_indices_count[n];
  }
}

void lfu_update_counts_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    Tensor unique_indices_count,
    Tensor lfu_state) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      unique_indices, unique_indices_length, unique_indices_count, lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(unique_indices.get_device());

  const int32_t N = unique_indices.size(0);
  AT_DISPATCH_INDEX_TYPES(
      unique_indices.scalar_type(), "lfu_update_counts_cuda", [&] {
        lfu_update_counts_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads),
                get_max_thread_blocks_for_cache_kernels_()),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            unique_indices_count
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            lfu_state.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

constexpr int32_t kCacheSetBits = 24;
constexpr int32_t kLFUCounterBits = 40;
static_assert(kCacheSetBits + kLFUCounterBits == 8 * sizeof(int64_t), "");

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lfu_cache_find_uncached_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const int32_t* __restrict__ N_unique,
    int64_t max_indices,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    uint64_t* __restrict__ cache_sets,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        lfu_state) {
  const int32_t C = lxu_cache_state.size(0);

  for (int32_t n = blockIdx.x * blockDim.y + threadIdx.y; n < *N_unique;
       n += gridDim.x * blockDim.y) {
    const int64_t idx = unique_indices[n];
    if (idx == max_indices) {
      // cache_sets are initialized with sentinel values in
      // lfu_cache_find_uncached_cuda
      continue;
    }
    const uint32_t cache_set = cache_slot(idx, C);

    const auto slot = threadIdx.x;
    const bool found = ::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx;

#ifdef __HIP_PLATFORM_HCC__
    if (!__any_sync(0xFFFFFFFFFFFFFFFF, found)) {
#else
    if (!__any_sync(0xFFFFFFFF, found)) {
#endif
      if (threadIdx.x == 0) {
        // sort so the highest LFUs come first in the segment.
        // assume lfu_state[idx] <= 2^40 - 1 and cache_set < 2^24 -1
        cache_sets[n] =
            ((static_cast<uint64_t>(cache_set) << kLFUCounterBits)) |
            ((static_cast<uint64_t>(1) << kLFUCounterBits) - 1 -
             lfu_state[idx]);
      }
    }
  }
}

std::pair<Tensor, Tensor> lfu_cache_find_uncached_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    int64_t max_indices,
    Tensor lxu_cache_state,
    Tensor lfu_state) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      unique_indices, unique_indices_length, lxu_cache_state, lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(unique_indices.get_device());

  auto cache_sets = full_like(
      unique_indices,
      static_cast<int64_t>(
          static_cast<uint64_t>(lxu_cache_state.size(0)) << kLFUCounterBits),
      unique_indices.options().dtype(at::kLong));
  const int32_t N = unique_indices.numel();
  auto sorted_cache_sets = empty_like(cache_sets);
  auto cache_set_sorted_unique_indices = empty_like(unique_indices);

  AT_DISPATCH_INDEX_TYPES(
      unique_indices.scalar_type(), "lfu_cache_find_uncached_cuda", [&] {
        // Find uncached indices
        lfu_cache_find_uncached_kernel<<<
            std::min(
                div_round_up(N, kMaxThreads / kWarpSize),
                get_max_thread_blocks_for_cache_kernels_()),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            max_indices,
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            (uint64_t*)cache_sets.data_ptr<int64_t>(),
            lfu_state.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // Sort the cache sets and ids
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            (uint64_t*)cache_sets.data_ptr<int64_t>(),
            (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1) + kLFUCounterBits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            unique_indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            (uint64_t*)cache_sets.data_ptr<int64_t>(),
            (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1) + kLFUCounterBits,
            at::cuda::getCurrentCUDAStream(),
            false));
      });
  return {sorted_cache_sets, cache_set_sorted_unique_indices};
}

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kCacheMaxThreads) void lfu_cache_insert_kernel(
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const uint64_t* __restrict__ sorted_cache_sets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        lfu_state,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  const int32_t C = lxu_cache_state.size(0);
  for (int32_t n = blockIdx.x * blockDim.y + threadIdx.y; n < *N_unique;
       n += gridDim.x * blockDim.y) {
    // check if this warp is responsible for this whole segment.
    const bool segment_start =
        (n == 0 ||
         (sorted_cache_sets[n - 1] >> kLFUCounterBits) !=
             (sorted_cache_sets[n] >> kLFUCounterBits));

    if (!segment_start) {
      // don't have *warp* divergence since we launch full warps in blockDim.x,
      // so we can just exit this warp entirely.
      continue;
    }
    const uint32_t cache_set = (sorted_cache_sets[n] >> kLFUCounterBits);
    if (cache_set == C) {
      // ignore the already-existing elements
      continue;
    }

    int32_t SL = 1;
    while (n + SL < *N_unique &&
           (sorted_cache_sets[n + SL] >> kLFUCounterBits) == cache_set) {
      SL += 1;
    }

    // now, we need to insert the (unique!) values in indices[n:n + SL] into
    // our slots.
    const int32_t slot = threadIdx.x;
    const int64_t current_idx = lxu_cache_state[cache_set][slot];
    const int64_t current_lfu_cost =
        (current_idx != static_cast<int64_t>(kCacheStateInvalid))
        ? lfu_state[current_idx]
        : -1;
    int64_t costs[1] = {current_lfu_cost};
    int32_t slots[1] = {slot};

    BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
    const int32_t sorted_slot = slots[0];
    const int64_t sorted_lfu_cost = costs[0];

    for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
      const int32_t insert_slot = shfl_sync(sorted_slot, l);
      const int64_t insert_current_lfu_cost = shfl_sync(sorted_lfu_cost, l);
      const int64_t insert_idx = cache_set_sorted_indices[n + l];
      const int64_t insert_lfu_cost = lfu_state[insert_idx];

      if (insert_current_lfu_cost > insert_lfu_cost) {
        // don't insert.
        // all subsequent `current_lfu_cost` values are greater, and all
        // subsequent `insert_lfu_cost` values are smaller, so we can exit
        // early here.
        break;
      }
      const int32_t t_insert = cache_index_table_map[insert_idx];
      const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
      const int64_t weights_offset_insert = weights_offsets[t_insert];
      const int32_t D_start_insert = D_offsets[t_insert];
      const int32_t D_end_insert = D_offsets[t_insert + 1];
      const int32_t D_insert = D_end_insert - D_start_insert;

      // not empty
      if (insert_current_lfu_cost != -1) {
        // ensure that threadIdx.x is the only thread reading/writing to
        // lxu_cache_state
        int64_t current_idx =
            threadIdx.x == 0 ? lxu_cache_state[cache_set][insert_slot] : 0;
        current_idx = shfl_sync(current_idx, 0);
        const int32_t t_current = cache_index_table_map[current_idx];
        const int64_t idx_current =
            current_idx - cache_hash_size_cumsum[t_current];
        const int64_t weights_offset_current = weights_offsets[t_current];
        const int32_t D_start_current = D_offsets[t_current];
        const int32_t D_end_current = D_offsets[t_current + 1];
        const int32_t D_current = D_end_current - D_start_current;

        int32_t D_emb = D_current;
        if (std::is_same<emb_t, uint8_t>::value) {
          D_emb += kINT8QparamsBytes;
        }
        auto weight_row = WeightRow<emb_t, cache_t, cache_t>(
            &weights[weights_offset_current + idx_current * D_emb + 0],
            &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
            D_current,
            nullptr);
        if (!std::is_same<emb_t, float>::value && stochastic_rounding) {
          StochasticRoundingRNGState state;
          // different for every *run* and every *thread*.
          auto stochastic_rounding_seeds =
              at::cuda::philox::unpack(stochastic_rounding_philox_args);
          stochastic_rounding_init(
              std::get<0>(stochastic_rounding_seeds) ^
                  std::get<1>(stochastic_rounding_seeds),
              (blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
               threadIdx.x) *
                      kWarpSize +
                  l,
              &state);
          weight_row.set_stoc_state(&state);
        }

        float2 qparams;
        at::acc_type<cache_t, true> local_min =
            std::numeric_limits<at::acc_type<cache_t, true>>::max();
        at::acc_type<cache_t, true> local_max =
            std::numeric_limits<at::acc_type<cache_t, true>>::lowest();
        if (std::is_same<emb_t, uint8_t>::value) {
          for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
            Vec4T<cache_t> cache_weights_vec =
                weight_row.load(d * 4, qparams); // qparams not used
            local_max = max(local_max, vec4_max(cache_weights_vec));
            local_min = min(local_min, vec4_min(cache_weights_vec));
          }
          qparams = warp_find_qparams(local_min, local_max);
          if (threadIdx.x == 0) {
            weight_row.store_qparams(qparams);
          }
        }
        for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
          Vec4T<cache_t> cache_weights_vec = weight_row.load(d * 4, qparams);
          weight_row.evict(cache_weights_vec, d * 4, qparams);
        }
      }
      // insert into cache
      int32_t D_emb = D_insert;
      if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
      }
      auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
          &weights[weights_offset_insert + idx_insert * D_emb + 0],
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
          D_insert,
          nullptr);

      auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
          &weights[weights_offset_insert + idx_insert * D_emb + 0],
          nullptr,
          D_insert,
          nullptr);

      float2 qparams;
      if (std::is_same<emb_t, uint8_t>::value) {
        qparams = weight_row_emb.load_qparams();
      }
      for (int32_t d = threadIdx.x; d * 4 < D_insert; d += blockDim.x) {
        auto row = weight_row_emb.load(d * 4, qparams);
        weight_row_cache.store(row, d * 4, qparams);
      }
      if (threadIdx.x == 0) {
        lxu_cache_state[cache_set][insert_slot] = insert_idx;
      }
    }
  }
}

void lfu_cache_insert_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    Tensor lfu_state,
    bool stochastic_rounding) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const int32_t N = cache_set_sorted_unique_indices.numel();

  DISPATCH_EMB_CACHE_TYPES(
      weights.scalar_type(),
      lxu_cache_weights.scalar_type(),
      "lfu_cache_insert_kernel_2",
      ([&] {
        at::PhiloxCudaState rng_engine_inputs;
        if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
          auto gen = at::cuda::detail::getDefaultCUDAGenerator();
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)
                                  ->philox_cuda_state(4);
        }

        lfu_cache_insert_kernel<emb_t, cache_t>
            <<<std::min(
                   div_round_up(N, kCacheMaxThreads / kWarpSize),
                   get_max_thread_blocks_for_cache_kernels_()),
               dim3(kWarpSize, kCacheMaxThreads / kWarpSize),
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                cache_hash_size_cumsum
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                cache_index_table_map
                    .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
                weights_offsets
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
                cache_set_sorted_unique_indices
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                unique_indices_length.data_ptr<int32_t>(),
                lxu_cache_state
                    .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
                lxu_cache_weights
                    .packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                lfu_state
                    .packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                stochastic_rounding,
                rng_engine_inputs);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

} // namespace

DLL_PUBLIC void lfu_cache_populate_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    Tensor lfu_state,
    bool stochastic_rounding) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      linear_cache_indices,
      lxu_cache_state,
      lxu_cache_weights,
      lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  // get unqiue indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(
          linear_cache_indices, total_cache_hash_size, true);

  // update lfu counts
  lfu_update_counts_cuda(
      unique_indices, unique_indices_length, *unique_indices_count, lfu_state);

  // find uncached indices
  auto cache_sets_and_unique_indices = lfu_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      lfu_state);
  const auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  const auto cache_set_sorted_unique_indices =
      cache_sets_and_unique_indices.second;

  // insert caching weights
  lfu_cache_insert_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lfu_state,
      stochastic_rounding);
}

namespace {

// In `lfu_cache_insert_kernel`, we use `emb_t` and `cache_t` for the
// high-precision cache implementation, where we can have {FP32, FP16, INT8}
// for embedding precision (data types), and {FP32, FP16} for cache precision
// (data types).
//
// In `lfu_cache_insert_byte_kernel`, we only use uint8_t for the both embedding
// and cache data type (conforming to the inference TBE kernel logics).
// - We pass in `weights_tys` to denote the real data types for the embeddings:
// {FP32, FP16, INT8, INT4, INT2}. For example, FP32 is 4 byte element in the
// byte tensor, and INT4 is half byte element in the byte tensor.
// - We only assume that the embedding and cache have the same precisions (the
// real "precision" is determined by `weights_tys` although the data types are
// uint8_t only). Basically no "high-precision cache" support for now.
// - The insert/evict of embedding row from the cache are done in a byte-by-byte
// manner.
template <typename index_t>
__global__
__launch_bounds__(kCacheMaxThreads) void lfu_cache_insert_byte_kernel(
    at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
        weights_tys,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const uint64_t* __restrict__ sorted_cache_sets,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        lfu_state,
    const int64_t row_alignment) {
  const int32_t C = lxu_cache_state.size(0);
  for (int32_t n = blockIdx.x * blockDim.y + threadIdx.y; n < *N_unique;
       n += gridDim.x * blockDim.y) {
    // check if this warp is responsible for this whole segment.
    const bool segment_start =
        (n == 0 ||
         (sorted_cache_sets[n - 1] >> kLFUCounterBits) !=
             (sorted_cache_sets[n] >> kLFUCounterBits));

    if (!segment_start) {
      // don't have *warp* divergence since we launch full warps in blockDim.x,
      // so we can just exit this warp entirely.
      continue;
    }
    const uint32_t cache_set = (sorted_cache_sets[n] >> kLFUCounterBits);
    if (cache_set == C) {
      // ignore the already-existing elements
      continue;
    }

    int32_t SL = 1;
    while (n + SL < *N_unique &&
           (sorted_cache_sets[n + SL] >> kLFUCounterBits) == cache_set) {
      SL += 1;
    }

    // now, we need to insert the (unique!) values in indices[n:n + SL] into
    // our slots.
    const int32_t slot = threadIdx.x;
    const int64_t current_idx = lxu_cache_state[cache_set][slot];
    const int64_t current_lfu_cost =
        (current_idx != static_cast<int64_t>(kCacheStateInvalid))
        ? lfu_state[current_idx]
        : -1;
    int64_t costs[1] = {current_lfu_cost};
    int32_t slots[1] = {slot};

    BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
    const int32_t sorted_slot = slots[0];
    const int64_t sorted_lfu_cost = costs[0];

    for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
      const int32_t insert_slot = shfl_sync(sorted_slot, l);
      const int64_t insert_current_lfu_cost = shfl_sync(sorted_lfu_cost, l);
      const index_t insert_idx = cache_set_sorted_indices[n + l];
      const int64_t insert_lfu_cost = lfu_state[insert_idx];

      if (insert_current_lfu_cost > insert_lfu_cost) {
        // don't insert.
        // all subsequent `current_lfu_cost` values are greater, and all
        // subsequent `insert_lfu_cost` values are smaller, so we can exit
        // early here.
        break;
      }
      const int32_t t_insert = cache_index_table_map[insert_idx];
      const SparseType weight_ty_insert =
          static_cast<SparseType>(weights_tys[t_insert]);
      const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
      const int64_t weights_offset_insert = weights_offsets[t_insert];
      const int32_t D_start_insert = D_offsets[t_insert];
      const int32_t D_end_insert = D_offsets[t_insert + 1];
      const int32_t D_insert = D_end_insert - D_start_insert;

      const int32_t D_insert_bytes = nbit::padded_row_size_in_bytes(
          D_insert, weight_ty_insert, row_alignment);

      // insert into cache. Note that nbit::padded_row_size_in_bytes pad each
      // row with row_alignment (16 bytes on GPUs) So each row will be multiple
      // of 16 bytes (uint4 = 32bit x 4 = 16 bytes).
      auto row = reinterpret_cast<const uint4*>(
          &weights[weights_offset_insert + idx_insert * D_insert_bytes + 0]);
      auto cache_row = reinterpret_cast<uint4*>(
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0]);
      for (int32_t d = threadIdx.x; d * sizeof(uint4) < D_insert_bytes;
           d += blockDim.x) {
        cache_row[d] = row[d];
      }
      if (threadIdx.x == 0) {
        lxu_cache_state[cache_set][insert_slot] = insert_idx;
      }
    }
  }
}

void lfu_cache_insert_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    Tensor lfu_state,
    int64_t row_alignment) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const int32_t N = cache_set_sorted_unique_indices.numel();

  AT_DISPATCH_INDEX_TYPES(
      cache_set_sorted_unique_indices.scalar_type(),
      "lfu_cache_insert_byte_cuda",
      [&] {
        lfu_cache_insert_byte_kernel<<<
            std::min(
                div_round_up(N, kCacheMaxThreads / kWarpSize),
                get_max_thread_blocks_for_cache_kernels_()),
            dim3(kWarpSize, kCacheMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(),
            cache_hash_size_cumsum
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            cache_index_table_map
                .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
            weights_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
            cache_set_sorted_unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(),
            lfu_state.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            row_alignment);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace

DLL_PUBLIC void lfu_cache_populate_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    Tensor lfu_state,
    int64_t row_alignment) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      linear_cache_indices,
      lxu_cache_state,
      lxu_cache_weights,
      lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  // get unqiue indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(
          linear_cache_indices, total_cache_hash_size, true);

  // update lfu counts
  lfu_update_counts_cuda(
      unique_indices, unique_indices_length, *unique_indices_count, lfu_state);

  // find uncached indices
  const auto cache_sets_and_unique_indices = lfu_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      lfu_state);
  const auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  const auto cache_set_sorted_unique_indices =
      cache_sets_and_unique_indices.second;

  // insert caching weights
  lfu_cache_insert_byte_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lfu_state,
      row_alignment);
}

namespace {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_lookup_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    int64_t invalid_index,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const bool gather_cache_stats,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats) {
  const int32_t C = lxu_cache_state.size(0);
  const int32_t N = linear_cache_indices.size(0);
  const int32_t n0 =
      blockIdx.x * blockDim.y * blockDim.x + threadIdx.y * blockDim.x;
  if (n0 >= N) {
    return;
  }

  int32_t cache_location = kCacheLocationMissing;
  int32_t n_indices = 0;
  int32_t n_hits = 0;
  const auto slot = threadIdx.x;
  for (int i = 0; i < blockDim.x; ++i) {
    int32_t n = n0 + i;
    const int64_t idx = linear_cache_indices[n0 + i];
    if (n >= N || idx == invalid_index) {
      continue;
    }
    const int32_t cache_set = cache_slot(idx, C);
    n_indices++;
    const bool found =
        (::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx);
#ifdef __HIP_PLATFORM_HCC__
    // FIXME: __ballot_sync with mask isn't supported by HIP yet.
    // See https://fburl.com/fvy7j0lq for the similar context.
    // assert false here with https://fburl.com/pfm7enw2
    assert(false);
    const auto bitmap = __ballot(found);
    if (bitmap) {
      const auto way = __ffsll(bitmap) - 1;
#else
    const auto bitmap = __ballot_sync(0xFFFFFFFF, found);
    if (bitmap) {
      // LSB == 1 hence we need to subtract one to get lane ID.
      const auto way = __ffs(bitmap) - 1;
#endif
      if (i == threadIdx.x) {
        cache_location = cache_set * kWarpSize + way;
      }
      n_hits++;
    }
  }

  const int32_t n = n0 + threadIdx.x;
  if (n < N) {
    lxu_cache_locations[n] = cache_location;
  }
  if (gather_cache_stats && threadIdx.x == 0 && n_indices > n_hits) {
    atomicAdd(
        &uvm_cache_stats[uvm_cache_stats_index::num_conflict_misses],
        (n_indices - n_hits));
  }
}

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void direct_mapped_lxu_cache_lookup_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    int64_t invalid_index,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations) {
  const int32_t C = lxu_cache_state.size(0);
  const int32_t N = linear_cache_indices.size(0);

  CUDA_KERNEL_LOOP(n, N) {
    int32_t cache_location = kCacheLocationMissing;
    const auto slot = 0;

    const int64_t idx = linear_cache_indices[n];
    if (idx == invalid_index) {
      continue;
    }

    const int32_t cache_set = cache_slot(idx, C);
    const bool found =
        (::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx);
    if (found) {
      cache_location = cache_set;
    }
    lxu_cache_locations[n] = cache_location;
  }
}

} // namespace

DLL_PUBLIC Tensor lxu_cache_lookup_cuda(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    int64_t invalid_index,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_cache_indices, lxu_cache_state);
  Tensor uvm_cache_stats_ =
      at::empty({0}, linear_cache_indices.options().dtype(at::kInt));
  if (gather_cache_stats) {
    TORCH_CHECK(uvm_cache_stats.has_value());
    uvm_cache_stats_ = uvm_cache_stats.value();
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(linear_cache_indices.get_device());

  const auto N = linear_cache_indices.numel();
  auto lxu_cache_locations = empty_like(
      linear_cache_indices, linear_cache_indices.options().dtype(at::kInt));
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return lxu_cache_locations;
  }

  const dim3 threads(kWarpSize, kMaxThreads / kWarpSize);
  const dim3 blocks(div_round_up(N, kMaxThreads));

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(), "lxu_cache_lookup_cuda", [&] {
        lxu_cache_lookup_kernel<<<
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            linear_cache_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            invalid_index,
            lxu_cache_locations
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            gather_cache_stats,
            uvm_cache_stats_
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return lxu_cache_locations;
}

DLL_PUBLIC Tensor direct_mapped_lxu_cache_lookup_cuda(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    int64_t invalid_index) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_cache_indices, lxu_cache_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(linear_cache_indices.get_device());

  const auto N = linear_cache_indices.numel();
  auto lxu_cache_locations = empty_like(
      linear_cache_indices, linear_cache_indices.options().dtype(at::kInt));
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return lxu_cache_locations;
  }

  const dim3 blocks(div_round_up(N, kMaxThreads));

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(),
      "direct_mapped_lxu_cache_lookup_cuda",
      [&] {
        direct_mapped_lxu_cache_lookup_kernel<<<
            blocks,
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            linear_cache_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            invalid_index,
            lxu_cache_locations
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return lxu_cache_locations;
}

int get_sm_count_() {
  cudaDeviceProp* deviceProp =
      at::cuda::getDeviceProperties(c10::cuda::current_device());
  return deviceProp->multiProcessorCount;
}

__global__ __launch_bounds__(kMaxThreads) void get_cache_indices_kernel(
    int32_t blocks_per_table,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_indices_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        logical_table_ids,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        buffer_ids,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        linear_cache_indices) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  const int32_t t_i = blockIdx.x / blocks_per_table;
  const int32_t threads_per_table = blocks_per_table * blockDim.x;
  const int32_t idx_table = index % threads_per_table;
  const int32_t logical_id = logical_table_ids[t_i];
  const int32_t buffer_id = buffer_ids[t_i];

  const int64_t num_indices =
      pruned_indices_offsets[buffer_id + 1] - pruned_indices_offsets[buffer_id];

  if (num_indices <= 0) {
    return;
  }

  const int64_t indices_per_thread =
      div_round_up(num_indices, threads_per_table);
  const int64_t start = idx_table * indices_per_thread;
  const int64_t end = min(start + indices_per_thread, num_indices);

  if (start >= num_indices) {
    return;
  }

  const int64_t pruned_indices_offset = pruned_indices_offsets[buffer_id];
  const int64_t* pruned_indices_table = &pruned_indices[pruned_indices_offset];
  int64_t* linear_cache_indices_table =
      &linear_cache_indices[pruned_indices_offset];

  const auto max_offset =
      __ldg(&cache_hash_size_cumsum[cache_hash_size_cumsum.size(0) - 1]);
  const auto curr_offset = __ldg(&cache_hash_size_cumsum[logical_id]);

  for (int64_t i = start; i < end; i++) {
    if (curr_offset >= 0) {
      linear_cache_indices_table[i] = curr_offset + pruned_indices_table[i];
    } else {
      linear_cache_indices_table[i] = max_offset;
    }
  }
}

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kMaxThreads) void reset_weight_momentum_kernel(
    int32_t blocks_per_table,
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    at::PackedTensorAccessor64<
        at::acc_type<cache_t, true>,
        1,
        at::RestrictPtrTraits> momentum1_dev,
    at::PackedTensorAccessor64<
        at::acc_type<cache_t, true>,
        1,
        at::RestrictPtrTraits> momentum1_uvm,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        momentum1_placements,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        momentum1_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        pruned_indices_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        logical_table_ids,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        buffer_ids,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  const int32_t t_i = blockIdx.x / blocks_per_table;
  const int32_t buffer_id = buffer_ids[t_i];
  const int64_t num_indices =
      pruned_indices_offsets[buffer_id + 1] - pruned_indices_offsets[buffer_id];

  if (num_indices <= 0) {
    return;
  }

  const int32_t logical_id = logical_table_ids[t_i];
  int32_t D = D_offsets[logical_id + 1] - D_offsets[logical_id];
  const int32_t chunk4s_per_row = D / 4;
  const int64_t total_chunk4s_per_table = num_indices * chunk4s_per_row;

  const int32_t threads_per_table = blocks_per_table * blockDim.x;
  const int64_t chunk4s_per_thread =
      div_round_up(total_chunk4s_per_table, threads_per_table);
  const int32_t idx_table = index % threads_per_table;
  const int64_t start = idx_table * chunk4s_per_thread;
  const int64_t end = min(start + chunk4s_per_thread, total_chunk4s_per_table);

  if (start >= total_chunk4s_per_table) {
    return;
  }

  int32_t D_emb = D;
  if (std::is_same<emb_t, uint8_t>::value) {
    D_emb += kINT8QparamsBytes;
  }

  at::acc_type<cache_t, true>* __restrict__ momentum1;
  const auto momentum1_placement =
      static_cast<PlacementType>(momentum1_placements[logical_id]);
  int64_t momentum1_offset = momentum1_offsets[logical_id];
  if (momentum1_placement == PlacementType::DEVICE) {
    momentum1 = &momentum1_dev[momentum1_offset];
  } else {
    momentum1 = &momentum1_uvm[momentum1_offset];
  }

  emb_t* __restrict__ weights{nullptr};
  cache_t* __restrict__ cache_weights{nullptr};
  const auto weights_placement =
      static_cast<PlacementType>(weights_placements[logical_id]);
  int64_t weights_offset = weights_offsets[logical_id];

  const int64_t pruned_indices_offset = pruned_indices_offsets[buffer_id];
  const int64_t* pruned_indices_table = &pruned_indices[pruned_indices_offset];

  for (int64_t i = start; i < end; i++) {
    int64_t idx = i / chunk4s_per_row;
    int64_t pruned_index = pruned_indices_table[idx];

    if (weights_placement == PlacementType::DEVICE) {
      weights = &dev_weights[weights_offset + pruned_index * D_emb];
    } else {
      weights = &uvm_weights[weights_offset + pruned_index * D_emb];
    }
    if (weights_placement == PlacementType::MANAGED_CACHING) {
      int32_t cache_idx = lxu_cache_locations[pruned_indices_offset + idx];
      if (cache_idx != kCacheLocationMissing) {
        cache_weights = &lxu_cache_weights[cache_idx][0];
      }
    }

    auto weight_row_template =
        WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
            weights, cache_weights, D, nullptr);

    // reset momentum1
    const int32_t d = (i % chunk4s_per_row) * 4;
    if (d == 0) {
      momentum1[pruned_index] = 0;
    }

    // reset weight
    float2 qparams_new = {1.0, 0.0}; // scaler=1.0, and offset=0.0, for int8.
    Vec4T<at::acc_type<cache_t, true>> weight_new; // 0 weight
    weight_row_template.store(
        weight_new,
        d,
        qparams_new); // qparams_new not used if type is not int8
  }
}

DLL_PUBLIC void reset_weight_momentum_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor momentum1_dev,
    Tensor momentum1_uvm,
    Tensor momentum1_placements,
    Tensor momentum1_offsets,
    Tensor D_offsets,
    Tensor pruned_indices,
    Tensor pruned_indices_offsets,
    Tensor logical_table_ids,
    Tensor buffer_ids,
    Tensor cache_hash_size_cumsum,
    Tensor lxu_cache_state,
    int64_t total_cache_hash_size) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      dev_weights,
      uvm_weights,
      lxu_cache_weights,
      weights_placements,
      weights_offsets,
      momentum1_dev,
      momentum1_uvm,
      momentum1_placements,
      momentum1_offsets,
      D_offsets,
      pruned_indices,
      pruned_indices_offsets,
      logical_table_ids,
      buffer_ids,
      cache_hash_size_cumsum,
      lxu_cache_state);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(dev_weights.get_device());

  const int64_t num_pruned_indices = pruned_indices.size(0);
  const int32_t num_pruned_tables = buffer_ids.size(0);
  const int32_t blocks_per_table = get_sm_count_();

  auto lxu_cache_locations =
      at::zeros({num_pruned_indices}, pruned_indices.options().dtype(at::kInt));
  lxu_cache_locations.fill_(kCacheLocationMissing);

  if (total_cache_hash_size > 0) {
    // Get corresponding cache indices of pruned indices
    auto linear_cache_indices = at::zeros(
        {num_pruned_indices}, pruned_indices.options().dtype(at::kLong));

    get_cache_indices_kernel<<<
        num_pruned_tables * blocks_per_table,
        kMaxThreads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        blocks_per_table,
        cache_hash_size_cumsum
            .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
        pruned_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
        pruned_indices_offsets
            .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
        logical_table_ids
            .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
        buffer_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
        linear_cache_indices
            .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Look up cache locations
    Tensor uvm_cache_stats =
        at::empty({0}, lxu_cache_weights.options().dtype(at::kInt));
    lxu_cache_locations = lxu_cache_lookup_cuda(
        linear_cache_indices,
        lxu_cache_state,
        total_cache_hash_size,
        false, // gather_cache_stats
        uvm_cache_stats);
  }

  // Reset weight and momentum of pruned rows
  DISPATCH_EMB_CACHE_TYPES(
      dev_weights.scalar_type(),
      lxu_cache_weights.scalar_type(),
      "reset_weight_momentum_kernel",
      ([&] {
        reset_weight_momentum_kernel<emb_t, cache_t><<<
            num_pruned_tables * blocks_per_table,
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            blocks_per_table,
            dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
            uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
            weights_placements
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            weights_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            momentum1_dev.packed_accessor64<
                at::acc_type<cache_t, true>,
                1,
                at::RestrictPtrTraits>(),
            momentum1_uvm.packed_accessor64<
                at::acc_type<cache_t, true>,
                1,
                at::RestrictPtrTraits>(),
            momentum1_placements
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            momentum1_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            pruned_indices
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            pruned_indices_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            logical_table_ids
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            buffer_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_locations
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}
